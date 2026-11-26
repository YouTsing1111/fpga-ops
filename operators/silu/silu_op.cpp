#include "silu_op.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "../../app/fp16_utils.h"
#include "../../iq/iq_id_defs.h"
#include "../../iq/modules/fpadd_iq.h"
#include "../../iq/modules/fpmul_iq.h"
#include "../../iq/modules/fptofp_iq.h"
#include "../../iq/modules/intadd_iq.h"
#include "../../iq/modules/logical_iq.h"
#include "../common/op_registry.h"
#include "../exp/exp_decompose.h"

namespace aiacc {
namespace op {

using namespace exp_iq;
using namespace exp_math;

namespace {

constexpr uint8_t kRtovrOutFpToFpS = 7;
constexpr uint8_t kRtovrInDrFpToFpD = 6;

constexpr uint8_t kBankFp16In = 2;     // Phase0: FP16 x @bank2（与 Exp/LSU LDB 对齐）
constexpr uint8_t kBankExpIn = 2;      // Phase0 out / Phase1 Exp in: FP32 -x
constexpr uint8_t kBankExpOut = 3;     // Phase1: e^{-x}；Phase2: 被 D=1+e^{-x} 覆写
constexpr uint8_t kBankConst = 4;      // Phase2 常数段（与 Exp 共享 bank4）
constexpr uint32_t kBankConstZero = 6;   // bank4@6 = int_zero = 0x00000000（Bank7 清零源）
constexpr uint32_t kBankConstOne = 4;    // bank4@4 = 1.0（Exp 常数表已有）
constexpr uint32_t kBankConstTwo = 8;    // bank4@8 = 2.0（Exp 表中 Softmax 保留槽，LSU 覆写）
constexpr uint32_t kBankConstMagic = 9;  // bank4@9 = 0x7EF311C7（Exp 表中 Softmax 保留槽，LSU 覆写）
constexpr uint32_t kBankConstNegOne = 14;  // bank4@14 = -1.0f（Phase0/dense upcast 取负）
constexpr uint8_t kBankDenom = 3;      // ★ SiLU: D=1+e^{-x} 覆写 bank3（e^{-x} 已死），
                                       //   把 bank5 让给 x_fp32 全程存活（sigmoid 的 D 在 bank5）
constexpr uint8_t kBankXfp32 = 5;      // Phase0 写入 x_fp32，此后无人覆写，活到最终乘
constexpr uint8_t kBankNewton = 6;     // Newton x_n → σ(x) → 最终被 silu_fp32 覆写
constexpr uint32_t kNewtonMagic = 0x7EF311C7u;
constexpr uint8_t kBankFinalOut = 7;   // Phase2: 刷0 + FPtoFP 输出 @bank7

uint8_t get_smc_id_en(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("smc_id_en");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty()) return 0;
    return static_cast<uint8_t>(it->second.para_data_i32[0] & 0x1F);
}

uint32_t get_newton_iters(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("newton_iters");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty())
        return kSiluDefaultNewtonIters;
    int v = it->second.para_data_i32[0];
    if (v < 1) v = 1;
    if (v > 3) v = 3;
    return static_cast<uint32_t>(v);
}

bool is_fp16_req(const app::AppOperatorRequest& req) {
    auto it = req.para_inputs.find("input_tensor");
    return it != req.para_inputs.end() &&
           it->second.para_dtype == app::AppDataType::FLOAT16;
}

std::vector<float> collect_x(const app::AppOperatorRequest& req, uint32_t n) {
    std::vector<float> out(n, 0.f);
    auto it = req.para_inputs.find("input_tensor");
    if (it == req.para_inputs.end()) return out;
    const auto& t = it->second;
    if (!t.para_data_f32.empty()) {
        for (uint32_t i = 0; i < n && i < t.para_data_f32.size(); ++i)
            out[i] = t.para_data_f32[i];
    }
    return out;
}

/** Build FP32 Exp request: input = -x (already upconverted from FP16 semantics). */
app::AppOperatorRequest make_exp_neg_x_request(const app::AppOperatorRequest& silu_req) {
    app::AppOperatorRequest exp_req = silu_req;
    exp_req.para_operator = "Exp";

    const uint32_t n = silu_req.para_dimensions.para_n * silu_req.para_dimensions.para_c *
                       silu_req.para_dimensions.para_h * silu_req.para_dimensions.para_w;
    auto x = collect_x(silu_req, n);

    // Force FP32 for Exp path (reuse proven FP32 Exp).
    app::AppTensorDesc input;
    input.para_name = "input_tensor";
    input.para_dtype = app::AppDataType::FLOAT32;
    input.para_shape = {silu_req.para_dimensions.para_n, silu_req.para_dimensions.para_c,
                        silu_req.para_dimensions.para_h, silu_req.para_dimensions.para_w};
    input.para_data_f32.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        // FP16 I/O semantics: round-trip x through half then negate for exp(-x).
        const float x_h = app::float16_to_float32(app::float32_to_float16(x[i]));
        input.para_data_f32[i] = -x_h;
    }
    exp_req.para_inputs["input_tensor"] = input;
    return exp_req;
}

float newton_recip_magic(float d) {
    uint32_t bits = 0;
    std::memcpy(&bits, &d, sizeof(bits));
    bits = kNewtonMagic - bits;
    float x0 = 0.f;
    std::memcpy(&x0, &bits, sizeof(x0));
    return x0;
}

float newton_recip(float d, uint32_t iters) {
    float x = newton_recip_magic(d);
    for (uint32_t i = 0; i < iters; ++i)
        x = x * (2.0f - d * x);
    return x;
}

void append_sched(std::vector<SchedEntry>& s, uint32_t cyc, uint64_t code) {
    s.push_back({cyc, code});
}

/** Hold RtoVR select every cycle in [c0, c1) — required; one-shot select is not enough. */
void hold_rtovr(std::vector<SchedEntry>& s, uint32_t c0, uint32_t c1, uint8_t src) {
    const uint64_t code = rtovr_code(src);
    for (uint32_t t = c0; t < c1; ++t) append_sched(s, t, code);
}

/** VIN/VOUT 直通开关（FPGA 稀疏掩码）：读 para_parameters["vin_vout"]，默认关。 */
bool vin_vout_mode(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("vin_vout");
    return it != req.para_parameters.end() && !it->second.para_data_i32.empty() &&
           it->second.para_data_i32[0] != 0;
}

/** vv 使能流：body 全程 0x3F（vld + smc_id_en=31 阈值），emitter 自动压成 loops。 */
void fill_vv_enable_streams(std::vector<SchedEntry>& vin0,
                            std::vector<SchedEntry>& vin1,
                            std::vector<SchedEntry>& vout0,
                            uint32_t body) {
    for (uint32_t t = 0; t < body; ++t) {
        append_sched(vin0, t, kVinVoutEnableCode);
        append_sched(vin1, t, kVinVoutEnableCode);
        append_sched(vout0, t, kVinVoutEnableCode);
    }
}

/**
 * FP binary op micro-schedule (align Exp Horner phases):
 *   read@0 → compute@3 → DR consumer gap → UR write@13
 * RtoVR ports held across the whole window.
 */
constexpr uint32_t kSiluUrSettle = 3;      // == kUrReadToComputeIssue
constexpr uint32_t kSiluFpOpSpan = 13;     // == kStageCycles
constexpr uint32_t kSiluWritePhase = 10;   // == kFinalWritePhase within span
constexpr uint32_t kSiluFptofpSpan = 13;
constexpr uint32_t kSiluFptofpWrite = 10;

uint16_t pack_intadd_sub_code(uint8_t smc_en) {
    iq::IntAddCru iadd{};
    iadd.para_inst_valid = true;
    iadd.para_precision_s0 = 3;
    iadd.para_precision_s1 = 3;
    iadd.para_precision_s2 = 3;
    iadd.para_sign_s0 = true;
    iadd.para_sign_s1 = false;
    iadd.para_sign_s2 = false;
    iadd.para_sign_d = true;
    iadd.para_is_sub = true;
    return static_cast<uint16_t>(iq::IntAddIq::pack_cru_intadd(iadd)) |
           (static_cast<uint16_t>(smc_en & 0x1Fu) << 13);
}

/** 片上 x0 = float_bits(MAGIC - float_bits(denom))，写 bank6（对齐 Sigmoid/Softmax Norm）。 */
uint32_t append_magic_x0_prologue(SiluPostInstructions& out,
                                  uint32_t c,
                                  uint8_t smc_id_en,
                                  uint64_t intadd_code) {
    const uint32_t S = c;
    const uint32_t end = S + kSiluFpOpSpan;

    append_sched(out.sched_ur_r0, S, ur_read_code(kBankDenom, 0, smc_id_en));
    append_sched(out.sched_ur_r1, S,
                 ur_read_code(kBankConst, kBankConstMagic, smc_id_en));

    if (out.para_use_vin_vout) {
        // vv：D→VIN0、MAGIC→VIN1（捕获@S+1，消费@S+4）；x0 经 VOUT0 回写
        append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
        append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));
        hold_rtovr(out.sched_rtovr_0, S, S + 5, kRtovrInDrVin1D);  // S0 = MAGIC
        hold_rtovr(out.sched_rtovr_1, S, S + 5, kRtovrInDrVin0D);  // S1 = bits(D)
        append_sched(out.sched_intadd, S + 4, intadd_code);
        append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrIntAddD0));
        hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
        append_sched(out.sched_ur_w0, S + 11,
                     ur_write_code(kBankNewton, 0, smc_id_en));
        return S + kSiluFpOpSpan + 2;
    }

    hold_rtovr(out.sched_rtovr_0, S, end, kRtovrInUrRdata0 + 1);
    hold_rtovr(out.sched_rtovr_1, S, end, kRtovrInUrRdata0);

    const uint32_t iadd_issue = S + kSiluUrSettle;
    append_sched(out.sched_intadd, iadd_issue, intadd_code);

    const uint32_t iadd_dr = iadd_issue + kIntaddLatency;
    hold_rtovr(out.sched_rtovr_35, iadd_dr, end, kRtovrInDrIntAddD0);
    append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                 ur_write_code(kBankNewton, 0, smc_id_en));

    return S + kSiluFpOpSpan + 2;
}

}  // namespace

// ---------------------------------------------------------------------------
// SiluPostInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SiluPostInstructions::serialize() const {
    std::vector<uint8_t> r;
    auto push = [&](uint32_t v) {
        r.push_back(v & 0xFF);
        r.push_back((v >> 8) & 0xFF);
        r.push_back((v >> 16) & 0xFF);
        r.push_back((v >> 24) & 0xFF);
    };
    push(para_body_cycles);
    push(para_total_steps);
    push(para_total_elements);
    push(para_total_cycles);
    push(para_newton_iters);
    return r;
}

std::vector<IqInstructionFile> SiluPostInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t steps = para_total_steps;
    const uint32_t body = para_body_cycles;

    auto add_iq = [&](const std::vector<SchedEntry>& sched, uint32_t iq_id,
                      const std::string& name, uint32_t code_bits, bool addr_inc) {
        if (sched.empty()) return;
        IqInstructionFile f;
        f.para_iq_id = iq_id;
        f.para_iq_name = name;
        f.para_width_bits = 41 + code_bits;
        f.para_data = emit_exp_embed_iq(sched, code_bits, addr_inc, steps, body,
                                        "SiluPost");
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17, true);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10, false);
    add_iq(sched_fptofp, iq::IQ_ID_FP_TO_FP, iq::IQ_NAME_FP_TO_FP, 10, false);
    add_iq(sched_intadd, iq::IQ_ID_INTADD, iq::IQ_NAME_INTADD, 18, false);
    add_iq(sched_rtovr_0, iq::IQ_ID_RTOVR_BASE + 0, "rtovr-0", 7, false);
    add_iq(sched_rtovr_1, iq::IQ_ID_RTOVR_BASE + 1, "rtovr-1", 7, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false);
    add_iq(sched_rtovr_7, iq::IQ_ID_RTOVR_BASE + kRtovrOutFpToFpS, "rtovr-7", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    // 同一份微码须同时过全交叉仿真与 FPGA 稀疏掩码：vv 开时多出 LOGIC + 六份文件
    if (para_use_vin_vout) {
        add_iq(sched_logic, iq::IQ_ID_LOGIC, iq::IQ_NAME_LOGIC, 11, false);
        add_iq(sched_rtovr_11, iq::IQ_ID_RTOVR_BASE + 11, "rtovr-11", 7, false);
        add_iq(sched_rtovr_12, iq::IQ_ID_RTOVR_BASE + 12, "rtovr-12", 7, false);
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
        add_iq(sched_vin0,  iq::IQ_ID_VIN0,  "68_vin0",  7, false);
        add_iq(sched_vin1,  iq::IQ_ID_VIN1,  "69_vin1",  7, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false);
    }
    return files;
}

// ---------------------------------------------------------------------------
// SiluDenseAdapterInstructions（dense 8×fp16/word ⇄ fp32 适配，需求形状用）
// ---------------------------------------------------------------------------

std::vector<uint8_t> SiluDenseAdapterInstructions::serialize() const {
    std::vector<uint8_t> r;
    auto push = [&](uint32_t v) {
        r.push_back(v & 0xFF);
        r.push_back((v >> 8) & 0xFF);
        r.push_back((v >> 16) & 0xFF);
        r.push_back((v >> 24) & 0xFF);
    };
    push(para_body_cycles);
    push(para_total_steps);
    push(para_total_elements);
    push(para_total_cycles);
    return r;
}

std::vector<IqInstructionFile> SiluDenseAdapterInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t steps = para_total_steps;
    const uint32_t body = para_body_cycles;

    auto add_iq = [&](const std::vector<SchedEntry>& sched, uint32_t iq_id,
                      const std::string& name, uint32_t code_bits, bool addr_inc) {
        if (sched.empty()) return;
        IqInstructionFile f;
        f.para_iq_id = iq_id;
        f.para_iq_name = name;
        f.para_width_bits = 41 + code_bits;
        f.para_data = emit_exp_embed_iq(sched, code_bits, addr_inc, steps, body,
                                        "SiluDenseAdapter");
        files.push_back(f);
    };

    // ur_r0/ur_w0 数据口随 step 递增（烘焙基址 + lev2 计数；同通道多基址已验证）；
    // ur_r1 只读 bank4 常数（固定地址）→ 不递增。
    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8, false);
    add_iq(sched_fptofp, iq::IQ_ID_FP_TO_FP, iq::IQ_NAME_FP_TO_FP, 10, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_7, iq::IQ_ID_RTOVR_BASE + kRtovrOutFpToFpS, "rtovr-7", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    if (para_use_vin_vout) {
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
        add_iq(sched_vin0,  iq::IQ_ID_VIN0,  "68_vin0",  7, false);
        add_iq(sched_vin1,  iq::IQ_ID_VIN1,  "69_vin1",  7, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false);
    }
    return files;
}

SiluDenseAdapterInstructions SiluOp::compile_dense_upcast(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t hi_base) {
    // body=60 = 4 块×15（cvt-lo@0, mul-lo@15, cvt-hi@30, mul-hi@45）。
    // 每 dense word w：
    //   bank7@w --FPtoFP src_low-->  bank5@w          (x_fp32，elems 8w+0..3)
    //   bank5@w × (-1@bank4:14)  --> bank2@w          (-x，供 Exp)
    //   bank7@w --FPtoFP src_high--> bank5@(hi_base+w)（elems 8w+4..7）
    //   bank5@(hi_base+w) × (-1) --> bank2@(hi_base+w)
    // mul 读 bank5@addr 于 S+15，距 cvt 写（vv@S+12/非vv@S+10）≥3 拍，与
    // SigmoidOp::compile_upcast 相邻块间依赖完全一致（已验证）。
    SiluDenseAdapterInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;

    iq::FpToFpCru up[2];
    for (auto& u : up) {
        u.para_inst_valid = true;
        u.para_src_is_32b = false;   // FP16
        u.para_dst_is_32b = true;    // → FP32
        u.para_dst_high = false;
    }
    up[0].para_src_high = false;     // src_low  → elems 8w+0..3
    up[1].para_src_high = true;      // src_high → elems 8w+4..7
    uint64_t up_code[2];
    for (int h = 0; h < 2; ++h)
        up_code[h] = (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(up[h])) |
                      (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = true;
    mul.para_dst_precision_is_32b = true;
    mul.para_smc_id_en = smc_id_en;
    const uint64_t mul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    // cvt 块：读 bank7@w（addr_inc +w），写 bank5@(dst_base+w)
    auto emit_cvt = [&](uint32_t S, uint64_t code, uint32_t dst_base) {
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankFinalOut, 0, smc_id_en));
        if (vv) {
            // VIN0 捕获@S+1，消费@S+4；VOUT0 捕获@S+9（FPTOFP 延迟边界），写@S+12
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
            hold_rtovr(out.sched_rtovr_7, S, S + 5, kRtovrInDrVin0D);
            append_sched(out.sched_fptofp, S + 4, code);
            append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrFpToFpD));
            hold_rtovr(out.sched_rtovr_35, S + 9, S + 12, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 12,
                         ur_write_code(kBankXfp32, dst_base, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_7, S, S + kSiluFptofpSpan, kRtovrInUrRdata0);
            append_sched(out.sched_fptofp, S + kSiluUrSettle, code);
            hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                       S + kSiluFptofpSpan, kRtovrInDrFpToFpD);
            append_sched(out.sched_ur_w0, S + kSiluFptofpWrite,
                         ur_write_code(kBankXfp32, dst_base, smc_id_en));
        }
    };

    // mul 块：读 bank5@(addr+w) 与 bank4@14(-1.0，固定地址)，写 bank2@(addr+w)
    auto emit_mul = [&](uint32_t S, uint32_t addr) {
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankXfp32, addr, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstNegOne,
                                                      smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
            append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));
            hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
            hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
            append_sched(out.sched_fpmul, S + 4, mul_code);
            append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));
            hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 11,
                         ur_write_code(kBankExpIn, addr, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_3, S, S + kSiluFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 1);
            append_sched(out.sched_fpmul, S + kSiluUrSettle, mul_code);
            hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                       S + kSiluFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                         ur_write_code(kBankExpIn, addr, smc_id_en));
        }
    };

    emit_cvt(0,  up_code[0], 0);         // lo: bank5@w
    emit_mul(15, 0);                     //     bank2@w
    emit_cvt(30, up_code[1], hi_base);   // hi: bank5@(hi_base+w)
    emit_mul(45, hi_base);               //     bank2@(hi_base+w)

    out.para_body_cycles = 60;
    out.para_total_cycles = 60 * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }
    std::cout << "[Silu] dense upcast body=60 cyc/word, steps=" << total_steps
              << " hi_base=" << hi_base << (vv ? " [vin_vout]" : "") << std::endl;
    return out;
}

SiluDenseAdapterInstructions SiluOp::compile_dense_downcast_hi(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t src_base) {
    // body=15：读 bank6@(src_base+w)（silu_fp32 高半组，elems 8w+4..7）
    // → FPtoFP fp32→fp16 dest_high → bank7@w 高半（wbe=0xCCCC）。
    // dest_high 的 D 线低16位 = output_data_r 残留，被 wbe 掩掉不进 SRAM；
    // 低半由 SiluPost 的 dest_low 写（wbe=0x3333，前置整字刷0）备好。
    static constexpr uint16_t kWbeHigh16 = 0xCCCCu;
    SiluDenseAdapterInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;

    iq::FpToFpCru cvt{};
    cvt.para_inst_valid = true;
    cvt.para_src_is_32b = true;    // FP32
    cvt.para_dst_is_32b = false;   // → FP16
    cvt.para_src_high = false;
    cvt.para_dst_high = true;      // dest_high：写每 lane 高16位
    const uint64_t dn_code =
        (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(cvt)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    const uint32_t S = 0;
    append_sched(out.sched_ur_r0, S, ur_read_code(kBankNewton, src_base, smc_id_en));
    if (vv) {
        append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
        hold_rtovr(out.sched_rtovr_7, S, S + 5, kRtovrInDrVin0D);
        append_sched(out.sched_fptofp, S + 4, dn_code);
        append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrFpToFpD));
        hold_rtovr(out.sched_rtovr_35, S + 9, S + 12, kRtovrInDrVout0D);
        append_sched(out.sched_ur_w0, S + 12,
                     ur_write_code(kBankFinalOut, 0, smc_id_en, kWbeHigh16));
    } else {
        hold_rtovr(out.sched_rtovr_7, S, S + kSiluFptofpSpan, kRtovrInUrRdata0);
        append_sched(out.sched_fptofp, S + kSiluUrSettle, dn_code);
        hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                   S + kSiluFptofpSpan, kRtovrInDrFpToFpD);
        append_sched(out.sched_ur_w0, S + kSiluFptofpWrite,
                     ur_write_code(kBankFinalOut, 0, smc_id_en, kWbeHigh16));
    }

    out.para_body_cycles = 15;
    out.para_total_cycles = 15 * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }
    std::cout << "[Silu] dense downcast-hi body=15 cyc/word, steps=" << total_steps
              << " src_base=" << src_base << (vv ? " [vin_vout]" : "") << std::endl;
    return out;
}

// ---------------------------------------------------------------------------
// SiluInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SiluInstructions::serialize() const {
    std::vector<uint8_t> r = stage0_exp.serialize();
    auto push = [&](uint32_t v) {
        r.push_back(v & 0xFF);
        r.push_back((v >> 8) & 0xFF);
        r.push_back((v >> 16) & 0xFF);
        r.push_back((v >> 24) & 0xFF);
    };
    push(para_num_stages);
    push(para_iq_depth_limit);
    push(para_phase0_max_insts);
    push(para_stage0_max_insts);
    push(para_stage1_max_insts);
    push(1u);  // onchip_upcast 恒为 1（复用 SigmoidOp::compile_upcast）
    return r;
}

std::vector<IqInstructionFile> SiluInstructions::get_iq_files() const {
    return stage0_exp.get_iq_files();
}

// ---------------------------------------------------------------------------
// SiluOp
// ---------------------------------------------------------------------------

std::string SiluOp::get_name() const { return "Silu"; }

uint32_t SiluOp::max_iq_inst_count(const std::vector<IqInstructionFile>& files) {
    uint32_t mx = 0;
    for (const auto& f : files) {
        const uint32_t nbytes = (f.para_width_bits + 7) / 8;
        if (nbytes == 0) continue;
        const uint32_t n = static_cast<uint32_t>(f.para_data.size() / nbytes);
        if (n > mx) mx = n;
    }
    return mx;
}

SiluPostInstructions SiluOp::compile_postprocess(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t newton_iters) {
    // Stage1 body (EMBED fold via emit_exp_embed_iq): IQ depth ≈ body, independent of steps.
    // Pipeline (FP32), RtoVR held every cycle (Exp Horner style — one-shot select is unsafe):
    //   +1:     e^{-x} + 1 → D @bank3（覆写 e^{-x}，已死）
    //   magic:  x0 = float_bits(MAGIC - bits(D)) → bank6（片上，对齐 Sigmoid）
    //   Newton: x = x*(2 - D*x)  (iters)
    //           [Window 1 zero-overhead zeroing]: ur_r1 read bank4@6 (=0) -> ur_w0 write bank7
    //   ★ 最终乘: silu = x_fp32(bank5) * σ(bank6) → bank6（覆写 σ）
    //   FPtoFP: FP32 → FP16 (dest_low native layout) → bank7 (已清零的高16位保持0x0000)
    //
    // Port map (multiword addr_inc): R0/R2/W0 inc; R1 fixed (const bank4).

    SiluPostInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_newton_iters = newton_iters;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = true;
    mul.para_dst_precision_is_32b = true;
    mul.para_smc_id_en = smc_id_en;
    const uint64_t mul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    iq::FpAddCru add;
    add.para_inst_valid = true;
    add.para_src_precision_32b = true;
    add.para_dst_precision_32b = true;
    add.para_update_status = false;
    add.para_is_sub = false;
    const uint64_t add_code =
        (static_cast<uint64_t>(iq::FpAddIq::pack_cru_fpadd(add)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    iq::FpAddCru sub;
    sub.para_inst_valid = true;
    sub.para_src_precision_32b = true;
    sub.para_dst_precision_32b = true;
    sub.para_update_status = false;
    sub.para_is_sub = true;
    const uint64_t sub_code =
        (static_cast<uint64_t>(iq::FpAddIq::pack_cru_fpadd(sub)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    iq::FpToFpCru cvt{};
    cvt.para_inst_valid = true;
    cvt.para_src_is_32b = true;
    cvt.para_dst_is_32b = false;  // → FP16
    cvt.para_src_high = false;
    cvt.para_dst_high = false;  // dest_low: FP16 in low-16 of each FP32 lane
    // CRU_FPTOFP_W=10: bits[4:0] opcode, bits[9:5] smc_id_en
    const uint64_t fptofp_code =
        (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(cvt)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    // vv 专用：bank7 刷零的合法通路（UR→UVR 直拷在稀疏掩码非法）——
    // int_zero 经 VIN1 进 LOGIC，OR(0,0)=0 直通，再经 VOUT0 写 bank7。
    iq::LogicalCru lor{};
    lor.para_inst_valid = true;
    lor.para_precision_32b = true;
    lor.para_opcode = iq::LogicalOp::OR;   // OR(x,x)=x 直通
    lor.para_smc_id_en = smc_id_en;
    const uint64_t or_code = iq::LogicalIq::pack_cru_logical(lor);

    auto emit_fptofp_to_bank7 = [&](SiluPostInstructions& o, uint32_t& c,
                                    uint8_t src_bank) {
        // read@0, fptofp@3, write@10 within span (softmax-like margin)
        // dest=bank7: bank7 已被 Newton window1 的零开销刷0，高16位为0x0000，满足 strict golden
        // 注意 fptofp_array.v dest_low 语义：FP32→FP16 时 D 线高16位 =
        // output_data_r 寄存器残留，并非 SRAM 原值！
        // 故此处 wbe 必须只开每 lane 低 2 字节（0x3333），不让残留进 SRAM；
        // 高 16 位由 Newton window1 的整字清零写提前备好。
        static constexpr uint16_t kSiluFinalWbeLow16 = 0x3333u;
        append_sched(o.sched_ur_r0, c, ur_read_code(src_bank, 0, smc_id_en));
        if (vv) {
            // vv：VIN0 捕获@c+1，FPTOFP@c+4；VOUT0 捕获@发射+5（FPTOFP 延迟边界，softmax 同款）
            append_sched(o.sched_rtovr_14, c + 1, rtovr_code(kRtovrInUrRdata0));
            hold_rtovr(o.sched_rtovr_7, c, c + 5, kRtovrInDrVin0D);
            append_sched(o.sched_fptofp, c + 4, fptofp_code);
            append_sched(o.sched_rtovr_16, c + 9, rtovr_code(kRtovrInDrFpToFpD));
            hold_rtovr(o.sched_rtovr_35, c + 9, c + 12, kRtovrInDrVout0D);
            append_sched(o.sched_ur_w0, c + 12,
                         ur_write_code(kBankFinalOut, 0, smc_id_en, kSiluFinalWbeLow16));
        } else {
            hold_rtovr(o.sched_rtovr_7, c, c + kSiluFptofpSpan, kRtovrInUrRdata0);
            append_sched(o.sched_fptofp, c + kSiluUrSettle, fptofp_code);
            hold_rtovr(o.sched_rtovr_35, c + kSiluUrSettle + kFpmulDrLatencyRtl,
                       c + kSiluFptofpSpan, kRtovrInDrFpToFpD);
            append_sched(o.sched_ur_w0, c + kSiluFptofpWrite,
                         ur_write_code(kBankFinalOut, 0, smc_id_en, kSiluFinalWbeLow16));
        }
        c += kSiluFptofpSpan;
    };

    uint32_t c = 0;

    // ---- +1: D = e^{-x} + 1.0 (FpAdd S0←UR0 e^{-x}, S1←UR1 1.0)；写回 bank3 覆写 e^{-x} ----
    {
        const uint32_t S = c;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankExpOut, 0, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstOne, smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← e^{-x}
            append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN1 ← 1.0
            hold_rtovr(out.sched_rtovr_5, S, S + 5, kRtovrInDrVin0D);
            hold_rtovr(out.sched_rtovr_6, S, S + 5, kRtovrInDrVin1D);
            append_sched(out.sched_fpadd, S + 4, add_code);
            append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpAddD));
            hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 11,
                         ur_write_code(kBankDenom, 0, smc_id_en));  // D → bank3
        } else {
            hold_rtovr(out.sched_rtovr_5, S, S + kSiluFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_6, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 1);
            append_sched(out.sched_fpadd, S + kSiluUrSettle, add_code);
            hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpaddDrLatencyRtl,
                       S + kSiluFpOpSpan, kRtovrInDrFpAddD);
            append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                         ur_write_code(kBankDenom, 0, smc_id_en));  // D → bank3（覆写 e^{-x}）
        }
        c = S + kSiluFpOpSpan + 2;  // UR write settle before magic / Newton reads denom
    }

    const uint64_t intadd_sub_code = pack_intadd_sub_code(smc_id_en);
    c = append_magic_x0_prologue(out, c, smc_id_en, intadd_sub_code);

    // ---- Newton iters ----
    // R0=denom (inc), R2=x_n (inc), R1=const 2.0 (no inc)
    for (uint32_t it = 0; it < newton_iters; ++it) {
        // t = denom * x_n
        {
            const uint32_t S = c;
            append_sched(out.sched_ur_r0, S, ur_read_code(kBankDenom, 0, smc_id_en));
            append_sched(out.sched_ur_r2, S, ur_read_code(kBankNewton, 0, smc_id_en));
            if (vv) {
                append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← D
                append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 2));  // VIN1 ← x_n
                hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
                hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
                append_sched(out.sched_fpmul, S + 4, mul_code);
                // FPMUL DR 粘滞，持窗到 u-block 消费（下一窗 FPADD S1）
                hold_rtovr(out.sched_rtovr_6, S + 8, S + kSiluFpOpSpan, kRtovrInDrFpMulD);
                // 注意：非 vv 在本窗零开销刷 bank7（UR→UVR 直拷）；该路由稀疏掩码非法，
                // vv 挪到 x-block 经 VIN1→LOGIC→VOUT0（见下）
            } else {
                hold_rtovr(out.sched_rtovr_3, S, S + kSiluFpOpSpan, kRtovrInUrRdata0);
                hold_rtovr(out.sched_rtovr_4, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 2);
                append_sched(out.sched_fpmul, S + kSiluUrSettle, mul_code);
                // Keep mul DR routed on S1 of next add; also park rtovr-6 on mul DR after issue
                hold_rtovr(out.sched_rtovr_6, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                           S + kSiluFpOpSpan, kRtovrInDrFpMulD);

                // Zero-overhead bank7 zeroing: 在 window1 ALU 空闲中，利用 ur_r1 读
                // bank4@6(int_zero=0x00000000, addr fixed) → rtovr_35(UR rdata1) → ur_w0(addr inc) 写 bank7。
                // 后续 FPtoFP dest_low 只写低16位，bank7 高16位保持 0x0000。
                append_sched(out.sched_ur_r1, S,
                             ur_read_code(kBankConst, kBankConstZero, smc_id_en));
                hold_rtovr(out.sched_rtovr_35, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 1);
                append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                             ur_write_code(kBankFinalOut, 0, smc_id_en));
            }
            c = S + kSiluFpOpSpan;
        }
        // u = 2 - t  (S0←UR1 const2, S1←mul DR). Re-read const; mul DR still held from above
        // → new window: read const, hold S0=UR1 & S1=mulDR, sub@3, write not needed (stay in DR)
        {
            const uint32_t S = c;
            append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstTwo, smc_id_en));
            if (vv) {
                append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN0 ← 2.0
                hold_rtovr(out.sched_rtovr_5, S, S + 5, kRtovrInDrVin0D);
                hold_rtovr(out.sched_rtovr_6, S, S + 5, kRtovrInDrFpMulD);  // t（DR 已粘滞）
                append_sched(out.sched_fpadd, S + 4, sub_code);
                hold_rtovr(out.sched_rtovr_4, S + 8, S + kSiluFpOpSpan, kRtovrInDrFpAddD);
            } else {
                hold_rtovr(out.sched_rtovr_5, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 1);
                hold_rtovr(out.sched_rtovr_6, S, S + kSiluFpOpSpan, kRtovrInDrFpMulD);
                append_sched(out.sched_fpadd, S + kSiluUrSettle, sub_code);
                // After sub, FpAdd DR holds u; route to mul S1 next
                hold_rtovr(out.sched_rtovr_4, S + kSiluUrSettle + kFpaddDrLatencyRtl,
                           S + kSiluFpOpSpan, kRtovrInDrFpAddD);
            }
            c = S + kSiluFpOpSpan;
        }
        // x = x_n * u  (S0←UR0 x_n, S1←add DR)
        {
            const uint32_t S = c;
            append_sched(out.sched_ur_r0, S, ur_read_code(kBankNewton, 0, smc_id_en));
            if (vv) {
                // vv 的 bank7 刷零挪进本窗：VIN1 捕 int_zero → LOGIC OR(0,0) → VOUT0。
                // VOUT0 串行复用：@S+8 捕 FPMUL_D(x) → 写 bank6@S+11；@S+9 捕 LOGIC_D(0)
                // → 写 bank7@S+12（VOUT0_D 于捕获+3 翻转到 0，两写各取其时）。
                append_sched(out.sched_ur_r1, S,
                             ur_read_code(kBankConst, kBankConstZero, smc_id_en));
                append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← x_n
                append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN1 ← 0
                hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
                hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrFpAddD);  // u（DR 已粘滞）
                append_sched(out.sched_fpmul, S + 4, mul_code);
                hold_rtovr(out.sched_rtovr_11, S, S + 5, kRtovrInDrVin1D);  // LOGIC S0 = 0
                hold_rtovr(out.sched_rtovr_12, S, S + 5, kRtovrInDrVin1D);  // LOGIC S1 = 0
                append_sched(out.sched_logic, S + 4, or_code);
                append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));
                append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrLogicD));
                hold_rtovr(out.sched_rtovr_35, S + 8, S + 12, kRtovrInDrVout0D);
                append_sched(out.sched_ur_w0, S + 11,
                             ur_write_code(kBankNewton, 0, smc_id_en));
                append_sched(out.sched_ur_w0, S + 12,
                             ur_write_code(kBankFinalOut, 0, smc_id_en));
            } else {
                hold_rtovr(out.sched_rtovr_3, S, S + kSiluFpOpSpan, kRtovrInUrRdata0);
                hold_rtovr(out.sched_rtovr_4, S, S + kSiluFpOpSpan, kRtovrInDrFpAddD);
                append_sched(out.sched_fpmul, S + kSiluUrSettle, mul_code);
                hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                           S + kSiluFpOpSpan, kRtovrInDrFpMulD);
                append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                             ur_write_code(kBankNewton, 0, smc_id_en));
            }
            c = S + kSiluFpOpSpan + 2;  // settle before next iter / 最终乘 reads σ
        }
    }

    // ---- ★ SiLU 最终乘：silu_fp32 = x_fp32(bank5) * σ(bank6) → bank6（覆写 σ）----
    // 镜像 Newton window1 的已验证模式：S0←UR rdata0，S1←UR rdata2，DR→ur_w0。
    // bank5 的 x_fp32 自 Phase0 写入后全程无人覆写（D 已挪 bank3）。
    {
        const uint32_t S = c;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankXfp32, 0, smc_id_en));
        append_sched(out.sched_ur_r2, S, ur_read_code(kBankNewton, 0, smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← x
            append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 2));  // VIN1 ← σ
            hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
            hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
            append_sched(out.sched_fpmul, S + 4, mul_code);
            append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));
            hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 11,
                         ur_write_code(kBankNewton, 0, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_3, S, S + kSiluFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSiluFpOpSpan, kRtovrInUrRdata0 + 2);
            append_sched(out.sched_fpmul, S + kSiluUrSettle, mul_code);
            hold_rtovr(out.sched_rtovr_35, S + kSiluUrSettle + kFpmulDrLatencyRtl,
                       S + kSiluFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_ur_w0, S + kSiluWritePhase,
                         ur_write_code(kBankNewton, 0, smc_id_en));
        }
        c = S + kSiluFpOpSpan + 2;  // settle 后 FPtoFP 读 bank6
    }

    // ---- FPtoFP FP32→FP16 → bank7（Newton window1 刷0 后，高16位为0）----
    emit_fptofp_to_bank7(out, c, kBankNewton);

    out.para_body_cycles = c;
    out.para_total_cycles = c * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }

    std::cout << "[Silu] Stage1 post body=" << out.para_body_cycles
              << " cyc/word, steps=" << total_steps
              << ", newton_iters=" << out.para_newton_iters << std::endl;
    return out;
}

SiluInstructions SiluOp::compile_two_stage(const app::AppOperatorRequest& req) {
    const auto exp_req = make_exp_neg_x_request(req);
    SiluInstructions out;
    out.stage0_exp = ExpOp::compile_merged(exp_req);

    const uint8_t smc_en = get_smc_id_en(req);
    const uint32_t iters = get_newton_iters(req);
    const uint32_t steps = out.stage0_exp.para_total_steps;
    const uint32_t elems = out.stage0_exp.para_total_elements;

    // Phase0 原样复用 Sigmoid：FPtoFP FP16→FP32 → bank5 = x_fp32；FPMUL×(-1) → bank2 = -x。
    // SiLU 语义下 bank5 的 x_fp32 之后不再被覆写（D 挪 bank3），活到 Phase2 最终乘。
    out.phase0_upcast = SigmoidOp::compile_upcast(req, steps, elems, smc_en);
    out.stage1_post = compile_postprocess(req, steps, elems, smc_en, iters);

    out.para_phase0_max_insts = max_iq_inst_count(out.phase0_upcast.get_iq_files());
    out.para_stage0_max_insts = max_iq_inst_count(out.stage0_exp.get_iq_files());
    out.para_stage1_max_insts = max_iq_inst_count(out.stage1_post.get_iq_files());
    out.para_phase0_fits = out.para_phase0_max_insts <= kSiluIqDepthLimit;
    out.para_stage0_fits = out.para_stage0_max_insts <= kSiluIqDepthLimit;
    out.para_stage1_fits = out.para_stage1_max_insts <= kSiluIqDepthLimit;
    out.para_num_stages = 3u;

    std::cout << "[Silu] stages=" << out.para_num_stages
              << " Phase0 max_iq=" << out.para_phase0_max_insts
              << (out.para_phase0_fits ? " OK" : " OVER")
              << " Phase1(Exp) max_iq=" << out.para_stage0_max_insts
              << (out.para_stage0_fits ? " OK" : " OVER")
              << " Phase2(post) max_iq=" << out.para_stage1_max_insts
              << (out.para_stage1_fits ? " OK" : " OVER")
              << " (limit=" << kSiluIqDepthLimit << ")" << std::endl;

    if (!out.para_phase0_fits || !out.para_stage0_fits || !out.para_stage1_fits) {
        std::cerr << "[Silu][WARN] IQ depth over limit" << std::endl;
    }
    return out;
}

OpInstructionsBase* SiluOp::compile(const app::AppOperatorRequest& req) {
    auto* insts = new SiluInstructions();
    *insts = compile_two_stage(req);
    return insts;
}

bool SiluOp::generate_data(const app::AppOperatorRequest& req,
                           std::vector<uint8_t>& input_bin,
                           std::vector<uint8_t>& other_bin,
                           std::vector<uint8_t>& alpha_bin) {
    // Stage0 host data = Exp layout with FP32 -x (half-rounded).
    const auto exp_req = make_exp_neg_x_request(req);
    ExpOp exp;
    const bool ok = exp.generate_data(exp_req, input_bin, other_bin, alpha_bin);
    std::cout << "[Silu] generate_data: Stage0 Exp bins (FP32 -x from FP16 semantics)"
              << std::endl;
    return ok;
}

bool SiluOp::calculate_golden(const app::AppOperatorRequest& req,
                              std::vector<uint8_t>& golden_bin) {
    const uint32_t n = req.para_dimensions.para_n * req.para_dimensions.para_c *
                       req.para_dimensions.para_h * req.para_dimensions.para_w;
    auto x = collect_x(req, n);
    const bool fp16 = is_fp16_req(req);
    const uint32_t lanes = fp16 ? 8u : 4u;
    const uint32_t steps = (n + lanes - 1) / lanes;
    const uint32_t words = std::max(steps, 1u);

    golden_bin.assign(static_cast<size_t>(words) * 16u, 0);
    for (uint32_t i = 0; i < n; ++i) {
        const float x_h = app::float16_to_float32(app::float32_to_float16(x[i]));
        // Phase1 复用 Exp 1.0.5：e^{-x} = host_exp_f32(-x_h)，含 k≤-126→0 / k≥127→+Inf
        const float e = host_exp_f32(-x_h);
        // Phase2：σ = newton_recip(1+e)，silu = x_h * σ（FP32 单乘，与 RTL 乘序一致）
        float sigma = 0.f;
        if (std::isinf(e) && e > 0.f) {
            sigma = 0.f;  // sigmoid(x) → 0 when exp(-x) 上溢
        } else {
            sigma = newton_recip(1.0f + e, kSiluDefaultNewtonIters);
        }
        const float y32 = x_h * sigma;
        const size_t off = static_cast<size_t>(i) * (fp16 ? 2u : 4u);
        if (fp16) {
            const uint16_t h = app::float32_to_float16(y32);
            if (off + 1 < golden_bin.size()) {
                golden_bin[off] = static_cast<uint8_t>(h & 0xFF);
                golden_bin[off + 1] = static_cast<uint8_t>((h >> 8) & 0xFF);
            }
        } else {
            uint32_t bits = 0;
            std::memcpy(&bits, &y32, 4);
            if (off + 3 < golden_bin.size()) {
                golden_bin[off] = bits & 0xFF;
                golden_bin[off + 1] = (bits >> 8) & 0xFF;
                golden_bin[off + 2] = (bits >> 16) & 0xFF;
                golden_bin[off + 3] = (bits >> 24) & 0xFF;
            }
        }
    }
    // Pad unused FP16 lanes with silu(0) = 0*σ(0) = 0
    if (fp16) {
        const uint16_t half = app::float32_to_float16(0.0f);
        for (uint32_t i = n; i < words * lanes; ++i) {
            const size_t off = static_cast<size_t>(i) * 2u;
            golden_bin[off] = half & 0xFF;
            golden_bin[off + 1] = (half >> 8) & 0xFF;
        }
    }
    return true;
}

void SiluOp::fill_random_request(app::AppOperatorRequest& req,
                                 std::mt19937& rng,
                                 std::uniform_real_distribution<float>&,
                                 const std::string& dtype_str) {
    req.para_operator = "Silu";
    // Keep |x| moderate so exp(-x) stays in FP16/FP32 comfort zone.
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
    const uint32_t total = req.para_dimensions.para_n * req.para_dimensions.para_c *
                           req.para_dimensions.para_h * req.para_dimensions.para_w;
    app::AppTensorDesc input;
    input.para_name = "input_tensor";
    input.para_dtype = (dtype_str == "fp16") ? app::AppDataType::FLOAT16
                                              : app::AppDataType::FLOAT32;
    input.para_shape = {req.para_dimensions.para_n, req.para_dimensions.para_c,
                        req.para_dimensions.para_h, req.para_dimensions.para_w};
    input.para_data_f32.reserve(total);
    for (uint32_t i = 0; i < total; ++i) input.para_data_f32.push_back(dist(rng));
    req.para_inputs["input_tensor"] = input;
}

REGISTER_OP(SiluOp);

}  // namespace op
}  // namespace aiacc
