#include "sigmoid_op.h"

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
constexpr uint8_t kBankExpOut = 3;     // e^{-x} in / intermediate
constexpr uint8_t kBankConst = 4;      // Phase2 常数段（与 Exp 共享 bank4）
constexpr uint32_t kBankConstZero = 6;   // bank4@6 = int_zero = 0x00000000（Bank7 清零源）
constexpr uint32_t kBankConstOne = 4;    // bank4@4 = 1.0（Exp 常数表已有）
constexpr uint32_t kBankConstTwo = 8;    // bank4@8 = 2.0（Exp 表中 Softmax 保留槽，LSU 覆写）
constexpr uint32_t kBankConstMagic = 9;  // bank4@9 = 0x7EF311C7（Exp 表中 Softmax 保留槽，LSU 覆写）
constexpr uint32_t kBankConstNegOne = 14;// bank4@14 = -1.0f（Phase0 x * (-1)）
constexpr uint8_t kBankDenom = 5;      // Phase2: 1+e^{-x}; Phase0 tmp: FP32 x
constexpr uint8_t kBankNewton = 6;     // Newton x_n；默认片上 magic 写 x0
constexpr uint32_t kNewtonMagic = 0x7EF311C7u;
constexpr uint8_t kBankFinalOut = 7;   // Phase2: 刷0 + FPtoFP 输出 @bank7
constexpr uint8_t kBankFp32Y = 7;      // optional host FP32 y for FPtoFP-only path

uint8_t get_smc_id_en(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("smc_id_en");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty()) return 0;
    return static_cast<uint8_t>(it->second.para_data_i32[0] & 0x1F);
}

uint32_t get_newton_iters(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("newton_iters");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty())
        return kSigmoidDefaultNewtonIters;
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
app::AppOperatorRequest make_exp_neg_x_request(const app::AppOperatorRequest& sig_req) {
    app::AppOperatorRequest exp_req = sig_req;
    exp_req.para_operator = "Exp";

    const uint32_t n = sig_req.para_dimensions.para_n * sig_req.para_dimensions.para_c *
                       sig_req.para_dimensions.para_h * sig_req.para_dimensions.para_w;
    auto x = collect_x(sig_req, n);

    // Force FP32 for Exp path (reuse proven FP32 Exp).
    app::AppTensorDesc input;
    input.para_name = "input_tensor";
    input.para_dtype = app::AppDataType::FLOAT32;
    input.para_shape = {sig_req.para_dimensions.para_n, sig_req.para_dimensions.para_c,
                        sig_req.para_dimensions.para_h, sig_req.para_dimensions.para_w};
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

/** VIN/VOUT 直通开关（FPGA 稀疏掩码）：读 para_parameters["vin_vout"]，默认关。
 *  sigmoid 自身 LSU 不置位（非 vv 位级不变）；silu LSU 置位后 Phase0 走 vv。 */
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
constexpr uint32_t kSigUrSettle = 3;       // == kUrReadToComputeIssue
constexpr uint32_t kSigFpOpSpan = 13;      // == kStageCycles
constexpr uint32_t kSigWritePhase = 10;    // == kFinalWritePhase within span
constexpr uint32_t kSigFptofpSpan = 13;
constexpr uint32_t kSigFptofpWrite = 10;

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

/** 片上 x0 = float_bits(MAGIC - float_bits(denom))，写 bank6（对齐 Softmax Norm）。 */
uint32_t append_magic_x0_prologue(SigmoidPostInstructions& out,
                                  uint32_t c,
                                  uint8_t smc_id_en,
                                  uint64_t intadd_code) {
    const uint32_t S = c;
    const uint32_t end = S + kSigFpOpSpan;

    append_sched(out.sched_ur_r0, S, ur_read_code(kBankDenom, 0, smc_id_en));
    append_sched(out.sched_ur_r1, S,
                 ur_read_code(kBankConst, kBankConstMagic, smc_id_en));

    hold_rtovr(out.sched_rtovr_0, S, end, kRtovrInUrRdata0 + 1);
    hold_rtovr(out.sched_rtovr_1, S, end, kRtovrInUrRdata0);

    const uint32_t iadd_issue = S + kSigUrSettle;
    append_sched(out.sched_intadd, iadd_issue, intadd_code);

    const uint32_t iadd_dr = iadd_issue + kIntaddLatency;
    hold_rtovr(out.sched_rtovr_35, iadd_dr, end, kRtovrInDrIntAddD0);
    append_sched(out.sched_ur_w0, S + kSigWritePhase,
                 ur_write_code(kBankNewton, 0, smc_id_en));

    return S + kSigFpOpSpan + 2;
}

}  // namespace

// ---------------------------------------------------------------------------
// SigmoidUpcastInstructions (Phase0: FP16→FP32 + neg)
// ---------------------------------------------------------------------------

std::vector<uint8_t> SigmoidUpcastInstructions::serialize() const {
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

std::vector<IqInstructionFile> SigmoidUpcastInstructions::get_iq_files() const {
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
                                        "SigmoidUpcast");
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8, false);
    add_iq(sched_fptofp, iq::IQ_ID_FP_TO_FP, iq::IQ_NAME_FP_TO_FP, 10, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_7, iq::IQ_ID_RTOVR_BASE + kRtovrOutFpToFpS, "rtovr-7", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    // 同一份微码须同时过全交叉仿真与 FPGA 稀疏掩码：vv 开时多出六份文件
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

// ---------------------------------------------------------------------------
// SigmoidPostInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SigmoidPostInstructions::serialize() const {
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

std::vector<IqInstructionFile> SigmoidPostInstructions::get_iq_files() const {
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
                                        "SigmoidPost");
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
    return files;
}

// ---------------------------------------------------------------------------
// SigmoidInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SigmoidInstructions::serialize() const {
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
    push(para_onchip_upcast ? 1u : 0u);
    return r;
}

std::vector<IqInstructionFile> SigmoidInstructions::get_iq_files() const {
    return stage0_exp.get_iq_files();
}

// ---------------------------------------------------------------------------
// SigmoidOp
// ---------------------------------------------------------------------------

std::string SigmoidOp::get_name() const { return "Sigmoid"; }

uint32_t SigmoidOp::max_iq_inst_count(const std::vector<IqInstructionFile>& files) {
    uint32_t mx = 0;
    for (const auto& f : files) {
        const uint32_t nbytes = (f.para_width_bits + 7) / 8;
        if (nbytes == 0) continue;
        const uint32_t n = static_cast<uint32_t>(f.para_data.size() / nbytes);
        if (n > mx) mx = n;
    }
    return mx;
}

SigmoidPostInstructions SigmoidOp::compile_postprocess(
    const app::AppOperatorRequest& /*req*/,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t newton_iters) {
    // Stage1 body (EMBED fold via emit_exp_embed_iq): IQ depth ≈ body, independent of steps.
    // Pipeline (FP32), RtoVR held every cycle (Exp Horner style — one-shot select is unsafe):
    //   +1:     D + 1 → denom @bank5
    //   magic:  x0 = float_bits(MAGIC - bits(denom)) → bank6（片上，对齐 Softmax）
    //   Newton: x = x*(2 - denom*x)  (iters)
    //           [Window 1 zero-overhead zeroing]: ur_r1 read bank4@6 (=0) -> ur_w0 write bank7
    //   FPtoFP: FP32 → FP16 (dest_low native layout) → bank7 (已清零的高16位保持0x0000)
    //
    // Port map (multiword addr_inc): R0/R2/W0 inc; R1 fixed (const bank4).

    SigmoidPostInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_newton_iters = newton_iters;
    out.para_smc_id_en = smc_id_en;

    const bool fptofp_only = []() {
        const char* e = std::getenv("SIGMOID_STAGE1_FPTOFP_ONLY");
        return e && e[0] == '1' && e[1] == '\0';
    }();

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

    auto emit_fptofp_to_bank7 = [&](SigmoidPostInstructions& o, uint32_t& c,
                                    uint8_t src_bank) {
        // read@0, fptofp@3, write@10 within span (softmax-like margin)
        // dest=bank7: bank7 已被 Newton window1 的零开销刷0，高16位为0x0000，满足 strict golden
        // 注意 fptofp_array.v dest_low 语义：FP32→FP16 时 D 线高16位 =
        // output_data_r 寄存器残留（= 本 step Phase0 的 fp32 x），并非 SRAM 原值！
        // 故此处 wbe 必须只开每 lane 低 2 字节（0x3333），不让残留进 SRAM；
        // 高 16 位由 185/226 拍的整字清零写提前备好。
        static constexpr uint16_t kSigFinalWbeLow16 = 0x3333u;
        append_sched(o.sched_ur_r0, c, ur_read_code(src_bank, 0, smc_id_en));
        hold_rtovr(o.sched_rtovr_7, c, c + kSigFptofpSpan, kRtovrInUrRdata0);
        append_sched(o.sched_fptofp, c + kSigUrSettle, fptofp_code);
        hold_rtovr(o.sched_rtovr_35, c + kSigUrSettle + kFpmulDrLatencyRtl,
                   c + kSigFptofpSpan, kRtovrInDrFpToFpD);
        append_sched(o.sched_ur_w0, c + kSigFptofpWrite,
                     ur_write_code(kBankFinalOut, 0, smc_id_en, kSigFinalWbeLow16));
        c += kSigFptofpSpan;
    };

    uint32_t c = 0;

    if (fptofp_only || newton_iters == 0u) {
        out.para_newton_iters = 0;
        emit_fptofp_to_bank7(out, c, kBankFp32Y);
        out.para_body_cycles = c;
        out.para_total_cycles = c * total_steps;
        std::cout << "[Sigmoid] Stage1 FPtoFP-only body=" << out.para_body_cycles
                  << " cyc/word, steps=" << total_steps << std::endl;
        return out;
    }

    // ---- +1: denom = D + 1.0 (FpAdd S0←UR0 e^{-x}, S1←UR1 1.0) ----
    {
        const uint32_t S = c;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankExpOut, 0, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstOne, smc_id_en));
        hold_rtovr(out.sched_rtovr_5, S, S + kSigFpOpSpan, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_6, S, S + kSigFpOpSpan, kRtovrInUrRdata0 + 1);
        append_sched(out.sched_fpadd, S + kSigUrSettle, add_code);
        hold_rtovr(out.sched_rtovr_35, S + kSigUrSettle + kFpaddDrLatencyRtl,
                   S + kSigFpOpSpan, kRtovrInDrFpAddD);
        append_sched(out.sched_ur_w0, S + kSigWritePhase,
                     ur_write_code(kBankDenom, 0, smc_id_en));
        c = S + kSigFpOpSpan + 2;  // UR write settle before magic / Newton reads denom
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
            hold_rtovr(out.sched_rtovr_3, S, S + kSigFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSigFpOpSpan, kRtovrInUrRdata0 + 2);
            append_sched(out.sched_fpmul, S + kSigUrSettle, mul_code);
            // Keep mul DR routed on S1 of next add; also park rtovr-6 on mul DR after issue
            hold_rtovr(out.sched_rtovr_6, S + kSigUrSettle + kFpmulDrLatencyRtl,
                       S + kSigFpOpSpan, kRtovrInDrFpMulD);

            // Zero-overhead bank7 zeroing: 在 window1 ALU 空闲中，利用 ur_r1 读
            // bank4@6(int_zero=0x00000000, addr fixed) → rtovr_35(UR rdata1) → ur_w0(addr inc) 写 bank7。
            // 后续 FPtoFP dest_low 只写低16位，bank7 高16位保持 0x0000。
            append_sched(out.sched_ur_r1, S,
                         ur_read_code(kBankConst, kBankConstZero, smc_id_en));
            hold_rtovr(out.sched_rtovr_35, S, S + kSigFpOpSpan, kRtovrInUrRdata0 + 1);
            append_sched(out.sched_ur_w0, S + kSigWritePhase,
                         ur_write_code(kBankFinalOut, 0, smc_id_en));

            c = S + kSigFpOpSpan;
        }
        // u = 2 - t  (S0←UR1 const2, S1←mul DR). Re-read const; mul DR still held from above
        // → new window: read const, hold S0=UR1 & S1=mulDR, sub@3, write not needed (stay in DR)
        {
            const uint32_t S = c;
            append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstTwo, smc_id_en));
            hold_rtovr(out.sched_rtovr_5, S, S + kSigFpOpSpan, kRtovrInUrRdata0 + 1);
            hold_rtovr(out.sched_rtovr_6, S, S + kSigFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_fpadd, S + kSigUrSettle, sub_code);
            // After sub, FpAdd DR holds u; route to mul S1 next
            hold_rtovr(out.sched_rtovr_4, S + kSigUrSettle + kFpaddDrLatencyRtl,
                       S + kSigFpOpSpan, kRtovrInDrFpAddD);
            c = S + kSigFpOpSpan;
        }
        // x = x_n * u  (S0←UR0 x_n, S1←add DR)
        {
            const uint32_t S = c;
            append_sched(out.sched_ur_r0, S, ur_read_code(kBankNewton, 0, smc_id_en));
            hold_rtovr(out.sched_rtovr_3, S, S + kSigFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSigFpOpSpan, kRtovrInDrFpAddD);
            append_sched(out.sched_fpmul, S + kSigUrSettle, mul_code);
            hold_rtovr(out.sched_rtovr_35, S + kSigUrSettle + kFpmulDrLatencyRtl,
                       S + kSigFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_ur_w0, S + kSigWritePhase,
                         ur_write_code(kBankNewton, 0, smc_id_en));
            c = S + kSigFpOpSpan + 2;  // settle before next iter / FPtoFP reads x_n
        }
    }

    // ---- FPtoFP FP32→FP16 → bank7（Window1 刷0 后，高16位为0）----
    emit_fptofp_to_bank7(out, c, kBankNewton);

    out.para_body_cycles = c;
    out.para_total_cycles = c * total_steps;

    // Body cycles are wall-clock; IQ depth after embed compress is << body.
    // Hard fallback only if uncompressed-style occupancy would exceed INSTQ.
    if (out.para_body_cycles > kSigmoidIqDepthLimit * 4u) {
        std::cout << "[Sigmoid] Stage1 body=" << out.para_body_cycles
                  << " absurdly long → fallback FPtoFP-only" << std::endl;
        out = SigmoidPostInstructions{};
        out.para_total_steps = total_steps;
        out.para_total_elements = total_elems;
        out.para_newton_iters = 0;
        out.para_smc_id_en = smc_id_en;
        c = 0;
        emit_fptofp_to_bank7(out, c, kBankFp32Y);
        out.para_body_cycles = c;
        out.para_total_cycles = c * total_steps;
    }

    std::cout << "[Sigmoid] Stage1 post body=" << out.para_body_cycles
              << " cyc/word, steps=" << total_steps
              << ", newton_iters=" << out.para_newton_iters << std::endl;
    return out;
}

SigmoidUpcastInstructions SigmoidOp::compile_upcast(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en) {
    // Phase0 body (EMBED):
    //   1) FPtoFP: FP16 src_low @bank2 → FP32 @bank5
    //   2) FPMUL:  bank5 * (-1 @bank4:14) → FP32 -x @bank2 (Exp input，覆盖 FP16)
    //   （-1.0f 已从 bank7 移至 bank4@14，bank7 全段预留给 Phase2 清零+FP16 输出）
    SigmoidUpcastInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;

    iq::FpToFpCru up{};
    up.para_inst_valid = true;
    up.para_src_is_32b = false;  // FP16
    up.para_dst_is_32b = true;   // → FP32
    up.para_src_high = false;    // src_low
    up.para_dst_high = false;
    const uint64_t up_code =
        (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(up)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = true;
    mul.para_dst_precision_is_32b = true;
    mul.para_smc_id_en = smc_id_en;
    const uint64_t mul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    uint32_t c = 0;

    // ---- FPtoFP FP16→FP32 → bank5 ----
    {
        const uint32_t S = c;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankFp16In, 0, smc_id_en));
        if (vv) {
            // vv：VIN0 捕获@S+1，消费@S+4（捕获+3，exp 实测）；VOUT0 回写（FPTOFP 捕获@发射+5）
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
            hold_rtovr(out.sched_rtovr_7, S, S + 5, kRtovrInDrVin0D);
            append_sched(out.sched_fptofp, S + 4, up_code);
            append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrFpToFpD));
            hold_rtovr(out.sched_rtovr_35, S + 9, S + 12, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 12,
                         ur_write_code(kBankDenom, 0, smc_id_en));  // tmp @bank5
        } else {
            hold_rtovr(out.sched_rtovr_7, S, S + kSigFptofpSpan, kRtovrInUrRdata0);
            append_sched(out.sched_fptofp, S + kSigUrSettle, up_code);
            hold_rtovr(out.sched_rtovr_35, S + kSigUrSettle + kFpmulDrLatencyRtl,
                       S + kSigFptofpSpan, kRtovrInDrFpToFpD);
            append_sched(out.sched_ur_w0, S + kSigFptofpWrite,
                         ur_write_code(kBankDenom, 0, smc_id_en));  // tmp @bank5
        }
        c = S + kSigFptofpSpan + 2;
    }

    // ---- FPMUL: x * (-1) → bank2 ----
    {
        const uint32_t S = c;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankDenom, 0, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankConst, kBankConstNegOne, smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← tmp
            append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN1 ← -1
            hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
            hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
            append_sched(out.sched_fpmul, S + 4, mul_code);
            append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));
            hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 11,
                         ur_write_code(kBankExpIn, 0, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_3, S, S + kSigFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSigFpOpSpan, kRtovrInUrRdata0 + 1);
            append_sched(out.sched_fpmul, S + kSigUrSettle, mul_code);
            hold_rtovr(out.sched_rtovr_35, S + kSigUrSettle + kFpmulDrLatencyRtl,
                       S + kSigFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_ur_w0, S + kSigWritePhase,
                         ur_write_code(kBankExpIn, 0, smc_id_en));
        }
        c = S + kSigFpOpSpan + 2;
    }

    out.para_body_cycles = c;
    out.para_total_cycles = c * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }
    std::cout << "[Sigmoid] Phase0 upcast body=" << out.para_body_cycles
              << " cyc/word (FPtoFP FP16→FP32 + mul -1), steps=" << total_steps
              << std::endl;
    return out;
}

SigmoidInstructions SigmoidOp::compile_two_stage(const app::AppOperatorRequest& req) {
    const bool onchip_upcast = []() {
        const char* e = std::getenv("SIGMOID_ONCHIP_UPCAST");
        return e && e[0] == '1' && e[1] == '\0';
    }();

    const auto exp_req = make_exp_neg_x_request(req);
    SigmoidInstructions out;
    out.para_onchip_upcast = onchip_upcast;
    out.stage0_exp = ExpOp::compile_merged(exp_req);

    const uint8_t smc_en = get_smc_id_en(req);
    const uint32_t iters = get_newton_iters(req);
    const uint32_t steps = out.stage0_exp.para_total_steps;
    const uint32_t elems = out.stage0_exp.para_total_elements;

    out.phase0_upcast = compile_upcast(req, steps, elems, smc_en);
    out.stage1_post = compile_postprocess(req, steps, elems, smc_en, iters);

    out.para_phase0_max_insts = max_iq_inst_count(out.phase0_upcast.get_iq_files());
    out.para_stage0_max_insts = max_iq_inst_count(out.stage0_exp.get_iq_files());
    out.para_stage1_max_insts = max_iq_inst_count(out.stage1_post.get_iq_files());
    out.para_phase0_fits = out.para_phase0_max_insts <= kSigmoidIqDepthLimit;
    out.para_stage0_fits = out.para_stage0_max_insts <= kSigmoidIqDepthLimit;
    out.para_stage1_fits = out.para_stage1_max_insts <= kSigmoidIqDepthLimit;
    out.para_num_stages = onchip_upcast ? 3u : 2u;

    std::cout << "[Sigmoid] stages=" << out.para_num_stages
              << " onchip_upcast=" << (onchip_upcast ? "1" : "0")
              << " Phase0 max_iq=" << out.para_phase0_max_insts
              << (out.para_phase0_fits ? " OK" : " OVER")
              << " Phase1(Exp) max_iq=" << out.para_stage0_max_insts
              << (out.para_stage0_fits ? " OK" : " OVER")
              << " Phase2(post) max_iq=" << out.para_stage1_max_insts
              << (out.para_stage1_fits ? " OK" : " OVER")
              << " (limit=" << kSigmoidIqDepthLimit << ")" << std::endl;

    if (!out.para_phase0_fits || !out.para_stage0_fits || !out.para_stage1_fits) {
        std::cerr << "[Sigmoid][WARN] IQ depth over limit" << std::endl;
    }
    return out;
}

OpInstructionsBase* SigmoidOp::compile(const app::AppOperatorRequest& req) {
    auto* insts = new SigmoidInstructions();
    *insts = compile_two_stage(req);
    return insts;
}

bool SigmoidOp::generate_data(const app::AppOperatorRequest& req,
                              std::vector<uint8_t>& input_bin,
                              std::vector<uint8_t>& other_bin,
                              std::vector<uint8_t>& alpha_bin) {
    // Stage0 host data = Exp layout with FP32 -x (half-rounded).
    const auto exp_req = make_exp_neg_x_request(req);
    ExpOp exp;
    const bool ok = exp.generate_data(exp_req, input_bin, other_bin, alpha_bin);
    std::cout << "[Sigmoid] generate_data: Stage0 Exp bins (FP32 -x from FP16 semantics)"
              << std::endl;
    return ok;
}

bool SigmoidOp::calculate_golden(const app::AppOperatorRequest& req,
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
        float y32 = 0.f;
        if (std::isinf(e) && e > 0.f) {
            y32 = 0.f;  // sigmoid(x) → 0 when exp(-x) 上溢
        } else {
            y32 = newton_recip(1.0f + e, kSigmoidDefaultNewtonIters);
        }
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
    // Pad unused FP16 lanes with sigmoid(0)=0.5
    if (fp16) {
        const uint16_t half = app::float32_to_float16(0.5f);
        for (uint32_t i = n; i < words * lanes; ++i) {
            const size_t off = static_cast<size_t>(i) * 2u;
            golden_bin[off] = half & 0xFF;
            golden_bin[off + 1] = (half >> 8) & 0xFF;
        }
    }
    return true;
}

void SigmoidOp::fill_random_request(app::AppOperatorRequest& req,
                                    std::mt19937& rng,
                                    std::uniform_real_distribution<float>&,
                                    const std::string& dtype_str) {
    req.para_operator = "Sigmoid";
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

REGISTER_OP(SigmoidOp);

}  // namespace op
}  // namespace aiacc
