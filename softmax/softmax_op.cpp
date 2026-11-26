
#include "softmax_op.h"

#include "softmax_golden.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

#include "../../app/fp16_utils.h"
#include "../../iq/iq_id_defs.h"
#include "../../iq/modules/fpadd_iq.h"
#include "../../iq/modules/fpmul_iq.h"
#include "../../iq/modules/fp32addtree8to1_iq.h"
#include "../../iq/modules/intadd_iq.h"
#include "../../iq/modules/logical_iq.h"
#include "../common/op_registry.h"
#include "../exp/exp_decompose.h"

namespace aiacc {
namespace op {

using namespace exp_iq;
using namespace exp_math;
using namespace softmax_golden;

namespace {

constexpr uint8_t kBankRawX = 2;       // 原始 x 或 x'（Phase0 入/出）
constexpr uint8_t kBankExpOut = 3;    // e^{x'} / y
constexpr uint8_t kBankConst = 4;      // Exp const @word0..7；Newton 2.0 @word8（e2e 与 Exp 共存）
constexpr uint32_t kBankConstNewtonTwo = 8;
constexpr uint32_t kBankConstNewtonMagic = 9;  // 0x7EF311C7 按 float 位模式存 UR
constexpr uint8_t kBankSum = 5;        // max / S 四 lane 广播
constexpr uint8_t kBankNewton = 6;     // Newton x_n / inv
// vv（host 预载 bank5）word 布局：@0=S 广播，@1=m 广播（非 vv：S 由片上 sum 写 @0）
constexpr uint32_t kBankSumSumWord = 0;
constexpr uint32_t kBankSumMaxWord = 1;

constexpr uint8_t kRtovrInDrFp32AddTree8D = 28;
constexpr uint8_t kRtovrOutFp32AddTree8S0 = 32;
constexpr uint8_t kRtovrOutFp32AddTree8S1 = 33;

constexpr uint32_t kSmUrSettle = 3;       // == kUrReadToComputeIssue
constexpr uint32_t kSmFpOpSpan = 13;      // == kStageCycles
constexpr uint32_t kSmWritePhase = 10;    // == kFinalWritePhase within span
constexpr uint32_t kSmAddTreeLatency = 3;  // ADDTREE8 RTL valid_pipe[2] → issue 后 3 拍 DR 有效
constexpr uint32_t kSmAddTreeIssueGap = 1; // UR settle 后再等 1 拍再发 ADDTREE8（对齐 matmul rtovr→tree）

uint8_t get_smc_id_en(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("smc_id_en");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty()) return 0;
    return static_cast<uint8_t>(it->second.para_data_i32[0] & 0x1F);
}

uint32_t get_newton_iters(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("newton_iters");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty())
        return kSoftmaxDefaultNewtonIters;
    int v = it->second.para_data_i32[0];
    if (v < 1) v = 1;
    if (v > 4) v = 4;
    return static_cast<uint32_t>(v);
}

bool host_inv_mode() {
    const char* e = std::getenv("SOFTMAX_HOST_INV");
    return e && e[0] == '1' && e[1] == '\0';
}

bool host_max_mode() {
    const char* e = std::getenv("SOFTMAX_HOST_MAX");
    if (!e) return false;  // 默认片上：bank2=raw x，跑 Phase0 max-sub
    return e[0] == '1' && e[1] == '\0';
}

bool host_sum_mode() {
    const char* e = std::getenv("SOFTMAX_HOST_SUM");
    if (!e) return false;  // 默认片上：跑 Phase1.5 sum reduce
    return e[0] == '1' && e[1] == '\0';
}

bool host_x0_mode() {
    const char* e = std::getenv("SOFTMAX_HOST_X0");
    if (!e) return false;  // 默认片上：Norm magic 前导 → bank6
    return e[0] == '1' && e[1] == '\0';
}

bool magic_only_mode() {
    const char* e = std::getenv("SOFTMAX_MAGIC_ONLY");
    return e && e[0] == '1' && e[1] == '\0';
}

/** VIN/VOUT 直通开关（FPGA USE_RTOVR_SPARSE_MASK 唯一合法通路；全交叉仿真同码兼容）。
 *  默认关：非 vv 路径位级不变。LSU 侧经 req.para_parameters["vin_vout"] 置位，
 *  同一 req 拷贝流入 make_exp_xprime_request → exp vv 同步开启。 */
bool vin_vout_mode(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("vin_vout");
    if (it == req.para_parameters.end() || it->second.para_data_i32.empty()) return false;
    return it->second.para_data_i32[0] != 0;
}

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

uint32_t logical_w(const app::AppOperatorRequest& req) {
    return req.para_dimensions.para_n * req.para_dimensions.para_c *
           req.para_dimensions.para_h * req.para_dimensions.para_w;
}

std::vector<float> collect_x(const app::AppOperatorRequest& req, uint32_t n) {
    std::vector<float> out(n, 0.f);
    auto it = req.para_inputs.find("input_tensor");
    if (it == req.para_inputs.end()) return out;
    const auto& t = it->second;
    for (uint32_t i = 0; i < n && i < t.para_data_f32.size(); ++i)
        out[i] = t.para_data_f32[i];
    return out;
}

/** Phase1 Exp：输入已是 FP32 x'（Host max-sub 后）。 */
app::AppOperatorRequest make_exp_xprime_request(const app::AppOperatorRequest& sm_req,
                                                const std::vector<float>& x_prime) {
    app::AppOperatorRequest exp_req = sm_req;
    exp_req.para_operator = "Exp";

    app::AppTensorDesc input;
    input.para_name = "input_tensor";
    input.para_dtype = app::AppDataType::FLOAT32;
    input.para_shape = {sm_req.para_dimensions.para_n, sm_req.para_dimensions.para_c,
                        sm_req.para_dimensions.para_h, sm_req.para_dimensions.para_w};
    input.para_data_f32 = x_prime;
    exp_req.para_inputs["input_tensor"] = input;
    return exp_req;
}

void append_sched(std::vector<SchedEntry>& s, uint32_t cyc, uint64_t code) {
    s.push_back({cyc, code});
}

void hold_rtovr(std::vector<SchedEntry>& s, uint32_t c0, uint32_t c1, uint8_t src) {
    const uint64_t code = rtovr_code(src);
    for (uint32_t t = c0; t < c1; ++t) append_sched(s, t, code);
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

/** 片上 x0 = bits_as_f32(MAGIC - bits(S))，结果 broadcast 写 bank6@0。
 *  UR 直送 INTADD（FPTOINT 是数值截断）；INTADD 输出已是 float 位模式，直写 UR（不用 INTTOFP）。
 *  vv：S→VIN0、MAGIC→VIN1（捕获@读+1，消费@读+4 = 捕获+3，exp 实测边界）；
 *  x0 经 VOUT0 回写（DR 粘滞捕获@发射+4，写=捕获+3）。 */
uint32_t append_magic_x0_prologue(SoftmaxNormInstructions& out,
                                  uint32_t c,
                                  uint8_t smc_id_en,
                                  uint64_t intadd_code) {
    const uint32_t S = c;
    const uint32_t end = S + kSmFpOpSpan;

    append_sched(out.sched_ur_r0, S, ur_read_code(kBankSum, kBankSumSumWord, smc_id_en));
    append_sched(out.sched_ur_r1, S,
                 ur_read_code(kBankConst, kBankConstNewtonMagic, smc_id_en));

    if (out.para_use_vin_vout) {
        append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← S
        append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN1 ← MAGIC
        hold_rtovr(out.sched_rtovr_0, S, S + 5, kRtovrInDrVin1D);  // S0 = MAGIC
        hold_rtovr(out.sched_rtovr_1, S, S + 5, kRtovrInDrVin0D);  // S1 = bits(S)
        append_sched(out.sched_intadd, S + 4, intadd_code);
        append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrIntAddD0));  // VOUT0 ← x0
        hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
        append_sched(out.sched_ur_w0, S + 11, ur_write_code(kBankNewton, 0, smc_id_en));
        out.para_run_magic_in_body = true;
        return S + kSmFpOpSpan + 2;
    }

    hold_rtovr(out.sched_rtovr_0, S, end, kRtovrInUrRdata0 + 1);
    hold_rtovr(out.sched_rtovr_1, S, end, kRtovrInUrRdata0);

    const uint32_t iadd_issue = S + kSmUrSettle;
    append_sched(out.sched_intadd, iadd_issue, intadd_code);

    const uint32_t iadd_dr = iadd_issue + kIntaddLatency;
    hold_rtovr(out.sched_rtovr_35, iadd_dr, end, kRtovrInDrIntAddD0);
    append_sched(out.sched_ur_w0, S + kSmWritePhase,
                 ur_write_code(kBankNewton, 0, smc_id_en));

    out.para_run_magic_in_body = true;
    return S + kSmFpOpSpan + 2;
}

constexpr uint8_t kBankColAcc = 6;     // Phase0a 列 max scratch @0
// LSU v3（load_store_v3/softmax）：bank4@10..12 是 exp k-flush 的 kMin/kMax/+Inf（@13=RN
// 魔法字），两算子共用一份 bank4 镜像时必须让位。掩码挪到 @15..18，hbm 常量表同步放置。
// 注：op/softmax 独立 pack（pack_softmax_rtl_case.py）若要复用本仓库该常量需同步。
constexpr uint32_t kBankConstMaskBase = 15;  // bank4@15..18 = lane 掩码 1/0
constexpr uint32_t kBankScalarBase = 20;     // bank6@20..23 标量 [ck,0,0,0]
constexpr uint32_t kBankScratchBroadcastA = 24;
constexpr uint32_t kBankScratchBroadcastB = 29;
constexpr uint32_t kBankScratchM01 = 25;
constexpr uint32_t kBankScratchM23 = 26;
constexpr uint32_t kBankScratchM = 27;

constexpr uint32_t kSmMaxCompareSpan = 13;  // UR 读 → compare → SELECT → 写回
constexpr uint32_t kSmMaxSelectIssue = 6;   // FPADD check@3 + SELECT@6（对齐 Exp flush）
constexpr uint32_t kSmExtractSpan = 18;     // FPMUL 掩码 + ADDTREE8 抽单 lane → scalar@lane0

/** DS 4×4 转置路径：STB 读 4 word + DS buffer + LDB 写回（模型，待 DS RTL 标定）。 */
constexpr uint32_t kSmDsStb4WordCycles = 12u;
constexpr uint32_t kSmDsTranspose4x4Cycles = 32u;
constexpr uint32_t kSmDsLdb4WordCycles = 12u;
constexpr uint32_t kSmUrWordCopyCycles = 8u;  // UR @0 → @dst 复制一行

uint16_t pack_logic_select_great(uint8_t smc_en) {
    iq::LogicalCru logic{};
    logic.para_inst_valid = true;
    logic.para_precision_32b = true;
    logic.para_opcode = iq::LogicalOp::SELECT_GREAT;
    logic.para_smc_id_en = smc_en;
    return iq::LogicalIq::pack_cru_logical(logic);
}

uint64_t pack_fpadd_check_code(uint8_t smc_en) {
    iq::FpAddCru chk{};
    chk.para_inst_valid = true;
    chk.para_src_precision_32b = true;
    chk.para_dst_precision_32b = true;
    chk.para_update_status = true;
    chk.para_is_sub = true;
    return (static_cast<uint64_t>(iq::FpAddIq::pack_cru_fpadd(chk)) |
            (static_cast<uint64_t>(smc_en & 0x1Fu) << 5));
}

}  // namespace

// ---------------------------------------------------------------------------
// SoftmaxMaxReduceInstructions (Phase0a: max reduce → bank5)
// ---------------------------------------------------------------------------

std::vector<uint8_t> SoftmaxMaxReduceInstructions::serialize() const {
    std::vector<uint8_t> r;
    auto push = [&](uint32_t v) {
        r.push_back(v & 0xFF);
        r.push_back((v >> 8) & 0xFF);
        r.push_back((v >> 16) & 0xFF);
        r.push_back((v >> 24) & 0xFF);
    };
    push(para_body_cycles);
    push(para_col_body_cycles);
    push(para_lane_tree_cycles);
    push(para_phase_b_replicate_cycles);
    push(para_phase_b_ds_cycles);
    push(para_phase_b_compare_cycles);
    push(para_phase_b_broadcast_cycles);
    push(para_phase_b_mode);
    push(para_total_steps);
    push(para_total_elements);
    push(para_total_cycles);
    return r;
}

std::vector<IqInstructionFile> SoftmaxMaxReduceInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t body = para_body_cycles;

    auto add_iq = [&](const std::vector<SchedEntry>& sched, uint32_t iq_id,
                      const std::string& name, uint32_t code_bits) {
        if (sched.empty()) return;
        IqInstructionFile f;
        f.para_iq_id = iq_id;
        f.para_iq_name = name;
        f.para_width_bits = 41 + code_bits;
        // 列 tournament + lane 树已按 word 展开地址；整段 flat 一次 trigger。
        f.para_data = emit_exp_embed_iq(sched, code_bits, false, 1u, body,
                                        "SoftmaxMaxReduce");
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 10);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10);
    add_iq(sched_logic, iq::IQ_ID_LOGIC, iq::IQ_NAME_LOGIC, 11);
    add_iq(sched_addtree8, iq::IQ_ID_FP32_ADDTREE_8TO1, iq::IQ_NAME_FP32_ADDTREE_8TO1, 8);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7);
    add_iq(sched_rtovr_11, iq::IQ_ID_RTOVR_BASE + 11, "rtovr-11", 7);
    add_iq(sched_rtovr_12, iq::IQ_ID_RTOVR_BASE + 12, "rtovr-12", 7);
    add_iq(sched_rtovr_13, iq::IQ_ID_RTOVR_BASE + 13, "rtovr-13", 7);
    add_iq(sched_rtovr_32, iq::IQ_ID_RTOVR_BASE + 32, "rtovr-32", 7);
    add_iq(sched_rtovr_33, iq::IQ_ID_RTOVR_BASE + 33, "rtovr-33", 7);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7);
    return files;
}

SoftmaxMaxReduceInstructions SoftmaxOp::compile_max_reduce(
    const app::AppOperatorRequest& /*req*/,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    SoftmaxMaxReducePhaseBMode phase_b_mode,
    bool skip_phase_b) {
    SoftmaxMaxReduceInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    out.para_col_embed_steps = 1u;
    out.para_phase_b_mode = static_cast<uint8_t>(phase_b_mode);

    const uint64_t fpadd_check = pack_fpadd_check_code(smc_id_en);
    const uint64_t logic_sel_great = pack_logic_select_great(smc_id_en);

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = true;
    mul.para_dst_precision_is_32b = true;
    mul.para_smc_id_en = smc_id_en;
    const uint64_t fpmul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    iq::Fp32AddTree8To1Cru tree{};
    tree.para_inst_valid = true;
    const auto tree_code_for = [&](uint8_t dst_lane) -> uint64_t {
        tree.para_dst_reg_enc = dst_lane & 0x3u;
        return (static_cast<uint64_t>(
                    iq::Fp32AddTree8To1Iq::pack_cru_fp32addtree8to1(tree)) |
                (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 3));
    };

    const auto wbe_lane = [](uint8_t dst_lane) -> uint16_t {
        return static_cast<uint16_t>(0xFu << (4u * (dst_lane & 0x3u)));
    };

    /** 同 lane 两两 max(acc, word) → bank6@0；s0=acc, s1=word, SELECT_GREAT 取较大。 */
    const auto emit_col_merge = [&](uint32_t start_c, uint32_t word_addr) -> uint32_t {
        const uint32_t end = start_c + kSmMaxCompareSpan;
        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankRawX, word_addr, smc_id_en));
        append_sched(out.sched_ur_r1, start_c, ur_read_code(kBankColAcc, 0, smc_id_en));
        hold_rtovr(out.sched_rtovr_5, start_c, end, kRtovrInUrRdata0 + 1);  // acc → s0
        hold_rtovr(out.sched_rtovr_6, start_c, end, kRtovrInUrRdata0);      // word → s1
        append_sched(out.sched_fpadd, start_c + kSmUrSettle, fpadd_check);
        hold_rtovr(out.sched_rtovr_11, start_c + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   start_c + kSmMaxSelectIssue + kLogicLatency, kRtovrInUrRdata0 + 1);
        hold_rtovr(out.sched_rtovr_12, start_c + kSmUrSettle, start_c + kSmMaxSelectIssue + 1u,
                   kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_13, start_c + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   start_c + kSmMaxSelectIssue + kLogicLatency, kRtovrInDrFpAddSt);
        append_sched(out.sched_logic, start_c + kSmMaxSelectIssue, logic_sel_great);
        hold_rtovr(out.sched_rtovr_35, start_c + kSmMaxSelectIssue + kLogicLatency, end,
                   kRtovrInDrLogicD);
        append_sched(out.sched_ur_w0, start_c + kSmWritePhase,
                     ur_write_code(kBankColAcc, 0, smc_id_en));
        return end + 2;
    };

    /** FPMUL 掩码 + ADDTREE8：cols@bank6@0 的 lane k → bank6@dst 的 [vk,0,0,0]。 */
    const auto emit_extract_lane = [&](uint32_t start_c, uint8_t lane_k,
                                       uint32_t dst_addr) -> uint32_t {
        const uint32_t tree_issue = start_c + kSmUrSettle + kSmAddTreeIssueGap;
        const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
        const uint32_t tree_write = tree_dr + 2;
        const uint32_t end = tree_write + 3;

        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankColAcc, 0, smc_id_en));
        append_sched(out.sched_ur_r1, start_c,
                     ur_read_code(kBankConst, kBankConstMaskBase + lane_k, smc_id_en));
        hold_rtovr(out.sched_rtovr_3, start_c, start_c + kSmUrSettle + 1, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_4, start_c, start_c + kSmUrSettle + 1, kRtovrInUrRdata0 + 1);
        append_sched(out.sched_fpmul, start_c + kSmUrSettle, fpmul_code);
        hold_rtovr(out.sched_rtovr_32, start_c + kSmUrSettle + kFpmulDrLatencyRtl,
                   tree_issue + 1, kRtovrInDrFpMulD);
        hold_rtovr(out.sched_rtovr_33, start_c, tree_issue + 1, kRtovrInTiedZero);
        append_sched(out.sched_addtree8, tree_issue, tree_code_for(0));
        hold_rtovr(out.sched_rtovr_35, tree_dr, end, kRtovrInDrFp32AddTree8D);
        append_sched(out.sched_ur_w0, tree_write,
                     ur_write_code(kBankColAcc, dst_addr, smc_id_en, wbe_lane(0)));
        return end;
    };

    /** 标量 [v,0,0,0]@src → 四 lane 广播到 dst word（ADDTREE8 + wbe×4）。 */
    const auto emit_broadcast_scalar = [&](uint32_t start_c, uint32_t src_addr,
                                           uint32_t dst_addr) -> uint32_t {
        uint32_t c = start_c;
        for (uint8_t dst = 0; dst < 4; ++dst) {
            const uint32_t tree_issue = c + kSmUrSettle + kSmAddTreeIssueGap;
            const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
            const uint32_t tree_write = tree_dr + 2;
            append_sched(out.sched_ur_r1, c, ur_read_code(kBankColAcc, src_addr, smc_id_en));
            hold_rtovr(out.sched_rtovr_32, c, tree_issue + 1, kRtovrInUrRdata0 + 1);
            hold_rtovr(out.sched_rtovr_33, c, tree_issue + 1, kRtovrInTiedZero);
            append_sched(out.sched_addtree8, tree_issue, tree_code_for(dst));
            hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
            append_sched(out.sched_ur_w0, tree_write,
                         ur_write_code(kBankColAcc, dst_addr, smc_id_en, wbe_lane(dst)));
            c = tree_write + 3;
        }
        return c;
    };

    /** 两标量 word@src_a/src_b（lane0 有效）→ max 写 src_dst lane0。 */
    const auto emit_max_scalar_pair = [&](uint32_t start_c, uint32_t src_a, uint32_t src_b,
                                          uint32_t src_dst) -> uint32_t {
        uint32_t c = start_c;
        c = emit_broadcast_scalar(c, src_a, kBankScratchBroadcastA);
        c = emit_broadcast_scalar(c, src_b, kBankScratchBroadcastB);
        const uint32_t S = c;
        const uint32_t end = S + kSmMaxCompareSpan;
        append_sched(out.sched_ur_r0, S,
                     ur_read_code(kBankColAcc, kBankScratchBroadcastA, smc_id_en));
        append_sched(out.sched_ur_r1, S,
                     ur_read_code(kBankColAcc, kBankScratchBroadcastB, smc_id_en));
        hold_rtovr(out.sched_rtovr_5, S, end, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_6, S, end, kRtovrInUrRdata0 + 1);
        append_sched(out.sched_fpadd, S + kSmUrSettle, fpadd_check);
        hold_rtovr(out.sched_rtovr_11, S + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   S + kSmMaxSelectIssue + kLogicLatency, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_12, S + kSmUrSettle, S + kSmMaxSelectIssue + 1u,
                   kRtovrInUrRdata0 + 1);
        hold_rtovr(out.sched_rtovr_13, S + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   S + kSmMaxSelectIssue + kLogicLatency, kRtovrInDrFpAddSt);
        append_sched(out.sched_logic, S + kSmMaxSelectIssue, logic_sel_great);
        hold_rtovr(out.sched_rtovr_35, S + kSmMaxSelectIssue + kLogicLatency, end,
                   kRtovrInDrLogicD);
        append_sched(out.sched_ur_w0, S + kSmWritePhase,
                     ur_write_code(kBankColAcc, src_dst, smc_id_en, wbe_lane(0)));
        return end + 2;
    };

    /** 两 broadcast 行 word@row_a/row_b → max 写 dst（四 lane 同值，全 wbe 写回）。 */
    const auto emit_max_broadcast_row_pair = [&](uint32_t start_c, uint32_t row_a,
                                                 uint32_t row_b, uint32_t dst) -> uint32_t {
        const uint32_t end = start_c + kSmMaxCompareSpan;
        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankColAcc, row_a, smc_id_en));
        append_sched(out.sched_ur_r1, start_c, ur_read_code(kBankColAcc, row_b, smc_id_en));
        hold_rtovr(out.sched_rtovr_5, start_c, end, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_6, start_c, end, kRtovrInUrRdata0 + 1);
        append_sched(out.sched_fpadd, start_c + kSmUrSettle, fpadd_check);
        hold_rtovr(out.sched_rtovr_11, start_c + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   start_c + kSmMaxSelectIssue + kLogicLatency, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_12, start_c + kSmUrSettle, start_c + kSmMaxSelectIssue + 1u,
                   kRtovrInUrRdata0 + 1);
        hold_rtovr(out.sched_rtovr_13, start_c + kSmUrSettle + kFpaddDrLatencyRtl - 1u,
                   start_c + kSmMaxSelectIssue + kLogicLatency, kRtovrInDrFpAddSt);
        append_sched(out.sched_logic, start_c + kSmMaxSelectIssue, logic_sel_great);
        hold_rtovr(out.sched_rtovr_35, start_c + kSmMaxSelectIssue + kLogicLatency, end,
                   kRtovrInDrLogicD);
        append_sched(out.sched_ur_w0, start_c + kSmWritePhase,
                     ur_write_code(kBankColAcc, dst, smc_id_en, 0xFFFFu));
        return end + 2;
    };

    /** bank6@src 复制到 @dst（UR 读→写，供 DS 转置前凑 4 行相同 [c0,c1,c2,c3]）。 */
    const auto emit_ur_copy_word = [&](uint32_t start_c, uint32_t src, uint32_t dst) -> uint32_t {
        const uint32_t write_c = start_c + kSmUrSettle;
        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankColAcc, src, smc_id_en));
        hold_rtovr(out.sched_rtovr_35, start_c, write_c + 1, kRtovrInUrRdata0);
        append_sched(out.sched_ur_w0, write_c,
                     ur_write_code(kBankColAcc, dst, smc_id_en, 0xFFFFu));
        return start_c + kSmUrWordCopyCycles;
    };

    /** m@bank6@scalar → bank5 四 lane 广播。 */
    const auto emit_broadcast_m_to_bank5 = [&](uint32_t start_c, uint32_t scalar_addr) -> uint32_t {
        uint32_t c = start_c;
        for (uint8_t dst = 0; dst < 4; ++dst) {
            const uint32_t tree_issue = c + kSmUrSettle + kSmAddTreeIssueGap;
            const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
            const uint32_t tree_write = tree_dr + 2;
            append_sched(out.sched_ur_r1, c, ur_read_code(kBankColAcc, scalar_addr, smc_id_en));
            hold_rtovr(out.sched_rtovr_32, c, tree_issue + 1, kRtovrInUrRdata0 + 1);
            hold_rtovr(out.sched_rtovr_33, c, tree_issue + 1, kRtovrInTiedZero);
            append_sched(out.sched_addtree8, tree_issue, tree_code_for(dst));
            hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
            append_sched(out.sched_ur_w0, tree_write,
                         ur_write_code(kBankSum, 0, smc_id_en, wbe_lane(dst)));
            c = tree_write + 3;
        }
        return c;
    };

    // ---- Phase A：列方向 tournament（pack 预置 bank6@0=-inf；每 word 同 lane max）----
    uint32_t c = 0;
    for (uint32_t wi = 0; wi < total_steps; ++wi) {
        c = emit_col_merge(c, wi);
    }
    out.para_col_body_cycles = kSmMaxCompareSpan;

    // ---- Phase B：列 max [c0,c1,c2,c3] → m@bank5 ----
    // skip_phase_b=true：纯 Phase A 列累积（acc 跨调用驻留 bank6@0），Phase B 由末次调用做。
    const uint32_t lane_tree_start = c;
    if (skip_phase_b) {
        out.para_phase_b_replicate_cycles = 0;
        out.para_phase_b_ds_cycles = 0;
        out.para_phase_b_compare_cycles = 0;
        out.para_phase_b_broadcast_cycles = 0;
    } else if (phase_b_mode == SoftmaxMaxReducePhaseBMode::DS_TRANSPOSE) {
        // B1：@0=[c0,c1,c2,c3] 复制到 @1..@3（4 行相同，DS 转置输入）
        const uint32_t rep_start = c;
        for (uint32_t dst = 1; dst < 4; ++dst) {
            c = emit_ur_copy_word(c, 0, dst);
        }
        out.para_phase_b_replicate_cycles = c - rep_start;

        // B2：STB→DS 4×4 转置→LDB 写回 bank6@0..3（周期模型，IQ 待 LSU+DS 接入）
        const uint32_t ds_start = c;
        c += kSmDsStb4WordCycles + kSmDsTranspose4x4Cycles + kSmDsLdb4WordCycles;
        out.para_phase_b_ds_cycles = c - ds_start;
        // 转置后 layout：Row_i @i = [ci,ci,ci,ci]

        // B3：3 步 tournament（行已 broadcast，直接 SELECT_GREAT）
        const uint32_t cmp_start = c;
        c = emit_max_broadcast_row_pair(c, 0, 1, kBankScratchM01);
        c = emit_max_broadcast_row_pair(c, 2, 3, kBankScratchM23);
        c = emit_max_broadcast_row_pair(c, kBankScratchM01, kBankScratchM23, kBankScratchM);
        out.para_phase_b_compare_cycles = c - cmp_start;

        const uint32_t bcast_start = c;
        c = emit_broadcast_m_to_bank5(c, kBankScratchM);
        out.para_phase_b_broadcast_cycles = c - bcast_start;
    } else {
        for (uint8_t lk = 0; lk < 4; ++lk) {
            c = emit_extract_lane(c, lk, kBankScalarBase + lk);
        }
        const uint32_t cmp_start = c;
        c = emit_max_scalar_pair(c, kBankScalarBase + 0, kBankScalarBase + 1, kBankScratchM01);
        c = emit_max_scalar_pair(c, kBankScalarBase + 2, kBankScalarBase + 3, kBankScratchM23);
        c = emit_max_scalar_pair(c, kBankScratchM01, kBankScratchM23, kBankScratchM);
        out.para_phase_b_compare_cycles = c - cmp_start;

        const uint32_t bcast_start = c;
        c = emit_broadcast_m_to_bank5(c, kBankScratchM);
        out.para_phase_b_broadcast_cycles = c - bcast_start;
    }

    out.para_lane_tree_cycles = c - lane_tree_start;
    out.para_body_cycles = c;
    out.para_total_cycles = c;

    const char* mode_str =
        (phase_b_mode == SoftmaxMaxReducePhaseBMode::DS_TRANSPOSE) ? "ds_transpose"
                                                                    : "extract_broadcast";
    std::cout << "[Softmax] Phase0a max-reduce mode=" << mode_str
              << " col_body=" << out.para_col_body_cycles
              << " cyc/word, steps=" << total_steps
              << " phase_b=" << out.para_lane_tree_cycles
              << " (rep=" << out.para_phase_b_replicate_cycles
              << " ds=" << out.para_phase_b_ds_cycles
              << " cmp=" << out.para_phase_b_compare_cycles
              << " bcast_m=" << out.para_phase_b_broadcast_cycles << ")"
              << " total_body=" << out.para_body_cycles << std::endl;
    return out;
}

// ---------------------------------------------------------------------------
// SoftmaxMaxSubInstructions (Phase0b: x' = x - broadcast(max@bank5))
// ---------------------------------------------------------------------------

std::vector<uint8_t> SoftmaxMaxSubInstructions::serialize() const {
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

std::vector<IqInstructionFile> SoftmaxMaxSubInstructions::get_iq_files() const {
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
        f.para_data = emit_exp_embed_iq(sched, code_bits, addr_inc, steps, body, "SoftmaxMaxSub");
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    if (para_use_vin_vout) {
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
        add_iq(sched_vin0, iq::IQ_ID_VIN0, "68_vin0", 7, false);
        add_iq(sched_vin1, iq::IQ_ID_VIN1, "69_vin1", 7, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false);
    }
    return files;
}

// ---------------------------------------------------------------------------
// SoftmaxSumReduceInstructions (Phase1.5: Σe → bank5@0)
// ---------------------------------------------------------------------------

std::vector<uint8_t> SoftmaxSumReduceInstructions::serialize() const {
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

std::vector<IqInstructionFile> SoftmaxSumReduceInstructions::get_iq_files() const {
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
        f.para_data = emit_exp_embed_iq(sched, code_bits, addr_inc, steps, body, "SoftmaxSum");
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10, false);
    add_iq(sched_addtree8, iq::IQ_ID_FP32_ADDTREE_8TO1, iq::IQ_NAME_FP32_ADDTREE_8TO1, 8, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false);
    add_iq(sched_rtovr_32, iq::IQ_ID_RTOVR_BASE + 32, "rtovr-32", 7, false);
    add_iq(sched_rtovr_33, iq::IQ_ID_RTOVR_BASE + 33, "rtovr-33", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    return files;
}

// ---------------------------------------------------------------------------
// SoftmaxNormInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SoftmaxNormInstructions::serialize() const {
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
    push(para_host_inv ? 1u : 0u);
    return r;
}

std::vector<IqInstructionFile> SoftmaxNormInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t steps = para_total_steps;
    const uint32_t body = para_body_cycles;

    auto add_iq = [&](const std::vector<SchedEntry>& sched, uint32_t iq_id,
                      const std::string& name, uint32_t code_bits, bool addr_inc,
                      uint32_t inc_from_cycle = 0u) {
        if (sched.empty()) return;
        IqInstructionFile f;
        f.para_iq_id = iq_id;
        f.para_iq_name = name;
        f.para_width_bits = 41 + code_bits;
        f.para_data = emit_exp_embed_iq(sched, code_bits, addr_inc, steps, body,
                                        "SoftmaxNorm", inc_from_cycle);
        files.push_back(f);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true,
           para_ur_r0_inc_from_cycle);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10, false);
    add_iq(sched_fptoint, iq::IQ_ID_FP_TO_INT, iq::IQ_NAME_FP_TO_INT, 10, false);
    add_iq(sched_intadd, iq::IQ_ID_INTADD, iq::IQ_NAME_INTADD, 18, false);
    add_iq(sched_inttofp, iq::IQ_ID_INT_TO_FP, iq::IQ_NAME_INT_TO_FP, 11, false);
    add_iq(sched_rtovr_0, iq::IQ_ID_RTOVR_BASE + 0, "rtovr-0", 7, false);
    add_iq(sched_rtovr_1, iq::IQ_ID_RTOVR_BASE + 1, "rtovr-1", 7, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false);
    add_iq(sched_rtovr_9, iq::IQ_ID_RTOVR_BASE + 9, "rtovr-9", 7, false);
    add_iq(sched_rtovr_10, iq::IQ_ID_RTOVR_BASE + 10, "rtovr-10", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    if (para_use_vin_vout) {
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
        add_iq(sched_vin0, iq::IQ_ID_VIN0, "68_vin0", 7, false);
        add_iq(sched_vin1, iq::IQ_ID_VIN1, "69_vin1", 7, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false);
    }
    return files;
}

// ---------------------------------------------------------------------------
// SoftmaxInstructions
// ---------------------------------------------------------------------------

std::vector<uint8_t> SoftmaxInstructions::serialize() const {
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
    push(para_sum_max_insts);
    push(para_stage1_max_insts);
    uint32_t sum_bits = 0, max_bits = 0;
    std::memcpy(&sum_bits, &para_preview_sum, sizeof(float));
    std::memcpy(&max_bits, &para_preview_max, sizeof(float));
    push(sum_bits);
    push(max_bits);
    push(para_host_max ? 1u : 0u);
    push(para_host_sum ? 1u : 0u);
    return r;
}

std::vector<IqInstructionFile> SoftmaxInstructions::get_iq_files() const {
    return stage0_exp.get_iq_files();
}

// ---------------------------------------------------------------------------
// SoftmaxOp
// ---------------------------------------------------------------------------

std::string SoftmaxOp::get_name() const { return "Softmax"; }

uint32_t SoftmaxOp::max_iq_inst_count(const std::vector<IqInstructionFile>& files) {
    uint32_t mx = 0;
    for (const auto& f : files) {
        const uint32_t nbytes = (f.para_width_bits + 7) / 8;
        if (nbytes == 0) continue;
        const uint32_t n = static_cast<uint32_t>(f.para_data.size() / nbytes);
        if (n > mx) mx = n;
    }
    return mx;
}

bool SoftmaxOp::parse_i32_param(const app::AppOperatorRequest& req,
                                const std::string& key,
                                int32_t& out) {
    auto it = req.para_parameters.find(key);
    if (it == req.para_parameters.end()) return false;
    if (!it->second.para_data_i32.empty()) {
        out = it->second.para_data_i32[0];
        return true;
    }
    if (!it->second.para_data_f32.empty()) {
        out = static_cast<int32_t>(it->second.para_data_f32[0]);
        return true;
    }
    return false;
}

bool SoftmaxOp::compute_stable_fp32(const app::AppOperatorRequest& req,
                                    std::vector<float>& x_prime,
                                    float& sum_exp,
                                    float& inv_sum,
                                    float* out_max) {
    const uint32_t n = logical_w(req);
    const auto x = collect_x(req, n);
    if (x.size() < n) return false;
    const StableVec s = decompose_stable_fp32(x);
    x_prime = s.x_prime;
    sum_exp = s.sum_exp;
    inv_sum = s.inv_sum;
    if (out_max) *out_max = s.max_v;
    return true;
}

SoftmaxMaxSubInstructions SoftmaxOp::compile_maxsub(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en) {
    // Phase0b: x' = x - broadcast(m@bank5) → bank2
    SoftmaxMaxSubInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    out.para_use_vin_vout = vin_vout_mode(req);

    iq::FpAddCru sub{};
    sub.para_inst_valid = true;
    sub.para_src_precision_32b = true;
    sub.para_dst_precision_32b = true;
    sub.para_update_status = false;
    sub.para_is_sub = true;
    const uint64_t sub_code =
        (static_cast<uint64_t>(iq::FpAddIq::pack_cru_fpadd(sub)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    const uint32_t S = 0;
    if (out.para_use_vin_vout) {
        // vv：m 由 host 预载 bank5@1；x→VIN0、m→VIN1，FPADD sub → VOUT0 → bank2。
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankRawX, 0, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankSum, kBankSumMaxWord, smc_id_en));
        append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← x
        append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN1 ← m
        hold_rtovr(out.sched_rtovr_5, S, S + 5, kRtovrInDrVin0D);
        hold_rtovr(out.sched_rtovr_6, S, S + 5, kRtovrInDrVin1D);
        append_sched(out.sched_fpadd, S + 4, sub_code);
        append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpAddD));  // VOUT0 ← x'
        hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
        append_sched(out.sched_ur_w0, S + 11, ur_write_code(kBankRawX, 0, smc_id_en));
    } else {
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankRawX, 0, smc_id_en));
        append_sched(out.sched_ur_r1, S, ur_read_code(kBankSum, 0, smc_id_en));
        // FPADD 口：RtoVR-5/6（非 FPMUL 的 3/4）
        hold_rtovr(out.sched_rtovr_5, S, S + kSmFpOpSpan, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_6, S, S + kSmFpOpSpan, kRtovrInUrRdata0 + 1);
        append_sched(out.sched_fpadd, S + kSmUrSettle, sub_code);
        hold_rtovr(out.sched_rtovr_35, S + kSmUrSettle + kFpaddDrLatencyRtl,
                   S + kSmFpOpSpan, kRtovrInDrFpAddD);
        append_sched(out.sched_ur_w0, S + kSmWritePhase,
                     ur_write_code(kBankRawX, 0, smc_id_en));
    }

    out.para_body_cycles = S + kSmFpOpSpan + 2;
    out.para_total_cycles = out.para_body_cycles * total_steps;
    if (out.para_use_vin_vout) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }

    std::cout << "[Softmax] Phase0 max-sub body=" << out.para_body_cycles
              << " cyc/word, steps=" << total_steps
              << (out.para_use_vin_vout ? " (vin_vout)" : "") << std::endl;
    return out;
}

SoftmaxSumReduceInstructions SoftmaxOp::compile_sum_reduce(
    const app::AppOperatorRequest& /*req*/,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    bool skip_broadcast) {
    // Phase1.5 per word（外层 EMBED × steps）：
    //   SMC ADDTREE8：result = ΣS0[0..3] + ΣS1[0..3]；DR 仅 dst lane 有 32b 结果。
    //   steps==1：S0=e@bank3, S1=0 → 用 wbe 四次写 bank5 四 lane 同 S
    //   steps>1：S0=e, S1=[acc,0,0,0]@bank6 → 写回 bank6；再广播 acc→bank5
    SoftmaxSumReduceInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;

    iq::Fp32AddTree8To1Cru tree{};
    tree.para_inst_valid = true;

    const auto tree_code_for = [&](uint8_t dst_lane) -> uint64_t {
        tree.para_dst_reg_enc = dst_lane & 0x3u;
        return (static_cast<uint64_t>(
                    iq::Fp32AddTree8To1Iq::pack_cru_fp32addtree8to1(tree)) |
                (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 3));
    };

    // UR write byte-enable：只更新 dst lane 的 4 字节，保留同 word 其它 lane
    const auto wbe_lane = [](uint8_t dst_lane) -> uint16_t {
        return static_cast<uint16_t>(0xFu << (4u * (dst_lane & 0x3u)));
    };

    const auto emit_tree_on_e = [&](uint32_t start_c, uint8_t dst_lane) -> uint32_t {
        const uint32_t tree_issue = start_c + kSmUrSettle + kSmAddTreeIssueGap;
        const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
        const uint32_t tree_write = tree_dr + 2;
        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankExpOut, 0, smc_id_en));
        hold_rtovr(out.sched_rtovr_32, start_c, tree_issue + 1, kRtovrInUrRdata0);
        hold_rtovr(out.sched_rtovr_33, start_c, tree_issue + 1, kRtovrInTiedZero);
        append_sched(out.sched_addtree8, tree_issue, tree_code_for(dst_lane));
        hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
        append_sched(out.sched_ur_w0, tree_write,
                     ur_write_code(kBankSum, 0, smc_id_en, wbe_lane(dst_lane)));
        return tree_write + 3;
    };

    /** 多 word：S0=e@bank3, S1=acc@bank6 → new_acc@bank6（dst=0）。 */
    const auto emit_acc_plus_e_tree = [&](uint32_t start_c) -> uint32_t {
        const uint32_t tree_issue = start_c + kSmUrSettle + kSmAddTreeIssueGap;
        const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
        const uint32_t tree_write = tree_dr + 2;
        append_sched(out.sched_ur_r0, start_c, ur_read_code(kBankExpOut, 0, smc_id_en));
        append_sched(out.sched_ur_r1, start_c, ur_read_code(kBankNewton, 0, smc_id_en));
        hold_rtovr(out.sched_rtovr_32, start_c, tree_issue + 1, kRtovrInUrRdata0);      // S0 = e
        hold_rtovr(out.sched_rtovr_33, start_c, tree_issue + 1, kRtovrInUrRdata0 + 1);  // S1 = acc
        append_sched(out.sched_addtree8, tree_issue, tree_code_for(0));
        hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
        // DR 为 sticky：只更新 dst lane，其它 lane 保留旧值。广播 dst=0..3 后
        // DR=[acc,acc,acc,acc]；若 wbe=0xFFFF 整字写回，下一拍 S1 会把 3×旧 acc
        // 再加进树（rtl≈27.6）。必须只写 lane0，保持 bank6=[acc,0,0,0]。
        append_sched(out.sched_ur_w0, tree_write,
                     ur_write_code(kBankNewton, 0, smc_id_en, wbe_lane(0)));
        return tree_write + 3;
    };

    /** bank6@0 标量 acc → bank5 指定 lane（S0=acc, S1=0；wbe 只改该 lane）。 */
    const auto emit_broadcast_acc_to_bank5 = [&](uint32_t start_c, uint8_t dst_lane) -> uint32_t {
        const uint32_t tree_issue = start_c + kSmUrSettle + kSmAddTreeIssueGap;
        const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
        const uint32_t tree_write = tree_dr + 2;
        append_sched(out.sched_ur_r1, start_c, ur_read_code(kBankNewton, 0, smc_id_en));
        hold_rtovr(out.sched_rtovr_32, start_c, tree_issue + 1, kRtovrInUrRdata0 + 1);
        hold_rtovr(out.sched_rtovr_33, start_c, tree_issue + 1, kRtovrInTiedZero);
        append_sched(out.sched_addtree8, tree_issue, tree_code_for(dst_lane));
        hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
        append_sched(out.sched_ur_w0, tree_write,
                     ur_write_code(kBankSum, 0, smc_id_en, wbe_lane(dst_lane)));
        return tree_write + 3;
    };

    uint32_t c = 0;
    if (total_steps <= 1u) {
        for (uint8_t dst = 0; dst < 4; ++dst) {
            c = emit_tree_on_e(c, dst);
        }
    } else {
        c = emit_acc_plus_e_tree(0);
        // skip_broadcast=true：纯跨 word/跨段累积（acc 驻留 bank6@0），
        // S→bank5 广播由末次 compile_sum_broadcast() 统一做。
        if (!skip_broadcast) {
            for (uint8_t dst = 0; dst < 4; ++dst) {
                c = emit_broadcast_acc_to_bank5(c, dst);
            }
        }
    }

    out.para_body_cycles = c;
    out.para_total_cycles = out.para_body_cycles * total_steps;

    std::cout << "[Softmax] Phase1.5 sum-reduce body=" << out.para_body_cycles
              << " cyc/word, steps=" << total_steps << std::endl;
    return out;
}

SoftmaxSumReduceInstructions SoftmaxOp::compile_sum_broadcast(uint8_t smc_id_en) {
    // 跨段累积（compile_sum_reduce(..., skip_broadcast=true)）完成后，
    // bank6@0 标量 Σ → bank5 四 lane 广播 S。单次触发，无外层 step fold。
    SoftmaxSumReduceInstructions out;
    out.para_total_steps = 0;
    out.para_total_elements = 0;
    out.para_smc_id_en = smc_id_en;

    iq::Fp32AddTree8To1Cru tree{};
    tree.para_inst_valid = true;
    const auto tree_code_for = [&](uint8_t dst_lane) -> uint64_t {
        tree.para_dst_reg_enc = dst_lane & 0x3u;
        return (static_cast<uint64_t>(
                    iq::Fp32AddTree8To1Iq::pack_cru_fp32addtree8to1(tree)) |
                (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 3));
    };
    const auto wbe_lane = [](uint8_t dst_lane) -> uint16_t {
        return static_cast<uint16_t>(0xFu << (4u * (dst_lane & 0x3u)));
    };

    uint32_t c = 0;
    for (uint8_t dst = 0; dst < 4; ++dst) {
        const uint32_t tree_issue = c + kSmUrSettle + kSmAddTreeIssueGap;
        const uint32_t tree_dr = tree_issue + kSmAddTreeLatency;
        const uint32_t tree_write = tree_dr + 2;
        append_sched(out.sched_ur_r1, c, ur_read_code(kBankNewton, 0, smc_id_en));
        hold_rtovr(out.sched_rtovr_32, c, tree_issue + 1, kRtovrInUrRdata0 + 1);
        hold_rtovr(out.sched_rtovr_33, c, tree_issue + 1, kRtovrInTiedZero);
        append_sched(out.sched_addtree8, tree_issue, tree_code_for(dst));
        hold_rtovr(out.sched_rtovr_35, tree_dr, tree_write + 1, kRtovrInDrFp32AddTree8D);
        append_sched(out.sched_ur_w0, tree_write,
                     ur_write_code(kBankSum, 0, smc_id_en, wbe_lane(dst)));
        c = tree_write + 3;
    }

    out.para_body_cycles = c;
    out.para_total_cycles = c;
    std::cout << "[Softmax] Phase1.5 sum-broadcast body=" << out.para_body_cycles
              << " cyc (acc@bank6@0 -> S@bank5@0)" << std::endl;
    return out;
}

SoftmaxNormInstructions SoftmaxOp::compile_normalize(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t newton_iters) {
    // Phase2 body（EMBED fold）：
    //   可选 magic(x0)：FPTOINT(S) → INTADD(MAGIC-bits) → INTTOFP → bank6
    //   Newton×N: 1/S on bank5(S) → bank6(inv)
    //   FPMUL:    e @bank3 × inv @bank6 → y @bank3
    // Host 预载：bank5=S 广播；bank6=magic(S)（SOFTMAX_HOST_X0=1）；
    //            bank4@8=2.0，bank4@9=MAGIC（Exp 占 word0..7）
    // vv：所有 UR 源经 VIN0/VIN1 捕获（捕获@读+1，消费@读+4）；FPMUL/INTADD 结果
    //     经 VOUT0 回写（捕获@发射+4，写=捕获+3）；块间 DR 直供窗口不变（合法源）。

    SoftmaxNormInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_newton_iters = newton_iters;
    out.para_smc_id_en = smc_id_en;
    out.para_host_inv = host_inv_mode();
    out.para_host_x0 = host_x0_mode();
    out.para_magic_only = magic_only_mode();
    out.para_use_vin_vout = vin_vout_mode(req);
    const bool vv = out.para_use_vin_vout;

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = true;
    mul.para_dst_precision_is_32b = true;
    mul.para_smc_id_en = smc_id_en;
    const uint64_t mul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    iq::FpAddCru sub;
    sub.para_inst_valid = true;
    sub.para_src_precision_32b = true;
    sub.para_dst_precision_32b = true;
    sub.para_update_status = false;
    sub.para_is_sub = true;
    const uint64_t sub_code =
        (static_cast<uint64_t>(iq::FpAddIq::pack_cru_fpadd(sub)) |
         (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    const uint64_t intadd_sub_code = pack_intadd_sub_code(smc_id_en);

    uint32_t c = 0;

    if (!out.para_host_x0) {
        c = append_magic_x0_prologue(out, c, smc_id_en, intadd_sub_code);
    }

    if (out.para_magic_only) {
        out.para_ur_r0_inc_from_cycle = c;
        out.para_body_cycles = c;
        out.para_total_cycles = c * total_steps;
        out.para_newton_iters = 0;
        if (vv) {
            fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                                   out.para_body_cycles);
        }
        std::cout << "[Softmax] Stage1 magic-x0-only body=" << out.para_body_cycles
                  << " cyc/word, steps=" << total_steps << std::endl;
        return out;
    }

    if (!out.para_host_inv && newton_iters > 0u) {
        for (uint32_t it = 0; it < newton_iters; ++it) {
            // t = S * x_n
            {
                const uint32_t S = c;
                append_sched(out.sched_ur_r0, S, ur_read_code(kBankSum, kBankSumSumWord, smc_id_en));
                append_sched(out.sched_ur_r2, S, ur_read_code(kBankNewton, 0, smc_id_en));
                if (vv) {
                    append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← S
                    append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 2));  // VIN1 ← x_n
                    hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
                    hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
                    append_sched(out.sched_fpmul, S + 4, mul_code);
                    // FPMUL_D@S+9 起粘滞；u 块 FPADD@S+17 采样（寄存器保持）
                    hold_rtovr(out.sched_rtovr_6, S + 8, S + kSmFpOpSpan, kRtovrInDrFpMulD);
                } else {
                    hold_rtovr(out.sched_rtovr_3, S, S + kSmFpOpSpan, kRtovrInUrRdata0);
                    hold_rtovr(out.sched_rtovr_4, S, S + kSmFpOpSpan, kRtovrInUrRdata0 + 2);
                    append_sched(out.sched_fpmul, S + kSmUrSettle, mul_code);
                    hold_rtovr(out.sched_rtovr_6, S + kSmUrSettle + kFpmulDrLatencyRtl,
                               S + kSmFpOpSpan, kRtovrInDrFpMulD);
                }
                c = S + kSmFpOpSpan;
            }
            // u = 2 - t
            {
                const uint32_t S = c;
                append_sched(out.sched_ur_r1, S,
                             ur_read_code(kBankConst, kBankConstNewtonTwo, smc_id_en));
                if (vv) {
                    append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0 + 1));  // VIN0 ← 2.0
                    hold_rtovr(out.sched_rtovr_5, S, S + 5, kRtovrInDrVin0D);
                    // rtovr_6 沿用 t 块保持的 FPMUL_D（本块不重发）
                    append_sched(out.sched_fpadd, S + 4, sub_code);
                    hold_rtovr(out.sched_rtovr_4, S + 8, S + kSmFpOpSpan, kRtovrInDrFpAddD);
                } else {
                    hold_rtovr(out.sched_rtovr_5, S, S + kSmFpOpSpan, kRtovrInUrRdata0 + 1);
                    hold_rtovr(out.sched_rtovr_6, S, S + kSmFpOpSpan, kRtovrInDrFpMulD);
                    append_sched(out.sched_fpadd, S + kSmUrSettle, sub_code);
                    hold_rtovr(out.sched_rtovr_4, S + kSmUrSettle + kFpaddDrLatencyRtl,
                               S + kSmFpOpSpan, kRtovrInDrFpAddD);
                }
                c = S + kSmFpOpSpan;
            }
            // x = x_n * u → bank6
            {
                const uint32_t S = c;
                append_sched(out.sched_ur_r0, S, ur_read_code(kBankNewton, 0, smc_id_en));
                if (vv) {
                    append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← x_n
                    hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
                    hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrFpAddD);
                    append_sched(out.sched_fpmul, S + 4, mul_code);
                    append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));  // VOUT0 ← x_{n+1}
                    hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
                    append_sched(out.sched_ur_w0, S + 11,
                                 ur_write_code(kBankNewton, 0, smc_id_en));
                } else {
                    hold_rtovr(out.sched_rtovr_3, S, S + kSmFpOpSpan, kRtovrInUrRdata0);
                    hold_rtovr(out.sched_rtovr_4, S, S + kSmFpOpSpan, kRtovrInDrFpAddD);
                    append_sched(out.sched_fpmul, S + kSmUrSettle, mul_code);
                    hold_rtovr(out.sched_rtovr_35, S + kSmUrSettle + kFpmulDrLatencyRtl,
                               S + kSmFpOpSpan, kRtovrInDrFpMulD);
                    append_sched(out.sched_ur_w0, S + kSmWritePhase,
                                 ur_write_code(kBankNewton, 0, smc_id_en));
                }
                c = S + kSmFpOpSpan + 2;
            }
        }
        out.para_run_newton_in_body = true;
    } else {
        out.para_newton_iters = 0;
    }

    // ---- y = e × inv → bank3 ----
    {
        const uint32_t S = c;
        out.para_ur_r0_inc_from_cycle = S;
        append_sched(out.sched_ur_r0, S, ur_read_code(kBankExpOut, 0, smc_id_en));
        append_sched(out.sched_ur_r2, S, ur_read_code(kBankNewton, 0, smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));      // VIN0 ← e
            append_sched(out.sched_rtovr_15, S + 1, rtovr_code(kRtovrInUrRdata0 + 2));  // VIN1 ← inv
            hold_rtovr(out.sched_rtovr_3, S, S + 5, kRtovrInDrVin0D);
            hold_rtovr(out.sched_rtovr_4, S, S + 5, kRtovrInDrVin1D);
            append_sched(out.sched_fpmul, S + 4, mul_code);
            append_sched(out.sched_rtovr_16, S + 8, rtovr_code(kRtovrInDrFpMulD));  // VOUT0 ← y
            hold_rtovr(out.sched_rtovr_35, S + 8, S + 11, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 11,
                         ur_write_code(kBankExpOut, 0, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_3, S, S + kSmFpOpSpan, kRtovrInUrRdata0);
            hold_rtovr(out.sched_rtovr_4, S, S + kSmFpOpSpan, kRtovrInUrRdata0 + 2);
            append_sched(out.sched_fpmul, S + kSmUrSettle, mul_code);
            hold_rtovr(out.sched_rtovr_35, S + kSmUrSettle + kFpmulDrLatencyRtl,
                       S + kSmFpOpSpan, kRtovrInDrFpMulD);
            append_sched(out.sched_ur_w0, S + kSmWritePhase,
                         ur_write_code(kBankExpOut, 0, smc_id_en));
        }
        c = S + kSmFpOpSpan + 2;
    }

    out.para_body_cycles = c;
    out.para_total_cycles = c * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }

    if (out.para_body_cycles > kSoftmaxIqDepthLimit * 4u) {
        std::cerr << "[Softmax] Stage1 body=" << out.para_body_cycles
                  << " too long for IQ" << std::endl;
    }

    std::cout << "[Softmax] Stage1 norm body=" << out.para_body_cycles
              << " cyc/word, steps=" << total_steps
              << ", newton_iters=" << out.para_newton_iters
              << ", host_inv=" << (out.para_host_inv ? "1" : "0")
              << ", host_x0=" << (out.para_host_x0 ? "1" : "0")
              << ", magic_only=" << (out.para_magic_only ? "1" : "0") << std::endl;
    return out;
}

SoftmaxInstructions SoftmaxOp::compile_two_stage(const app::AppOperatorRequest& req) {
    SoftmaxInstructions out;
    out.para_host_max = host_max_mode();
    out.para_host_sum = host_sum_mode();

    std::vector<float> x_prime;
    float sum_exp = 0.f, inv_sum = 0.f, max_v = 0.f;
    if (!compute_stable_fp32(req, x_prime, sum_exp, inv_sum, &max_v)) {
        throw std::runtime_error("Softmax: compute_stable_fp32 failed");
    }
    out.para_preview_sum = sum_exp;
    out.para_preview_max = max_v;

    const uint32_t w = logical_w(req);
    std::vector<float> exp_in = pad_vector_fp32(x_prime, w);
    const auto exp_req = make_exp_xprime_request(req, exp_in);
    out.stage0_exp = ExpOp::compile_merged(exp_req);

    const uint8_t smc_en = get_smc_id_en(req);
    const uint32_t iters = get_newton_iters(req);
    const uint32_t steps = out.stage0_exp.para_total_steps;
    const uint32_t elems = out.stage0_exp.para_total_elements;

    if (!out.para_host_max) {
        out.phase0_max_reduce =
            compile_max_reduce(req, steps, elems, smc_en);
        out.phase0_maxsub = compile_maxsub(req, steps, elems, smc_en);
        out.para_phase0_max_insts = std::max(
            max_iq_inst_count(out.phase0_max_reduce.get_iq_files()),
            max_iq_inst_count(out.phase0_maxsub.get_iq_files()));
        out.para_phase0_fits = out.para_phase0_max_insts <= kSoftmaxIqDepthLimit;
    }

    if (!out.para_host_sum) {
        out.stage_sum_reduce = compile_sum_reduce(req, steps, elems, smc_en);
        out.para_sum_max_insts = max_iq_inst_count(out.stage_sum_reduce.get_iq_files());
        out.para_sum_fits = out.para_sum_max_insts <= kSoftmaxIqDepthLimit;
    }

    out.stage1_norm = compile_normalize(req, steps, elems, smc_en, iters);
    out.para_stage0_max_insts = max_iq_inst_count(out.stage0_exp.get_iq_files());
    out.para_stage1_max_insts = max_iq_inst_count(out.stage1_norm.get_iq_files());
    out.para_stage0_fits = out.para_stage0_max_insts <= kSoftmaxIqDepthLimit;
    out.para_stage1_fits = out.para_stage1_max_insts <= kSoftmaxIqDepthLimit;

    uint32_t nstages = 2;
    if (!out.para_host_max) ++nstages;
    if (!out.para_host_sum) ++nstages;
    out.para_num_stages = nstages;

    std::cout << "[Softmax] compile: W=" << w << " words=" << softmax_words(w)
              << " pad=" << kSoftmaxPadValue
              << " host_max=" << (out.para_host_max ? "1" : "0")
              << " host_sum=" << (out.para_host_sum ? "1" : "0")
              << " Phase0(maxsub) iq=" << out.para_phase0_max_insts
              << " Phase1(Exp) iq=" << out.para_stage0_max_insts
              << " Phase1.5(sum) iq=" << out.para_sum_max_insts
              << " Phase2(norm) iq=" << out.para_stage1_max_insts
              << " S=" << sum_exp << " m=" << max_v << std::endl;

    if (!out.para_phase0_fits || !out.para_stage0_fits || !out.para_sum_fits ||
        !out.para_stage1_fits) {
        std::cerr << "[Softmax][WARN] IQ depth over limit" << std::endl;
    }
    return out;
}

OpInstructionsBase* SoftmaxOp::compile(const app::AppOperatorRequest& req) {
    auto* insts = new SoftmaxInstructions();
    *insts = compile_two_stage(req);
    return insts;
}

bool SoftmaxOp::generate_data(const app::AppOperatorRequest& req,
                              std::vector<uint8_t>& input_bin,
                              std::vector<uint8_t>& other_bin,
                              std::vector<uint8_t>& alpha_bin) {
    const uint32_t w = logical_w(req);
    const auto x = collect_x(req, w);
    if (x.size() < w) return false;

    const bool fp16 = [&]() {
        auto it = req.para_inputs.find("input_tensor");
        return it != req.para_inputs.end() &&
               it->second.para_dtype == app::AppDataType::FLOAT16;
    }();

    std::vector<float> x_prime;
    float sum_exp = 0.f, inv_sum = 0.f, max_v = 0.f;
    if (!compute_stable_fp32(req, x_prime, sum_exp, inv_sum, &max_v)) return false;

    std::vector<float> bank2_data;
    if (host_max_mode()) {
        bank2_data = fp16 ? pad_vector_fp16_semantics(x_prime, w)
                          : pad_vector_fp32(x_prime, w);
    } else {
        bank2_data = fp16 ? pad_vector_fp16_semantics(x, w) : pad_vector_fp32(x, w);
    }

    fill_merged_exp_host_data(bank2_data, fp16, input_bin, other_bin, alpha_bin);

    std::cout << "[Softmax] generate_data: W=" << w << " padded=" << softmax_padded_w(w)
              << " pad=" << kSoftmaxPadValue
              << " host_max=" << (host_max_mode() ? "1(x')" : "0(raw x)")
              << " host_sum=" << (host_sum_mode() ? "1" : "0")
              << " S=" << sum_exp << " m=" << max_v << std::endl;
    return true;
}

bool SoftmaxOp::calculate_golden(const app::AppOperatorRequest& req,
                                 std::vector<uint8_t>& golden_bin) {
    const uint32_t n = req.para_dimensions.para_n * req.para_dimensions.para_c *
                       req.para_dimensions.para_h * req.para_dimensions.para_w;
    const auto x = collect_x(req, n);
    if (x.size() < n) return false;

    const uint32_t newton_iters = get_newton_iters(req);
    const auto y = softmax_host_fp32(x, newton_iters);

    const bool fp16_out = [&]() {
        auto it = req.para_inputs.find("input_tensor");
        return it != req.para_inputs.end() &&
               it->second.para_dtype == app::AppDataType::FLOAT16;
    }();

    golden_bin.clear();
    for (uint32_t i = 0; i < n; ++i) {
        if (fp16_out) {
            const uint16_t h = app::float32_to_float16(y[i]);
            golden_bin.push_back(h & 0xFFu);
            golden_bin.push_back((h >> 8) & 0xFFu);
        } else {
            uint32_t bits = 0;
            std::memcpy(&bits, &y[i], sizeof(float));
            golden_bin.push_back(bits & 0xFFu);
            golden_bin.push_back((bits >> 8) & 0xFFu);
            golden_bin.push_back((bits >> 16) & 0xFFu);
            golden_bin.push_back((bits >> 24) & 0xFFu);
        }
    }
    return true;
}

void SoftmaxOp::fill_random_request(app::AppOperatorRequest& req,
                                    std::mt19937& rng,
                                    std::uniform_real_distribution<float>&,
                                    const std::string& dtype_str) {
    req.para_operator = "Softmax";
    std::uniform_real_distribution<float> dist(-5.0f, 3.3f);
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

    app::AppTensorDesc dim_desc;
    dim_desc.para_name = "dim";
    dim_desc.para_dtype = app::AppDataType::INT32;
    dim_desc.para_shape = {1};
    dim_desc.para_data_i32 = {-1};
    req.para_parameters["dim"] = dim_desc;
}

REGISTER_OP(SoftmaxOp);

}  // namespace op
}  // namespace aiacc
