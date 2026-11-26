#include "exp_preprocess_op.h"

#include <algorithm>
#include <iostream>

#include "../../iq/iq_id_defs.h"
#include "../../iq/modules/fpadd_iq.h"
#include "../../iq/modules/fpmul_iq.h"
#include "../../iq/modules/fptoint_iq.h"
#include "../../iq/modules/intadd_iq.h"
#include "../../iq/modules/inttofp_iq.h"
#include "../../iq/modules/logical_iq.h"
#include "../common/op_registry.h"

namespace aiacc {
namespace op {

using namespace exp_iq;
using namespace exp_math;

namespace {

// 半整数归约：n=RN(tw) 魔法数 + √2 SELECT；k-flush 见 compile_kflush_p2_overlay（与 Horner 并行）。
// Core body=56（WriteScaleScratch@53）。N=10k 证明：尾部/魔法数再压会导致奇数 n（√2）相对 Host 全错。
// MAGIC bits=0x4B400000。
constexpr uint32_t kPreReadPhase = 0;
constexpr uint32_t kPreFpmulTPhase = 3;
constexpr uint32_t kPreReadMagicPhase = 6;
constexpr uint32_t kPreFpaddTwPhase = 9;
constexpr uint32_t kPreFpaddTwMagicPhase = 15;
constexpr uint32_t kPreIntaddNPhase = 21;
constexpr uint32_t kPreFpaddTw2Phase = 22;
constexpr uint32_t kPreReadIntOnePhase = 22;
constexpr uint32_t kPreInttofpNPhase = 23;
constexpr uint32_t kPreLogicAndOddPhase = 25;
constexpr uint32_t kPreInttofpOddPhase = 27;
constexpr uint32_t kPreFpaddDiffPhase = 28;
constexpr uint32_t kPreLogicShrMPhase = 28;
constexpr uint32_t kPreReadHalf2Phase = 28;
constexpr uint32_t kPreReadBiasPhase = 29;
constexpr uint32_t kPreIntaddExpPhase = 32;
constexpr uint32_t kPreWriteKfPhase = 33;
constexpr uint32_t kPreFpmulRPhase = 34;
constexpr uint32_t kPreReadShiftPhase = 34;
constexpr uint32_t kPreLogicShlScalePhase = 37;
constexpr uint32_t kPreWriteRPhase = 41;
constexpr uint32_t kPreReadOddZeroPhase = 41;
constexpr uint32_t kPreReadSqrt2Phase = 42;
constexpr uint32_t kPreFpaddOddCheckPhase = 44;
constexpr uint32_t kPreFpmulScaleOddPhase = 45;
constexpr uint32_t kPreLogicSelectScalePhase = 50;
constexpr uint32_t kPreWriteScaleScratchPhase = 53;  // core 结束 → body=56



uint16_t pack_fpadd_code(bool fp16, bool is_sub, bool update_status, uint8_t smc_en) {
    iq::FpAddCru add{};
    add.para_inst_valid = true;
    add.para_src_precision_32b = !fp16;
    add.para_dst_precision_32b = !fp16;
    add.para_update_status = update_status;
    add.para_is_sub = is_sub;
    return static_cast<uint16_t>(iq::FpAddIq::pack_cru_fpadd(add)) |
           (static_cast<uint16_t>(smc_en & 0x1Fu) << 5);
}

uint16_t pack_logic_code(bool fp16, iq::LogicalOp op, uint8_t smc_en) {
    iq::LogicalCru logic{};
    logic.para_inst_valid = true;
    logic.para_precision_32b = !fp16;
    logic.para_opcode = op;
    logic.para_smc_id_en = smc_en;
    return iq::LogicalIq::pack_cru_logical(logic);
}

uint16_t pack_intadd_code(bool is_sub, uint8_t smc_en) {
    iq::IntAddCru iadd{};
    iadd.para_inst_valid = true;
    iadd.para_precision_s0 = 3;
    iadd.para_precision_s1 = 3;
    iadd.para_precision_s2 = 3;
    iadd.para_sign_s0 = true;
    iadd.para_sign_s1 = false;
    iadd.para_sign_s2 = false;
    iadd.para_sign_d = true;
    iadd.para_is_sub = is_sub;
    return static_cast<uint16_t>(iq::IntAddIq::pack_cru_intadd(iadd)) |
           (static_cast<uint16_t>(smc_en & 0x1Fu) << 13);
}

}  // namespace

std::string ExpPreprocessOp::get_name() const { return "ExpPreprocess"; }

void ExpPreprocessOp::apply_smc_params(ExpPreprocessParams& params,
                                       const app::AppOperatorRequest& req) {
    auto get_param_i32 = [&](const std::string& key, int def) -> int {
        auto it = req.para_parameters.find(key);
        if (it != req.para_parameters.end() && !it->second.para_data_i32.empty()) {
            return it->second.para_data_i32[0];
        }
        return def;
    };
    const int num_smc = get_param_i32("num_smc", 1);
    int smc_id_en = get_param_i32("smc_id_en", -1);
    if (smc_id_en < 0) smc_id_en = (num_smc > 1) ? (num_smc - 1) : 0;
    params.para_smc_id_en = static_cast<uint8_t>(smc_id_en & 0x1F);

    int rtovr_ext = get_param_i32("rtovr_read_extend_loops", -1);
    if (rtovr_ext < 0) rtovr_ext = (num_smc > 1) ? (num_smc - 1) : 0;
    if (rtovr_ext < 0) rtovr_ext = 0;
    if (rtovr_ext > 1023) rtovr_ext = 1023;
    params.para_rtovr_read_extend_loops = static_cast<uint16_t>(rtovr_ext);

    params.para_use_vin_vout = (get_param_i32("vin_vout", 0) != 0);
}

ExpPreprocessParams ExpPreprocessOp::parse_request(
    const app::AppOperatorRequest& req) const {
    ExpPreprocessParams p;
    p.para_n = req.para_dimensions.para_n;
    p.para_c = req.para_dimensions.para_c;
    p.para_h = req.para_dimensions.para_h;
    p.para_w = req.para_dimensions.para_w;

    auto input_it = req.para_inputs.find("input_tensor");
    if (input_it != req.para_inputs.end() &&
        input_it->second.para_dtype == app::AppDataType::FLOAT16) {
        p.para_dtype = 1;
    }
    return p;
}

ExpPreprocessInstructions ExpPreprocessOp::compile_preprocess(
    const ExpPreprocessParams& params) const {
    ExpPreprocessInstructions result;
    const uint32_t total_elems = params.para_n * params.para_c *
                                 params.para_h * params.para_w;
    const uint32_t total_steps =
        (total_elems + params.get_elems_per_word() - 1) / params.get_elems_per_word();

    result.para_total_steps = total_steps;
    result.para_total_elements = total_elems;

    const bool fp16 = (params.para_dtype == 1);
    const uint8_t smc_en = params.para_smc_id_en;
    const bool vv = params.para_use_vin_vout;
    // vv 实测：VIN 捕获@C → VIN_D 从 C+3 起有效（原假设 +1 全链差 2 拍）。
    // 对策：core 计算/写发射统一 +1（UR 读与 VIN 捕获不动；路由窗上界自带 +1
    // 余量恰好覆盖），VOUT0 写按捕获+3 落位，body 56→57。
    const uint32_t vs = vv ? 1u : 0u;
    const uint32_t cru_comp = smc_cru_delay_comp(smc_en);
    const uint32_t body_cycles = kPreprocessCoreCycles + cru_comp + vs;
    result.para_body_cycles = body_cycles;
    result.para_total_cycles = body_cycles * total_steps;

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = !fp16;
    mul.para_dst_precision_is_32b = !fp16;
    mul.para_smc_id_en = smc_en;
    result.para_fpmul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    result.para_fpadd_sub_code = pack_fpadd_code(fp16, true, false, smc_en);
    result.para_fpadd_add_code = pack_fpadd_code(fp16, false, false, smc_en);
    result.para_fpadd_check_code = pack_fpadd_code(fp16, true, true, smc_en);

    iq::FpToIntCru f2i{};
    f2i.para_inst_valid = true;
    f2i.para_src_is_32b = !fp16;
    f2i.para_dst_is_32b = !fp16;
    result.para_fptoint_code =
        static_cast<uint16_t>(iq::FpToIntIq::pack_cru_fptoint(f2i)) |
        (static_cast<uint16_t>(smc_en & 0x1Fu) << 5);

    iq::IntToFpCru i2f{};
    i2f.para_inst_valid = true;
    i2f.para_src_is_32b = !fp16;
    i2f.para_dst_is_32b = !fp16;
    i2f.para_src_signed = true;
    result.para_inttofp_code =
        static_cast<uint16_t>(iq::IntToFpIq::pack_cru_inttofp(i2f)) |
        (static_cast<uint16_t>(smc_en & 0x1Fu) << 6);

    result.para_intadd_sub_code = pack_intadd_code(true, smc_en);
    result.para_intadd_exp_code = pack_intadd_code(false, smc_en);

    result.para_logic_select_less_code =
        pack_logic_code(fp16, iq::LogicalOp::SELECT_LESS, smc_en);
    result.para_logic_select_eq_code =
        pack_logic_code(fp16, iq::LogicalOp::SELECT_EQUAL, smc_en);
    result.para_logic_select_great_code =
        pack_logic_code(fp16, iq::LogicalOp::SELECT_GREAT, smc_en);
    result.para_logic_shift_code =
        pack_logic_code(fp16, iq::LogicalOp::LOGIC_LEFT_SHIFT, smc_en);
    result.para_logic_and_code =
        pack_logic_code(fp16, iq::LogicalOp::AND, smc_en);
    result.para_logic_shr_code =
        pack_logic_code(fp16, iq::LogicalOp::ARITH_RIGHT_SHIFT, smc_en);

    std::cout << "[ExpPreprocess] core: n=RN(tw) magic, r, scale0*(√2?); "
              << "k-flush deferred→P2∥Horner; raw_x B"
              << static_cast<int>(params.para_raw_x_bank) << "@" << params.para_raw_x_addr
              << " -> r B" << static_cast<int>(params.para_r_bank)
              << ", scale_scratch@" << params.para_scale_scratch_addr
              << ", magic B" << static_cast<int>(params.para_log2e_bank)
              << "@" << params.para_round_magic_addr
              << ", steps=" << total_steps
              << ", per_elem_cycles=" << body_cycles
              << ", smc_cru_delay_comp=" << cru_comp << std::endl;

    // --- UR reads（core；flush 常数在 overlay）---
    result.sched_ur_r0.push_back({kPreReadPhase,
        ur_read_code(params.para_raw_x_bank, params.para_raw_x_addr, smc_en)});
    result.sched_ur_r1.push_back({kPreReadPhase,
        ur_read_code(params.para_log2e_bank, params.para_log2e_addr, smc_en)});
    result.sched_ur_r1.push_back({kPreReadHalf2Phase,
        ur_read_code(params.para_log2e_bank, params.para_half_addr, smc_en)});
    result.sched_ur_r1.push_back({kPreReadSqrt2Phase,
        ur_read_code(params.para_log2e_bank, params.para_sqrt2_addr, smc_en)});

    result.sched_ur_r2.push_back({kPreReadMagicPhase,
        ur_read_code(params.para_log2e_bank, params.para_round_magic_addr, smc_en)});
    result.sched_ur_r2.push_back({kPreReadIntOnePhase,
        ur_read_code(params.para_log2e_bank, params.para_int_one_addr, smc_en)});
    result.sched_ur_r2.push_back({kPreReadBiasPhase,
        ur_read_code(params.para_log2e_bank, params.para_bias127_addr, smc_en)});
    result.sched_ur_r2.push_back({kPreReadShiftPhase,
        ur_read_code(params.para_log2e_bank, params.para_shift23_addr, smc_en)});
    result.sched_ur_r2.push_back({kPreReadOddZeroPhase,
        ur_read_code(params.para_log2e_bank, params.para_int_zero_addr, smc_en)});

    // --- compute（core；vv 统一 +1 对齐 VIN_D 的 C+3 生效沿）---
    result.sched_fpmul.push_back({kPreFpmulTPhase + vs, result.para_fpmul_code});
    result.sched_fpmul.push_back({kPreFpmulRPhase + vs, result.para_fpmul_code});
    result.sched_fpmul.push_back({kPreFpmulScaleOddPhase + vs, result.para_fpmul_code});

    result.sched_fpadd.push_back({kPreFpaddTwPhase + vs, result.para_fpadd_add_code});
    result.sched_fpadd.push_back({kPreFpaddTwMagicPhase + vs, result.para_fpadd_add_code});
    result.sched_fpadd.push_back({kPreFpaddTw2Phase + vs, result.para_fpadd_add_code});
    result.sched_fpadd.push_back({kPreFpaddDiffPhase + vs, result.para_fpadd_sub_code});
    result.sched_fpadd.push_back({kPreFpaddOddCheckPhase + vs, result.para_fpadd_check_code});

    result.sched_inttofp.push_back({kPreInttofpOddPhase + vs, result.para_inttofp_code});
    result.sched_inttofp.push_back({kPreInttofpNPhase + vs, result.para_inttofp_code});

    result.sched_intadd.push_back({kPreIntaddNPhase + vs, result.para_intadd_sub_code});
    result.sched_intadd.push_back({kPreIntaddExpPhase + vs, result.para_intadd_exp_code});

    result.sched_logic.push_back({kPreLogicAndOddPhase + vs, result.para_logic_and_code});
    result.sched_logic.push_back({kPreLogicShrMPhase + vs, result.para_logic_shr_code});
    result.sched_logic.push_back({kPreLogicShlScalePhase + vs, result.para_logic_shift_code});
    result.sched_logic.push_back({kPreLogicSelectScalePhase + vs, result.para_logic_select_eq_code});

    result.sched_ur_w0.push_back({kPreWriteRPhase + vs,
        ur_write_code(params.para_r_bank, params.para_r_addr, smc_en)});
    result.sched_ur_w1.push_back({kPreWriteKfPhase + vs,
        ur_write_code(params.para_k_scratch_bank, params.para_k_scratch_addr, smc_en)});
    // VIN/VOUT：scale 写回经 VOUT0（SELECT@51 → DR@53 → 捕获@53 → VOUT0_D@56 →
    // 写@56 = 捕获+3 边界；与 flush WriteScale / final y 同口型）
    result.sched_ur_w1.push_back({vv ? kPreWriteScaleScratchPhase + 3u
                                     : kPreWriteScaleScratchPhase,
        ur_write_code(params.para_k_scratch_bank, params.para_scale_scratch_addr, smc_en)});

    for (uint32_t p = 0; p < body_cycles; ++p) {
        const auto ext = [&](uint32_t phase) { return phase + cru_comp; };

        // VIN/VOUT（FPGA 稀疏掩码）：空闲默认源必须合法——计算口选 VIN0_D(11)、
        // LOGIC_ST 口选 FPADD_ST(5)、UVR 写口选 VOUT0_D(13)（原默认 30 在 FPGA 上是死链）
        const uint8_t r_def   = vv ? kRtovrInDrVin0D  : kRtovrInUrRdata0;
        const uint8_t r13_def = vv ? kRtovrInDrFpAddSt : kRtovrInUrRdata0;
        const uint8_t r35_def = vv ? kRtovrInDrVout0D : kRtovrInUrRdata0;
        uint8_t r0 = r_def;
        uint8_t r1 = r_def;
        uint8_t r3 = r_def;
        uint8_t r4 = r_def;
        uint8_t r5 = r_def;
        uint8_t r6 = r_def;
        uint8_t r10 = r_def;
        uint8_t r11 = r_def;
        uint8_t r12 = r_def;
        uint8_t r13 = r13_def;
        uint8_t r35 = r35_def;
        uint8_t r36 = r35_def;

        // x/log2e → FPMUL t（短持到 FPMUL issue）
        if (p <= ext(kPreFpmulTPhase) + kRtovrRouteLatency) {
            r3 = vv ? kRtovrInDrVin0D : ur_source_for_port(0);
            r4 = vv ? kRtovrInDrVin1D : ur_source_for_port(1);
        }
        // Tw = t+t
        if (p >= kPreFpmulTPhase + kFpmulDrLatencyRtl &&
            p < ext(kPreFpaddTwPhase) + 1u) {
            r5 = kRtovrInDrFpMulD;
            r6 = kRtovrInDrFpMulD;
        }
        // y = tw + MAGIC（MAGIC 挂 r6；勿冲掉 Tw 的 r6=t）
        if (p >= kPreFpaddTwPhase + kFpaddDrLatencyRtl &&
            p < ext(kPreFpaddTwMagicPhase) + 1u) {
            r5 = kRtovrInDrFpAddD;
            r6 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);
        }
        // n = bits(y) − MAGIC：FPADD DR 位型直送 INTADD（同 Softmax magic）
        if (p >= kPreFpaddTwMagicPhase + kFpaddDrLatencyRtl &&
            p < ext(kPreIntaddNPhase) + 1u) {
            r0 = kRtovrInDrFpAddD;
        }
        if (p >= kPreReadMagicPhase + kUrReadToComputeIssue - 1u &&
            p < ext(kPreIntaddNPhase) + 1u) {
            r1 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);
        }
        // Tw2：IntaddN 采样后再覆写 FPADD DR
        if (p >= kPreIntaddNPhase &&
            p < ext(kPreFpaddTw2Phase) + 1u) {
            r5 = kRtovrInDrFpMulD;
            r6 = kRtovrInDrFpMulD;
        }
        // AND / SHR：n & 1 / n >> 1
        if (p >= kPreIntaddNPhase + kIntaddLatency &&
            p < ext(kPreLogicShrMPhase) + 1u) {
            r11 = kRtovrInDrIntAddD0;
        }
        if (p >= kPreReadIntOnePhase + 1u &&
            p < ext(kPreLogicShrMPhase) + 1u) {
            r12 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);  // int_one（Bias 前）
        }
        if (p >= kPreLogicAndOddPhase + kLogicLatency - 1u &&
            p < ext(kPreInttofpOddPhase) + 1u) {
            r10 = kRtovrInDrLogicD;
        }
        if (p >= kPreIntaddNPhase + kIntaddLatency &&
            p < ext(kPreInttofpNPhase) + 1u) {
            r10 = kRtovrInDrIntAddD0;  // InttofpN 覆盖同拍 Logic 选通
        }
        // Diff = tw − n_f
        if (p >= kPreFpaddTw2Phase + kFpaddDrLatencyRtl &&
            p < ext(kPreFpaddDiffPhase) + 1u) {
            r5 = kRtovrInDrFpAddD;
        }
        if (p >= kPreInttofpNPhase + kInttofpLatency - 1u &&
            p < ext(kPreFpaddDiffPhase) + 1u) {
            r6 = kRtovrInDrIntToFpD;
        }
        // IntaddExp：m + bias
        if (p >= kPreLogicShrMPhase + kLogicLatency &&
            p < ext(kPreIntaddExpPhase) + 1u) {
            r0 = kRtovrInDrLogicD;
        }
        if (p >= kPreReadBiasPhase + kUrReadToComputeIssue - 1u &&
            p < ext(kPreIntaddExpPhase) + 1u) {
            r1 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);
        }
        // W1 写整数 k（SHR Logic）；须在 SHL 覆盖 Logic 之前
        // vv：数据经 VOUT0 捕获（rtovr-16@31 → D@34），写@34 由 r36 默认 13 覆盖
        if (!vv && p >= kPreLogicShrMPhase + kLogicLatency &&
            p <= ext(kPreWriteKfPhase)) {
            r36 = kRtovrInDrLogicD;
        }
        // SelectScale 后暂存 scale → bank0@201（vv：VOUT0 捕获@53 → D@56，写@56，默认 13 覆盖）
        if (!vv && p >= kPreLogicSelectScalePhase + kLogicLatency &&
            p <= ext(kPreWriteScaleScratchPhase)) {
            r36 = kRtovrInDrLogicD;
        }
        // SHL：r11=exp，r12=shift23
        if (p >= kPreIntaddExpPhase + kIntaddLatency &&
            p < ext(kPreLogicShlScalePhase) + 1u) {
            r11 = kRtovrInDrIntAddD0;
        }
        if (p >= kPreReadShiftPhase + kUrReadToComputeIssue - 1u &&
            p < ext(kPreLogicShlScalePhase) + 1u) {
            r12 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);
        }
        // FPMUL r：Diff · half
        if (p >= kPreFpmulRPhase - kRtovrRouteLatency &&
            p <= ext(kPreFpmulRPhase)) {
            r3 = kRtovrInDrFpAddD;
            r4 = vv ? kRtovrInDrVin0D : ur_source_for_port(1);
        }
        // FPMUL scale_odd：scale0 · √2
        if (p >= kPreLogicShlScalePhase + kLogicLatency &&
            p <= ext(kPreFpmulScaleOddPhase)) {
            r3 = kRtovrInDrLogicD;
        }
        if (p >= kPreReadSqrt2Phase + kUrReadToComputeIssue - 1u &&
            p <= ext(kPreFpmulScaleOddPhase)) {
            r4 = vv ? kRtovrInDrVin0D : ur_source_for_port(1);
        }
        // OddCheck：odd_f vs 0
        if (p >= kPreInttofpOddPhase + kInttofpLatency &&
            p >= kPreFpmulRPhase + 1u &&
            p < ext(kPreFpaddOddCheckPhase) + 1u) {
            r5 = kRtovrInDrIntToFpD;
        }
        if (p >= kPreReadOddZeroPhase + kUrReadToComputeIssue - 1u &&
            p < ext(kPreFpaddOddCheckPhase) + 1u) {
            r6 = vv ? kRtovrInDrVin1D : ur_source_for_port(2);
        }
        // SELECT(scale0, scale_odd)
        if (p >= kPreLogicShlScalePhase + kLogicLatency &&
            p < ext(kPreLogicSelectScalePhase) + 1u) {
            r11 = kRtovrInDrLogicD;
        }
        if (p >= kPreFpmulScaleOddPhase + kFpmulDrLatencyRtl &&
            p < ext(kPreLogicSelectScalePhase) + 1u) {
            r12 = kRtovrInDrFpMulD;
        }
        if (p >= kPreFpaddOddCheckPhase + kFpaddDrLatencyRtl - 1u &&
            p < ext(kPreLogicSelectScalePhase) + 1u) {
            r13 = kRtovrInDrFpAddSt;
        }
        // WriteR（vv：VOUT0 捕获@39 → D@42，写@42 由 r35 默认 13 覆盖）
        if (!vv && p >= kPreFpmulRPhase + kFpmulLatencyTvgen &&
            p <= ext(kPreWriteRPhase)) {
            r35 = kRtovrInDrFpMulD;
        }

        result.sched_rtovr_0.push_back({p, rtovr_code(r0)});
        result.sched_rtovr_1.push_back({p, rtovr_code(r1)});
        result.sched_rtovr_3.push_back({p, rtovr_code(r3)});
        result.sched_rtovr_4.push_back({p, rtovr_code(r4)});
        result.sched_rtovr_5.push_back({p, rtovr_code(r5)});
        result.sched_rtovr_6.push_back({p, rtovr_code(r6)});
        result.sched_rtovr_10.push_back({p, rtovr_code(r10)});
        result.sched_rtovr_11.push_back({p, rtovr_code(r11)});
        result.sched_rtovr_12.push_back({p, rtovr_code(r12)});
        result.sched_rtovr_13.push_back({p, rtovr_code(r13)});
        result.sched_rtovr_35.push_back({p, rtovr_code(r35)});
        result.sched_rtovr_36.push_back({p, rtovr_code(r36)});
    }

    // VIN/VOUT 捕获表（FPGA 稀疏掩码：UR_RDATA 只能经 VIN0/1 进计算，DR 只能经 VOUT0 回 UVR）。
    // 实测时序（RTL 探针）：捕获@C → route 生效@C+1 → RtoVR 寄存器@C+2 → IPR →
    // VIN_D/VOUT0_D 从 C+3 起有效（非原假设的 +1）。故 UR 读 issue+1 拍捕获不变，
    // 消费端发射统一 +1（见 vs）；未被捕获的拍该通道 code=0 冻结，
    // VIN_D 保持到下一次捕获（RtoVR 输出寄存器 + IPR 均为 vld 门控保持语义）。
    if (vv) {
        // VIN0 = 数据 lane：x / half / √2；stage0 的 0 取 R2 最后一次读(int_zero@41)的 rdata 驻留
        result.sched_rtovr_14.push_back({kPreReadPhase + 1u,
            rtovr_code(ur_source_for_port(0))});                    // 1:  x(R0@0)
        result.sched_rtovr_14.push_back({kPreReadHalf2Phase + 1u,
            rtovr_code(ur_source_for_port(1))});                    // 29: half(R1@28)
        result.sched_rtovr_14.push_back({kPreReadSqrt2Phase + 1u,
            rtovr_code(ur_source_for_port(1))});                    // 43: sqrt2(R1@42)
        result.sched_rtovr_14.push_back({body_cycles - 1u,
            rtovr_code(ur_source_for_port(2))});                    // 55: zero(R2@41 驻留)
        // VIN1 = 常数 lane：log2e / MAGIC / int_one / bias / shift23 / int_zero
        result.sched_rtovr_15.push_back({kPreReadPhase + 1u,
            rtovr_code(ur_source_for_port(1))});                    // 1:  log2e(R1@0)
        result.sched_rtovr_15.push_back({kPreReadMagicPhase + 1u,
            rtovr_code(ur_source_for_port(2))});                    // 7:  MAGIC(R2@6)
        result.sched_rtovr_15.push_back({kPreReadIntOnePhase + 1u,
            rtovr_code(ur_source_for_port(2))});                    // 23: int_one(R2@22)
        result.sched_rtovr_15.push_back({kPreReadBiasPhase + 1u,
            rtovr_code(ur_source_for_port(2))});                    // 30: bias(R2@29)
        result.sched_rtovr_15.push_back({kPreReadShiftPhase + 1u,
            rtovr_code(ur_source_for_port(2))});                    // 35: shift23(R2@34)
        result.sched_rtovr_15.push_back({kPreReadOddZeroPhase + 1u,
            rtovr_code(ur_source_for_port(2))});                    // 42: int_zero(R2@41)
        // VOUT0：k_f(SHR@29→DR@31→捕获31→D@34, 写@34) / r(FPMUL@35→DR@40→捕获39→D@42, 写@42)
        //        / scale(SELECT@51→DR@53→捕获53→D@56, 写@56)——捕获须采到 DR 生效后的拍
        result.sched_rtovr_16.push_back({kPreLogicShrMPhase + 3u,
            rtovr_code(kRtovrInDrLogicD)});                         // 31: k_f
        result.sched_rtovr_16.push_back({kPreFpmulRPhase + kFpmulLatencyTvgen + 1u,
            rtovr_code(kRtovrInDrFpMulD)});                         // 39: r（FPMUL@35→DR@40，捕获须 ≥39）
        result.sched_rtovr_16.push_back({kPreLogicSelectScalePhase + 3u,
            rtovr_code(kRtovrInDrLogicD)});                         // 53: scale
    }

    return result;
}

ExpPreprocessInstructions ExpPreprocessOp::compile_kflush_p2_overlay(
    const ExpPreprocessParams& params) const {
    // ∥Horner：已验证口型（check→SELECT=+6）嵌进 FPMUL 后空隙；oflow 放 S3.FPMUL 后。
    ExpPreprocessInstructions out;
    const bool fp16 = (params.para_dtype == 1);
    const uint8_t smc_en = params.para_smc_id_en;
    const bool vv = params.para_use_vin_vout;
    // vv：与 core 同原则，flush 计算/写统一 +1（UR 读与 VIN 捕获不动，
    // 对齐实测 C+3 生效沿；VOUT0 捕获不挪——DR sticky，写跟到 捕获+3）
    const uint32_t vs = vv ? 1u : 0u;

    iq::IntToFpCru i2f{};
    i2f.para_inst_valid = true;
    i2f.para_src_is_32b = !fp16;
    i2f.para_dst_is_32b = !fp16;
    i2f.para_src_signed = true;
    out.para_inttofp_code =
        static_cast<uint16_t>(iq::IntToFpIq::pack_cru_inttofp(i2f)) |
        (static_cast<uint16_t>(smc_en & 0x1Fu) << 6);
    out.para_fpadd_check_code = pack_fpadd_code(fp16, true, true, smc_en);
    out.para_logic_select_less_code =
        pack_logic_code(fp16, iq::LogicalOp::SELECT_LESS, smc_en);

    out.sched_ur_r0.push_back({kFlushReadKf,
        ur_read_code(params.para_k_scratch_bank, params.para_k_scratch_addr, smc_en)});
    out.sched_ur_r0.push_back({kFlushReadKMin,
        ur_read_code(params.para_log2e_bank, params.para_k_min_addr, smc_en)});
    out.sched_ur_r0.push_back({kFlushReadZero,
        ur_read_code(params.para_log2e_bank, params.para_int_zero_addr, smc_en)});
    // vv：scale 读挪 R0（R1 须全 P2 驻留 r——r 捕获与 flush VIN1 捕获错峰的前提）；
    // 非 vv 保持 R1 直读不动（行为位级不变）
    (vv ? out.sched_ur_r0 : out.sched_ur_r1).push_back({kFlushReadScale,
        ur_read_code(params.para_k_scratch_bank, params.para_scale_scratch_addr, smc_en)});
    out.sched_ur_r0.push_back({kFlushReadKf2,
        ur_read_code(params.para_k_scratch_bank, params.para_k_scratch_addr, smc_en)});
    out.sched_ur_r0.push_back({kFlushReadKMax,
        ur_read_code(params.para_log2e_bank, params.para_k_max_addr, smc_en)});
    out.sched_ur_r0.push_back({kFlushReadPosInf,
        ur_read_code(params.para_log2e_bank, params.para_pos_inf_addr, smc_en)});

    out.sched_inttofp.push_back({kFlushInttofpK + vs, out.para_inttofp_code});
    out.sched_inttofp.push_back({kFlushInttofpK2 + vs, out.para_inttofp_code});
    out.sched_fpadd.push_back({kFlushUflowCheck + vs, out.para_fpadd_check_code});
    out.sched_fpadd.push_back({kFlushOflowCheck + vs, out.para_fpadd_check_code});
    out.sched_logic.push_back({kFlushSelectUflow + vs, out.para_logic_select_less_code});
    out.sched_logic.push_back({kFlushSelectOflow + vs, out.para_logic_select_less_code});
    out.sched_ur_w0.push_back({kFlushWriteScale + vs,
        ur_write_code(params.para_coeff_bank, params.para_scale_word_base, smc_en)});

    auto hold = [](std::vector<SchedEntry>& s, uint32_t c0, uint32_t c1, uint8_t src) {
        const uint64_t code = rtovr_code(src);
        for (uint32_t t = c0; t < c1; ++t) s.push_back({t, code});
    };

    // uflow：长持到 check（与 Horner S1 的 r5 窗错开——S1.r5 已收窄到 FPADD 前）
    // vv：UR 源全部改 VIN_D（k_f→VIN1，k_min/scale→VIN0，zero→VIN1），捕获见函数尾
    hold(out.sched_rtovr_10, kFlushReadKf + 1u, kFlushInttofpK + 1u,
         vv ? kRtovrInDrVin1D : ur_source_for_port(0));
    hold(out.sched_rtovr_5, kFlushInttofpK + kInttofpLatency, kFlushUflowCheck + 1u,
         kRtovrInDrIntToFpD);
    hold(out.sched_rtovr_6, kFlushReadKMin + 1u, kFlushUflowCheck + 1u,
         vv ? kRtovrInDrVin0D : ur_source_for_port(0));
    hold(out.sched_rtovr_11, kFlushReadZero + 1u, kFlushSelectUflow + 1u,
         vv ? kRtovrInDrVin1D : ur_source_for_port(0));
    hold(out.sched_rtovr_12, kFlushReadScale + 1u, kFlushSelectUflow + 1u,
         vv ? kRtovrInDrVin0D : ur_source_for_port(1));
    hold(out.sched_rtovr_13, kFlushUflowCheck + kFpaddDrLatencyRtl - 1u, kFlushSelectUflow + 1u,
         kRtovrInDrFpAddSt);
    hold(out.sched_rtovr_10, kFlushReadKf2 + 1u, kFlushInttofpK2 + 1u,
         vv ? kRtovrInDrVin1D : ur_source_for_port(0));
    // oflow：r5/r6 只短持 check 窗，避开 S2/S3 Horner FPADD 口型；Inttofp DR / R0(KMax) 自身保持
    hold(out.sched_rtovr_5, kFlushOflowCheck - kRtovrRouteLatency, kFlushOflowCheck + 1u,
         kRtovrInDrIntToFpD);
    hold(out.sched_rtovr_6, kFlushOflowCheck - kRtovrRouteLatency, kFlushOflowCheck + 1u,
         vv ? kRtovrInDrVin0D : ur_source_for_port(0));
    hold(out.sched_rtovr_11, kFlushSelectUflow + kLogicLatency, kFlushSelectOflow + 1u,
         kRtovrInDrLogicD);
    hold(out.sched_rtovr_12, kFlushReadPosInf + 1u, kFlushSelectOflow + 1u,
         vv ? kRtovrInDrVin0D : ur_source_for_port(0));
    hold(out.sched_rtovr_13, kFlushOflowCheck + kFpaddDrLatencyRtl - 1u, kFlushSelectOflow + 1u,
         kRtovrInDrFpAddSt);
    // vv：写回数据经 VOUT0（SELECT@41 → DR@43，捕获@42 于 cycle43 采到 DR →
    // VOUT0_D@45 → 路由 13{43,44}，写@45 = 捕获+3 边界）
    if (vv) {
        hold(out.sched_rtovr_35, kFlushSelectOflow + kLogicLatency + 1u,
             kFlushWriteScale + 1u, kRtovrInDrVout0D);
    } else {
        hold(out.sched_rtovr_35, kFlushSelectOflow + kLogicLatency,
             kFlushWriteScale + 1u, kRtovrInDrLogicD);
    }

    // vv 捕获表（P2 相对相位；与 Horner 的 coeff(VIN0)/r(VIN1) 占用区间手工错峰，
    // D=捕获+3 生效并驻留到下一次捕获；消费端发射已统一 +1，全部 ≥C+3）
    if (vv) {
        out.sched_rtovr_15.push_back({kFlushReadKf + 1u,
            rtovr_code(ur_source_for_port(0))});      // 8:  k_f → VIN1（INTTOFP@11，D@11 边界）
        out.sched_rtovr_14.push_back({kFlushReadKMin + 1u,
            rtovr_code(ur_source_for_port(0))});      // 12: k_min → VIN0（FPADD@15，D@15 边界）
        out.sched_rtovr_15.push_back({kFlushReadZero + 1u,
            rtovr_code(ur_source_for_port(0))});      // 17: zero → VIN1（LOGIC@21，D@20）
        out.sched_rtovr_14.push_back({kFlushReadScale + 1u,
            rtovr_code(ur_source_for_port(0))});      // 18: scale(vv 读已挪 R0) → VIN0（LOGIC@21，D@21 边界）
        out.sched_rtovr_15.push_back({kFlushReadKf2 + 1u,
            rtovr_code(ur_source_for_port(0))});      // 24: k_f2 → VIN1（INTTOFP@27，D@27 边界）
        out.sched_rtovr_14.push_back({kFlushReadKMax + 1u,
            rtovr_code(ur_source_for_port(0))});      // 28: k_max → VIN0（FPADD@35，D@31）
        out.sched_rtovr_14.push_back({kFlushReadPosInf + 1u,
            rtovr_code(ur_source_for_port(0))});      // 37: pos_inf → VIN0（LOGIC@41，D@40）
        // SELECT DR sticky：捕获@42 于 cycle43 采到（SELECT@41→DR@43），VOUT0_D@45 → 写@45
        out.sched_rtovr_16.push_back({kFlushSelectOflow + kLogicLatency,
            rtovr_code(kRtovrInDrLogicD)});           // 42: SELECT DR → VOUT0（w0@45）
    }

    out.para_body_cycles = kFlushWriteScale + vs + 3u;
    return out;
}

OpInstructionsBase* ExpPreprocessOp::compile(const app::AppOperatorRequest& req) {
    ExpPreprocessParams params = parse_request(req);
    apply_smc_params(params, req);
    auto* insts = new ExpPreprocessInstructions();
    *insts = compile_preprocess(params);
    // 单测路径：core 后串行挂上 k-flush（无 Horner 可并行）
    const ExpPreprocessInstructions flush = compile_kflush_p2_overlay(params);
    const uint32_t core = insts->para_body_cycles;
    append_sched_offset(insts->sched_ur_r0, flush.sched_ur_r0, core);
    append_sched_offset(insts->sched_ur_r1, flush.sched_ur_r1, core);
    append_sched_offset(insts->sched_ur_w0, flush.sched_ur_w0, core);
    append_sched_offset(insts->sched_inttofp, flush.sched_inttofp, core);
    append_sched_offset(insts->sched_fpadd, flush.sched_fpadd, core);
    append_sched_offset(insts->sched_logic, flush.sched_logic, core);
    append_sched_offset(insts->sched_rtovr_0, flush.sched_rtovr_0, core);
    append_sched_offset(insts->sched_rtovr_1, flush.sched_rtovr_1, core);
    append_sched_offset(insts->sched_rtovr_5, flush.sched_rtovr_5, core);
    append_sched_offset(insts->sched_rtovr_6, flush.sched_rtovr_6, core);
    append_sched_offset(insts->sched_rtovr_10, flush.sched_rtovr_10, core);
    append_sched_offset(insts->sched_rtovr_11, flush.sched_rtovr_11, core);
    append_sched_offset(insts->sched_rtovr_12, flush.sched_rtovr_12, core);
    append_sched_offset(insts->sched_rtovr_13, flush.sched_rtovr_13, core);
    append_sched_offset(insts->sched_rtovr_35, flush.sched_rtovr_35, core);
    insts->para_body_cycles = core + flush.para_body_cycles;
    insts->para_total_cycles = insts->para_body_cycles * insts->para_total_steps;
    return insts;
}

std::vector<IqInstructionFile> ExpPreprocessInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t total_steps = para_total_steps;
    const uint32_t body_cycles = para_body_cycles;
    auto add_iq = [&](const std::vector<SchedEntry>& sched,
                      uint32_t iq_id, const std::string& iq_name,
                      uint32_t code_bits,
                      bool outer_addr_inc = false) {
        if (sched.empty()) return;
        IqInstructionFile file;
        file.para_iq_id = iq_id;
        file.para_iq_name = iq_name;
        file.para_width_bits = 41 + code_bits;
        file.para_data = emit_preprocess_embed_iq(sched, code_bits, outer_addr_inc,
                                                  total_steps, body_cycles);
        files.push_back(file);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_ur_w1, iq::IQ_ID_UR_WDATA_1, iq::IQ_NAME_UR_WDATA_1, 33, false);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8, false);
    add_iq(sched_fptoint, iq::IQ_ID_FP_TO_INT, iq::IQ_NAME_FP_TO_INT, 10, false);
    add_iq(sched_inttofp, iq::IQ_ID_INT_TO_FP, iq::IQ_NAME_INT_TO_FP, 11, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10, false);
    add_iq(sched_intadd, iq::IQ_ID_INTADD, iq::IQ_NAME_INTADD, 18, false);
    add_iq(sched_logic, iq::IQ_ID_LOGIC, iq::IQ_NAME_LOGIC, 11, false);
    add_iq(sched_rtovr_0, iq::IQ_ID_RTOVR_BASE + 0, "rtovr-0", 7, false);
    add_iq(sched_rtovr_1, iq::IQ_ID_RTOVR_BASE + 1, "rtovr-1", 7, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false);
    add_iq(sched_rtovr_9, iq::IQ_ID_RTOVR_BASE + 9, "rtovr-9", 7, false);
    add_iq(sched_rtovr_10, iq::IQ_ID_RTOVR_BASE + 10, "rtovr-10", 7, false);
    add_iq(sched_rtovr_11, iq::IQ_ID_RTOVR_BASE + 11, "rtovr-11", 7, false);
    add_iq(sched_rtovr_12, iq::IQ_ID_RTOVR_BASE + 12, "rtovr-12", 7, false);
    add_iq(sched_rtovr_13, iq::IQ_ID_RTOVR_BASE + 13, "rtovr-13", 7, false);
    add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
    add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
    add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    add_iq(sched_rtovr_36, iq::IQ_ID_RTOVR_BASE + 36, "rtovr-36", 7, false);
    return files;
}

std::vector<uint8_t> ExpPreprocessInstructions::serialize() const {
    std::vector<uint8_t> result;
    auto push_u32 = [&](uint32_t v) {
        result.push_back(static_cast<uint8_t>(v & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    };
    push_u32(para_total_steps);
    push_u32(para_total_elements);
    push_u32(para_total_cycles);
    return result;
}

bool ExpPreprocessOp::generate_data(const app::AppOperatorRequest& req,
                                    std::vector<uint8_t>& input_bin,
                                    std::vector<uint8_t>& other_bin,
                                    std::vector<uint8_t>& alpha_bin) {
    const auto params = parse_request(req);
    const uint32_t total_elems =
        params.para_n * params.para_c * params.para_h * params.para_w;
    const bool fp16 = (params.para_dtype == 1);
    const std::vector<float> raw_x = collect_or_make_input_data(req, total_elems);

    fill_raw_x_bin(raw_x, fp16, alpha_bin);
    input_bin.clear();
    other_bin.clear();
    fill_preprocess_const_bank4(other_bin, fp16);

    std::cout << "[ExpPreprocess] generate_data: alpha=raw_x bank2, "
              << "other=const bank4(+sqrt2@7), bank0/bank1 left zero for RTL write"
              << std::endl;
    return true;
}

bool ExpPreprocessOp::calculate_golden(const app::AppOperatorRequest& req,
                                       std::vector<uint8_t>& golden_bin) {
    golden_bin.clear();
    const auto params = parse_request(req);
    const bool fp16 = (params.para_dtype == 1);
    const uint32_t total_elems =
        params.para_n * params.para_c * params.para_h * params.para_w;
    const std::vector<float> raw_x = collect_or_make_input_data(req, total_elems);

    std::vector<uint8_t> golden_r;
    build_preprocess_golden_bank0(raw_x, fp16, golden_r);
    golden_bin = std::move(golden_r);
    return true;
}

void ExpPreprocessOp::fill_random_request(app::AppOperatorRequest& req,
                                          std::mt19937& rng,
                                          std::uniform_real_distribution<float>&,
                                          const std::string& dtype_str) {
    req.para_operator = "ExpPreprocess";
    std::uniform_real_distribution<float> exp_dist(0.1f, 8.0f);
    const uint32_t total = req.para_dimensions.para_n * req.para_dimensions.para_c *
                           req.para_dimensions.para_h * req.para_dimensions.para_w;
    app::AppTensorDesc input;
    input.para_name = "input_tensor";
    input.para_dtype = (dtype_str == "fp16") ? app::AppDataType::FLOAT16
                                              : app::AppDataType::FLOAT32;
    input.para_shape = {req.para_dimensions.para_n, req.para_dimensions.para_c,
                        req.para_dimensions.para_h, req.para_dimensions.para_w};
    input.para_data_f32.reserve(total);
    for (uint32_t i = 0; i < total; ++i) input.para_data_f32.push_back(exp_dist(rng));
    req.para_inputs["input_tensor"] = input;
}

REGISTER_OP(ExpPreprocessOp);

}  // namespace op
}  // namespace aiacc
