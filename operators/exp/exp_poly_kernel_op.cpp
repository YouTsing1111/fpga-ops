#include "exp_poly_kernel_op.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "../../iq/iq_id_defs.h"
#include "../../iq/modules/fpadd_iq.h"
#include "../../iq/modules/fpmul_iq.h"
#include "../common/op_registry.h"

namespace aiacc {
namespace op {

using namespace exp_iq;
using namespace exp_math;

std::string ExpPolyKernelOp::get_name() const { return "ExpPolyKernel"; }

void ExpPolyKernelOp::apply_smc_params(ExpPolyKernelParams& params,
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

ExpPolyKernelParams ExpPolyKernelOp::parse_request(
    const app::AppOperatorRequest& req) const {
    ExpPolyKernelParams p;
    p.para_n = req.para_dimensions.para_n;
    p.para_c = req.para_dimensions.para_c;
    p.para_h = req.para_dimensions.para_h;
    p.para_w = req.para_dimensions.para_w;
    p.para_num_terms = kPolyTerms;

    auto input_it = req.para_inputs.find("input_tensor");
    if (input_it != req.para_inputs.end() &&
        input_it->second.para_dtype == app::AppDataType::FLOAT16) {
        p.para_dtype = 1;
    }

    const uint32_t total_elems = p.para_n * p.para_c * p.para_h * p.para_w;
    const uint32_t total_steps =
        (total_elems + p.get_elems_per_word() - 1) / p.get_elems_per_word();
    const uint32_t coeff_words = kPolyTerms + total_steps;

    if (req.para_layout_mode == "bank_conflict") {
        p.para_r_bank = static_cast<uint8_t>(
            (req.para_bc_input_bank >= 0) ? req.para_bc_input_bank : 0);
        p.para_coeff_bank = static_cast<uint8_t>(
            (req.para_bc_other_bank >= 0) ? req.para_bc_other_bank : 1);
        p.para_output_bank = static_cast<uint8_t>(
            (req.para_bc_output_bank >= 0) ? req.para_bc_output_bank : 3);
        p.para_r_addr = 0;
        p.para_coeff_addr = (p.para_coeff_bank == p.para_r_bank) ? total_steps : 0;
        p.para_output_addr = 0;
        if (p.para_output_bank == p.para_r_bank) p.para_output_addr += total_steps;
        if (p.para_output_bank == p.para_coeff_bank) p.para_output_addr += coeff_words;
    } else {
        p.para_r_bank = 0;
        p.para_r_addr = 0;
        p.para_coeff_bank = 1;
        p.para_coeff_addr = 0;
        p.para_output_bank = 3;
        p.para_output_addr = 0;
    }
    return p;
}

ExpPolyKernelInstructions ExpPolyKernelOp::compile_kernel(
    const ExpPolyKernelParams& params,
    const char* fold_mode_override) const {
    ExpPolyKernelInstructions result;
    const uint32_t total_elems = params.para_n * params.para_c *
                                 params.para_h * params.para_w;
    const uint32_t total_steps =
        (total_elems + params.get_elems_per_word() - 1) / params.get_elems_per_word();

    result.para_total_steps = total_steps;
    result.para_total_elements = total_elems;
    // vv 直通：Final 尾链 y 写@60（捕获+3 边界），body 59→61
    const bool vv = params.para_use_vin_vout;
    result.para_use_vin_vout = vv;
    const uint32_t per_elem_cycles =
        kStageCycles * kNumStages + (vv ? kFinalCyclesVv : kFinalCycles);
    result.para_total_cycles = per_elem_cycles * total_steps;
    result.para_block_cycles = kStageCycles;
    result.para_num_terms = kPolyTerms;
    result.para_smc_id_en = params.para_smc_id_en;
    result.para_rtovr_read_extend_loops = params.para_rtovr_read_extend_loops;

    iq::FpMulCru mul{};
    mul.para_inst_valid = true;
    mul.para_src_precision_is_32b = (params.para_dtype == 0);
    mul.para_dst_precision_is_32b = (params.para_dtype == 0);
    mul.para_smc_id_en = params.para_smc_id_en;
    result.para_fpmul_code = iq::FpMulIq::pack_cru_fpmul(mul);

    iq::FpAddCru add{};
    add.para_inst_valid = true;
    add.para_src_precision_32b = (params.para_dtype == 0);
    add.para_dst_precision_32b = (params.para_dtype == 0);
    add.para_update_status = false;
    add.para_is_sub = false;
    result.para_fpadd_code =
        static_cast<uint16_t>(iq::FpAddIq::pack_cru_fpadd(add)) |
        (static_cast<uint16_t>(params.para_smc_id_en & 0x1Fu) << 5);

    const uint8_t smc_en = params.para_smc_id_en;
    const uint32_t scale_addr = params.para_coeff_addr + kPolyTerms;

    // 一律 unroll：DR 链 + stage0 tied-zero（无 ACC@bank2 fold）
    (void)fold_mode_override;
    const bool use_acc = false;

    // UR 端口（同拍多读必须分 bank；R0 在 Horner/Final 分时复用）：
    //   R0 ← bank2 ACC（Horner）/ bank1 scale（Final）
    //   R1 ← bank0 r
    //   R2 ← bank1 coeff
    //   W0 ← bank2 ACC（Final@kAccWriteInFinalPhase）/ bank3 y（Final@kFinalWritePhase）
    // 同拍禁令：R0 读 bank2 与 W0 写 bank2；Final 内 ACC 写@1、y 写@10，错开。
    std::cout << "[ExpPolyKernel] y = 2^k * poly(r): r B"
              << static_cast<int>(params.para_r_bank) << "@" << params.para_r_addr
              << ", coeff B" << static_cast<int>(params.para_coeff_bank)
              << "@" << params.para_coeff_addr << ", scale@" << scale_addr
              << ", output B" << static_cast<int>(params.para_output_bank)
              << "@" << params.para_output_addr
              << ", steps=" << total_steps
              << ", per_elem_cycles=" << per_elem_cycles;
    if (use_acc) {
        std::cout << ", acc B" << static_cast<int>(params.para_acc_bank)
                  << "@" << params.para_acc_addr
                  << " (ACC@bank2, write@Final+" << kAccWriteInFinalPhase << ")";
    } else {
        std::cout << " (unroll DR+tied_zero)";
    }
    std::cout << std::endl;

    for (uint32_t stage = 0; stage < kNumStages; ++stage) {
        const uint32_t S = stage * kStageCycles;
        result.sched_ur_r1.push_back({S + kReadPhase,
            ur_read_code(params.para_r_bank, params.para_r_addr, smc_en)});
        result.sched_ur_r2.push_back({S + kReadPhase,
            ur_read_code(params.para_coeff_bank, params.para_coeff_addr + stage, smc_en)});

        if (use_acc) {
            result.sched_ur_r0.push_back({S + kReadPhase,
                ur_read_code(params.para_acc_bank, params.para_acc_addr, smc_en)});
        }

        // unroll：stage0=tied_zero（vv：VIN0 里 P1 驻留的 int_zero，sparse 掩码下 TiedZero 不可达）；
        // 其后接上一级 FPADD_DR
        const uint8_t mul_s0 = use_acc
            ? ur_source_for_port(0)
            : ((stage == 0u) ? (vv ? kRtovrInDrVin0D : kRtovrInTiedZero)
                             : kRtovrInDrFpAddD);
        // RtoVR 尽量短持：只覆盖 consumer issue 窗，避免整 stage 空挂（访存/互连气泡）。
        // mul_s0：须从 stage 起点挂到 FPMUL（上一级 DR 可能在 stage 头几拍才稳定）。
        for (uint32_t p = 0; p <= kFpmulPhase; ++p) {
            result.sched_rtovr_3.push_back({S + p, rtovr_code(mul_s0)});
        }
        // r：持到 FPMUL；coeff→FPADD 短持，腾出中段 r5 给 flush uflow check@15
        // vv：r→VIN1、coeff→VIN0（捕获在下方 vv 块；实测 C+3 生效，落点见捕获表注释）
        for (uint32_t p = 0; p <= kFpmulPhase; ++p) {
            result.sched_rtovr_4.push_back({S + p,
                rtovr_code(vv ? kRtovrInDrVin1D : ur_source_for_port(1))});
        }
        {
            const uint32_t r5_lo =
                (kFpaddPhase > kRtovrRouteLatency) ? (kFpaddPhase - kRtovrRouteLatency) : 0u;
            for (uint32_t p = r5_lo; p <= kFpaddPhase; ++p) {
                result.sched_rtovr_5.push_back({S + p,
                    rtovr_code(vv ? kRtovrInDrVin0D : ur_source_for_port(2))});
            }
        }
        for (uint32_t p = kFpmulPhase + kFpmulDrLatencyRtl; p <= kFpaddPhase; ++p) {
            result.sched_rtovr_6.push_back({S + p, rtovr_code(kRtovrInDrFpMulD)});
        }
        result.sched_fpmul.push_back({S + kFpmulPhase, result.para_fpmul_code});
        result.sched_fpadd.push_back({S + kFpaddPhase, result.para_fpadd_code});
    }

    {
        const uint32_t F = kStageCycles * kNumStages;
        // Final：P 在 FPADD DR；×scale。R0 改读 scale（与 Horner 读 ACC 分时）
        // vv：实测 捕获@C→VIN_D@C+3，scale 读提前到 F-2（捕获@F-1→D@F+2，FPMUL@F+3 有 1 拍余量；
        //     R0 在 flush pos_inf@36 之后空闲，且 flush 写 scale@45 < 读本拍）
        const uint32_t rd_scale = F + kFinalReadPhase - (vv ? 2u : 0u);
        result.sched_ur_r0.push_back({rd_scale,
            ur_read_code(params.para_coeff_bank, scale_addr, smc_en)});
        // P × scale：短持到 FPMUL；写回短持 VOUT0_D→UR-W
        // vv：scale→VIN1（捕获@49→D@52，FPMUL@53）；y：FPMUL@53→DR@58→捕获@57→VOUT0_D@60→写@60
        for (uint32_t p = 0; p <= kFinalFpmulPhase; ++p) {
            result.sched_rtovr_3.push_back({F + p, rtovr_code(kRtovrInDrFpAddD)});
            result.sched_rtovr_4.push_back({F + p,
                rtovr_code(vv ? kRtovrInDrVin1D : ur_source_for_port(0))});
        }
        if (use_acc) {
            for (uint32_t p = 0; p <= kAccWriteInFinalPhase; ++p) {
                result.sched_rtovr_35.push_back({F + p, rtovr_code(kRtovrInDrFpAddD)});
            }
        }
        // vv：写@60 须 mux 在 cycle60 选中 VOUT0_D（code@59），窗 [7,9] 覆盖
        const uint32_t w_lo = kFinalFpmulPhase + kFpmulDrLatencyRtl + (vv ? 1u : 0u);
        const uint32_t w_hi = kFinalWritePhase + (vv ? 1u : 0u);
        for (uint32_t p = w_lo; p <= w_hi; ++p) {
            result.sched_rtovr_35.push_back({F + p,
                rtovr_code(vv ? kRtovrInDrVout0D : kRtovrInDrFpMulD)});
        }
        if (use_acc) {
            result.sched_ur_w0.push_back({F + kAccWriteInFinalPhase,
                ur_write_code(params.para_acc_bank, params.para_acc_addr, smc_en)});
        }
        result.sched_fpmul.push_back({F + kFinalFpmulPhase, result.para_fpmul_code});
        result.sched_ur_w0.push_back({F + kFinalWritePhase + (vv ? 2u : 0u),
            ur_write_code(params.para_output_bank, params.para_output_addr, smc_en)});
    }

    // VIN/VOUT 捕获表（Horner 部分；flush 常数捕获在 kflush overlay 里）。
    // coeff→VIN0：S0 晚三拍（zero 先消费@3）、S1 让给 k_min@12、S3 晚到 @35
    // （flush k_max@28→D@31 须活到 oflow FPADD@35 采样；coeff3@35→D@38 踩 FPADD@38 边界）。
    // r→VIN1：flush 的 k_f/zero/k_f2 捕获（@8/17/24）也落 VIN1，逐 stage 错峰——
    // S1@10（k_f D@11 给 INTTOFP@11 后立刻装回 r，D@13 踩 FPMUL@13 边界）、
    // S2@20（zero D@20 给 LOGIC@21，D@23 踩 FPMUL@23）、S3@28（k_f2 D@27 给 INTTOFP@27，
    // D@31 ≤ FPMUL@33）；S0/S4 维持读+1。前提：R1 全 P2 驻留 r（flush scale 已挪 R0）。
    if (vv) {
        static const uint8_t coeff_cap_off[kNumStages] = {4, 4, 1, 5, 1};
        static const uint8_t r_cap_abs[kNumStages] = {1, 10, 20, 28, 41};
        for (uint32_t stage = 0; stage < kNumStages; ++stage) {
            const uint32_t S = stage * kStageCycles;
            result.sched_rtovr_14.push_back({S + coeff_cap_off[stage],
                rtovr_code(ur_source_for_port(2))});   // coeff → VIN0
            result.sched_rtovr_15.push_back({r_cap_abs[stage],
                rtovr_code(ur_source_for_port(1))});   // r → VIN1（错峰 flush，见上注释）
        }
        const uint32_t F = kStageCycles * kNumStages;
        result.sched_rtovr_15.push_back({F + kFinalReadPhase - 1u,
            rtovr_code(ur_source_for_port(0))});       // 49: final scale(R0@48) → VIN1（D@52）
        result.sched_rtovr_16.push_back({F + kFinalFpmulPhase + kFpmulDrLatencyRtl + 1u,
            rtovr_code(kRtovrInDrFpMulD)});            // 57: final FPMUL DR@58 → VOUT0（D@60，写@60）
    }
    return result;
}

OpInstructionsBase* ExpPolyKernelOp::compile(const app::AppOperatorRequest& req) {
    ExpPolyKernelParams params = parse_request(req);
    apply_smc_params(params, req);
    auto* insts = new ExpPolyKernelInstructions();
    *insts = compile_kernel(params);
    return insts;
}

std::vector<IqInstructionFile> ExpPolyKernelInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t total_steps = para_total_steps;
    auto add_iq = [&](const std::vector<SchedEntry>& sched,
                      uint32_t iq_id, const std::string& iq_name,
                      uint32_t code_bits,
                      bool addr_inc = false,
                      bool outer_addr_inc = false) {
        if (sched.empty()) return;
        IqInstructionFile file;
        file.para_iq_id = iq_id;
        file.para_iq_name = iq_name;
        file.para_width_bits = 41 + code_bits;
        file.para_data = emit_horner_kernel_embed_iq(
            sched, code_bits, addr_inc, outer_addr_inc, total_steps,
            "ExpPolyKernel",
            para_use_vin_vout ? (kStageCycles * kNumStages + kFinalCyclesVv) : 0u);
        files.push_back(file);
    };

    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, false, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, false, true);
    // coeff：内层 Horner lev1 递增（addr_inc=true）；不随 word 外层递增
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17, true, false);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, false, true);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7);
    add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7);
    add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7);
    add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7);
    return files;
}

std::vector<uint8_t> ExpPolyKernelInstructions::serialize() const {
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
    push_u32(para_block_cycles);
    push_u32(para_num_terms);
    return result;
}

bool ExpPolyKernelOp::generate_data(const app::AppOperatorRequest& req,
                                    std::vector<uint8_t>& input_bin,
                                    std::vector<uint8_t>& other_bin,
                                    std::vector<uint8_t>& alpha_bin) {
    // Phase-2 ACC@bank2：alpha/bank2 保持空/零作累加器初值（勿填原始 x）
    alpha_bin.clear();
    const auto params = parse_request(req);
    const uint32_t total_elems =
        params.para_n * params.para_c * params.para_h * params.para_w;
    const bool fp16 = (params.para_dtype == 1);
    const std::vector<float> raw_x = collect_or_make_input_data(req, total_elems);

    std::cout << "[ExpPolyKernel] generate_data: host decompose x->(r,2^k); "
              << "bank2 left clear for ACC@lev2" << std::endl;
    fill_kernel_banks_from_raw_x(raw_x, fp16, input_bin, other_bin);
    return true;
}

bool ExpPolyKernelOp::calculate_golden(const app::AppOperatorRequest& req,
                                       std::vector<uint8_t>& golden_bin) {
    golden_bin.clear();
    const auto params = parse_request(req);
    const bool fp16 = (params.para_dtype == 1);
    const uint32_t total_elems =
        params.para_n * params.para_c * params.para_h * params.para_w;
    const std::vector<float> raw_x = collect_or_make_input_data(req, total_elems);

    for (float x : raw_x) {
        // FP16：golden 必须先按 half 舍入再分解，与 RTL bank2 装载一致；否则 x 边界上 k 可能差 1。
        const float x_eff = fp16
            ? app::float16_to_float32(app::float32_to_float16(x))
            : x;
        const ExpReduced red = decompose_exp_input(x_eff);
        if (!fp16) {
            const float y = host_exp_f32(x_eff);
            uint32_t bits;
            std::memcpy(&bits, &y, sizeof(float));
            append_scalar(golden_bin, bits, 4);
        } else {
            append_scalar(golden_bin, host_exp_f16_bits(x), 2);
        }
    }
    return true;
}

void ExpPolyKernelOp::fill_random_request(app::AppOperatorRequest& req,
                                          std::mt19937& rng,
                                          std::uniform_real_distribution<float>&,
                                          const std::string& dtype_str) {
    req.para_operator = "ExpPolyKernel";
    std::uniform_real_distribution<float> exp_dist(-10.0f, 10.0f);
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

REGISTER_OP(ExpPolyKernelOp);

}  // namespace op
}  // namespace aiacc
