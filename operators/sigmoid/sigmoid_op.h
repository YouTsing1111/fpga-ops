#ifndef AIACC_SMI_IQ_OP_SIGMOID_OP_H_
#define AIACC_SMI_IQ_OP_SIGMOID_OP_H_

/**
 * @file sigmoid_op.h
 * @brief FP16 Sigmoid = 1/(1+exp(-x))，复用 FP32 Exp。
 *
 * 片上精度路径（LSU / SIGMOID_ONCHIP_UPCAST=1）：
 *   x(FP16 dest_low @bank2；LSU LDB 与 Exp 同 bank)
 *     --Phase0: FPtoFP FP16→FP32 → FPMUL×(-1)--> -x (FP32 @bank2)
 *     --Phase1: Exp --> e^{-x} (@bank3)
 *     --Phase2: +1, Newton, FPtoFP FP32→FP16 dest_low --> y (@bank3)
 *
 * 兼容路径（默认 smoke）：Host 仍可预写 FP32 -x @bank2，跳过 Phase0。
 *
 * FPtoFP dest_low / src_low：每 FP32 槽低 16bit 放一个 FP16（高 16=0）。
 */

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "../../app/app_json_parser.h"
#include "../common/op_instructions_base.h"
#include "../exp/exp_iq_utils.h"
#include "../exp/exp_op.h"

namespace aiacc {
namespace op {

constexpr uint32_t kSigmoidIqDepthLimit = 128u;  // smi_top.v: SMI_IQ_INSTQ_DEPTH=128
constexpr uint32_t kSigmoidDefaultNewtonIters = 2u;

struct SigmoidPostInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_fptofp;
    std::vector<exp_iq::SchedEntry> sched_intadd;
    std::vector<exp_iq::SchedEntry> sched_rtovr_0;
    std::vector<exp_iq::SchedEntry> sched_rtovr_1;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_7;   // FPtoFP S
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;

    uint32_t para_body_cycles{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint32_t para_newton_iters{kSigmoidDefaultNewtonIters};
    uint8_t para_smc_id_en{0};

    std::string get_operator_name() const override { return "SigmoidPost"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Phase0：FP16→FP32（FPtoFP）+ ×(-1) → Exp 输入 bank2。 */
struct SigmoidUpcastInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fptofp;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_7;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;
    // VIN/VOUT 直通（FPGA 稀疏掩码；默认关，非 vv 路径位级不变；sigmoid 自身不开，silu 用）
    std::vector<exp_iq::SchedEntry> sched_rtovr_14;
    std::vector<exp_iq::SchedEntry> sched_rtovr_15;
    std::vector<exp_iq::SchedEntry> sched_rtovr_16;
    std::vector<exp_iq::SchedEntry> sched_vin0;
    std::vector<exp_iq::SchedEntry> sched_vin1;
    std::vector<exp_iq::SchedEntry> sched_vout0;

    bool para_use_vin_vout{false};
    uint32_t para_body_cycles{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint8_t para_smc_id_en{0};

    std::string get_operator_name() const override { return "SigmoidUpcast"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** 容器：tvgen 默认返回 Phase1（Exp）；Phase0/2 单独取。 */
struct SigmoidInstructions : public OpInstructionsBase {
    SigmoidUpcastInstructions phase0_upcast;
    ExpMergedInstructions stage0_exp;       // Phase1
    SigmoidPostInstructions stage1_post;  // Phase2
    bool para_onchip_upcast{false};
    uint32_t para_num_stages{2};
    uint32_t para_iq_depth_limit{kSigmoidIqDepthLimit};
    uint32_t para_phase0_max_insts{0};
    uint32_t para_stage0_max_insts{0};
    uint32_t para_stage1_max_insts{0};
    bool para_phase0_fits{true};
    bool para_stage0_fits{true};
    bool para_stage1_fits{true};

    std::string get_operator_name() const override { return "Sigmoid"; }
    uint32_t get_total_steps() const override { return stage0_exp.get_total_steps(); }
    uint32_t get_total_elements() const override { return stage0_exp.get_total_elements(); }
    std::vector<uint8_t> serialize() const override;
    /** 默认吐 Phase1 Exp IQ（与旧 Stage0 兼容）。 */
    std::vector<IqInstructionFile> get_iq_files() const override;
    std::vector<IqInstructionFile> get_phase0_iq_files() const {
        return phase0_upcast.get_iq_files();
    }
    std::vector<IqInstructionFile> get_stage1_iq_files() const { return stage1_post.get_iq_files(); }
};

class SigmoidOp : public OpBase {
public:
    std::string get_name() const override;
    OpInstructionsBase* compile(const app::AppOperatorRequest& req) override;
    bool generate_data(const app::AppOperatorRequest& req,
                       std::vector<uint8_t>& input_bin,
                       std::vector<uint8_t>& other_bin,
                       std::vector<uint8_t>& alpha_bin) override;
    bool calculate_golden(const app::AppOperatorRequest& req,
                          std::vector<uint8_t>& golden_bin) override;
    void fill_random_request(app::AppOperatorRequest& req,
                             std::mt19937& rng,
                             std::uniform_real_distribution<float>& data_dist,
                             const std::string& dtype_str) override;

    static SigmoidInstructions compile_two_stage(const app::AppOperatorRequest& req);
    static SigmoidUpcastInstructions compile_upcast(const app::AppOperatorRequest& req,
                                                    uint32_t total_steps,
                                                    uint32_t total_elems,
                                                    uint8_t smc_id_en);
    static SigmoidPostInstructions compile_postprocess(const app::AppOperatorRequest& req,
                                                       uint32_t total_steps,
                                                       uint32_t total_elems,
                                                       uint8_t smc_id_en,
                                                       uint32_t newton_iters);

    /** 统计单通道微指令条数（EMBED 展开后的队列深度占用）。 */
    static uint32_t max_iq_inst_count(const std::vector<IqInstructionFile>& files);
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_SIGMOID_OP_H_
