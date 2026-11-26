
#ifndef AIACC_SMI_IQ_OP_SOFTMAX_OP_H_
#define AIACC_SMI_IQ_OP_SOFTMAX_OP_H_

/**
 * @file softmax_op.h
 * @brief FP32 Softmax（dim=-1），复用 halfint FP32 Exp。
 *
 * 完整链路（分次 trigger）：
 *   Phase0a: max reduce → bank5（广播 m）
 *   Phase0b: x - m → x' @bank2
 *   Phase1:  Exp → e^{x'} @bank3
 *   Phase1.5: sum reduce → S @bank5
 *   Phase2:  Newton(1/S) + mul → y @bank3
 *
 * 联调：SOFTMAX_HOST_MAX=1 / SOFTMAX_HOST_SUM=1 跳过片上 reduce。
 * W 任意：内部 pad 到 ceil(W/4)*4，尾 lane = -1000.0f（依赖 Exp k-flush → e=0）。
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

constexpr uint32_t kSoftmaxIqDepthLimit = 128u;
constexpr uint32_t kSoftmaxDefaultNewtonIters = 3u;

/** Phase0a Phase B：列 max [c0..c3] → m@bank5 的实现策略。 */
enum class SoftmaxMaxReducePhaseBMode : uint8_t {
    /** FPMUL 掩码抽 lane + ADDTREE 广播 + 3×标量 max（当前默认）。 */
    EXTRACT_BROADCAST = 0,
    /** 复制 4 行 + DS 4×4 转置 + 直接行 max（STB/DS/LDB 周期为模型估计）。 */
    DS_TRANSPOSE = 1,
};

/** Phase0a：列 max  tournament → bank6@0；lane 树 → m 广播写 bank5。 */
struct SoftmaxMaxReduceInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_logic;
    std::vector<exp_iq::SchedEntry> sched_addtree8;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_11;
    std::vector<exp_iq::SchedEntry> sched_rtovr_12;
    std::vector<exp_iq::SchedEntry> sched_rtovr_13;
    std::vector<exp_iq::SchedEntry> sched_rtovr_32;
    std::vector<exp_iq::SchedEntry> sched_rtovr_33;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;

    uint32_t para_body_cycles{0};
    uint32_t para_col_body_cycles{0};
    uint32_t para_lane_tree_cycles{0};
    uint32_t para_phase_b_replicate_cycles{0};
    uint32_t para_phase_b_ds_cycles{0};
    uint32_t para_phase_b_compare_cycles{0};
    uint32_t para_phase_b_broadcast_cycles{0};
    uint8_t para_phase_b_mode{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint32_t para_col_embed_steps{0};
    uint8_t para_smc_id_en{0};

    std::string get_operator_name() const override { return "SoftmaxMaxReduce"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Phase0b：x' = x - broadcast(max@bank5)。 */
struct SoftmaxMaxSubInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;
    // VIN/VOUT 直通（FPGA 稀疏掩码；默认关，非 vv 路径位级不变）
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

    std::string get_operator_name() const override { return "SoftmaxMaxSub"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Phase1.5：Σ e → bank5（标量广播）；bank6 仍由 pack 写 magic(S)。 */
struct SoftmaxSumReduceInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_addtree8;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_32;
    std::vector<exp_iq::SchedEntry> sched_rtovr_33;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;

    uint32_t para_body_cycles{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint8_t para_smc_id_en{0};

    std::string get_operator_name() const override { return "SoftmaxSumReduce"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Phase2：1/S（Newton）+ e×inv → y；可选片上 magic(x0) 前导。 */
struct SoftmaxNormInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_fptoint;
    std::vector<exp_iq::SchedEntry> sched_intadd;
    std::vector<exp_iq::SchedEntry> sched_inttofp;
    std::vector<exp_iq::SchedEntry> sched_rtovr_0;
    std::vector<exp_iq::SchedEntry> sched_rtovr_1;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_9;
    std::vector<exp_iq::SchedEntry> sched_rtovr_10;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;
    // VIN/VOUT 直通（FPGA 稀疏掩码；默认关，非 vv 路径位级不变）
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
    uint32_t para_newton_iters{kSoftmaxDefaultNewtonIters};
    bool para_host_inv{false};
    bool para_host_x0{true};
    bool para_magic_only{false};
    bool para_run_magic_in_body{false};
    bool para_run_newton_in_body{false};
    uint8_t para_smc_id_en{0};
    /** ur_r0 仅对 abs_cycle >= 此值的读随外层 word 递增（bank3 e）；S/x0 读在此前固定 bank5/6@0。 */
    uint32_t para_ur_r0_inc_from_cycle{0};

    std::string get_operator_name() const override { return "SoftmaxNorm"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** 容器：Exp + 可选 Phase0/1.5 + Norm。 */
struct SoftmaxInstructions : public OpInstructionsBase {
    SoftmaxMaxReduceInstructions phase0_max_reduce;
    SoftmaxMaxSubInstructions phase0_maxsub;
    ExpMergedInstructions stage0_exp;
    SoftmaxSumReduceInstructions stage_sum_reduce;
    SoftmaxNormInstructions stage1_norm;

    uint32_t para_num_stages{2};
    uint32_t para_iq_depth_limit{kSoftmaxIqDepthLimit};
    uint32_t para_phase0_max_insts{0};
    uint32_t para_stage0_max_insts{0};
    uint32_t para_sum_max_insts{0};
    uint32_t para_stage1_max_insts{0};
    bool para_phase0_fits{true};
    bool para_stage0_fits{true};
    bool para_sum_fits{true};
    bool para_stage1_fits{true};
    bool para_host_max{true};
    bool para_host_sum{true};
    float para_preview_sum{0.f};
    float para_preview_max{0.f};

    std::string get_operator_name() const override { return "Softmax"; }
    uint32_t get_total_steps() const override { return stage0_exp.get_total_steps(); }
    uint32_t get_total_elements() const override { return stage0_exp.get_total_elements(); }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
    std::vector<IqInstructionFile> get_phase0a_iq_files() const {
        return phase0_max_reduce.get_iq_files();
    }
    std::vector<IqInstructionFile> get_phase0_iq_files() const {
        return phase0_maxsub.get_iq_files();
    }
    std::vector<IqInstructionFile> get_sum_iq_files() const {
        return stage_sum_reduce.get_iq_files();
    }
    std::vector<IqInstructionFile> get_stage1_iq_files() const {
        return stage1_norm.get_iq_files();
    }
};

class SoftmaxOp : public OpBase {
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

    static SoftmaxInstructions compile_two_stage(const app::AppOperatorRequest& req);
    static SoftmaxMaxReduceInstructions compile_max_reduce(
        const app::AppOperatorRequest& req,
        uint32_t total_steps,
        uint32_t total_elems,
        uint8_t smc_id_en,
        SoftmaxMaxReducePhaseBMode phase_b_mode =
            SoftmaxMaxReducePhaseBMode::EXTRACT_BROADCAST,
        bool skip_phase_b = false);
    static SoftmaxMaxSubInstructions compile_maxsub(const app::AppOperatorRequest& req,
                                                  uint32_t total_steps,
                                                  uint32_t total_elems,
                                                  uint8_t smc_id_en);
    static SoftmaxSumReduceInstructions compile_sum_reduce(
        const app::AppOperatorRequest& req,
        uint32_t total_steps,
        uint32_t total_elems,
        uint8_t smc_id_en,
        bool skip_broadcast = false);
    /** 仅 Phase B 广播：Σ(跨段累积 bank6@0) → S@bank5（配合 skip_broadcast=true 的多次 compile_sum_reduce）。 */
    static SoftmaxSumReduceInstructions compile_sum_broadcast(uint8_t smc_id_en);
    static SoftmaxNormInstructions compile_normalize(const app::AppOperatorRequest& req,
                                                     uint32_t total_steps,
                                                     uint32_t total_elems,
                                                     uint8_t smc_id_en,
                                                     uint32_t newton_iters);

    static bool compute_stable_fp32(const app::AppOperatorRequest& req,
                                    std::vector<float>& x_prime,
                                    float& sum_exp,
                                    float& inv_sum,
                                    float* out_max = nullptr);

    static uint32_t max_iq_inst_count(const std::vector<IqInstructionFile>& files);

private:
    static bool parse_i32_param(const app::AppOperatorRequest& req,
                                const std::string& key,
                                int32_t& out);
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_SOFTMAX_OP_H_
