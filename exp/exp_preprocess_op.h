#ifndef AIACC_SMI_IQ_OP_EXP_PREPROCESS_OP_H_
#define AIACC_SMI_IQ_OP_EXP_PREPROCESS_OP_H_

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "../../app/app_json_parser.h"
#include "../common/op_base.h"
#include "../common/op_instructions_base.h"
#include "exp_decompose.h"
#include "exp_iq_utils.h"

namespace aiacc {
namespace op {

struct ExpPreprocessParams {
    uint32_t para_n{1};
    uint32_t para_c{1};
    uint32_t para_h{1};
    uint32_t para_w{4};
    uint8_t para_dtype{0};

    uint8_t para_raw_x_bank{2};
    uint8_t para_r_bank{0};
    uint8_t para_coeff_bank{1};
    uint32_t para_raw_x_addr{0};
    uint32_t para_r_addr{0};
    uint32_t para_coeff_addr{0};
    uint8_t para_log2e_bank{4};
    uint32_t para_log2e_addr{0};
    uint32_t para_bias127_addr{1};
    uint32_t para_shift23_addr{2};
    uint32_t para_half_addr{3};
    uint32_t para_one_addr{4};
    uint32_t para_int_one_addr{5};
    uint32_t para_int_zero_addr{6};
    uint32_t para_sqrt2_addr{7};
    // bank4@8..9 留给 Softmax；Exp k-flush 常数：
    uint32_t para_k_min_addr{10};   // -126.0f
    uint32_t para_k_max_addr{11};   // 127.0f
    uint32_t para_pos_inf_addr{12}; // +Inf
    // bank4@13：RN 魔法数（FP32 bits=0x4B400000=1.5·2²³；兼作 INT 减数）
    uint32_t para_round_magic_addr{13};
    uint8_t para_k_scratch_bank{0};
    uint32_t para_k_scratch_addr{200};  // 暂存整数 k；W1 addr_inc=false；flush 时再 Inttofp
    uint32_t para_scale_scratch_addr{201};  // scale 暂存：SelectScale / uflow 后供 SELECT（UR/UR）
    uint32_t para_scale_word_base{5};

    uint8_t para_smc_id_en{0};
    uint16_t para_rtovr_read_extend_loops{0};

    // VIN/VOUT 直通（FPGA USE_RTOVR_SPARSE_MASK）；默认 false 保持全交叉仿真旧行为，
    // 保护 sigmoid/silu/softmax 的 compile_merged 复用方零变化
    bool para_use_vin_vout{false};

    uint32_t get_elems_per_word() const {
        return (para_dtype == 0) ? 4u : 8u;
    }
};

struct ExpPreprocessInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_ur_w1;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_fptoint;
    std::vector<exp_iq::SchedEntry> sched_inttofp;
    std::vector<exp_iq::SchedEntry> sched_intadd;
    std::vector<exp_iq::SchedEntry> sched_logic;
    std::vector<exp_iq::SchedEntry> sched_rtovr_0;
    std::vector<exp_iq::SchedEntry> sched_rtovr_1;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    std::vector<exp_iq::SchedEntry> sched_rtovr_9;
    std::vector<exp_iq::SchedEntry> sched_rtovr_10;
    std::vector<exp_iq::SchedEntry> sched_rtovr_11;
    std::vector<exp_iq::SchedEntry> sched_rtovr_12;
    std::vector<exp_iq::SchedEntry> sched_rtovr_13;
    // VIN/VOUT 捕获通道（仅 para_use_vin_vout 时填充；默认恒 vld=0 冻结）
    std::vector<exp_iq::SchedEntry> sched_rtovr_14;  // VIN0_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_15;  // VIN1_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_16;  // VOUT0_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;
    std::vector<exp_iq::SchedEntry> sched_rtovr_36;

    uint16_t para_fpmul_code{0};
    uint16_t para_fpadd_sub_code{0};
    uint16_t para_fpadd_add_code{0};
    uint16_t para_fpadd_check_code{0};
    uint16_t para_fptoint_code{0};
    uint16_t para_inttofp_code{0};
    uint16_t para_intadd_sub_code{0};
    uint16_t para_intadd_exp_code{0};
    uint16_t para_logic_select_less_code{0};
    uint16_t para_logic_select_eq_code{0};
    uint16_t para_logic_select_great_code{0};
    uint16_t para_logic_shift_code{0};
    uint16_t para_logic_and_code{0};
    uint16_t para_logic_shr_code{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint32_t para_body_cycles{0};

    std::string get_operator_name() const override { return "ExpPreprocess"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

class ExpPreprocessOp : public OpBase {
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

    ExpPreprocessParams parse_request(const app::AppOperatorRequest& req) const;
    ExpPreprocessInstructions compile_preprocess(const ExpPreprocessParams& params) const;
    /** P2 相对相位的 k-flush（0/Inf），与 Horner 并行；Merged 时 offset=p1_body。 */
    ExpPreprocessInstructions compile_kflush_p2_overlay(
        const ExpPreprocessParams& params) const;
    static void apply_smc_params(ExpPreprocessParams& params,
                                 const app::AppOperatorRequest& req);

private:
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_EXP_PREPROCESS_OP_H_
