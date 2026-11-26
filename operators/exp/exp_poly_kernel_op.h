#ifndef AIACC_SMI_IQ_OP_EXP_POLY_KERNEL_OP_H_
#define AIACC_SMI_IQ_OP_EXP_POLY_KERNEL_OP_H_

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

struct ExpPolyKernelParams {
    uint32_t para_n{1};
    uint32_t para_c{1};
    uint32_t para_h{1};
    uint32_t para_w{4};
    uint8_t para_dtype{0};

    uint8_t para_r_bank{0};
    uint8_t para_coeff_bank{1};
    uint8_t para_acc_bank{2};   // P2 Horner 累加器（复用已废弃的 x bank）
    uint8_t para_output_bank{3};
    uint32_t para_r_addr{0};
    uint32_t para_coeff_addr{0};
    uint32_t para_acc_addr{0};
    uint32_t para_output_addr{0};

    uint32_t para_num_terms{5};
    uint8_t para_smc_id_en{0};
    uint16_t para_rtovr_read_extend_loops{0};
    // 同 ExpPreprocessParams：FPGA 稀疏掩码 VIN/VOUT 直通开关
    bool para_use_vin_vout{false};

    uint32_t get_elems_per_word() const {
        return (para_dtype == 0) ? 4u : 8u;
    }
};

struct ExpPolyKernelInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fpadd;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;
    std::vector<exp_iq::SchedEntry> sched_rtovr_5;
    std::vector<exp_iq::SchedEntry> sched_rtovr_6;
    // VIN/VOUT 捕获通道（仅 para_use_vin_vout）
    std::vector<exp_iq::SchedEntry> sched_rtovr_14;  // VIN0_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_15;  // VIN1_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_16;  // VOUT0_S
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;

    uint16_t para_fpmul_code{0};
    uint16_t para_fpadd_code{0};
    uint8_t para_smc_id_en{0};
    uint16_t para_rtovr_read_extend_loops{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};
    uint32_t para_block_cycles{0};
    uint32_t para_num_terms{5};
    // vv 直通时 body=61（kFinalCyclesVv），get_iq_files 发射长度要用它
    bool para_use_vin_vout{false};

    std::string get_operator_name() const override { return "ExpPolyKernel"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

class ExpPolyKernelOp : public OpBase {
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

    ExpPolyKernelParams parse_request(const app::AppOperatorRequest& req) const;
    // fold_mode_override：保留参数兼容；实现一律 unroll（忽略 override）
    ExpPolyKernelInstructions compile_kernel(
        const ExpPolyKernelParams& params,
        const char* fold_mode_override = nullptr) const;
    static void apply_smc_params(ExpPolyKernelParams& params,
                                 const app::AppOperatorRequest& req);

private:
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_EXP_POLY_KERNEL_OP_H_
