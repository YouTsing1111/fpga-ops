#ifndef AIACC_SMI_IQ_OP_EXP_OP_H_
#define AIACC_SMI_IQ_OP_EXP_OP_H_

#include <random>
#include <string>
#include <vector>

#include "../../app/app_json_parser.h"
#include "../common/op_instructions_base.h"
#include "exp_poly_kernel_op.h"
#include "exp_preprocess_op.h"

namespace aiacc {
namespace op {

/** Single-stream merged P1+P2 microcode (one stage, one IQ set). */
struct ExpMergedInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_r2;
    std::vector<exp_iq::SchedEntry> sched_ur_r3;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_ur_w1;  // P1：暂存 k_f @bank0/200（addr_inc=false）
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
    // VIN/VOUT（FPGA 稀疏掩码）：14/15/16 为捕获路由（默认 vld=0 冻结），
    // vin0/vin1/vout0 为 IQ 68/69/70 使能流（0x3F 常开）
    std::vector<exp_iq::SchedEntry> sched_rtovr_14;
    std::vector<exp_iq::SchedEntry> sched_rtovr_15;
    std::vector<exp_iq::SchedEntry> sched_rtovr_16;
    std::vector<exp_iq::SchedEntry> sched_vin0;
    std::vector<exp_iq::SchedEntry> sched_vin1;
    std::vector<exp_iq::SchedEntry> sched_vout0;
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;
    std::vector<exp_iq::SchedEntry> sched_rtovr_36;  // 半整数 P1 默认源占位

    bool para_use_vin_vout{false};
    uint32_t para_p1_body_cycles{0};
    uint32_t para_p2_body_cycles{0};
    uint32_t para_merged_body_cycles{0};
    uint32_t para_total_steps{0};
    uint32_t para_total_elements{0};
    uint32_t para_total_cycles{0};

    std::string get_operator_name() const override { return "Exp"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Dense 适配微码（8×fp16/word ⇄ fp32 工作区），供 LSU 需求形状（1×1×11008 fp16）
 *  直通使用，全部复用已验证块（时序逐拍镜像 silu dense adapter / sigmoid upcast）。
 *  布局（与 silu/softmax dense 一致）：128bit = [lane0: e0|e4, lane1: e1|e5,
 *  lane2: e2|e6, lane3: e3|e7]；低16 = 前半 4 elems，高16 = 后半 4 elems。
 *  compile_dense_upcast（body=30，steps=nd）每 dense word w：
 *    cvt-lo: bank7@w → FPtoFP src_low  → bank2@w          （x_fp32 低半组）
 *    cvt-hi: bank7@w → FPtoFP src_high → bank2@(hi_base+w)（x_fp32 高半组）
 *  compile_dense_downcast（body=30，steps=nd）：
 *    lo: bank3@(src_base+w)      → FPtoFP dest_low  → bank7@w（wbe=0x3333）
 *    hi: bank3@(src_base+nd+w)   → FPtoFP dest_high → bank7@w（wbe=0xCCCC）
 *  两写字节域互补，合成完整 dense word；[nd,256) 保持 UR 零初始化。 */
struct ExpDenseAdapterInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fptofp;
    std::vector<exp_iq::SchedEntry> sched_rtovr_7;   // FPtoFP S
    std::vector<exp_iq::SchedEntry> sched_rtovr_35;  // UVR wdata 源
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

    std::string get_operator_name() const override { return "ExpDenseAdapter"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

class ExpOp : public OpBase {
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

    static ExpMergedInstructions compile_merged(const app::AppOperatorRequest& req);
    /** LSU dense（需求 fp16 形状）：bank7 dense → bank2 fp32 lo/hi 两组。 */
    static ExpDenseAdapterInstructions compile_dense_upcast(
        const app::AppOperatorRequest& req,
        uint32_t total_steps,
        uint32_t total_elems,
        uint8_t smc_id_en,
        uint32_t hi_base);
    /** LSU dense：bank3 y_fp32 lo/hi 两组 → bank7 dense（wbe 0x3333/0xCCCC 互补）。 */
    static ExpDenseAdapterInstructions compile_dense_downcast(
        const app::AppOperatorRequest& req,
        uint32_t total_steps,
        uint32_t total_elems,
        uint8_t smc_id_en,
        uint32_t src_base);
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_EXP_OP_H_
