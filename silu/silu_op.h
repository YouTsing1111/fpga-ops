#ifndef AIACC_SMI_IQ_OP_SILU_OP_H_
#define AIACC_SMI_IQ_OP_SILU_OP_H_

/**
 * @file silu_op.h
 * @brief FP16 SiLU(x) = x * sigmoid(x) = x / (1 + e^{-x})，复用 FP32 Exp 与 Sigmoid Phase0。
 *
 * 片上精度路径（LSU）：
 *   x(FP16 dest_low @bank2；LSU LDB 与 Exp/Sigmoid 同 bank)
 *     --Phase0: FPtoFP FP16→FP32 → bank5 = x_fp32（全程存活！）；
 *               FPMUL×(-1) → -x (FP32 @bank2)
 *     --Phase1: Exp --> e^{-x} (@bank3)
 *     --Phase2: +1 → D=1+e^{-x}（覆写 bank3，e^{-x} 已死）；
 *               Newton×2 → σ(x) @bank6；
 *               ★ FPMUL x_fp32(bank5) × σ(bank6) → silu_fp32 @bank6；
 *               FPtoFP FP32→FP16 dest_low → y (@bank7)
 *
 * 与 Sigmoid 的唯一差异：D 从 bank5 挪到 bank3（bank5 留给 x_fp32 活到 Phase2 末尾），
 * Phase2 尾部多一个 FPMUL 窗口（+15 拍/词）。
 *
 * FPtoFP dest_low / src_low：每 FP32 槽低 16bit 放一个 FP16（高 16=0）。
 * bank7 高 16 位清零 + wbe=0x3333 语义与 Sigmoid 完全一致（见其 README §3）。
 */

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "../../app/app_json_parser.h"
#include "../common/op_instructions_base.h"
#include "../exp/exp_iq_utils.h"
#include "../exp/exp_op.h"
#include "../sigmoid/sigmoid_op.h"

namespace aiacc {
namespace op {

constexpr uint32_t kSiluIqDepthLimit = 128u;  // smi_top.v: SMI_IQ_INSTQ_DEPTH=128
constexpr uint32_t kSiluDefaultNewtonIters = 2u;

struct SiluPostInstructions : public OpInstructionsBase {
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
    // VIN/VOUT 直通（FPGA 稀疏掩码；默认关，非 vv 路径位级不变）
    // vv 下 bank7 刷零改走 LOGIC OR(0,0)（UR→UVR 直拷在稀疏掩码非法），
    // 且与 Newton x_n 写同窗串行复用 VOUT0（FPMUL_D 捕获先于 LOGIC_D）
    std::vector<exp_iq::SchedEntry> sched_logic;
    std::vector<exp_iq::SchedEntry> sched_rtovr_11;  // LOGIC S0
    std::vector<exp_iq::SchedEntry> sched_rtovr_12;  // LOGIC S1
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
    uint32_t para_newton_iters{kSiluDefaultNewtonIters};
    uint8_t para_smc_id_en{0};

    std::string get_operator_name() const override { return "SiluPost"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** Dense 适配微码（8×fp16/word ⇄ fp32 工作区），供 LSU 需求形状（1×1×11008 fp16
 *  连续直通）使用。dense word 布局（与 softmax streaming 一致）：
 *    128bit = [lane0: e0|e4, lane1: e1|e5, lane2: e2|e6, lane3: e3|e7]
 *    低16 = 前半 4 elems，高16 = 后半 4 elems。
 *  compile_dense_upcast（body=60，steps=nd）每 dense word w 两块链：
 *    cvt-lo: 读 bank7@w → FPtoFP src_low  → bank5@w        (x_fp32 低半组)
 *    mul-lo: bank5@w × (-1 @bank4:14)     → bank2@w        (-x 低半组，供 Exp)
 *    cvt-hi: 读 bank7@w → FPtoFP src_high → bank5@(hi_base+w)
 *    mul-hi: bank5@(hi_base+w) × (-1)     → bank2@(hi_base+w)
 *  compile_dense_downcast_hi（body=15，steps=nd）：
 *    读 bank6@(src_base+w)(silu_fp32 高半组) → FPtoFP fp32→fp16 dest_high
 *    → bank7@w（wbe=0xCCCC；低半由 SiluPost 的 dest_low wbe=0x3333 写备好，
 *    两写字节域不重叠，合成完整 dense word）。
 *  时序逐拍镜像 SigmoidOp::compile_upcast / SiluOp::compile_postprocess 的
 *  已验证块（vv: 读@S→VIN捕@S+1→发射@S+4→VOUT捕@S+8/9→写@S+11/12；
 *  非vv: 读@S→发射@S+3→DR粘滞→写@S+10）。 */
struct SiluDenseAdapterInstructions : public OpInstructionsBase {
    std::vector<exp_iq::SchedEntry> sched_ur_r0;
    std::vector<exp_iq::SchedEntry> sched_ur_r1;
    std::vector<exp_iq::SchedEntry> sched_ur_w0;
    std::vector<exp_iq::SchedEntry> sched_fpmul;
    std::vector<exp_iq::SchedEntry> sched_fptofp;
    std::vector<exp_iq::SchedEntry> sched_rtovr_3;   // FPMUL S0
    std::vector<exp_iq::SchedEntry> sched_rtovr_4;   // FPMUL S1
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

    std::string get_operator_name() const override { return "SiluDenseAdapter"; }
    uint32_t get_total_steps() const override { return para_total_steps; }
    uint32_t get_total_elements() const override { return para_total_elements; }
    std::vector<uint8_t> serialize() const override;
    std::vector<IqInstructionFile> get_iq_files() const override;
};

/** 容器：Phase0 复用 SigmoidUpcastInstructions，Phase1 复用 ExpMerged。 */
struct SiluInstructions : public OpInstructionsBase {
    SigmoidUpcastInstructions phase0_upcast;    // Phase0（SigmoidOp::compile_upcast 原样）
    ExpMergedInstructions stage0_exp;           // Phase1
    SiluPostInstructions stage1_post;           // Phase2（含最终 x·σ 乘窗）
    uint32_t para_num_stages{3};
    uint32_t para_iq_depth_limit{kSiluIqDepthLimit};
    uint32_t para_phase0_max_insts{0};
    uint32_t para_stage0_max_insts{0};
    uint32_t para_stage1_max_insts{0};
    bool para_phase0_fits{true};
    bool para_stage0_fits{true};
    bool para_stage1_fits{true};

    std::string get_operator_name() const override { return "Silu"; }
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

class SiluOp : public OpBase {
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

    static SiluInstructions compile_two_stage(const app::AppOperatorRequest& req);
    static SiluPostInstructions compile_postprocess(const app::AppOperatorRequest& req,
                                                    uint32_t total_steps,
                                                    uint32_t total_elems,
                                                    uint8_t smc_id_en,
                                                    uint32_t newton_iters);

    /** Dense upcast：bank7@w(8×fp16 dense) → bank5/bank2 fp32（lo@[w]、hi@[hi_base+w]）。
     *  steps=nd（dense words/SMC），body=60。 */
    static SiluDenseAdapterInstructions compile_dense_upcast(
        const app::AppOperatorRequest& req, uint32_t total_steps,
        uint32_t total_elems, uint8_t smc_id_en, uint32_t hi_base);
    /** Dense downcast-hi：bank6@(src_base+w) fp32 → bank7@w 高半（wbe=0xCCCC）。
     *  steps=nd，body=15。 */
    static SiluDenseAdapterInstructions compile_dense_downcast_hi(
        const app::AppOperatorRequest& req, uint32_t total_steps,
        uint32_t total_elems, uint8_t smc_id_en, uint32_t src_base);

    /** 统计单通道微指令条数（EMBED 展开后的队列深度占用）。 */
    static uint32_t max_iq_inst_count(const std::vector<IqInstructionFile>& files);
};

}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_SILU_OP_H_
