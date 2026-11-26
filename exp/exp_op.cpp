#include "exp_op.h"

#include <cstdlib>
#include <iostream>

#include "../../iq/iq_id_defs.h"
#include "../../iq/modules/fptofp_iq.h"
#include "../common/op_registry.h"
#include "exp_decompose.h"
#include "exp_iq_utils.h"

namespace aiacc {
namespace op {

using namespace exp_iq;
using namespace exp_math;

namespace {
// dense adapter 常量（与 silu dense 同值；时序逐拍镜像已验证块）。
constexpr uint8_t kRtovrOutFpToFpS = 7;
constexpr uint8_t kRtovrInDrFpToFpD = 6;
constexpr uint8_t kExpBankFp16Io = 7;   // fp16 dense 输入/输出
constexpr uint8_t kExpBankXfp32 = 2;    // fp32 x（Exp preprocess 输入）
constexpr uint8_t kExpBankYfp32 = 3;    // fp32 y（Exp kernel 输出）
constexpr uint32_t kExpUrSettle = 3;        // == kUrReadToComputeIssue
constexpr uint32_t kExpFptofpSpan = 13;
constexpr uint32_t kExpFptofpWrite = 10;
constexpr uint16_t kWbeLow16 = 0x3333u;
constexpr uint16_t kWbeHigh16 = 0xCCCCu;

void append_sched(std::vector<SchedEntry>& s, uint32_t cyc, uint64_t code) {
    s.push_back({cyc, code});
}

/** Hold RtoVR select every cycle in [c0, c1) — required; one-shot select is not enough. */
void hold_rtovr(std::vector<SchedEntry>& s, uint32_t c0, uint32_t c1, uint8_t src) {
    const uint64_t code = rtovr_code(src);
    for (uint32_t t = c0; t < c1; ++t) append_sched(s, t, code);
}

bool vin_vout_mode(const app::AppOperatorRequest& req) {
    auto it = req.para_parameters.find("vin_vout");
    return it != req.para_parameters.end() && !it->second.para_data_i32.empty() &&
           it->second.para_data_i32[0] != 0;
}

void fill_vv_enable_streams(std::vector<SchedEntry>& vin0,
                            std::vector<SchedEntry>& vin1,
                            std::vector<SchedEntry>& vout0,
                            uint32_t body_cycles) {
    for (uint32_t t = 0; t < body_cycles; ++t) {
        vin0.push_back({t, kVinVoutEnableCode});
        vin1.push_back({t, kVinVoutEnableCode});
        vout0.push_back({t, kVinVoutEnableCode});
    }
}
}  // namespace

std::string ExpOp::get_name() const { return "Exp"; }

ExpMergedInstructions ExpOp::compile_merged(const app::AppOperatorRequest& req) {
    ExpPreprocessOp preprocess;
    ExpPolyKernelOp kernel;

    ExpPreprocessParams pre_params = preprocess.parse_request(req);
    ExpPreprocessOp::apply_smc_params(pre_params, req);

    ExpPolyKernelParams ker_params = kernel.parse_request(req);
    ExpPolyKernelOp::apply_smc_params(ker_params, req);

    const ExpPreprocessInstructions pre = preprocess.compile_preprocess(pre_params);
    const ExpPreprocessInstructions flush =
        preprocess.compile_kflush_p2_overlay(pre_params);

    const ExpPolyKernelInstructions ker =
        kernel.compile_kernel(ker_params, "unroll");

    const uint32_t p1_body = pre.para_body_cycles;
    // vv 直通：P1 body 57（core 内已含 +1），P2 body 61（Final y 写@60）
    const uint32_t p2_body = ker.para_block_cycles * kNumStages +
        (pre_params.para_use_vin_vout ? kFinalCyclesVv : kFinalCycles);
    const uint32_t merged_body = p1_body + p2_body;

    ExpMergedInstructions out;
    out.para_p1_body_cycles = p1_body;
    out.para_p2_body_cycles = p2_body;
    out.para_merged_body_cycles = merged_body;
    out.para_total_steps = pre.para_total_steps;
    out.para_total_elements = pre.para_total_elements;
    out.para_total_cycles = merged_body * pre.para_total_steps;

    auto merge3 = [&](std::vector<SchedEntry>& dst,
                      const std::vector<SchedEntry>& a,
                      const std::vector<SchedEntry>& b,
                      const std::vector<SchedEntry>& c) {
        merge_sched_pair(dst, a, b, p1_body);
        append_sched_offset(dst, c, p1_body);
    };

    merge3(out.sched_ur_r1, pre.sched_ur_r1, ker.sched_ur_r1, flush.sched_ur_r1);
    merge3(out.sched_ur_r0, pre.sched_ur_r0, ker.sched_ur_r0, flush.sched_ur_r0);
    merge_sched_pair(out.sched_ur_r2, pre.sched_ur_r2, ker.sched_ur_r2, p1_body);
    merge3(out.sched_ur_w0, pre.sched_ur_w0, ker.sched_ur_w0, flush.sched_ur_w0);
    out.sched_ur_w1 = pre.sched_ur_w1;
    merge_sched_pair(out.sched_fpmul, pre.sched_fpmul, ker.sched_fpmul, p1_body);
    merge3(out.sched_fpadd, pre.sched_fpadd, ker.sched_fpadd, flush.sched_fpadd);

    out.sched_fptoint = pre.sched_fptoint;
    merge_sched_pair(out.sched_inttofp, pre.sched_inttofp, flush.sched_inttofp, p1_body);
    out.sched_intadd = pre.sched_intadd;
    merge_sched_pair(out.sched_logic, pre.sched_logic, flush.sched_logic, p1_body);

    merge_sched_pair(out.sched_rtovr_0, pre.sched_rtovr_0, flush.sched_rtovr_0, p1_body);
    merge_sched_pair(out.sched_rtovr_1, pre.sched_rtovr_1, flush.sched_rtovr_1, p1_body);
    merge_sched_pair(out.sched_rtovr_3, pre.sched_rtovr_3, ker.sched_rtovr_3, p1_body);
    merge_sched_pair(out.sched_rtovr_4, pre.sched_rtovr_4, ker.sched_rtovr_4, p1_body);
    merge3(out.sched_rtovr_5, pre.sched_rtovr_5, ker.sched_rtovr_5, flush.sched_rtovr_5);
    merge3(out.sched_rtovr_6, pre.sched_rtovr_6, ker.sched_rtovr_6, flush.sched_rtovr_6);
    out.sched_rtovr_9 = pre.sched_rtovr_9;
    merge_sched_pair(out.sched_rtovr_10, pre.sched_rtovr_10, flush.sched_rtovr_10, p1_body);
    merge_sched_pair(out.sched_rtovr_11, pre.sched_rtovr_11, flush.sched_rtovr_11, p1_body);
    merge_sched_pair(out.sched_rtovr_12, pre.sched_rtovr_12, flush.sched_rtovr_12, p1_body);
    merge_sched_pair(out.sched_rtovr_13, pre.sched_rtovr_13, flush.sched_rtovr_13, p1_body);
    merge3(out.sched_rtovr_35, pre.sched_rtovr_35, ker.sched_rtovr_35, flush.sched_rtovr_35);
    out.sched_rtovr_36 = pre.sched_rtovr_36;

    // VIN/VOUT 捕获通道（P1 + P2∥flush 三段拼接）
    merge3(out.sched_rtovr_14, pre.sched_rtovr_14, ker.sched_rtovr_14, flush.sched_rtovr_14);
    merge3(out.sched_rtovr_15, pre.sched_rtovr_15, ker.sched_rtovr_15, flush.sched_rtovr_15);
    merge3(out.sched_rtovr_16, pre.sched_rtovr_16, ker.sched_rtovr_16, flush.sched_rtovr_16);

    // IQ 68/69/70 使能流：0x3F（smc_id_en=31 阈值覆盖 SMC0..31）全 body 常开
    out.para_use_vin_vout = pre_params.para_use_vin_vout;
    if (out.para_use_vin_vout) {
        for (uint32_t t = 0; t < merged_body; ++t) {
            out.sched_vin0.push_back({t, kVinVoutEnableCode});
            out.sched_vin1.push_back({t, kVinVoutEnableCode});
            out.sched_vout0.push_back({t, kVinVoutEnableCode});
        }
        std::cout << "[Exp] VIN/VOUT: FPGA sparse-mask 直通（rtovr-14/15/16 + iq 68/69/70）"
                  << std::endl;
    }

    std::cout << "[Exp] halfint merged: P1core=" << p1_body
              << " + P2 Horner∥kflush " << p2_body
              << (pre.para_total_steps > 1u ? " (multiword)" : " (steps=1)")
              << ", wall≈" << merged_body << " cyc/word"
              << ", steps=" << out.para_total_steps
              << " (√2@P1; 0/Inf∥Horner gaps; expect≈115, vv≈118)" << std::endl;
    return out;
}

// ---------------------------------------------------------------------------
// ExpDenseAdapterInstructions（LSU dense 需求形状的 cvt 适配；时序镜像 silu dense）
// ---------------------------------------------------------------------------

std::vector<uint8_t> ExpDenseAdapterInstructions::serialize() const {
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

std::vector<IqInstructionFile> ExpDenseAdapterInstructions::get_iq_files() const {
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
                                        "ExpDenseAdapter");
        files.push_back(f);
    };

    // upcast：r0 每 body 两次读 bank7@w（lo/hi 块同址）；downcast：r0=lo(bank3@w)、
    // r1=hi(bank3@nd+w) 两个递增基址 —— 均 addr_inc=true（同通道多基址已验证）。
    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17, true);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33, true);
    add_iq(sched_fptofp, iq::IQ_ID_FP_TO_FP, iq::IQ_NAME_FP_TO_FP, 10, false);
    add_iq(sched_rtovr_7, iq::IQ_ID_RTOVR_BASE + kRtovrOutFpToFpS, "rtovr-7", 7, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false);
    if (para_use_vin_vout) {
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false);
        add_iq(sched_vin0,  iq::IQ_ID_VIN0,  "68_vin0", 7, false);
        add_iq(sched_vin1,  iq::IQ_ID_VIN1,  "69_vin1", 7, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false);
    }
    return files;
}

ExpDenseAdapterInstructions ExpOp::compile_dense_upcast(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t hi_base) {
    // body=30 = 2 块×15（cvt-lo@0, cvt-hi@15）。每 dense word w：
    //   bank7@w --FPtoFP src_low-->  bank2@w           （x_fp32，elems 8w+0..3）
    //   bank7@w --FPtoFP src_high--> bank2@(hi_base+w) （x_fp32，elems 8w+4..7）
    // lo 写 @S+12（vv）/ @S+10，hi 块 @S=15 写 @S+27/25 —— 次块起点 30 > 27 ✔。
    ExpDenseAdapterInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;

    iq::FpToFpCru up[2];
    for (auto& u : up) {
        u.para_inst_valid = true;
        u.para_src_is_32b = false;   // FP16
        u.para_dst_is_32b = true;    // → FP32
        u.para_dst_high = false;
    }
    up[0].para_src_high = false;     // src_low  → elems 8w+0..3
    up[1].para_src_high = true;      // src_high → elems 8w+4..7
    uint64_t up_code[2];
    for (int h = 0; h < 2; ++h)
        up_code[h] = (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(up[h])) |
                      (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    // cvt 块：读 bank7@w（addr_inc +w），写 bank2@(dst_base+w)
    auto emit_cvt = [&](uint32_t S, uint64_t code, uint32_t dst_base) {
        append_sched(out.sched_ur_r0, S, ur_read_code(kExpBankFp16Io, 0, smc_id_en));
        if (vv) {
            // VIN0 捕获@S+1，消费@S+4；VOUT0 捕获@S+9（FPTOFP 延迟边界），写@S+12
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(kRtovrInUrRdata0));
            hold_rtovr(out.sched_rtovr_7, S, S + 5, kRtovrInDrVin0D);
            append_sched(out.sched_fptofp, S + 4, code);
            append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrFpToFpD));
            hold_rtovr(out.sched_rtovr_35, S + 9, S + 12, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 12,
                         ur_write_code(kExpBankXfp32, dst_base, smc_id_en));
        } else {
            hold_rtovr(out.sched_rtovr_7, S, S + kExpFptofpSpan, kRtovrInUrRdata0);
            append_sched(out.sched_fptofp, S + kExpUrSettle, code);
            hold_rtovr(out.sched_rtovr_35, S + kExpUrSettle + kFpmulDrLatencyRtl,
                       S + kExpFptofpSpan, kRtovrInDrFpToFpD);
            append_sched(out.sched_ur_w0, S + kExpFptofpWrite,
                         ur_write_code(kExpBankXfp32, dst_base, smc_id_en));
        }
    };

    emit_cvt(0,  up_code[0], 0);         // lo: bank2@w
    emit_cvt(15, up_code[1], hi_base);   // hi: bank2@(hi_base+w)

    out.para_body_cycles = 30;
    out.para_total_cycles = 30 * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }
    std::cout << "[Exp] dense upcast body=30 cyc/word, steps=" << total_steps
              << " hi_base=" << hi_base << (vv ? " [vin_vout]" : "") << std::endl;
    return out;
}

ExpDenseAdapterInstructions ExpOp::compile_dense_downcast(
    const app::AppOperatorRequest& req,
    uint32_t total_steps,
    uint32_t total_elems,
    uint8_t smc_id_en,
    uint32_t src_base) {
    // body=30 = 2 块×15（lo@0, hi@15）。每 dense word w：
    //   bank3@(src_base+w)     --FPtoFP fp32→fp16 dest_low-->  bank7@w（wbe=0x3333）
    //   bank3@(src_base+nd+w)  --FPtoFP fp32→fp16 dest_high--> bank7@w（wbe=0xCCCC）
    // 两写字节域互补合成完整 dense word；dest_high 的 D 线低16位残留被 wbe 掩掉。
    // r0/r1 各承担一个递增读基址（lo/hi 双 fp32 组）。
    ExpDenseAdapterInstructions out;
    out.para_total_steps = total_steps;
    out.para_total_elements = total_elems;
    out.para_smc_id_en = smc_id_en;
    const bool vv = vin_vout_mode(req);
    out.para_use_vin_vout = vv;
    const uint32_t nd = total_steps;

    iq::FpToFpCru dn[2];
    for (auto& c : dn) {
        c.para_inst_valid = true;
        c.para_src_is_32b = true;    // FP32
        c.para_dst_is_32b = false;   // → FP16
        c.para_src_high = false;
    }
    dn[0].para_dst_high = false;     // dest_low
    dn[1].para_dst_high = true;      // dest_high
    uint64_t dn_code[2];
    for (int h = 0; h < 2; ++h)
        dn_code[h] = (static_cast<uint64_t>(iq::FpToFpIq::pack_cru_fptofp(dn[h])) |
                      (static_cast<uint64_t>(smc_id_en & 0x1Fu) << 5));

    // 块：读口 dst_bank@src（addr_inc），写 bank7@0（wbe 区分半字）。
    // ur_port 必须与读调度所在通道一致：lo 块读走 r0→RDATA_0(30)，hi 块读走 r1→RDATA_1(31)；
    // 否则 VIN0 抓到另一口的驻留数据（hi16 写成 lo 组值的经典症状）。
    auto emit_blk = [&](uint32_t S, uint64_t code, uint8_t rd_bank, uint32_t src,
                        uint16_t wbe, std::vector<SchedEntry>& rd_sched,
                        uint32_t ur_port) {
        append_sched(rd_sched, S, ur_read_code(rd_bank, src, smc_id_en));
        if (vv) {
            append_sched(out.sched_rtovr_14, S + 1, rtovr_code(ur_port));
            hold_rtovr(out.sched_rtovr_7, S, S + 5, kRtovrInDrVin0D);
            append_sched(out.sched_fptofp, S + 4, code);
            append_sched(out.sched_rtovr_16, S + 9, rtovr_code(kRtovrInDrFpToFpD));
            hold_rtovr(out.sched_rtovr_35, S + 9, S + 12, kRtovrInDrVout0D);
            append_sched(out.sched_ur_w0, S + 12,
                         ur_write_code(kExpBankFp16Io, 0, smc_id_en, wbe));
        } else {
            hold_rtovr(out.sched_rtovr_7, S, S + kExpFptofpSpan, ur_port);
            append_sched(out.sched_fptofp, S + kExpUrSettle, code);
            hold_rtovr(out.sched_rtovr_35, S + kExpUrSettle + kFpmulDrLatencyRtl,
                       S + kExpFptofpSpan, kRtovrInDrFpToFpD);
            append_sched(out.sched_ur_w0, S + kExpFptofpWrite,
                         ur_write_code(kExpBankFp16Io, 0, smc_id_en, wbe));
        }
    };

    emit_blk(0,  dn_code[0], kExpBankYfp32, src_base, kWbeLow16, out.sched_ur_r0,
             kRtovrInUrRdata0);
    emit_blk(15, dn_code[1], kExpBankYfp32, src_base + nd, kWbeHigh16, out.sched_ur_r1,
             kRtovrInUrRdata0 + 1);

    out.para_body_cycles = 30;
    out.para_total_cycles = 30 * total_steps;
    if (vv) {
        fill_vv_enable_streams(out.sched_vin0, out.sched_vin1, out.sched_vout0,
                               out.para_body_cycles);
    }
    std::cout << "[Exp] dense downcast body=30 cyc/word, steps=" << total_steps
              << " src_base=" << src_base << (vv ? " [vin_vout]" : "") << std::endl;
    return out;
}

OpInstructionsBase* ExpOp::compile(const app::AppOperatorRequest& req) {
    auto* insts = new ExpMergedInstructions();
    *insts = compile_merged(req);
    return insts;
}

std::vector<uint8_t> ExpMergedInstructions::serialize() const {
    std::vector<uint8_t> result;
    auto push_u32 = [&](uint32_t v) {
        result.push_back(static_cast<uint8_t>(v & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
        result.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    };
    push_u32(para_p1_body_cycles);
    push_u32(para_p2_body_cycles);
    push_u32(para_merged_body_cycles);
    push_u32(para_total_steps);
    push_u32(para_total_elements);
    push_u32(para_total_cycles);
    return result;
}

std::vector<IqInstructionFile> ExpMergedInstructions::get_iq_files() const {
    std::vector<IqInstructionFile> files;
    const uint32_t total_steps = para_total_steps;
    const uint32_t p1_body = para_p1_body_cycles;
    const uint32_t p2_body = para_p2_body_cycles;

    // 一律：整段 P1|P2 → emit_exp_embed_iq（unroll）
    auto add_iq = [&](const std::vector<SchedEntry>& sched,
                      uint32_t iq_id, const std::string& iq_name,
                      uint32_t code_bits,
                      bool p1_outer_addr_inc,
                      bool p2_horner_addr_inc,
                      bool p2_outer_addr_inc,
                      bool suppress_flush_fixed_ur_inc = false) {
        if (sched.empty()) return;
        IqInstructionFile file;
        file.para_iq_id = iq_id;
        file.para_iq_name = iq_name;
        file.para_width_bits = 41 + code_bits;
        file.para_data = emit_merged_p1_then_fold_p2_iq(
            sched, code_bits, p1_body, total_steps,
            p1_outer_addr_inc, p2_horner_addr_inc, p2_outer_addr_inc, "Exp",
            suppress_flush_fixed_ur_inc, p2_body);
        files.push_back(file);
    };

    // p1_outer / p2_horner(lev1) / p2_outer(lev2) — 对齐 standalone P1/P2 习惯
    // R0/R1：∥flush 固定地址读关掉 outer inc，避免多字把 scratch/常数地址 +1
    add_iq(sched_ur_r0, iq::IQ_ID_UR_RDATA_0, iq::IQ_NAME_UR_RDATA_0, 17,
           true, false, true, true);
    add_iq(sched_ur_r1, iq::IQ_ID_UR_RDATA_1, iq::IQ_NAME_UR_RDATA_1, 17,
           false, false, true, true);
    add_iq(sched_ur_r2, iq::IQ_ID_UR_RDATA_2, iq::IQ_NAME_UR_RDATA_2, 17,
           false, true, false);  // P2 coeff：Horner lev1
    add_iq(sched_ur_r3, iq::IQ_ID_UR_RDATA_3, iq::IQ_NAME_UR_RDATA_3, 17,
           true, false, true);
    add_iq(sched_ur_w0, iq::IQ_ID_UR_WDATA_0, iq::IQ_NAME_UR_WDATA_0, 33,
           true, false, true);
    add_iq(sched_ur_w1, iq::IQ_ID_UR_WDATA_1, iq::IQ_NAME_UR_WDATA_1, 33,
           false, false, false);
    add_iq(sched_fpmul, iq::IQ_ID_FPMUL, iq::IQ_NAME_FPMUL, 8,
           false, false, false);
    add_iq(sched_fptoint, iq::IQ_ID_FP_TO_INT, iq::IQ_NAME_FP_TO_INT, 10,
           false, false, false);
    add_iq(sched_inttofp, iq::IQ_ID_INT_TO_FP, iq::IQ_NAME_INT_TO_FP, 11,
           false, false, false);
    add_iq(sched_fpadd, iq::IQ_ID_FPADD, iq::IQ_NAME_FPADD, 10,
           false, false, false);
    add_iq(sched_intadd, iq::IQ_ID_INTADD, iq::IQ_NAME_INTADD, 18,
           false, false, false);
    add_iq(sched_logic, iq::IQ_ID_LOGIC, iq::IQ_NAME_LOGIC, 11,
           false, false, false);
    add_iq(sched_rtovr_0, iq::IQ_ID_RTOVR_BASE + 0, "rtovr-0", 7, false, false, false);
    add_iq(sched_rtovr_1, iq::IQ_ID_RTOVR_BASE + 1, "rtovr-1", 7, false, false, false);
    add_iq(sched_rtovr_3, iq::IQ_ID_RTOVR_BASE + 3, "rtovr-3", 7, false, false, false);
    add_iq(sched_rtovr_4, iq::IQ_ID_RTOVR_BASE + 4, "rtovr-4", 7, false, false, false);
    add_iq(sched_rtovr_5, iq::IQ_ID_RTOVR_BASE + 5, "rtovr-5", 7, false, false, false);
    add_iq(sched_rtovr_6, iq::IQ_ID_RTOVR_BASE + 6, "rtovr-6", 7, false, false, false);
    add_iq(sched_rtovr_9, iq::IQ_ID_RTOVR_BASE + 9, "rtovr-9", 7, false, false, false);
    add_iq(sched_rtovr_10, iq::IQ_ID_RTOVR_BASE + 10, "rtovr-10", 7, false, false, false);
    add_iq(sched_rtovr_11, iq::IQ_ID_RTOVR_BASE + 11, "rtovr-11", 7, false, false, false);
    add_iq(sched_rtovr_12, iq::IQ_ID_RTOVR_BASE + 12, "rtovr-12", 7, false, false, false);
    add_iq(sched_rtovr_13, iq::IQ_ID_RTOVR_BASE + 13, "rtovr-13", 7, false, false, false);
    add_iq(sched_rtovr_35, iq::IQ_ID_RTOVR_BASE + 35, "rtovr-35", 7, false, false, false);
    add_iq(sched_rtovr_36, iq::IQ_ID_RTOVR_BASE + 36, "rtovr-36", 7, false, false, false);
    // 同一份微码须同时过全交叉仿真与 FPGA 稀疏掩码：两条模式下 emit 均合法；
    // 仅在 VIN/VOUT 开时多出六份文件（mul/sub 同款命名 → 打包流程识别）
    if (para_use_vin_vout) {
        add_iq(sched_rtovr_14, iq::IQ_ID_RTOVR_BASE + 14, "rtovr-14", 7, false, false, false);
        add_iq(sched_rtovr_15, iq::IQ_ID_RTOVR_BASE + 15, "rtovr-15", 7, false, false, false);
        add_iq(sched_rtovr_16, iq::IQ_ID_RTOVR_BASE + 16, "rtovr-16", 7, false, false, false);
        add_iq(sched_vin0,  iq::IQ_ID_VIN0,  "68_vin0",  7, false, false, false);
        add_iq(sched_vin1,  iq::IQ_ID_VIN1,  "69_vin1",  7, false, false, false);
        add_iq(sched_vout0, iq::IQ_ID_VOUT0, "70_vout0", 7, false, false, false);
    }
    return files;
}

bool ExpOp::generate_data(const app::AppOperatorRequest& req,
                          std::vector<uint8_t>& input_bin,
                          std::vector<uint8_t>& other_bin,
                          std::vector<uint8_t>& alpha_bin) {
    const uint32_t total_elems = req.para_dimensions.para_n * req.para_dimensions.para_c *
                                 req.para_dimensions.para_h * req.para_dimensions.para_w;
    const bool fp16 = [&]() {
        auto it = req.para_inputs.find("input_tensor");
        return it != req.para_inputs.end() &&
               it->second.para_dtype == app::AppDataType::FLOAT16;
    }();
    const std::vector<float> raw_x = collect_or_make_input_data(req, total_elems);

    fill_merged_exp_host_data(raw_x, fp16, input_bin, other_bin, alpha_bin);

    std::cout << "[Exp] generate_data: alpha=bank2 x, other=bank4 const, "
              << "input=bank1 coeff@0..4, bank0/1 scale+r left zero for RTL" << std::endl;
    return true;
}

bool ExpOp::calculate_golden(const app::AppOperatorRequest& req,
                             std::vector<uint8_t>& golden_bin) {
    ExpPolyKernelOp kernel;
    return kernel.calculate_golden(req, golden_bin);
}

void ExpOp::fill_random_request(app::AppOperatorRequest& req,
                                std::mt19937& rng,
                                std::uniform_real_distribution<float>&,
                                const std::string& dtype_str) {
    req.para_operator = "Exp";
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

REGISTER_OP(ExpOp);

}  // namespace op
}  // namespace aiacc
