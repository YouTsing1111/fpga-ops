#ifndef AIACC_SMI_IQ_OP_EXP_IQ_UTILS_H_
#define AIACC_SMI_IQ_OP_EXP_IQ_UTILS_H_

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace aiacc {
namespace op {
namespace exp_iq {

constexpr uint8_t kRtovrInDrFpMulD = 3;
constexpr uint8_t kRtovrInDrFpAddD = 4;
constexpr uint8_t kRtovrInDrFpToIntD = 8;
constexpr uint8_t kRtovrInDrIntToFpD = 9;
constexpr uint8_t kRtovrInDrLogicD = 10;
constexpr uint8_t kRtovrInDrIntAddD0 = 0;
constexpr uint8_t kRtovrInDrFpAddSt = 5;
constexpr uint8_t kRtovrInUrRdata0 = 30;
// RtoVR.v：input 34..NUM_INPUTS-1 硬绑 0；Horner stage0 用它清累加器初值
constexpr uint8_t kRtovrInTiedZero = 34;

constexpr uint8_t kRtovrOutFpMulS0 = 3;
constexpr uint8_t kRtovrOutFpMulS1 = 4;
constexpr uint8_t kRtovrOutFpAddS0 = 5;
constexpr uint8_t kRtovrOutFpAddS1 = 6;
constexpr uint8_t kRtovrOutFpToIntS = 9;
constexpr uint8_t kRtovrOutIntToFpS = 10;
constexpr uint8_t kRtovrOutLogicS0 = 11;
constexpr uint8_t kRtovrOutLogicS1 = 12;
constexpr uint8_t kRtovrOutIntAddS0 = 0;
constexpr uint8_t kRtovrOutIntAddS1 = 1;
constexpr uint8_t kRtovrOutUrWdata0 = 35;

// --- VIN/VOUT 直通（FPGA USE_RTOVR_SPARSE_MASK 下唯一合法 UR↔计算通路；全交叉仿真同码兼容）---
// 数值 = RtoVR.v IN_IDX/OUT_IDX（与 sub_op_constants.h / mul_op_constants.h 同值）
constexpr uint8_t kRtovrOutDvrVin0S = 14;   // 输出口：VIN0 暂存输入侧（稀疏掩码只收 UR_RDATA 30..33）
constexpr uint8_t kRtovrOutDvrVin1S = 15;   // 输出口：VIN1 暂存输入侧
constexpr uint8_t kRtovrOutDvrVout0S = 16;  // 输出口：VOUT0 暂存输入侧（稀疏掩码只收计算 DR 0..10）
constexpr uint8_t kRtovrInDrVin0D = 11;     // 路由源：VIN0 暂存输出
constexpr uint8_t kRtovrInDrVin1D = 12;     // 路由源：VIN1 暂存输出（LOGIC_ST 口 13 只收它）
constexpr uint8_t kRtovrInDrVout0D = 13;    // 路由源：VOUT0 暂存输出（UVR_WDATA 口 35..38 只收 13/28/29）
// IQ 68/69/70 使能 CRU code：{smc_id_en[5:1], vld[0]}；31 阈值语义覆盖 SMC0..31 全常开
// （IPR 常开使能时 dr_vin_d 每拍自 dvr_vin_s 重载；RtoVR 输出寄存器 vld=0 冻结保持，
//   故单发捕获脉冲即可让 VIN_D 稳定保持到下一次捕获）
constexpr uint32_t kVinVoutEnableCode = 0x3Fu;

// 捕获路由目标端口：lane0→VIN0_S(14)，lane1→VIN1_S(15)
inline uint8_t vin_capture_port(uint8_t lane) {
    return lane ? kRtovrOutDvrVin1S : kRtovrOutDvrVin0S;
}
// VIN_D 作为消费路由的源：lane0→VIN0_D(11)，lane1→VIN1_D(12)
inline uint8_t vin_consume_src(uint8_t lane) {
    return lane ? kRtovrInDrVin1D : kRtovrInDrVin0D;
}

constexpr uint32_t kStageCycles = 10;   // was 11：FPMUL→FPADD gap 6→5，砍 stage 尾 1 拍
constexpr uint32_t kNumStages = 5;
constexpr uint32_t kFinalCycles = 9;    // was 11：写回@8，再砍尾空
// vv（VIN/VOUT 直通）：实测 捕获@C → VIN_D/VOUT0_D 从 C+3 起有效（非 +1）。
// Final 尾链 y：FPMUL@53 → DR@58 → 捕获@57 → VOUT0_D@60 → 写@60，body 须 61。
constexpr uint32_t kFinalCyclesVv = 11;

constexpr uint32_t kReadPhase = 0;
constexpr uint32_t kFpmulPhase = 3;
constexpr uint32_t kFpaddPhase = 8;     // was 9：FPMUL@3 → FPADD@8（gap=5）
constexpr uint32_t kFinalReadPhase = 0;
constexpr uint32_t kFinalFpmulPhase = 3;
constexpr uint32_t kFinalWritePhase = 8;  // was 9：FPMUL@3→DR@6→route→写@8
// ACC 写回放在 Final 前两拍（同体循环里 Horner 之后）：
// FPADD@Horner8 → DR@11 → Final0 拉 RtoVR-35 → Final1 UR-W（+1 settle）
// 下一轮 Horner cyc0 再读 ACC，中间隔整段 Final，无 RAW settle 冒险。
// 禁止在 Horner 末拍写 ACC：与下 stage cyc0 读冲突。
constexpr uint32_t kAccRouteInFinalPhase = 0;
constexpr uint32_t kAccWriteInFinalPhase = 1;

// --- SMC 硬件延迟（Phase-1/2 调度依据，详见 op/exp/README.md §4.2 / FLOW_ANALYSIS.md）---
// RTL 寄存器级：FPMUL v_pipe[3]、FPADD valid_pipe[2] → 均为 issue 后 3 拍 dr 有效。
// TVGen 保守值（div/add/mul）：FPMUL_LATENCY=FPADD_LATENCY=4。
// Exp 实测调度：UR 读 issue → 计算 issue = +3；浮点 DR 消费者 issue 间隔试 gap=5。
constexpr uint32_t kUrReadToComputeIssue = 3;
constexpr uint32_t kFpDrConsumerGap = 5;   // FPMUL@3 → FPADD@8（Phase-2 Horner）
constexpr uint32_t kRtovrRouteLatency = 1;
constexpr uint32_t kFpmulDrLatencyRtl = 3;
constexpr uint32_t kFpaddDrLatencyRtl = 3;
constexpr uint32_t kFpmulLatencyTvgen = 4;
constexpr uint32_t kFpaddLatencyTvgen = 4;
constexpr uint32_t kFptointLatency = 1;    // 组合逻辑 + fpto_int_array 输出寄存器
constexpr uint32_t kInttofpLatency = 1;
constexpr uint32_t kIntaddLatency = 1;
constexpr uint32_t kLogicLatency = 2;      // bitwise_and LOGICAL_LATENCY

// 相邻 SMC 间 CRU 经 shiftup/shiftdn 级联，每跨 1 个 SMC 多 1 拍（SMCn 比 SMC0 晚 n 拍收到同槽 IQ）
constexpr uint32_t kSmcCruCascadeLatencyPerHop = 1;
// 半整数归约 + RN 魔法数：
// - Merged：P1 core=56（含√2，保持已验证相位）；P2=59（gap5）→ 墙钟 ≈115
// - Preprocess 单测：core 后串行挂同一 overlay
constexpr uint32_t kPreprocessCoreCycles = 56;   // WriteScaleScratch@53 + settle
constexpr uint32_t kPreprocessFullCycles = 103;  // core56 + overlay(WriteScale@44→+3)
constexpr uint32_t kPreprocessBaseCycles = kPreprocessCoreCycles;
constexpr uint32_t kKernelBodyCycles =
    kStageCycles * kNumStages + kFinalCycles;  // 10*5+9 = 59

// P2∥k-flush（相对 Horner；stage=10 / FPADD@8 重定时）
// S1.FPMUL@13 → uflow check@14；S3.FPMUL@33 → oflow check@34；check→SELECT=+6
// Horner FPADD update_status=0 → SELECT 可跨过中间 Horner FPADD
constexpr uint32_t kFlushReadKf = 7;
constexpr uint32_t kFlushInttofpK = 10;           // 近 S1 前；∥ 空隙准备
constexpr uint32_t kFlushReadKMin = 11;
constexpr uint32_t kFlushUflowCheck = 14;         // 后于 S1.FPMUL@13，不冲累加器 DR
constexpr uint32_t kFlushReadZero = 16;           // R0
constexpr uint32_t kFlushReadScale = 17;          // R1
constexpr uint32_t kFlushSelectUflow = 20;        // check+6（跨 S1.FPADD@18）
constexpr uint32_t kFlushReadKf2 = 23;            // ∥ S2.FPMUL@23
constexpr uint32_t kFlushInttofpK2 = 26;
constexpr uint32_t kFlushReadKMax = 27;           // 保持到 oflow check（勿中途改写 R0）
constexpr uint32_t kFlushOflowCheck = 34;         // 后于 S3.FPMUL@33
constexpr uint32_t kFlushReadPosInf = 36;
constexpr uint32_t kFlushSelectOflow = 40;        // check+6（跨 S3.FPADD@38）
constexpr uint32_t kFlushWriteScale = 44;         // Final 读 scale@50 前；避开 S4 coeff@40


inline uint32_t smc_cru_delay_comp(uint8_t smc_id_en) {
    // RTL 实验证实（2026-07-14，README §12.2.6）：多 SMC 的级联偏移由硬件 shiftup/shiftdn 自动
    // 完成，微码应与单 SMC 完全一致。此前按 smc_id_en 给 P1 body 补 n 拍 + 拉伸 RtoVR 路由窗口，
    // 反而让 P1 的 r→bank0 写全程丢失、Horner 塌缩成 c0（8-SMC 单字假过、多字 steps>=2 全错）。
    // 去掉补偿后 8-SMC 单字/多字（W=4/8/16）逐 lane 与 golden 一致。故补偿恒为 0。
    // 需临时复现旧(错误)行为做对比时设 EXP_ENABLE_CRU_COMP=1。
    if (std::getenv("EXP_ENABLE_CRU_COMP"))
        return (smc_id_en > 0u) ? static_cast<uint32_t>(smc_id_en) : 0u;
    return 0u;
}

inline uint32_t preprocess_body_cycles(uint8_t smc_id_en) {
    return kPreprocessBaseCycles + smc_cru_delay_comp(smc_id_en);
}

inline uint32_t ur_read_code(uint8_t bank, uint32_t addr, uint8_t smc_id_en) {
    return 1u | ((uint32_t)(bank & 0x7u) << 1) | ((addr & 0xFFu) << 4) |
           ((uint32_t)(smc_id_en & 0x1Fu) << 12);
}

inline uint64_t ur_write_code(uint8_t bank, uint32_t addr, uint8_t smc_id_en,
                             uint16_t wbe = 0xFFFF) {
    return 1u | ((uint64_t)(bank & 0x7u) << 1) | ((uint64_t)(addr & 0xFFu) << 4) |
           ((uint64_t)(wbe & 0xFFFFu) << 12) | ((uint64_t)(smc_id_en & 0x1Fu) << 28);
}

inline uint32_t rtovr_code(uint8_t source_id) {
    return 1u | ((uint32_t)(source_id & 0x3Fu) << 1);
}

// UR code：bit0=valid，[11:4]=addr。按字复制发射时给基址加 word 偏移。
inline uint64_t ur_code_add_addr(uint64_t code, uint32_t delta) {
    if ((code & 1u) == 0u || delta == 0u) return code;
    const uint32_t addr = static_cast<uint32_t>((code >> 4) & 0xFFu);
    const uint32_t new_addr = (addr + delta) & 0xFFu;
    return (code & ~static_cast<uint64_t>(0xFFu << 4)) |
           (static_cast<uint64_t>(new_addr) << 4);
}

inline uint8_t ur_source_for_port(uint8_t port) {
    return static_cast<uint8_t>(kRtovrInUrRdata0 + (port & 0x3u));
}

// EMBED 字段语义（硬件）：
//   embed      : 入口写入「层数」本身（1=lev1, 2=lev2, …）
//   embed_end  : 位掩码，从低位到高位；某位写 1 表示「该层在此拍作为循环结束边界」
//                bit0→lev1, bit1→lev2, …  可多位置 1 同时结束多层
//                不是「填 N 表示结束第 N 层」的层号字段（虽然单层结束时
//                结束 lev1=0b001、结束 lev2=0b010，数值碰巧等于 1/2）
//   inc_embed  : 同为位掩码，标记地址计算参与哪几层的 loop 计数
inline uint64_t serialize_header_41(uint8_t vld, uint8_t jump, uint8_t dly,
                                    uint8_t embed = 0, uint16_t loops = 0,
                                    uint8_t cond = 0, uint8_t inc = 0,
                                    uint8_t embed_end_bits = 0,
                                    uint8_t inc_embed_bits = 0) {
    uint64_t bits = 0;
    uint32_t pos = 0;
    const uint8_t embed_end = (embed == 0) ? (embed_end_bits & 0x3Fu) : 0u;
    const uint8_t inc_embed = (embed == 0) ? (inc_embed_bits & 0x3Fu) : 0u;

    bits |= (uint64_t)(inc_embed & 0x3F) << pos; pos += 6;
    bits |= (uint64_t)(inc & 0xF)        << pos; pos += 4;
    bits |= (uint64_t)(dly & 0x7)        << pos; pos += 3;
    bits |= (uint64_t)(cond & 0x7F)      << pos; pos += 7;
    bits |= (uint64_t)(loops & 0x3FF)    << pos; pos += 10;
    bits |= (uint64_t)(embed_end & 0x3F) << pos; pos += 6;
    bits |= (uint64_t)(embed & 0x7)      << pos; pos += 3;
    bits |= (uint64_t)(jump & 0x1)       << pos; pos += 1;
    bits |= (uint64_t)(vld & 0x1)        << pos; pos += 1;
    return bits;
}

inline std::vector<uint8_t> pack_instruction(uint8_t vld, uint8_t jump,
                                             uint8_t dly, uint64_t code,
                                             uint32_t code_bits,
                                             uint8_t embed = 0,
                                             uint16_t loops = 0,
                                             uint8_t cond = 0,
                                             uint8_t inc = 0,
                                             uint8_t embed_end_bits = 0,
                                             uint8_t inc_embed_bits = 0) {
    const uint64_t header = serialize_header_41(vld, jump, dly, embed, loops,
                                                cond, inc, embed_end_bits,
                                                inc_embed_bits);
    const uint32_t total_bits = 41 + code_bits;
    const uint32_t num_bytes = (total_bits + 7) / 8;
    std::vector<uint8_t> bytes;
    if (total_bits <= 64) {
        const uint64_t mask = (code_bits >= 64) ? ~0ULL : ((1ULL << code_bits) - 1ULL);
        const uint64_t total = (header << code_bits) | ((uint64_t)code & mask);
        for (int i = static_cast<int>(num_bytes) - 1; i >= 0; --i)
            bytes.push_back(static_cast<uint8_t>((total >> (i * 8)) & 0xFF));
    } else {
        const unsigned __int128 mask =
            (static_cast<unsigned __int128>(1) << code_bits) - 1;
        const unsigned __int128 total =
            (static_cast<unsigned __int128>(header) << code_bits) |
            (static_cast<unsigned __int128>(code) & mask);
        for (int i = static_cast<int>(num_bytes) - 1; i >= 0; --i)
            bytes.push_back(static_cast<uint8_t>((total >> (i * 8)) & 0xFF));
    }
    return bytes;
}

struct SchedEntry {
    uint32_t abs_cycle{0};
    uint64_t code{0};
};

/**
 * EXP_COMPRESS_NOP_LOOPS=0 关闭 body 内 loops 折叠。默认开启。
 * 开启时：连续相同 code（含 NOP=0，以及 RtoVR 同源长窗口）→ 1 条 + loops=(N-1)。
 * 硬件：embed=0 && loops!=0 时不推进 IQ 读地址，同码再发 loops+1 拍。
 */
inline bool compress_nop_loops_enabled() {
    const char* env = std::getenv("EXP_COMPRESS_NOP_LOOPS");
    if (!env) return true;
    return !(env[0] == '0' && env[1] == '\0');
}

/** 连续相同 code 的 run 长度；若给定 inc_split，则不得跨越该边界。 */
inline uint32_t same_code_run_len(const std::vector<uint64_t>& body, uint32_t p,
                                  uint32_t end, uint32_t inc_split = UINT32_MAX) {
    uint32_t run = 1u;
    while (p + run < end && body[p + run] == body[p]) {
        // outer_addr_inc 通道：inc_from_cycle 前后 inc_embed 语义不同，禁止跨边界合并
        if (inc_split != UINT32_MAX && p < inc_split && (p + run) >= inc_split)
            break;
        ++run;
    }
    return run;
}

// Phase-2 Horner IQ 发射：一律 kKernelBodyCycles 拍 unroll（DR+tied_zero）进 outer embed。
// embed_end / inc_embed：位掩码 bit0=lev1, bit1=lev2
inline std::vector<uint8_t> emit_horner_kernel_embed_iq(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    bool horner_addr_inc,
    bool outer_addr_inc,
    uint32_t total_steps,
    const char* tag,
    uint32_t flat_cycles_override = 0u) {  // vv 直通：P2 body 59→61，其余调用方恒 0
    const uint32_t flat_cycles =
        flat_cycles_override ? flat_cycles_override
                             : kStageCycles * kNumStages + kFinalCycles;  // == kKernelBodyCycles
    const uint32_t outer_loops = (total_steps > 1u) ? (total_steps - 1u) : 0u;
    std::vector<uint64_t> flat(flat_cycles, 0u);

    for (const auto& e : sched) {
        if (e.abs_cycle < flat_cycles) {
            if (flat[e.abs_cycle] != 0u && flat[e.abs_cycle] != e.code) {
                std::cerr << "[" << tag << "][ERR] conflicting body at cycle "
                          << e.abs_cycle << std::endl;
            }
            flat[e.abs_cycle] = e.code;
        }
    }

    std::vector<uint8_t> data;
    const uint8_t outer_embed_lvl = 2u;
    const uint8_t kEmbedEndLev1 = 1u << 0;
    const uint8_t kEmbedEndLev2 = 1u << 1;

    auto append_inst = [&](uint8_t vld, uint8_t jump, uint64_t code, uint8_t embed,
                           uint16_t loops, uint8_t inc, uint8_t embed_end,
                           uint8_t inc_embed) {
        auto inst = pack_instruction(vld, jump, 0, code, code_bits, embed, loops,
                                     0, inc, embed_end, inc_embed);
        data.insert(data.end(), inst.begin(), inst.end());
    };

    auto append_body_slot = [&](uint64_t code, bool last, uint8_t embed_end,
                                bool use_horner_lev, bool use_outer_lev) {
        uint8_t inc_embed = 0u;
        const bool active = ((code & 1u) != 0u) || last;
        if (use_horner_lev && horner_addr_inc && active) inc_embed |= kEmbedEndLev1;
        if (use_outer_lev && outer_addr_inc && active) inc_embed |= kEmbedEndLev2;
        append_inst(1, 0, code, 0u, 0u, 0u, embed_end, inc_embed);
    };

    // 在一段固定模板内压缩连续 NOP（Horner 13 / Final 13）
    // 注意：嵌套 EMBED 体内对非零 code 用 loops 可能与 EXE_EMBED0 捕获冲突，故此处只压 NOP。
    auto append_template_compressed = [&](const std::vector<uint64_t>& tmpl,
                                          bool use_horner_lev, bool use_outer_lev,
                                          uint8_t tail_embed_end) {
        const uint32_t n = static_cast<uint32_t>(tmpl.size());
        const bool compress = compress_nop_loops_enabled();
        for (uint32_t p = 0; p < n; ) {
            if (compress && tmpl[p] == 0u) {
                uint32_t run = 1u;
                while (p + run < n && tmpl[p + run] == 0u) ++run;
                const bool covers_tail = (p + run == n);
                const uint8_t embed_end = covers_tail ? tail_embed_end : 0u;
                const uint16_t loops =
                    (run > 1u) ? static_cast<uint16_t>(run - 1u) : 0u;
                // 与旧 last 槽一致：模板末拍即使 NOP 也参与 inc_embed（字/段地址推进）
                uint8_t inc_embed = 0u;
                if (covers_tail) {
                    if (use_horner_lev && horner_addr_inc) inc_embed |= kEmbedEndLev1;
                    if (use_outer_lev && outer_addr_inc) inc_embed |= kEmbedEndLev2;
                }
                append_inst(1, 0, 0u, 0u, loops, 0u, embed_end, inc_embed);
                p += run;
                continue;
            }
            const bool last = (p + 1u == n);
            append_body_slot(tmpl[p], last, last ? tail_embed_end : 0u,
                             use_horner_lev, use_outer_lev);
            ++p;
        }
    };

    // ---------- unroll only（单字无 outer；多字 lev2×steps）----------
    if (outer_loops > 0u) {
        append_inst(1, 0, 0, outer_embed_lvl, static_cast<uint16_t>(outer_loops),
                    outer_addr_inc ? 1u : 0u, 0u, 0u);
        append_template_compressed(flat, false, true, kEmbedEndLev2);
        append_inst(1, 1, 0, 0u, 0u, 0u, 0u, 0u);
        std::cout << "[" << tag << "] Horner EMBED: unroll "
                  << "(HW lev2×" << total_steps << "), iq_inst≈"
                  << (data.size() / ((41u + code_bits + 7u) / 8u)) << std::endl;
        return data;
    }

    append_template_compressed(flat, false, false, kEmbedEndLev1);
    append_inst(1, 1, 0, 0u, 0u, 0u, 0u, 0u);
    std::cout << "[" << tag << "] Horner EMBED: unroll steps=1, iq_inst≈"
              << (data.size() / ((41u + code_bits + 7u) / 8u)) << std::endl;
    return data;
}

inline bool is_flush_fixed_ur_read_rel(uint32_t rel_cyc) {
    return rel_cyc == kFlushReadKf || rel_cyc == kFlushReadKMin ||
           rel_cyc == kFlushReadZero || rel_cyc == kFlushReadScale ||
           rel_cyc == kFlushReadKf2 || rel_cyc == kFlushReadKMax ||
           rel_cyc == kFlushReadPosInf;
}

/**
 * Body loops 折叠：连续相同 code → 1 条 + loops=(N-1)（NOP 与 RtoVR 同源窗口同理）。
 * 硬件：embed=0 && loops!=0 时停留当前 IQ，同码再发 loops+1 拍。
 * 跨 IQ 各自压缩不影响墙钟对齐（各通道发射拍数仍 = body_cycles）。
 *
 * no_outer_inc_abs：这些绝对拍的有效 UR 读/写强制不带 outer addr_inc
 * （∥Horner 的 k-flush 固定地址常数/scratch，避免多字把 @201 读成 @202）。
 */
// flat_tail_no_embed_end=1：无外层循环（total_steps=1）的 flat body 末尾 run
// 不打 embed_end=1。该标记会触发 IIC 的外层循环检查（smi_iq_iic.v），而
// lev1..6 loop 寄存器跨 segment 残留、embed=0 指令又会清零对应 cnt —— 若本通道
// 此前跑过 embed>=1 的程序（如 exp/maxsub 的 embed=2），检查必失败，整个 flat
// body 会被 EXE_ARRAY_INST 重放（ur_w0 双写 bank7 事故根因）。flat 程序的终止
// 由末尾 jump 指令承担，embed_end 仅为外层循环服务，flat 路径安全去除。
inline std::vector<uint8_t> emit_preprocess_embed_iq(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    bool outer_addr_inc,
    uint32_t total_steps,
    uint32_t body_cycles,
    uint32_t inc_from_cycle = 0u,
    const std::vector<uint32_t>* no_outer_inc_abs = nullptr,
    bool flat_tail_no_embed_end = false) {
    // inc_from_cycle：只有 abs_cycle >= inc_from_cycle 的读/写才随外层 EMBED 递增地址。
    // 用于 merged UR-R1：P1 常量读（cyc<p1_body）固定地址、P2 的 r 读（cyc>=p1_body）逐 word 递增。
    const uint32_t outer_loops = (total_steps > 1u) ? (total_steps - 1u) : 0u;
    std::vector<uint64_t> body(body_cycles, 0u);

    for (const auto& e : sched) {
        if (e.abs_cycle < body_cycles) body[e.abs_cycle] = e.code;
    }

    auto skip_outer_inc = [&](uint32_t p) -> bool {
        if (!no_outer_inc_abs) return false;
        for (uint32_t c : *no_outer_inc_abs) {
            if (c == p) return true;
        }
        return false;
    };

    std::vector<uint8_t> data;
    const uint8_t outer_embed_lvl = 2u;          // embed 入口层号
    const uint8_t outer_embed_end = 1u << 1;     // embed_end 位掩码 bit1→lev2
    const uint8_t outer_inc_embed = outer_addr_inc ? outer_embed_end : 0u;
    const bool compress = compress_nop_loops_enabled();
    // flush 固定读若落在 inc 区，不能与相邻同码合并（inc 属性不同）
    const uint32_t inc_split =
        (outer_addr_inc && outer_loops > 0u) ? inc_from_cycle : UINT32_MAX;
    uint32_t iq_body_insts = 0u;
    uint32_t same_runs_compressed = 0u;
    uint32_t same_slots_saved = 0u;

    if (outer_loops > 0u) {
        auto outer_init = pack_instruction(1, 0, 0, 0, code_bits,
                                           outer_embed_lvl, outer_loops,
                                           0, outer_addr_inc ? 1u : 0u, 0u, 0u);
        data.insert(data.end(), outer_init.begin(), outer_init.end());
    }

    auto append_one = [&](uint32_t p, uint64_t code, uint16_t loops, uint8_t embed_end) {
        uint8_t inc_embed = 0u;
        if (outer_loops > 0u && outer_addr_inc && (p >= inc_from_cycle) &&
            !skip_outer_inc(p) &&
            (((code & 1u) != 0u) || embed_end == outer_embed_end)) {
            inc_embed = outer_inc_embed;
        }
        auto inst = pack_instruction(1, 0, 0, code, code_bits,
                                     0u, loops, 0, 0, embed_end, inc_embed);
        data.insert(data.end(), inst.begin(), inst.end());
        ++iq_body_insts;
    };

    for (uint32_t p = 0; p < body_cycles; ) {
        if (compress) {
            // 勿跨越「需要/不需要 outer inc」边界合并
            uint32_t run = same_code_run_len(body, p, body_cycles, inc_split);
            if (no_outer_inc_abs) {
                const bool p_skip = skip_outer_inc(p);
                for (uint32_t k = 1; k < run; ++k) {
                    if (skip_outer_inc(p + k) != p_skip) {
                        run = k;
                        break;
                    }
                }
            }
            const bool covers_tail = (p + run == body_cycles);
            const uint8_t embed_end =
                covers_tail ? (outer_loops > 0u ? outer_embed_end
                                                : (flat_tail_no_embed_end ? 0u : 1u))
                            : 0u;
            const uint16_t loops =
                (run > 1u) ? static_cast<uint16_t>(run - 1u) : 0u;
            append_one(p, body[p], loops, embed_end);
            if (run > 1u) {
                ++same_runs_compressed;
                same_slots_saved += (run - 1u);
            }
            p += run;
            continue;
        }

        const uint8_t embed_end_if_last =
            (p == body_cycles - 1u) ? (outer_loops > 0u ? outer_embed_end
                                                        : (flat_tail_no_embed_end ? 0u : 1u))
                                    : 0u;
        append_one(p, body[p], 0u, embed_end_if_last);
        ++p;
    }

    auto end_inst = pack_instruction(1, 1, 0, 0, code_bits);
    data.insert(data.end(), end_inst.begin(), end_inst.end());

    if (compress && (same_runs_compressed > 0u || body_cycles > 0u)) {
        const uint32_t nbytes = (41u + code_bits + 7u) / 8u;
        const uint32_t iq_total =
            nbytes ? static_cast<uint32_t>(data.size() / nbytes) : 0u;
        std::cout << "[ExpIQ] same-code+loops compress: body_cycles=" << body_cycles
                  << " body_iq=" << iq_body_insts
                  << " saved_slots=" << same_slots_saved
                  << " runs=" << same_runs_compressed
                  << " iq_total≈" << iq_total << std::endl;
    }
    return data;
}

inline uint32_t merged_exp_body_cycles(uint8_t smc_id_en) {
    return preprocess_body_cycles(smc_id_en) + kKernelBodyCycles;
}

// 前向声明：flat fallback 用
inline std::vector<uint8_t> emit_exp_embed_iq(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    bool outer_addr_inc,
    uint32_t total_steps,
    uint32_t body_cycles,
    const char* tag = "Exp",
    uint32_t inc_from_cycle = 0u,
    const std::vector<uint32_t>* no_outer_inc_abs = nullptr,
    bool flat_tail_no_embed_end = false);

/** 将 merged 绝对时间轴拆回 P1 / P2（P2 坐标回到 0..）。 */
inline void split_merged_sched(const std::vector<SchedEntry>& merged,
                               uint32_t p1_body,
                               std::vector<SchedEntry>& p1,
                               std::vector<SchedEntry>& p2) {
    p1.clear();
    p2.clear();
    for (const auto& e : merged) {
        if (e.abs_cycle < p1_body) {
            p1.push_back(e);
        } else {
            p2.push_back({e.abs_cycle - p1_body, e.code});
        }
    }
}

inline bool merged_fold_p2_enabled() {
    // 历史开关；Merged 现已统一 emit_exp_embed，此函数恒 true（保留供旧调用点）。
    return true;
}

/**
 * Merged 拼法：整段 P1|P2 → emit_exp_embed_iq（单字/多字同一路径，P2=unroll）。
 * UR-R0/R1：∥flush 固定地址读禁止 outer addr_inc（见 no_inc 列表）。
 */
inline std::vector<uint8_t> emit_merged_p1_then_fold_p2_iq(
    const std::vector<SchedEntry>& sched_merged,
    uint32_t code_bits,
    uint32_t p1_body,
    uint32_t total_steps,
    bool p1_outer_addr_inc,
    bool /*p2_horner_addr_inc*/,
    bool p2_outer_addr_inc,
    const char* tag,
    bool suppress_flush_fixed_ur_inc = false,
    uint32_t p2_body_override = 0u) {  // vv 直通：P2 body 59→61，其余调用方恒 0
    const uint32_t p2_body =
        p2_body_override ? p2_body_override : kKernelBodyCycles;
    std::vector<uint32_t> no_inc;
    if (suppress_flush_fixed_ur_inc) {
        const uint32_t rels[] = {
            kFlushReadKf, kFlushReadKMin, kFlushReadZero, kFlushReadScale,
            kFlushReadKf2, kFlushReadKMax, kFlushReadPosInf};
        for (uint32_t r : rels) no_inc.push_back(p1_body + r);
    }
    auto data = emit_exp_embed_iq(
        sched_merged, code_bits, p1_outer_addr_inc || p2_outer_addr_inc,
        total_steps, p1_body + p2_body, tag,
        p1_outer_addr_inc ? 0u : p1_body,
        suppress_flush_fixed_ur_inc ? &no_inc : nullptr);
    const uint32_t nbytes = (41u + code_bits + 7u) / 8u;
    const uint32_t iq_total =
        nbytes ? static_cast<uint32_t>(data.size() / nbytes) : 0u;
    std::cout << "[" << tag << "] merged P1|P2: p1_body=" << p1_body
              << " steps=" << total_steps
              << " p2=unroll" << p2_body << "(emit_exp_embed)"
              << " iq_inst≈" << iq_total << std::endl;
    return data;
}

inline void append_sched_offset(std::vector<SchedEntry>& dst,
                                const std::vector<SchedEntry>& src,
                                uint32_t cycle_offset) {
    for (const auto& e : src) {
        dst.push_back({e.abs_cycle + cycle_offset, e.code});
    }
}

inline void merge_sched_pair(std::vector<SchedEntry>& dst,
                             const std::vector<SchedEntry>& phase_a,
                             const std::vector<SchedEntry>& phase_b,
                             uint32_t phase_b_offset) {
    dst = phase_a;
    append_sched_offset(dst, phase_b, phase_b_offset);
}

inline void remap_rtovr_ur_port_in_range(std::vector<SchedEntry>& sched,
                                         uint32_t cycle_min,
                                         uint32_t cycle_max,
                                         uint8_t from_port,
                                         uint8_t to_port) {
    const uint64_t from_code = rtovr_code(ur_source_for_port(from_port));
    const uint64_t to_code = rtovr_code(ur_source_for_port(to_port));
    for (auto& e : sched) {
        if (e.abs_cycle >= cycle_min && e.abs_cycle < cycle_max && e.code == from_code) {
            e.code = to_code;
        }
    }
}

inline std::vector<uint8_t> emit_exp_embed_iq(
    const std::vector<SchedEntry>& sched,
    uint32_t code_bits,
    bool outer_addr_inc,
    uint32_t total_steps,
    uint32_t body_cycles,
    const char* /*tag*/,
    uint32_t inc_from_cycle,
    const std::vector<uint32_t>* no_outer_inc_abs,
    bool flat_tail_no_embed_end) {
    return emit_preprocess_embed_iq(sched, code_bits, outer_addr_inc, total_steps,
                                    body_cycles, inc_from_cycle, no_outer_inc_abs,
                                    flat_tail_no_embed_end);
}

}  // namespace exp_iq
}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_EXP_IQ_UTILS_H_
