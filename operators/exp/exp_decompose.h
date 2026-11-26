#ifndef AIACC_SMI_IQ_OP_EXP_DECOMPOSE_H_
#define AIACC_SMI_IQ_OP_EXP_DECOMPOSE_H_

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../../app/app_json_parser.h"
#include "../../app/fp16_utils.h"

namespace aiacc {
namespace op {
namespace exp_math {

constexpr float kLog2E = 1.44269504089f;
constexpr float kSqrt2 = 1.41421356237f;
/**
 * FP32 scale 拼装阈值（bank4 比较常数仍为 ±126/127）。
 * Host/RTL 有效 flush（SELECT_LESS 含等号）：k ≤ -126 → 0，k ≥ 127 → +Inf；
 * 即有限 scale 的合法 k 为 [-125, 126]。
 */
constexpr int kExpKMin = -126;
constexpr int kExpKMax = 127;
constexpr uint32_t kPolyTerms = 5;

// Remez（相对误差）逼近 2^r，r∈[-0.25,0.25]，次数 4（exp1.0.0 之后：半整数归约）。
// 目标：min max |(2^r − P(r)) / 2^r|；数值扫描 max rel ≈ 3e-7。
// （旧 [-0.5,0.5] 系数见 backup_exp1.0.0_DONE_20260717/exp_decompose.h）
constexpr std::array<float, kPolyTerms> kRemezCoeff = {{
    0.99999994f,
    0.69314606f,
    0.24023668f,
    0.05559393f,
    0.00943751f,
}};

struct ExpReduced {
    int k{0};       // m = n>>1，用于 2^m
    int n{0};       // RN(tw)（魔法数；旧 ⌊tw+0.5⌋ 已废弃）
    bool odd{false};
    float r{0.0f};  // ∈[-0.25,0.25]
    float scale{1.0f};
};

/** FP32 RN 魔法数：1.5·2^23；n = bitcast(tw+MAGIC) − 0x4B400000。 */
constexpr float kRoundMagic15 = 12582912.0f;  // 1.5f * 8388608.f
constexpr uint32_t kRoundMagicBiasBits = 0x4B400000u;

/**
 * 半整数归约（现行）：n = RN(tw)（IEEE ties-to-even，魔法数抽整），
 * r=(tw−n)/2，scale=2^(n>>1)·(√2 if n odd)。
 * 旧语义 n=⌊tw+0.5⌋ 已移除（仅存在于 backup_exp1.0.5_*）。
 */
inline ExpReduced decompose_exp_input(float x) {
    const float t = x * kLog2E;
    const float tw = 2.0f * t;
    const float y = tw + kRoundMagic15;
    uint32_t y_bits = 0;
    std::memcpy(&y_bits, &y, sizeof(y_bits));
    const int n = static_cast<int>(static_cast<int32_t>(y_bits - kRoundMagicBiasBits));
    const int m = n >> 1;  // 算术右移
    ExpReduced out;
    out.n = n;
    out.k = m;
    out.odd = ((n & 1) != 0);
    out.r = 0.5f * (tw - static_cast<float>(n));
    out.scale = std::ldexp(1.0f, m) * (out.odd ? kSqrt2 : 1.0f);
    return out;
}

inline float scale_from_int_k(int k) {
    return std::ldexp(1.0f, k);
}

inline float exp2r_horner_f32(float r) {
    float acc = 0.0f;
    for (int idx = 4; idx >= 0; --idx) {
        acc = acc * r + kRemezCoeff[static_cast<size_t>(idx)];
    }
    return acc;
}

inline uint16_t exp2r_horner_f16_bits(float r) {
    const uint16_t r_bits = app::float32_to_float16(r);
    const float r_val = app::float16_to_float32(r_bits);
    uint16_t acc_bits = app::float32_to_float16(0.0f);
    for (int idx = 4; idx >= 0; --idx) {
        const float acc_val = app::float16_to_float32(acc_bits);
        const float mul_val = app::float16_to_float32(
            app::float32_to_float16(acc_val * r_val));
        const float c_val = app::float16_to_float32(
            app::float32_to_float16(kRemezCoeff[static_cast<size_t>(idx)]));
        acc_bits = app::float32_to_float16(mul_val + c_val);
    }
    return acc_bits;
}

inline std::vector<float> collect_or_make_input_data(
    const app::AppOperatorRequest& req, uint32_t total_elems) {
    std::vector<float> data;
    data.reserve(total_elems);
    auto it = req.para_inputs.find("input_tensor");
    if (it != req.para_inputs.end() && !it->second.para_data_f32.empty()) {
        data = it->second.para_data_f32;
        if (data.size() < total_elems) data.resize(total_elems, 0.0f);
        if (data.size() > total_elems) data.resize(total_elems);
        return data;
    }
    for (uint32_t i = 0; i < total_elems; ++i) {
        const int v = static_cast<int>(i % 257u) - 128;
        data.push_back(static_cast<float>(v) / 16.0f);
    }
    return data;
}

template <typename T>
inline void append_scalar(std::vector<uint8_t>& out, T bits, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i)
        out.push_back(static_cast<uint8_t>((bits >> (i * 8)) & 0xFF));
}

struct PreprocessBankLayout {
    uint8_t para_raw_x_bank{2};
    uint8_t para_r_bank{0};
    uint8_t para_coeff_bank{1};
    uint32_t para_raw_x_addr{0};
    uint32_t para_r_addr{0};
    uint32_t para_coeff_addr{0};
    uint32_t para_scale_word_base{5};
};

struct KernelBankLayout {
    uint8_t para_r_bank{0};
    uint8_t para_coeff_bank{1};
    uint8_t para_output_bank{3};
    uint32_t para_r_addr{0};
    uint32_t para_coeff_addr{0};
    uint32_t para_output_addr{0};
};

inline void append_coeff_words(std::vector<uint8_t>& other_bin, bool fp16) {
    for (int k = static_cast<int>(kPolyTerms) - 1; k >= 0; --k) {
        const float c = kRemezCoeff[static_cast<size_t>(k)];
        if (!fp16) {
            uint32_t bits;
            std::memcpy(&bits, &c, sizeof(float));
            for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(other_bin, bits, 4);
        } else {
            uint16_t bits = app::float32_to_float16(c);
            for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(other_bin, bits, 2);
        }
    }
}

inline void fill_kernel_banks_from_raw_x(const std::vector<float>& raw_x,
                                         bool fp16,
                                         std::vector<uint8_t>& r_bin,
                                         std::vector<uint8_t>& coeff_scale_bin) {
    r_bin.clear();
    coeff_scale_bin.clear();
    std::vector<float> scales;
    scales.reserve(raw_x.size());
    for (float x : raw_x) {
        const ExpReduced red = decompose_exp_input(x);
        scales.push_back(red.scale);
        if (!fp16) {
            uint32_t bits;
            std::memcpy(&bits, &red.r, sizeof(float));
            append_scalar(r_bin, bits, 4);
        } else {
            append_scalar(r_bin, app::float32_to_float16(red.r), 2);
        }
    }
    append_coeff_words(coeff_scale_bin, fp16);
    const uint32_t lanes = fp16 ? 8u : 4u;
    for (size_t base = 0; base < scales.size(); base += lanes) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const size_t idx = base + lane;
            const float s = (idx < scales.size()) ? scales[idx] : 0.0f;
            if (!fp16) {
                uint32_t bits;
                std::memcpy(&bits, &s, sizeof(float));
                append_scalar(coeff_scale_bin, bits, 4);
            } else {
                append_scalar(coeff_scale_bin, app::float32_to_float16(s), 2);
            }
        }
    }
}

inline void fill_preprocess_const_bank4(std::vector<uint8_t>& bank4_bin, bool fp16) {
    bank4_bin.clear();
    const auto append_word_zeros = [&](uint32_t words) {
        const uint32_t lanes = fp16 ? 8u : 4u;
        const uint32_t lane_bytes = fp16 ? 2u : 4u;
        for (uint32_t w = 0; w < words; ++w) {
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                for (uint32_t b = 0; b < lane_bytes; ++b) bank4_bin.push_back(0);
            }
        }
    };

    if (!fp16) {
        uint32_t log2e_bits;
        std::memcpy(&log2e_bits, &kLog2E, sizeof(float));
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, log2e_bits, 4);
        const int32_t bias127 = 127;
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, bias127, 4);
        const int32_t shift23 = 23;
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, shift23, 4);
        const float half = 0.5f;
        uint32_t half_bits;
        std::memcpy(&half_bits, &half, sizeof(float));
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, half_bits, 4);
        const float one = 1.0f;
        uint32_t one_bits;
        std::memcpy(&one_bits, &one, sizeof(float));
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, one_bits, 4);
        const int32_t int_one = 1;
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, int_one, 4);
        const int32_t int_zero = 0;
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, int_zero, 4);
        uint32_t sqrt2_bits;
        std::memcpy(&sqrt2_bits, &kSqrt2, sizeof(float));
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, sqrt2_bits, 4);
        // word8..9：留给 Softmax Norm（2.0 / MAGIC）；Exp flush 用 10..12
        append_word_zeros(2);
        const float k_min_f = static_cast<float>(kExpKMin);
        const float k_max_f = static_cast<float>(kExpKMax);
        uint32_t kmin_bits = 0, kmax_bits = 0;
        std::memcpy(&kmin_bits, &k_min_f, sizeof(float));
        std::memcpy(&kmax_bits, &k_max_f, sizeof(float));
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, kmin_bits, 4);
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, kmax_bits, 4);
        const uint32_t pos_inf_bits = 0x7F800000u;  // IEEE754 +Inf
        for (uint32_t lane = 0; lane < 4; ++lane) append_scalar(bank4_bin, pos_inf_bits, 4);
        // word13：RN 魔法数（float 1.5·2²³ 与 int bias 同为 0x4B400000）
        for (uint32_t lane = 0; lane < 4; ++lane)
            append_scalar(bank4_bin, kRoundMagicBiasBits, 4);
        append_word_zeros(242);
    } else {
        // FP16 IEEE754：指数 bias=15、尾数 10 bit；组装 2^k 时用 (k+15)<<10 再按 half 解释。
        // （勿复用 FP32 的 127/23，否则 scale 全错。FP32 分支不变。）
        const uint16_t log2e_bits = app::float32_to_float16(kLog2E);
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, log2e_bits, 2);
        const int16_t bias15 = 15;
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, bias15, 2);
        const int16_t shift10 = 10;
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, shift10, 2);
        const uint16_t half_bits = app::float32_to_float16(0.5f);
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, half_bits, 2);
        const uint16_t one_bits = app::float32_to_float16(1.0f);
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, one_bits, 2);
        const int16_t int_one = 1;
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, int_one, 2);
        const int16_t int_zero = 0;
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, int_zero, 2);
        const uint16_t sqrt2_bits = app::float32_to_float16(kSqrt2);
        for (uint32_t lane = 0; lane < 8; ++lane) append_scalar(bank4_bin, sqrt2_bits, 2);
        append_word_zeros(5);  // word8..12（与 FP32 槽位对齐；Softmax 可覆写 8..9）
        // word13：FP16 RN 魔法数 1.5·2¹⁰（bits 0x6600，兼作 INT 减数）
        const uint16_t round_magic_f16 = 0x6600u;
        for (uint32_t lane = 0; lane < 8; ++lane)
            append_scalar(bank4_bin, round_magic_f16, 2);
        append_word_zeros(242);
    }
}

inline void build_preprocess_golden_bank0(const std::vector<float>& raw_x,
                                            bool fp16,
                                            std::vector<uint8_t>& golden_r_bin) {
    golden_r_bin.clear();
    for (float x : raw_x) {
        const ExpReduced red = decompose_exp_input(x);
        if (!fp16) {
            uint32_t bits;
            std::memcpy(&bits, &red.r, sizeof(float));
            append_scalar(golden_r_bin, bits, 4);
        } else {
            append_scalar(golden_r_bin, app::float32_to_float16(red.r), 2);
        }
    }
}

inline void build_preprocess_golden_bank1_scale(
    const std::vector<float>& raw_x,
    bool fp16,
    uint32_t scale_word_base,
    std::vector<uint8_t>& golden_scale_bin) {
    golden_scale_bin.clear();
    const uint32_t lanes = fp16 ? 8u : 4u;
    const uint32_t lane_bytes = fp16 ? 2u : 4u;
    const uint32_t prefix_words = scale_word_base;
    for (uint32_t w = 0; w < prefix_words; ++w) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            for (uint32_t b = 0; b < lane_bytes; ++b) golden_scale_bin.push_back(0);
        }
    }
    std::vector<float> scales;
    scales.reserve(raw_x.size());
    for (float x : raw_x) {
        const ExpReduced red = decompose_exp_input(x);
        float s = red.scale;
        if (red.k < kExpKMin) s = 0.0f;   // kMin 严格小于（对齐 RTL 实测）
        else if (red.k >= kExpKMax) s = std::numeric_limits<float>::infinity();
        scales.push_back(s);
    }
    for (size_t base = 0; base < scales.size(); base += lanes) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const size_t idx = base + lane;
            const float s = (idx < scales.size()) ? scales[idx] : 0.0f;
            if (!fp16) {
                uint32_t bits;
                std::memcpy(&bits, &s, sizeof(float));
                append_scalar(golden_scale_bin, bits, 4);
            } else {
                append_scalar(golden_scale_bin, app::float32_to_float16(s), 2);
            }
        }
    }
}

inline void fill_raw_x_bin(const std::vector<float>& raw_x,
                           bool fp16,
                           std::vector<uint8_t>& raw_x_bin) {
    raw_x_bin.clear();
    for (float x : raw_x) {
        if (!fp16) {
            uint32_t bits;
            std::memcpy(&bits, &x, sizeof(float));
            append_scalar(raw_x_bin, bits, 4);
        } else {
            append_scalar(raw_x_bin, app::float32_to_float16(x), 2);
        }
    }
}

inline void fill_merged_exp_host_data(const std::vector<float>& raw_x,
                                      bool fp16,
                                      std::vector<uint8_t>& coeff_bank1_bin,
                                      std::vector<uint8_t>& const_bank4_bin,
                                      std::vector<uint8_t>& raw_x_bank2_bin) {
    fill_raw_x_bin(raw_x, fp16, raw_x_bank2_bin);
    fill_preprocess_const_bank4(const_bank4_bin, fp16);
    coeff_bank1_bin.clear();
    append_coeff_words(coeff_bank1_bin, fp16);
}

// ---------------------------------------------------------------------------
// Host 黄金（Exp 算子数值 SSOT：Softmax / pack / 扫掠脚本均委托此处）
// 与 ExpOp::calculate_golden / Exp RTL 对齐；改 Remez / 分解只改本文件。
// ---------------------------------------------------------------------------

/** FP32 merged Exp Host 输出（= scale · Horner(r)；k 越界 flush 对齐 RTL）。 */
inline float host_exp_f32(float x) {
    if (std::isnan(x)) return x;
    const ExpReduced red = decompose_exp_input(x);
    // 本仓 RTL（magic-number P1）实测：kMin 侧严格小于才 flush，k=-126 走有限
    // （scale=2^-126 正常数，FPMUL 可出 denormal）；kMax 侧仍含等号。
    // 2026-08-26 LSU wc=64 验收：x≈-87.4(k=-126) RTL 出 1.105e-38(denormal)。
    if (red.k < kExpKMin) return 0.0f;
    if (red.k >= kExpKMax) return std::numeric_limits<float>::infinity();
    return red.scale * exp2r_horner_f32(red.r);
}

/** FP16 I/O：x 先 RN→half，再分解；输出 half 格 y（与 ExpPolyKernel golden 一致）。 */
inline uint16_t host_exp_f16_bits(float x) {
    const float x_eff =
        app::float16_to_float32(app::float32_to_float16(x));
    const ExpReduced red = decompose_exp_input(x_eff);
    const uint16_t p_bits = exp2r_horner_f16_bits(red.r);
    const float p = app::float16_to_float32(p_bits);
    const float s = app::float16_to_float32(app::float32_to_float16(red.scale));
    return app::float32_to_float16(p * s);
}

inline float host_exp_f16(float x) {
    return app::float16_to_float32(host_exp_f16_bits(x));
}

}  // namespace exp_math
}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_EXP_DECOMPOSE_H_
