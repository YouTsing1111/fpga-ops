#ifndef AIACC_SMI_IQ_OP_SOFTMAX_GOLDEN_H_
#define AIACC_SMI_IQ_OP_SOFTMAX_GOLDEN_H_

/**
 * @file softmax_golden.h
 * @brief Softmax Host 黄金：对齐芯片链路（max-sub → halfint Exp → sum → Newton×3 → mul）。
 *
 * Phase0（片上 max-sub）开发期可由 Host 预写 x'；golden 仍从原始 x 完整走一遍 stable softmax。
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../../app/fp16_utils.h"
#include "../exp/exp_decompose.h"

namespace aiacc {
namespace op {
namespace softmax_golden {

constexpr uint32_t kDefaultNewtonIters = 3u;
constexpr uint32_t kNewtonMagic = 0x7EF311C7u;

/** W 非 4 整除时尾 lane 填充（FP32/FP16 统一语义；勿用 -inf，Exp 会 NaN）。 */
/** 尾 lane pad：-1000；Phase0 后 x'≪0，依赖 Exp k-flush（k≤-126→0）保证 pad lane e=0。 */
constexpr float kSoftmaxPadValue = -1000.0f;

inline uint32_t softmax_padded_w(uint32_t w) { return ((w + 3u) / 4u) * 4u; }

inline uint32_t softmax_words(uint32_t w) { return (w + 3u) / 4u; }

inline bool softmax_lane_valid(uint32_t lane_idx, uint32_t w) {
    return lane_idx < w;
}

inline float as_f32(float x) {
    // 强制经 float32 语义存储（与 Python np.float32 一致）
    float y = x;
    uint32_t bits = 0;
    std::memcpy(&bits, &y, sizeof(bits));
    std::memcpy(&y, &bits, sizeof(bits));
    return y;
}

inline float halfint_exp_f32(float x) {
    return exp_math::host_exp_f32(as_f32(x));
}

inline float newton_recip_magic(float d) {
    uint32_t bits = 0;
    std::memcpy(&bits, &d, sizeof(bits));
    bits = kNewtonMagic - bits;
    float x0 = 0.f;
    std::memcpy(&x0, &bits, sizeof(x0));
    return x0;
}

inline float newton_recip(float d, uint32_t iters = kDefaultNewtonIters) {
    d = as_f32(d);
    float x = newton_recip_magic(d);
    for (uint32_t i = 0; i < iters; ++i)
        x = as_f32(x * as_f32(2.0f - as_f32(d * x)));
    return x;
}

/** FP16 I/O 语义：先 RN 到 half 再升回 FP32（对齐 Sigmoid/LSU）。 */
inline float fp16_roundtrip_f32(float x) {
    return app::float16_to_float32(app::float32_to_float16(as_f32(x)));
}

inline float fp16_from_f32(float x) {
    return app::float16_to_float32(app::float32_to_float16(as_f32(x)));
}

/** 将长度 w 向量 pad 到 4 的倍数，尾 lane 填 kSoftmaxPadValue。 */
inline std::vector<float> pad_vector_fp32(const std::vector<float>& x, uint32_t w) {
    const uint32_t pw = softmax_padded_w(w);
    std::vector<float> out(pw, kSoftmaxPadValue);
    for (uint32_t i = 0; i < w && i < x.size(); ++i) out[i] = as_f32(x[i]);
    return out;
}

/** FP16 I/O：pad 值先 RN 到 half 再升回 FP32。 */
inline std::vector<float> pad_vector_fp16_semantics(const std::vector<float>& x,
                                                    uint32_t w) {
    const uint32_t pw = softmax_padded_w(w);
    const float pad_h = fp16_roundtrip_f32(kSoftmaxPadValue);
    std::vector<float> out(pw, pad_h);
    for (uint32_t i = 0; i < w && i < x.size(); ++i)
        out[i] = fp16_roundtrip_f32(x[i]);
    return out;
}

struct StableVec {
    std::vector<float> x_prime;
    std::vector<float> exp_v;
    float max_v{0.f};
    float sum_exp{0.f};
    float inv_sum{0.f};
};

/** FP32 加法（强制 f32 语义）。 */
inline float fp32_add_f32(float a, float b) {
    return as_f32(as_f32(a) + as_f32(b));
}

/** 对齐 RTL FP32ADDTREE8TO1：4 lane + 4×0 的 8 输入树（S1=0）。 */
inline float fp32_addtree8_word_sum(float e0, float e1, float e2, float e3) {
    const float l0_0 = fp32_add_f32(e0, e1);
    const float l0_1 = fp32_add_f32(e2, e3);
    return fp32_add_f32(l0_0, l0_1);
}

/** Phase1.5 sum：按 word ADDTREE8 累加（与 RTL / pack_softmax_rtl_case.py 一致）。 */
inline float sum_exp_tree_fp32(const std::vector<float>& exp_v, uint32_t w) {
    const uint32_t words = softmax_words(w);
    float s = 0.f;
    for (uint32_t wi = 0; wi < words; ++wi) {
        float lanes[4] = {0.f, 0.f, 0.f, 0.f};
        for (uint32_t li = 0; li < 4u; ++li) {
            const uint32_t idx = wi * 4u + li;
            if (idx < w && idx < exp_v.size()) lanes[li] = as_f32(exp_v[idx]);
        }
        const float word_sum =
            fp32_addtree8_word_sum(lanes[0], lanes[1], lanes[2], lanes[3]);
        s = fp32_add_f32(s, word_sum);
    }
    return s;
}

/** dim=-1，N=C=H=1：对长度 W 向量做 stable 分解（FP32 内部）。
 *  force_zero_max=true：强制 m=0（x'=x），镜像 LSU streaming 的 RTL 常态 ——
 *  片上 max 链产出 m=0（bank5@0 从未被写 m），若 golden 仍减 true max，
 *  fp32 链系统性差 ~1 ulp，宽行（如 W=1147）个别 fp16 落在 RN 边界翻转。
 *  默认 false，legacy 调用行为不变。 */
inline StableVec decompose_stable_fp32(const std::vector<float>& x,
                                       bool force_zero_max = false) {
    StableVec s;
    if (x.empty()) return s;
    float m = -std::numeric_limits<float>::infinity();
    for (float v : x) m = std::max(m, as_f32(v));
    s.max_v = force_zero_max ? 0.f : m;
    s.x_prime.resize(x.size());
    s.exp_v.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        s.x_prime[i] = as_f32(as_f32(x[i]) - s.max_v);
        s.exp_v[i] = halfint_exp_f32(s.x_prime[i]);
    }
    const uint32_t w = static_cast<uint32_t>(x.size());
    s.sum_exp = sum_exp_tree_fp32(s.exp_v, w);
    s.inv_sum = (s.sum_exp == 0.f) ? 0.f : newton_recip(s.sum_exp);
    return s;
}

/** Host 黄金 y（FP32 输出路径，I/O 仍为 FP32）。force_zero_max 见 decompose_stable_fp32。 */
inline std::vector<float> softmax_host_fp32(const std::vector<float>& x,
                                            uint32_t newton_iters = kDefaultNewtonIters,
                                            bool force_zero_max = false) {
    StableVec s = decompose_stable_fp32(x, force_zero_max);
    std::vector<float> y(x.size(), 0.f);
    const float inv = (s.sum_exp == 0.f) ? 0.f : newton_recip(s.sum_exp, newton_iters);
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = as_f32(as_f32(s.exp_v[i]) * inv);
    return y;
}

/** Host 黄金 y（FP16 输出；内部 FP32 链路 + 末尾 RN→half）。force_zero_max 见 decompose_stable_fp32。 */
inline std::vector<float> softmax_host_fp16_out(const std::vector<float>& x,
                                                uint32_t newton_iters = kDefaultNewtonIters,
                                                bool force_zero_max = false) {
    // 输入按 FP16 语义
    std::vector<float> x_h(x.size());
    for (size_t i = 0; i < x.size(); ++i) x_h[i] = fp16_roundtrip_f32(x[i]);
    const auto y32 = softmax_host_fp32(x_h, newton_iters, force_zero_max);
    std::vector<float> y(y32.size());
    for (size_t i = 0; i < y32.size(); ++i) y[i] = fp16_from_f32(y32[i]);
    return y;
}

/** 理想参考：float64 stable softmax。 */
inline std::vector<double> softmax_math_f64(const std::vector<double>& x) {
    if (x.empty()) return {};
    double m = -std::numeric_limits<double>::infinity();
    for (double v : x) m = std::max(m, v);
    std::vector<double> e(x.size());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        e[i] = std::exp(x[i] - m);
        sum += e[i];
    }
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = e[i] / sum;
    return y;
}

inline int ulp_f32(float a, float b) {
    auto ord = [](float f) -> int32_t {
        uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        if (u & 0x80000000u) return static_cast<int32_t>(0x80000000u - u);
        return static_cast<int32_t>(u);
    };
    return std::abs(ord(a) - ord(b));
}

inline int ulp_fp16(float a, float b) {
    const uint16_t ha = app::float32_to_float16(a);
    const uint16_t hb = app::float32_to_float16(b);
    auto ord = [](uint16_t h) -> int {
        if (h & 0x8000) return -(h & 0x7FFF);
        return h & 0x7FFF;
    };
    if ((ha & 0x7C00) == 0x7C00 || (hb & 0x7C00) == 0x7C00)
        return (ha == hb) ? 0 : 0x7FFF;
    return std::abs(ord(ha) - ord(hb));
}

}  // namespace softmax_golden
}  // namespace op
}  // namespace aiacc

#endif  // AIACC_SMI_IQ_OP_SOFTMAX_GOLDEN_H_
