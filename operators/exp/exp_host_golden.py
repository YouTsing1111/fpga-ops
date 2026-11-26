#!/usr/bin/env python3
"""Exp Host 黄金（Python SSOT，对齐 exp_decompose.h::host_exp_f32 / host_exp_f16）。

Softmax pack / 扫掠 / 其它算子请 import 本模块，勿复制 Remez / 分解逻辑。

  from exp_host_golden import host_exp_f32, host_exp_f16, REMEZ_COEFF

C++ 侧 SSOT：op/exp/exp_decompose.h（kRemezCoeff / decompose_exp_input / host_exp_f32）

归约：n = RN(tw)（魔法数 1.5·2^23）；旧 ⌊tw+0.5⌋ 已移除。
"""
from __future__ import annotations

import math
import struct
from typing import Tuple

# 须与 exp_decompose.h kRemezCoeff / kLog2E / kSqrt2 / kRoundMagic* 一致
LOG2E = 1.44269504089
SQRT2 = 1.41421356237
REMEZ_COEFF = (0.99999994, 0.69314606, 0.24023668, 0.05559393, 0.00943751)
ROUND_MAGIC15 = 12582912.0  # 1.5 * 2**23
ROUND_MAGIC_BIAS = 0x4B400000


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _bits_f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(x)))[0]


def decompose_exp_input(x: float) -> Tuple[int, int, bool, float, float]:
    """对齐 exp_math::decompose_exp_input：n=RN(tw) 魔法数，全程 f32。"""
    xf = f32(x)
    t = f32(xf * f32(LOG2E))
    tw = f32(2.0 * t)
    y = f32(f32(tw) + f32(ROUND_MAGIC15))
    n = _bits_f32(y) - ROUND_MAGIC_BIAS
    if n >= 2**31:
        n -= 2**32
    m = n >> 1
    odd = (n & 1) != 0
    r = f32(0.5 * f32(tw - f32(n)))
    try:
        scale = f32(math.ldexp(1.0, m) * (SQRT2 if odd else 1.0))
    except OverflowError:
        scale = float("inf") if m > 0 else 0.0
    return n, m, odd, r, scale


def exp2r_horner_f32(r: float) -> float:
    """对齐 exp_math::exp2r_horner_f32：每步 RN 到 f32。"""
    acc = f32(0.0)
    rf = f32(r)
    for c in reversed(REMEZ_COEFF):
        acc = f32(f32(acc * rf) + f32(c))
    return acc


def host_exp_f32(x: float) -> float:
    """对齐 exp_math::host_exp_f32（f32 运算 + k 越界 flush）。

    Flush 与本仓 RTL（magic-number P1）实测一致：
      k < -126 → 0；k ≥ 127 → +Inf。
      （k = -126 走有限：scale=2^-126，FPMUL 可出 denormal；
        2026-08-26 LSU wc=64 验收确认，此前文档误记为含等号。）
    """
    xf = f32(x)
    if math.isnan(xf):
        return xf
    n, m, odd, r, scale = decompose_exp_input(xf)
    if m < -126:
        return 0.0
    if m >= 127:
        return float("inf")
    try:
        return f32(scale * exp2r_horner_f32(r))
    except OverflowError:
        return float("inf") if scale > 0 else 0.0


# 兼容旧名
halfint_exp = host_exp_f32


def _fp16_bits(f: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f32(f)))[0]
    sign = (bits >> 16) & 0x8000
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if exp == 0xFF:
        return sign | 0x7C00 | (0x200 if mant else 0)
    new_exp = exp - 127 + 15
    if new_exp >= 0x1F:
        return sign | 0x7C00
    if new_exp <= 0:
        if new_exp < -10:
            return sign
        mant |= 0x800000
        shift = 14 - new_exp
        half = mant >> shift
        if (mant >> (shift - 1)) & 1:
            half += 1
        return sign | (half & 0x7FFF)
    half_mant = mant >> 13
    round_bit = (mant >> 12) & 1
    sticky = (mant & 0xFFF) != 0
    if round_bit and (sticky or (half_mant & 1)):
        half_mant += 1
        if half_mant == 0x400:
            half_mant = 0
            new_exp += 1
            if new_exp >= 0x1F:
                return sign | 0x7C00
    return sign | (new_exp << 10) | (half_mant & 0x3FF)


def _bits_to_fp16(h: int) -> float:
    h &= 0xFFFF
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        if m == 0:
            return -0.0 if s else 0.0
        return ((-1) ** s) * (m / 1024.0) * (2 ** -14)
    if e == 31:
        return float("nan") if m else ((-1) ** s) * float("inf")
    return ((-1) ** s) * (1 + m / 1024.0) * (2 ** (e - 15))


def fp16_rt(x: float) -> float:
    return _bits_to_fp16(_fp16_bits(x))


fp16_bits = _fp16_bits


def exp2r_horner_f16_bits(r: float) -> int:
    r_bits = _fp16_bits(r)
    r_val = _bits_to_fp16(r_bits)
    acc_bits = _fp16_bits(0.0)
    for c in reversed(REMEZ_COEFF):
        acc_val = _bits_to_fp16(acc_bits)
        mul_val = _bits_to_fp16(_fp16_bits(acc_val * r_val))
        c_val = _bits_to_fp16(_fp16_bits(c))
        acc_bits = _fp16_bits(mul_val + c_val)
    return acc_bits


def host_exp_f16_bits(x: float) -> int:
    x_eff = fp16_rt(x)
    _, _, _, r, scale = decompose_exp_input(x_eff)
    p = _bits_to_fp16(exp2r_horner_f16_bits(r))
    s = _bits_to_fp16(_fp16_bits(scale))
    return _fp16_bits(p * s)


def host_exp_f16(x: float) -> float:
    return _bits_to_fp16(host_exp_f16_bits(x))
