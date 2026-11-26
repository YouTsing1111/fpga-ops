#!/usr/bin/env python3
"""Softmax Host 黄金 + 误差扫掠（Exp 委托 op/exp/exp_host_golden.py）。

用法:
  python3 op/softmax/softmax_host_golden.py
  python3 op/softmax/softmax_host_golden.py --w 128 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

_EXP_DIR = Path(__file__).resolve().parent.parent / "exp"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))
from exp_host_golden import (  # noqa: E402
    fp16_bits,
    fp16_rt,
    host_exp_f32 as halfint_exp,
)

NEWTON_ITERS = 3
MAGIC = 0x7EF311C7

OUT_DIR = Path(__file__).resolve().parent / "experiment_logs"


def f32(x: float) -> float:
    return float(np.float32(x))


def newton_recip(d: float, iters: int = NEWTON_ITERS) -> float:
    bits = struct.unpack("<I", struct.pack("<f", f32(d)))[0]
    bits = (MAGIC - bits) & 0xFFFFFFFF
    x = struct.unpack("<f", struct.pack("<I", bits))[0]
    for _ in range(iters):
        x = f32(x * f32(2.0 - f32(d * x)))
    return x


def softmax_host_fp32(xs: list[float]) -> tuple[list[float], dict]:
    _SOFTMAX_DIR = Path(__file__).resolve().parent
    if str(_SOFTMAX_DIR) not in sys.path:
        sys.path.insert(0, str(_SOFTMAX_DIR))
    from pack_softmax_rtl_case import softmax_host_fp32_tree  # noqa: E402

    xs = [f32(v) for v in xs]
    y, s = softmax_host_fp32_tree(xs)
    m = max(xs)
    xp = [f32(v - m) for v in xs]
    ev = [halfint_exp(v) for v in xp]
    inv = newton_recip(s) if s != 0.0 else 0.0
    meta = {"max": m, "x_prime": xp, "exp": ev, "sum": s, "inv": inv}
    return y, meta


def softmax_host_fp16_out(xs: list[float]) -> list[float]:
    x_h = [fp16_rt(v) for v in xs]
    y32, _ = softmax_host_fp32(x_h)
    return [fp16_rt(v) for v in y32]


def softmax_math_f64(xs: list[float], *, flush_align: bool = True) -> list[float]:
    """float64 stable softmax；默认 e 的 flush 与 Host Exp（k≤-126→0）对齐。"""
    xd = [float(v) for v in xs]
    m = max(xd)
    ev: list[float] = []
    for v in xd:
        xp = f32(f32(v) - f32(m))
        if flush_align and f32(halfint_exp(xp)) == 0.0:
            ev.append(0.0)
        else:
            ev.append(math.exp(v - m))
    s = sum(ev)
    if s == 0.0:
        return [0.0] * len(xs)
    return [e / s for e in ev]


def ulp_f32(a: float, b: float) -> int:
    def ord_f(f: float) -> int:
        u = struct.unpack("<I", struct.pack("<f", f32(f)))[0]
        return u if (u & 0x80000000) == 0 else int(0xFFFFFFFF - u)

    return abs(ord_f(a) - ord_f(b))


def ulp_fp16(a: float, b: float) -> int:
    ha, hb = fp16_bits(a), fp16_bits(b)

    def ord_h(h: int) -> int:
        return -(h & 0x7FFF) if h & 0x8000 else (h & 0x7FFF)

    if (ha & 0x7C00) == 0x7C00 or (hb & 0x7C00) == 0x7C00:
        return 0 if ha == hb else 0x7FFF
    return abs(ord_h(ha) - ord_h(hb))


def analyze_vectors(vectors: list[tuple[str, list[float]]]) -> dict:
    rows = []
    for name, xs in vectors:
        y_host, meta = softmax_host_fp32(xs)
        y_math = softmax_math_f64(xs)
        y_f16 = softmax_host_fp16_out(xs)
        y_math_f32 = [f32(v) for v in y_math]
        y_math_f16 = [fp16_rt(v) for v in y_math]

        ulp_vs_math_f32 = [ulp_f32(y_host[i], y_math_f32[i]) for i in range(len(xs))]
        ulp_vs_math_f16 = [ulp_fp16(y_f16[i], y_math_f16[i]) for i in range(len(xs))]

        rel_host = [
            abs(y_host[i] - y_math[i]) / max(abs(y_math[i]), 1e-30) for i in range(len(xs))
        ]
        rel_f16 = [
            abs(y_f16[i] - y_math[i]) / max(abs(y_math[i]), 1e-30) for i in range(len(xs))
        ]

        # Exp 段单独：halfint vs math.exp(x')
        ulp_exp = [
            ulp_f32(meta["exp"][i], f32(math.exp(meta["x_prime"][i])))
            for i in range(len(xs))
        ]

        rows.append(
            {
                "name": name,
                "W": len(xs),
                "max_ulp_y_fp32_vs_math_f32": max(ulp_vs_math_f32),
                "mean_ulp_y_fp32_vs_math_f32": sum(ulp_vs_math_f32) / len(ulp_vs_math_f32),
                "max_ulp_y_fp16_vs_math_f16": max(ulp_vs_math_f16),
                "mean_ulp_y_fp16_vs_math_f16": sum(ulp_vs_math_f16) / len(ulp_vs_math_f16),
                "max_rel_y_fp32_vs_math_f64": max(rel_host),
                "max_rel_y_fp16_vs_math_f64": max(rel_f16),
                "max_ulp_exp_vs_math_f32": max(ulp_exp),
                "sum_exp": meta["sum"],
                "max_x": max(xs),
                "min_x_prime": min(meta["x_prime"]),
            }
        )
    return {"rows": rows}


def default_vectors(w: int, seed: int) -> list[tuple[str, list[float]]]:
    rng = np.random.default_rng(seed)
    out: list[tuple[str, list[float]]] = []
    # 固定 smoke（attention 常见量级）
    out.append(("smoke_symmetric", [-2.0, -1.0, 0.0, 1.0][:w] if w <= 4 else list(rng.uniform(-3, 3, w))))
    out.append(("smoke_logits", [-0.5, 0.0, 0.25, 0.5][:w] if w <= 4 else list(rng.uniform(-1, 1, w))))
    # LLAMA 风格：x ∈ [-5, 3.3] 与 Sigmoid 测试区间一致
    out.append((f"uniform_-5_3.3_w{w}", list(rng.uniform(-5.0, 3.3, w))))
    # 尖锐：一个 dominant logit
    xs = [0.0] * w
    if w > 0:
        xs[0] = 8.0
    out.append((f"dominant_w{w}", xs))
    # 全相等
    out.append((f"all_zero_w{w}", [0.0] * w))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260720)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    vecs = default_vectors(args.w, args.seed)
    # 附加典型 attention 长度
    for w in (16, 32, 128):
        if w != args.w:
            vecs.extend(default_vectors(w, args.seed + w))

    report = analyze_vectors(vecs)
    stamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (OUT_DIR / f"softmax_golden_analysis_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Softmax Host Golden 误差扫掠",
        "",
        "Host 链路：`max` → `x'=x-max` → **halfint Exp** → **FP32 sum** → **Newton×3(1/S)** → **FP32 mul**；",
        "FP16 路径：输入/输出各多一次 RN→half。",
        "",
        "| 向量 | W | max ULP(y) FP32 | mean ULP FP32 | max ULP(y) FP16 | max ULP(exp) | max rel% FP32 |",
        "|------|---|-----------------|---------------|-----------------|--------------|---------------|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['name']} | {r['W']} | {r['max_ulp_y_fp32_vs_math_f32']} | "
            f"{r['mean_ulp_y_fp32_vs_math_f32']:.2f} | {r['max_ulp_y_fp16_vs_math_f16']} | "
            f"{r['max_ulp_exp_vs_math_f32']} | {r['max_rel_y_fp32_vs_math_f64']*100:.6e} |"
        )
    (out_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\n[OK] {out_dir}/summary.json ANALYSIS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
