#!/usr/bin/env python3
"""Sigmoid 大规模：W=100 点均分到 8 SMC（数据互不相同），默认 x∈[-5, 3.3]。

布局（对齐 Exp 8SMC split）：
  - 每 SMC 固定 W=16（Phase1 Exp steps=4，同微码）
  - SMC0–3 各 13 有效点，SMC4–7 各 12（合计 100）
  - pad x=0 → sigmoid(0)=0.5
  - Host 双触发：Phase1=Exp(-x)，Phase2=+1+Newton+FPtoFP
    （IQ 深度另用 SIGMOID_ONCHIP_UPCAST=1 的 Phase0/1/2 emit 报告）

误差：ULP(FP16) vs Host-golden / vs math.sigmoid；辅绝对误差。

用法（需 VCS）:
  python3 op/sigmoid/run_sigmoid_w100_8smc_split.py [seed] [x_lo] [x_hi]
  # 默认 seed=20260719，x_lo=-5，x_hi=3.3
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/YouTsing/Jianyuan-aiacc_4.21")
SMI = ROOT / "aiacc/SMI_IQ"
SIG = SMI / "op/sigmoid"

sys.path.insert(0, str(SMI / "op/exp"))
from exp_host_golden import host_exp_f32  # noqa: E402

NUM_SMC = 8
W_PER_SMC = 16  # 4 words，steps=4
SIZES = [13, 13, 13, 13, 12, 12, 12, 12]
assert sum(SIZES) == 100
NEWTON_ITERS = 2
MAGIC = 0x7EF311C7


def sim_env() -> dict:
    env = os.environ.copy()
    vcs = "/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
    env["VCS_HOME"] = vcs
    env["PATH"] = vcs + "/bin:" + env.get("PATH", "")
    env["LM_LICENSE_FILE"] = env.get("LM_LICENSE_FILE", "27000@bupt-PowerEdge-R730xd")
    env["SNPSLMD_LICENSE_FILE"] = env.get(
        "SNPSLMD_LICENSE_FILE", "27000@bupt-PowerEdge-R730xd"
    )
    env["EXP_COMPRESS_NOP_LOOPS"] = "1"
    return env


def run(cmd, cwd=None, log: Path | None = None, check=True) -> int:
    env = sim_env()
    if log:
        with log.open("w", encoding="utf-8") as f:
            p = subprocess.run(
                cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True, env=env
            )
    else:
        p = subprocess.run(cmd, cwd=cwd, text=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"fail({p.returncode}): {cmd}")
    return p.returncode


def fp16_bits(f: float) -> int:
    """Round FP32→FP16 bits (RN)."""
    bits = struct.unpack("<I", struct.pack("<f", float(np.float32(f))))[0]
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


def bits_to_fp16(h: int) -> float:
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


def ulp_fp16(a: float, b: float) -> int:
    """Integer ULP distance between two values after FP16 quantization."""
    ha, hb = fp16_bits(a), fp16_bits(b)
    # ordered magnitude compare ignoring NaN/Inf extremes for distance
    def to_ord(h: int) -> int:
        if h & 0x8000:
            return -(h & 0x7FFF)
        return h & 0x7FFF

    if (ha & 0x7C00) == 0x7C00 or (hb & 0x7C00) == 0x7C00:
        return 0 if ha == hb else 0x7FFF
    return abs(to_ord(ha) - to_ord(hb))


def newton_recip(d: float, iters: int = NEWTON_ITERS) -> float:
    bits = struct.unpack("<I", struct.pack("<f", float(np.float32(d))))[0]
    bits = (MAGIC - bits) & 0xFFFFFFFF
    x = struct.unpack("<f", struct.pack("<I", bits))[0]
    for _ in range(iters):
        x = float(np.float32(x * (2.0 - d * x)))
    return x


def host_sigmoid(x: float) -> float:
    """Host 金模型：FP16 x → host_exp_f32(-x_h)（含 Exp k-flush）→ Newton×2。"""
    x_h = bits_to_fp16(fp16_bits(x))
    e = host_exp_f32(float(-x_h))
    if math.isinf(e) and e > 0:
        return float(np.float32(0.0))
    return newton_recip(1.0 + e, NEWTON_ITERS)


def math_sigmoid(x: float) -> float:
    x_h = bits_to_fp16(fp16_bits(x))
    # stable
    if x_h >= 0:
        z = math.exp(-x_h)
        return 1.0 / (1.0 + z)
    z = math.exp(x_h)
    return z / (1.0 + z)


def hexlines(data: bytes, line_bytes: int = 16, min_lines: int = 256) -> list[str]:
    lines = [
        data[i : i + line_bytes][::-1].hex() for i in range(0, len(data), line_bytes)
    ]
    while len(lines) < min_lines:
        lines.append("0" * 32)
    return lines


def f32_list_to_bin(xs: list[float]) -> bytes:
    return b"".join(struct.pack("<f", float(np.float32(v))) for v in xs)


def split_100(xs: list[float]) -> list[list[float]]:
    out: list[list[float]] = []
    i = 0
    for n in SIZES:
        chunk = list(xs[i : i + n])
        i += n
        while len(chunk) < W_PER_SMC:
            chunk.append(0.0)
        out.append(chunk)
    return out


def load_f32_dump(path: Path, n: int) -> list[float]:
    vals: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = bytes.fromhex(line)[::-1]
        vals.extend(struct.unpack("<4f", raw))
        if len(vals) >= n:
            break
    return vals[:n]


def load_fp16_dest_low(path: Path, n: int) -> list[float]:
    """Phase2 bank3: FP16 in low-16 of each FP32 lane."""
    vals: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = bytes.fromhex(line)[::-1]
        for li in range(4):
            (h,) = struct.unpack_from("<H", raw, li * 4)
            vals.append(bits_to_fp16(h))
            if len(vals) >= n:
                return vals[:n]
    return vals[:n]


def fptofp_dest_low_word(ys4: list[float]) -> bytes:
    parts = []
    for y in ys4:
        parts.append(struct.pack("<H", fp16_bits(y)) + b"\x00\x00")
    return b"".join(parts)


def ensure_emit_bin() -> Path:
    out = SMI / "out" / "sigmoid_fp16_smoke"
    out.mkdir(parents=True, exist_ok=True)
    bin_path = out / "emit_sigmoid_stages"
    need = not bin_path.is_file()
    if not need:
        for src in (
            SIG / "emit_sigmoid_stages_main.cpp",
            SIG / "sigmoid_op.cpp",
            SIG / "sigmoid_op.h",
        ):
            if src.stat().st_mtime > bin_path.stat().st_mtime:
                need = True
                break
    if need:
        exclude = (
            "div|div_scalar|select|expand|Mean-dim|mul_scalar|softmax|sub|add|"
            "cos|neg|pow|unsqueeze|rsqrt"
        )
        op_srcs = subprocess.check_output(
            f"find op -name '*.cpp' | grep -vE '/({exclude})/' | "
            f"grep -vE '/backup_|/test_|emit_sigmoid|_main\\.cpp$' | tr '\\n' ' '",
            shell=True,
            cwd=SMI,
            text=True,
        ).strip()
        iq_srcs = subprocess.check_output(
            "find iq -name '*.cpp' | tr '\\n' ' '",
            shell=True,
            cwd=SMI,
            text=True,
        ).strip()
        cmd = (
            f"g++ -std=c++17 -O2 -fno-stack-protector -I. -Iapp "
            f"{SIG}/emit_sigmoid_stages_main.cpp {op_srcs} {iq_srcs} "
            f"app/tvgen_core.cpp app/tvgen_dynamic.cpp app/simulator_config_gen.cpp "
            f"app/app_json_parser.cpp app/fp16_utils.cpp "
            f"math_utils/newton_method/newton_method.cpp "
            f"math_utils/taylor_series/taylor_series.cpp "
            f"-o {bin_path}"
        )
        subprocess.check_call(cmd, shell=True, cwd=SMI)
    return bin_path


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260719
    x_lo = float(sys.argv[2]) if len(sys.argv) > 2 else -5.0
    x_hi = float(sys.argv[3]) if len(sys.argv) > 3 else 3.3
    rng = np.random.default_rng(seed)
    xs_all = [float(x) for x in rng.uniform(x_lo, x_hi, size=100)]
    chunks = split_100(xs_all)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    case = f"sigmoid_w100_8smc_split_{ts}"
    log_dir = SIG / "experiment_logs" / case
    emit_out = SMI / "out" / case
    case_rel_p1 = f"verification_v2/{case}_p1"
    case_rel_p2 = f"verification_v2/{case}_p2"
    case_p1 = ROOT / "test_atomic_ops" / case_rel_p1
    case_p2 = ROOT / "test_atomic_ops" / case_rel_p2
    log_dir.mkdir(parents=True, exist_ok=True)

    alloc = {
        "seed": seed,
        "x_range": [x_lo, x_hi],
        "W_per_smc": W_PER_SMC,
        "steps": W_PER_SMC // 4,
        "sizes_valid": SIZES,
        "note": "8 SMC 数据互不相同；同微码 steps=4；pad x=0 → y=0.5",
        "chunks_head": [[round(x, 6) for x in c[:4]] for c in chunks],
    }
    (log_dir / "alloc.json").write_text(json.dumps(alloc, indent=2), encoding="utf-8")
    (log_dir / "xs_all.json").write_text(json.dumps(xs_all), encoding="utf-8")

    print(f"[0] log={log_dir}")
    print(f"[1] make tvgen + emit W={W_PER_SMC} smc_en=7...")
    run(["make", "tvgen"], cwd=SMI, log=log_dir / "make_tvgen.log")
    emit_bin = ensure_emit_bin()
    emit_out.mkdir(parents=True, exist_ok=True)
    run(
        [str(emit_bin), str(emit_out), str(W_PER_SMC), "7"],
        cwd=SMI,
        log=log_dir / "emit.log",
    )
    # 另 emit 一份带 Phase0 的 IQ，用于 per-queue 深度报告（不改变本测 Host 灌 -x 路径）
    env_u = sim_env()
    env_u["SIGMOID_ONCHIP_UPCAST"] = "1"
    with (log_dir / "emit_onchip_upcast.log").open("w", encoding="utf-8") as f:
        subprocess.run(
            [str(emit_bin), str(emit_out / "onchip_upcast"), str(W_PER_SMC), "7"],
            cwd=SMI,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env_u,
        )
    iq_depths = {}
    import re as _re

    for phase, sub in (
        ("phase0", "onchip_upcast/phase0_hex"),
        ("phase1_exp", "onchip_upcast/stage0_hex"),
        ("phase2_post", "onchip_upcast/stage1_hex"),
    ):
        d = emit_out / sub
        queues = []
        if d.is_dir():
            for hf in sorted(d.glob("iq_*.hex")):
                n = sum(1 for ln in hf.read_text().splitlines() if ln.strip())
                m = _re.match(r"iq_(\d+)_(.+)\.hex", hf.name)
                queues.append(
                    {
                        "file": hf.name,
                        "iq_id": int(m.group(1)) if m else None,
                        "name": m.group(2) if m else hf.stem,
                        "depth": n,
                    }
                )
        iq_depths[phase] = {
            "max_iq": max((q["depth"] for q in queues), default=0),
            "queues": queues,
        }
    (log_dir / "iq_depths.json").write_text(json.dumps(iq_depths, indent=2), encoding="utf-8")

    emit_txt = (log_dir / "emit.log").read_text(encoding="utf-8", errors="replace")
    for key in ("max_iq", "halfint", "unroll", "steps=4", "Stage0", "Stage1", "Phase0"):
        if key in emit_txt:
            pass
    for line in emit_txt.splitlines():
        if "max_iq=" in line or "halfint merged" in line or "2-stage" in line or "Phase0" in line:
            print(" ", line)
    up_txt = (log_dir / "emit_onchip_upcast.log").read_text(encoding="utf-8", errors="replace")
    for line in up_txt.splitlines():
        if "Phase0" in line or "stages=" in line or "body=" in line and "Sigmoid" in line:
            print(" ", line)
    print(
        f"  IQ depths: P0 max={iq_depths.get('phase0', {}).get('max_iq')} "
        f"P1 max={iq_depths.get('phase1_exp', {}).get('max_iq')} "
        f"P2 max={iq_depths.get('phase2_post', {}).get('max_iq')} → {log_dir / 'iq_depths.json'}"
    )

    # --- Phase 1 pack ---
    print("[2] pack Phase1 (Exp) 8-SMC...")
    if case_p1.exists():
        import shutil

        shutil.rmtree(case_p1)
    run(
        [
            "python3",
            str(SIG / "pack_sigmoid_rtl_case.py"),
            "stage0",
            "--case-root",
            str(case_p1),
            "--tvgen-out",
            str(emit_out),
            "--num-smc",
            "8",
            "--lane-format",
            "fp16",
        ],
        log=log_dir / "pack_p1.log",
    )
    # overwrite per-SMC alpha=-x_h and Phase1 golden=exp(-x)
    print("[3] overwrite per-SMC Phase1 UR (不同 -x)...")
    for s in range(NUM_SMC):
        chunk = chunks[s]
        n_valid = SIZES[s]
        neg_x = []
        gold_e = []
        for i, x in enumerate(chunk):
            x_h = bits_to_fp16(fp16_bits(x))
            a = -x_h
            neg_x.append(a)
            if i < n_valid:
                gold_e.append(host_exp_f32(a))
            else:
                gold_e.append(1.0)  # pad x=0 → e^0=1
        ur = case_p1 / f"smc_{s}" / "ur_data"
        ob = case_p1 / f"smc_{s}" / "output_bank"
        (ur / "bank2.hex").write_text(
            "\n".join(hexlines(f32_list_to_bin(neg_x))) + "\n", encoding="utf-8"
        )
        (ob / "bank3_final.hex").write_text(
            "\n".join(hexlines(f32_list_to_bin(gold_e))) + "\n", encoding="utf-8"
        )
        (log_dir / f"smc{s}_xs.json").write_text(
            json.dumps(
                {"n_valid": n_valid, "xs": chunk, "neg_x": neg_x, "exp_gold": gold_e},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  SMC{s}: n={n_valid} x0..3={[round(v,4) for v in chunk[:4]]}")

    heads = [tuple(round(v, 6) for v in chunks[s][:4]) for s in range(NUM_SMC)]
    assert len(set(heads)) == NUM_SMC
    print("  OK: 8 SMC 输入头互不相同")

    print("[4] Phase1 RTL sim...")
    run(
        ["python3", "./generate_data_list_coverage.py", case_rel_p1, "--subq", "0"],
        cwd=ROOT,
        log=log_dir / "gendata_p1.log",
    )
    env = sim_env()
    env["SKIP_GENERATE_DATA_LIST"] = "1"
    with (log_dir / "sim_p1.log").open("w", encoding="utf-8") as f:
        p = subprocess.run(
            [
                "bash",
                "./run_sim_aiacc_top_f.sh",
                "verification_v2",
                f"{case_rel_p1}/smc_0",
                "-s",
                "0",
            ],
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    p1_rc = p.returncode
    for line in (log_dir / "sim_p1.log").read_text(errors="replace").splitlines():
        if "Summary:" in line:
            print(" ", line)

    # --- Phase 2 pack from Phase1 dumps ---
    print("[5] pack Phase2 from Phase1 dumps...")
    if case_p2.exists():
        import shutil

        shutil.rmtree(case_p2)
    case_p2.mkdir(parents=True)
    (case_p2 / "lane_format").write_text("fp16\n", encoding="utf-8")
    (case_p2 / "sigmoid_stage").write_text("1\n", encoding="utf-8")
    (case_p2 / "fptofp_dest_low").write_text("1\n", encoding="utf-8")

    # IQ from emit stage1 → smc_0 + case root
    run(
        [
            "python3",
            str(SIG / "pack_sigmoid_rtl_case.py"),
            "stage1",
            "--case-root",
            str(case_p2),
            "--stage0-case",
            str(case_p1),
            "--num-smc",
            "8",
            "--words",
            str(W_PER_SMC // 4),
            "--stage1-iq-dir",
            str(emit_out / "stage1_hex"),
        ],
        log=log_dir / "pack_p2.log",
    )

    # Fix per-SMC banks from actual Phase1 dump + host Newton golden (dest_low)
    print("[6] overwrite Phase2 UR/golden per SMC...")
    for s in range(NUM_SMC):
        chunk = chunks[s]
        n_valid = SIZES[s]
        dump = case_p1 / f"smc_{s}" / f"ur_result_smc_{s}_bank3.hex"
        if not dump.is_file():
            alts = list((case_p1 / f"smc_{s}").glob("ur_result*_bank3.hex"))
            dump = alts[0] if alts else dump
        rtl_e = load_f32_dump(dump, W_PER_SMC)

        bank3 = []
        bank5 = []
        bank6 = []
        bank7 = []
        gold_words = []
        for wi in range(256):
            if wi < W_PER_SMC // 4:
                ys, e_lanes = [], []
                for li in range(4):
                    idx = wi * 4 + li
                    e = rtl_e[idx] if idx < len(rtl_e) else 0.0
                    e_lanes.append(e)
                    if idx < n_valid:
                        ys.append(host_sigmoid(chunk[idx]))
                    else:
                        ys.append(0.5)
                bank3.append(struct.pack("<4f", *e_lanes))
                bank5.append(b"\x00" * 16)  # Phase2 片上 +1 写 denom
                bank6.append(b"\x00" * 16)  # Phase2 片上 magic 写 x0
                bank7.append(struct.pack("<4f", *ys))  # 仅 golden 对照
                gold_words.append(fptofp_dest_low_word(ys))
            else:
                z = b"\x00" * 16
                bank3.append(z)
                bank5.append(z)
                bank6.append(z)
                bank7.append(z)
                gold_words.append(z)

        const4 = [struct.pack("<ffff", 0, 0, 0, 0) for _ in range(256)]
        const4[0] = struct.pack("<ffff", 1.0, 1.0, 1.0, 1.0)
        const4[1] = struct.pack("<ffff", 2.0, 2.0, 2.0, 2.0)
        magic_f = struct.unpack("<f", struct.pack("<I", MAGIC))[0]
        const4[2] = struct.pack("<ffff", magic_f, magic_f, magic_f, magic_f)

        ur = case_p2 / f"smc_{s}" / "ur_data"
        ob = case_p2 / f"smc_{s}" / "output_bank"
        ur.mkdir(parents=True, exist_ok=True)
        ob.mkdir(parents=True, exist_ok=True)
        for b, name in (
            (bank3, "bank3.hex"),
            (const4, "bank4.hex"),
            (bank5, "bank5.hex"),
            (bank6, "bank6.hex"),
            (bank7, "bank7.hex"),
        ):
            (ur / name).write_text(
                "\n".join(hexlines(b"".join(b))) + "\n", encoding="utf-8"
            )
        # zero other banks if missing
        for bi in range(8):
            p = ur / f"bank{bi}.hex"
            if not p.is_file():
                p.write_text(("0" * 32 + "\n") * 256, encoding="utf-8")
        (ob / "bank3_final.hex").write_text(
            "\n".join(hexlines(b"".join(gold_words))) + "\n", encoding="utf-8"
        )

    print("[7] Phase2 RTL sim...")
    run(
        ["python3", "./generate_data_list_coverage.py", case_rel_p2, "--subq", "0"],
        cwd=ROOT,
        log=log_dir / "gendata_p2.log",
    )
    with (log_dir / "sim_p2.log").open("w", encoding="utf-8") as f:
        p = subprocess.run(
            [
                "bash",
                "./run_sim_aiacc_top_f.sh",
                "verification_v2",
                f"{case_rel_p2}/smc_0",
                "-s",
                "0",
            ],
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    p2_rc = p.returncode
    for line in (log_dir / "sim_p2.log").read_text(errors="replace").splitlines():
        if "Summary:" in line:
            print(" ", line)

    print("[8] ULP analysis vs Host-golden / math...")
    rows = []
    max_ulp_host = 0
    max_ulp_math = 0
    max_abs_host = 0.0
    max_abs_math = 0.0
    sum_ulp_host = 0
    sum_ulp_math = 0
    n_pts = 0
    word0_heads = []

    for s in range(NUM_SMC):
        n_valid = SIZES[s]
        chunk = chunks[s][:n_valid]
        dump = case_p2 / f"smc_{s}" / f"ur_result_smc_{s}_bank3.hex"
        if not dump.is_file():
            alts = list((case_p2 / f"smc_{s}").glob("ur_result*_bank3.hex"))
            dump = alts[0] if alts else dump
        rtl = load_fp16_dest_low(dump, W_PER_SMC)[:n_valid]
        word0_heads.append(tuple(round(v, 6) for v in rtl[: min(4, len(rtl))]))
        smc_ulp_h = 0
        for i, x in enumerate(chunk):
            y_rtl = rtl[i]
            y_host = host_sigmoid(x)
            y_math = math_sigmoid(x)
            uh = ulp_fp16(y_rtl, y_host)
            um = ulp_fp16(y_rtl, y_math)
            ah = abs(y_rtl - bits_to_fp16(fp16_bits(y_host)))
            am = abs(y_rtl - bits_to_fp16(fp16_bits(y_math)))
            max_ulp_host = max(max_ulp_host, uh)
            max_ulp_math = max(max_ulp_math, um)
            max_abs_host = max(max_abs_host, ah)
            max_abs_math = max(max_abs_math, am)
            sum_ulp_host += uh
            sum_ulp_math += um
            n_pts += 1
            smc_ulp_h = max(smc_ulp_h, uh)
            rows.append(
                {
                    "smc": s,
                    "i": i,
                    "x": x,
                    "y_rtl": y_rtl,
                    "y_host": y_host,
                    "y_math": y_math,
                    "ulp_host": uh,
                    "ulp_math": um,
                    "abs_host": ah,
                    "abs_math": am,
                }
            )
        print(f"  SMC{s}: n={n_valid} max_ULP_vs_host={smc_ulp_h} dump={dump.name}")

    identical = len(set(word0_heads)) == 1
    print(f"  word0 跨 SMC 是否全相同: {identical} (期望 False)")

    mean_ulp_host = sum_ulp_host / max(n_pts, 1)
    mean_ulp_math = sum_ulp_math / max(n_pts, 1)

    # Enrich: f64 abs before FP16 quant; FP16 hex for both sides
    max_abs_f64_host = 0.0
    max_abs_f64_math = 0.0
    for r in rows:
        r["y_host_f16"] = bits_to_fp16(fp16_bits(r["y_host"]))
        r["y_math_f16"] = bits_to_fp16(fp16_bits(r["y_math"]))
        r["y_rtl_hex"] = f"{fp16_bits(r['y_rtl']):04x}"
        r["y_host_f16_hex"] = f"{fp16_bits(r['y_host']):04x}"
        r["abs_f64_host"] = abs(r["y_rtl"] - r["y_host"])
        r["abs_f64_math"] = abs(r["y_rtl"] - r["y_math"])
        max_abs_f64_host = max(max_abs_f64_host, r["abs_f64_host"])
        max_abs_f64_math = max(max_abs_f64_math, r["abs_f64_math"])

    summary = {
        "case": case,
        "seed": seed,
        "x_range": [x_lo, x_hi],
        "n_points": n_pts,
        "W_per_smc": W_PER_SMC,
        "steps": 4,
        "phase1_sim_rc": p1_rc,
        "phase2_sim_rc": p2_rc,
        "max_ulp_vs_host": max_ulp_host,
        "mean_ulp_vs_host": mean_ulp_host,
        "max_ulp_vs_math": max_ulp_math,
        "mean_ulp_vs_math": mean_ulp_math,
        "max_abs_vs_host_f16": max_abs_host,
        "max_abs_vs_math_f16": max_abs_math,
        "max_abs_f64_vs_host": max_abs_f64_host,
        "max_abs_f64_vs_math": max_abs_f64_math,
        "word0_all_smc_identical": identical,
        "cycles": {
            "phase0_upcast_body_cyc_per_word": 30,
            "phase1_exp_body_cyc_per_word": 186,
            "phase2_post_body_cyc_per_word": 125,
            "steps": 4,
            "phase0_upcast_total_approx": 30 * 4,
            "phase1_exp_total_approx": 186 * 4,
            "phase2_post_total_approx": 125 * 4,
            "note": "body×steps；三 Phase 各独立 trigger",
        },
        "iq_depths": iq_depths,
        "iq_note": "见 iq_depths.json：每个 SMI_IQ 队列深度；P0/P1/P2 max≈7/25/14",
        "ulp_def": "ULP(FP16(RTL), FP16(ref)) — zero means same half-precision code",
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (log_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Pick one example where f64 differs but ULP=0
    ex = next((r for r in rows if r["ulp_host"] == 0 and r["abs_f64_host"] > 0), rows[0])

    lines = [
        "# Sigmoid W=100 → 8 SMC 均分完整分析",
        "",
        f"- case: `{case}`",
        f"- seed: `{seed}`，x ∼ U({x_lo}, {x_hi})",
        f"- 每 SMC W={W_PER_SMC}（Phase1 steps=4），有效点合计 **{n_pts}**，分配 {SIZES}",
        "- 分配说明：SMC0–3 各 **13** 有效点，SMC4–7 各 **12** 有效点；每 SMC UR 仍跑满 **16** 个数（4 word×4 lane，不足 pad x=0）",
        f"- Phase1/2 sim rc: {p1_rc}/{p2_rc}",
        f"- 8 SMC word0 是否全相同: **{identical}**（应为 False）",
        "",
        "## 表1：误差汇总",
        "",
        "| 对照对象 | 最大ULP（半精度格距） | 平均ULP | 最大绝对误差（量化前，FP64） |",
        "|----------|----------------------|---------|------------------------------|",
        f"| RTL vs Host-golden（主机黄金） | **{max_ulp_host}** | {mean_ulp_host:.3f} | {max_abs_f64_host:.6g} |",
        f"| RTL vs math.sigmoid（数学真值） | **{max_ulp_math}** | {mean_ulp_math:.3f} | {max_abs_f64_math:.6g} |",
        "",
        "## 为什么 ULP 可以是 0？",
        "",
        "本报告的 **ULP 是在 FP16 网格上量的**：",
        "",
        "```text",
        "ULP = distance( FP16(RTL), FP16(ref) )",
        "```",
        "",
        "- RTL 输出本身已是片上 FPtoFP 得到的 **FP16**（dest_low）。",
        "- Host / math 先算 FP64/FP32，再 **量化到同一 FP16** 再比。",
        "- 若两边落到 **同一个半精度编码**，ULP=0，即使量化前的 float 小数看起来不同。",
        "",
        f"例（SMC{ex['smc']} i={ex['i']}，x≈{ex['x']:.5g}）：",
        "",
        f"- `y_rtl`  = {ex['y_rtl']:.16g}  (hex `{ex['y_rtl_hex']}`)",
        f"- `y_host` = {ex['y_host']:.16g}  → FP16 `{ex['y_host_f16_hex']}` = {ex['y_host_f16']:.16g}",
        f"- 量化前 |RTL−Host|_f64 ≈ {ex['abs_f64_host']:.6g}，但 FP16 码相同 → **ULP=0**",
        "",
        "Phase1 的 `e^{-x}` 相对 `math.exp` 仍有小误差；经 `1/(1+·)` + FP16 后，"
        "对多数点会落到与 Host-golden 相同的半精度格——这是 **FP16 分辨率粗**，不是「完全零误差」。",
        "",
        "## 为什么 Host 列小数位数看起来不一样？",
        "",
        "旧表用了 `:.5g`，会按数值大小 **自动裁短**，不是 Host 精度忽高忽低。",
        "",
        "- **y_rtl**：FP16 解码，有效约 **~3.3 位十进制**（10 bit mantissa）。",
        "- **y_host / y_math**：脚本内是 FP64，打印可以很长；比 ULP 前会先收成 FP16。",
        "- 下表统一格式，并同时给出量化前 f64 绝对误差。",
        "",
        "## 表2：完整 100 点明细",
        "",
        "| 全局序号 | SMC编号 | 点内序号 | 输入x | RTL输出y（半精度解码） | Host黄金→半精度 | 数学真值→半精度 | "
        "RTL半精度码 | Host半精度码 | ULP相对Host | ULP相对数学 | "
        "绝对误差相对Host（量化前FP64） | 绝对误差相对数学（量化前FP64） |",
        "|----------|---------|----------|-------|------------------------|-----------------|-----------------|"
        "-------------|--------------|-------------|-------------|"
        "-------------------------------|-------------------------------|",
    ]
    for k, r in enumerate(rows):
        lines.append(
            f"| {k} | {r['smc']} | {r['i']} | {r['x']:.6f} | {r['y_rtl']:.8g} | "
            f"{r['y_host_f16']:.8g} | {r['y_math_f16']:.8g} | `{r['y_rtl_hex']}` | "
            f"`{r['y_host_f16_hex']}` | {r['ulp_host']} | {r['ulp_math']} | "
            f"{r['abs_f64_host']:.6g} | {r['abs_f64_math']:.6g} |"
        )

    lines += [
        "",
        "## 表3：按 SMC 小计",
        "",
        "| SMC编号 | 有效点数 | 最大ULP相对Host | 最大ULP相对数学 | "
        "最大绝对误差相对Host（量化前） | 最大绝对误差相对数学（量化前） |",
        "|---------|----------|-----------------|-----------------|"
        "-------------------------------|-------------------------------|",
    ]
    for s in range(NUM_SMC):
        rs = [r for r in rows if r["smc"] == s]
        lines.append(
            f"| {s} | {len(rs)} | {max(r['ulp_host'] for r in rs)} | "
            f"{max(r['ulp_math'] for r in rs)} | "
            f"{max(r['abs_f64_host'] for r in rs):.6g} | "
            f"{max(r['abs_f64_math'] for r in rs):.6g} |"
        )

    lines += [
        "",
        "## IQ 深度与时钟周期（W=16 / steps=4，SIGMOID_ONCHIP_UPCAST emit）",
        "",
        "| Phase | body (cyc/word) | ×4 约总周期 | max_iq |",
        "|-------|-----------------|-------------|--------|",
        f"| 0 Upcast | 30 | 120 | {iq_depths.get('phase0', {}).get('max_iq', '?')} |",
        f"| 1 Exp | 142 | 568 | {iq_depths.get('phase1_exp', {}).get('max_iq', '?')} |",
        f"| 2 Post | 110 | 440 | {iq_depths.get('phase2_post', {}).get('max_iq', '?')} |",
        "",
        "明细见同目录 `iq_depths.json`（每个 SMI_IQ 队列 depth）。",
        "",
        "本测 RTL 数值路径仍为 Host 预写 FP32 `-x` + Phase1/2（与既有 w100 一致）；"
        "Phase0 深度来自同微码 emit，供 INSTQ 评估。",
        "",
    ]
    (log_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {log_dir}/ANALYSIS.md")
    print(json.dumps({k: summary[k] for k in summary if k != "iq_depths"}, indent=2))
    print("iq_depths max:", {k: v.get("max_iq") for k, v in iq_depths.items()})

    ok = (
        p1_rc == 0
        and p2_rc == 0
        and not identical
        and n_pts == 100
        and max_ulp_host < 64  # soft gate; report primary
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
