# ★ Sigmoid 交付版本状态

| 项 | 内容 |
|----|------|
| **版本** | **halfint-exp-stage1（2026-07-19）** |
| **状态** | **可交付** — Compute（Phase0/1/2）已验证；**LSU L→C→S 全通**（见 load_store_v3） |
| **命名** | Phase 1 / Phase 2（对齐 Exp；详见 README） |
| **I/O** | FP16 dest_low；**升精度片上 FPtoFP**（Phase0）；降精度 FPtoFP dest_low（Phase2） |
| **依赖** | [`../exp`](../exp) **halfint-delivery**（不改 Exp 源） |
| **大规模误差** | 主尺度 **ULP(FP16)**；辅绝对误差（见 README） |
| **默认测区间** | x ∼ U(**-5, 3.3**)（`run_sigmoid_w100_8smc_split.py`） |
| **IQ（任意 W）** | Phase0 max=**7** / Phase1=**≈25** / Phase2=**≈14**（**强制多字 Exp unroll**，含 W=4） |
| **周期/字** | P0=**30** / P1=**142** / P2=**110** |
| **详述** | [`README.md`](./REAaDME.md) |

## 权威实验产物

| 场景 | 日志 |
|------|------|
| W=4（Phase1+Phase2 gate） | `experiment_logs/sigmoid_fp16_smoke_20260719_123923/` |
| W=8 多字 | `experiment_logs/sigmoid_fp16_smoke_20260719_123936/` |
| **W=100 → 8SMC** x∼U(**-5, 3.3**) | `experiment_logs/sigmoid_w100_8smc_split_20260719_165942/`（含 `iq_depths.json`） |
| **LSU DDR↔UR（L→C→S 全通）** | [`../../load_store_v3/sigmoid/`](../../load_store_v3/sigmoid/) · 权威 [`…/sigmoid_lsu_20260719_182641/`](../../load_store_v3/sigmoid/experiment_logs/sigmoid_lsu_20260719_182641/)（FP16 max_ulp=0）；踩坑详述见该目录 README |

W=100 摘要：pass=8/8；**max ULP(FP16) vs Host = 0**，vs math→f16 = 0；8 SMC word0 互不相同。  
（Phase1 Exp 相对 math.exp 仍有小误差，经 Newton+FP16 后落入同一半精度格。）

## IQ 深度（实测）

| 配置 | Phase1 max_iq | Phase2 max_iq | Phase2 body |
|------|---------------|---------------|-------------|
| W=4（steps=1） | ≈35 | ≈13 | 110 cyc/word |
| W=8（steps=2） | ≈25 | ≈14 | 110 cyc/word |

limit = `smi_top` `SMI_IQ_INSTQ_DEPTH=128`。
