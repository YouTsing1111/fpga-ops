# ★ Exp 交付版本状态

| 项 | 内容 |
|----|------|
| **版本** | **exp1.0.5（2026-07-21）** |
| **状态** | **可交付（已冻结备份）** — k 域 flush 闭环；Host/RTL 对齐；大范围评测完成 |
| **备份** | 交付冻结 [`backup_exp1.0.5_20260721/`](./backup_exp1.0.5_20260721/)；**流程优化前** [`backup_exp1.0.5_pre_flow_opt_20260723/`](./backup_exp1.0.5_pre_flow_opt_20260723/) · 流程账本 [`FLOW_ANALYSIS.md`](./FLOW_ANALYSIS.md) |
| **精度** | 对 Host：**100% ULP=0**（N=10000，x∈[-100,100]）；对 Math（安全域有限）max ULP≈**61**（算法间隙） |
| **算法** | 半整数归约 \(r\in[-0.25,0.25]\) + Remez + √2 + **k flush**；Host **n=RN(tw) 魔法数**（旧 ⌊tw+0.5⌋ 已删） |
| **详述** | [`README.md`](./README.md) · 流程 [`FLOW_ANALYSIS.md`](./FLOW_ANALYSIS.md) |

## ★ 重要结论

- **实现闭环**：硬件忠实复现金模型；再提升 vs Math 精度需 **改算法**，不是修微码对齐。  
- **Flush**：`k≤-126→0`，`k≥127→+Inf`（与 RTL 实测一致）。
- **Host 归约（2026-07-23）**：`n = RN(tw)`（`1.5·2²³` 魔法数）；常规域与旧 floor 一致，`tw=*.5` tie 最多差 3 ULP。  
- **RTL P1 魔法数 RN（2026-07-23）**：与 Host 同构（`tw+MAGIC` + `bits−MAGIC`）；**P1 108→97**；墙钟 **186→163**（含 P2 66）。需 RTL 冒烟确认。  
- **P2 RtoVR 压缩（2026-07-23）**：stage/final 13→11；P2 **78→66**。

## 权威实验产物

| 场景 | 日志 |
|------|------|
| **N=10000 / 8SMC / 26word（1.0.5 权威）** | `experiment_logs/exp_n10k_8smc_26w_20260721_011755/` |
| W=100 切块复测（1.0.5） | `experiment_logs/exp_w100_halfint_20260721_013730/` |
| W=100 → 8SMC（1.0.4） | `experiment_logs/exp_halfint_w100_8smc_split_20260719_175428/` |
| 单字统一 unroll（W=4） | `experiment_logs/singleword_unroll_20260719_173935/` |
| IQ 深度实测（W=4/W=16） | `experiment_logs/_iq_depth_w4_smc7.txt` / `_iq_depth_w16_smc7.txt` |

## 冻结对照线（勿混）

| 线 | 目录 | 说明 |
|----|------|------|
| **1.0.5 半整数 + k-flush** | `backup_exp1.0.5_20260721/` | **当前冻结** |
| 1.0.4 半整数交付 | `backup_exp1.0.4_20260719/` | 无 k-flush |
| 1.0.3 整数 k / r∈[-0.5,0.5] | `backup_exp1.0.3_20260718/` | IQ 深度改造参考；精度劣于半整数 |
| 半整数算法源 | `backup_halfint_pre_restore_20260718/` | 恢复半整数前的算法快照 |

更早演进归档：`docs_archive/README_ARCHIVE_pre_delivery_20260718.md`。
