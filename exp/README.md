# Exp（FP32）

半整数归约 + Remez Horner + k 域 flush。状态见 [`STATUS.md`](./STATUS.md)；长文/归档见 [`README_FULL.md`](./README_FULL.md)；时延账本见 [`FLOW_ANALYSIS.md`](./FLOW_ANALYSIS.md)。

仓库根说明：[../README.md](../README.md) · FPGA 包：[`../exp_lcs_fpga_20261125_125437/`](../exp_lcs_fpga_20261125_125437/)

---

## 算法（Host / RTL 同构）

```text
x → t=x·log2e → tw=2t → n=RN(tw)（1.5·2²³ 魔法数）
  → r=(tw−n)/2 ∈ [-0.25,0.25]
  → scale = 2^{n>>1} · (√2 if n odd)
  → k-flush：k≤−126→0；k≥127→+Inf
  → y = scale · Horner(r; c4..c0)
```

---

## 源码

| 文件 | 职责 |
|------|------|
| `exp_decompose.h` / `exp_host_golden.py` | Host 金模型 SSOT |
| `exp_preprocess_op.*` | P1：归约、√2 SELECT、k scratch |
| `exp_poly_kernel_op.*` | P2：Horner + ×scale |
| `exp_op.*` | Merged P1\|P2；flush∥Horner |
| `exp_iq_utils.h` | 延迟常量、emit、flush 相位 |

---

## 周期（本树微码口径）

| 段 | 约值 | 说明 |
|----|------|------|
| P1 core | **56** | 含 √2；flush 不串在墙钟上 |
| P2 | **59** | stage=10×5 + Final 9（FPMUL→FPADD gap=5） |
| **墙钟 / word** | **≈115** | flush 藏进 Horner 空隙 |

N=10k / 8SMC 回归目标：对 Host **100% ULP=0**（含 uflow/oflow）。P1 尾部再压易导致奇数 n（√2）全错，改 schedule 后务必大评测。

---

## Bank（Compute 默认）

| Bank | 内容 |
|------|------|
| 2 | 输入 x |
| 4 | 常数（log2e / MAGIC / √2 / flush 界…） |
| 1 @0..4 | Remez；其后每 word scale |
| 0 | r（+ scratch k/scale） |
| 3 | 输出 y |

---

## FPGA LCS

板上 Load–Compute–Store 交付包见：

[`../exp_lcs_fpga_20261125_125437/README.md`](../exp_lcs_fpga_20261125_125437/README.md)
