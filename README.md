# fpga-ops

GPLPU / AIACC 上的激活与归一化算子包：**Exp / Sigmoid / SiLU / Softmax**，含微码源码与 U280 LCS（Load–Compute–Store）FPGA 交付包。

仓库：[YouTsing1111/fpga-ops](https://github.com/YouTsing1111/fpga-ops)

---

## 目录一览

| 路径 | 内容 |
|------|------|
| [`exp/`](./exp) | FP32 Exp 微码与 Host 金模型（半整数归约 + Remez + k-flush） |
| [`sigmoid/`](./sigmoid) | FP16 Sigmoid（复用 Exp + Newton 倒数），8SMC / LSU |
| [`silu/`](./silu) | FP16 SiLU = \(x\cdot\sigma(x)\)，复用 Sigmoid 链路 |
| [`softmax/`](./softmax) | Softmax（max → Exp → sum → Newton），交付域见子目录 README |
| `*_lcs_fpga_*/` | 对应算子的 **FPGA LCS 打包**（IQ / HBM case / runner） |

源码目录偏 **SMI IQ 微码与仿真回归**；`*_lcs_fpga_*` 偏 **板上 Load→Compute→Store** 一键跑法。

---

## 算子状态（摘要）

| 算子 | 状态指针 | 要点 |
|------|----------|------|
| **Exp** | [`exp/STATUS.md`](./exp/STATUS.md) | halfint + k-flush；对 Host 目标 **ULP=0**；墙钟见源码约 **115 cyc/word**（P1∥flush + P2 Horner gap5） |
| **Sigmoid** | [`sigmoid/STATUS.md`](./sigmoid/STATUS.md) | Phase0/1/2 + LSU；依赖 Exp halfint |
| **SiLU** | [`silu/README.md`](./silu/README.md) | 在 Sigmoid 末尾多一次 \(x\cdot\sigma(x)\) |
| **Softmax** | [`softmax/STATUS.md`](./softmax/STATUS.md) | 中间复用 `ExpOp::compile_merged`；默认交付域见 README |

详细算法、Bank 布局、IQ 深度与回归脚本以各子目录 README / STATUS 为准。

---

## FPGA LCS 包怎么用

每个 `*_lcs_fpga_YYYYMMDD_HHMMSS/` 目录通常包含：

| 文件 | 作用 |
|------|------|
| `PACKAGE_INFO.txt` | 算子名、SMC 数、IQ id、endian / RTL 版本说明 |
| `PACKAGE_SHA256SUMS.txt` | 包内校验 |
| `case/` | `meta_v3.json` 等 case 元数据 |
| `aiacc_u280_case_runner.py` | U280 侧跑 case |
| `run_one_case.sh` | 薄封装启动 runner |

通用注意（详见各包 `PACKAGE_INFO.txt`）：

- Runner 使用 **`iq_inst_*.txt`**；`data_list` 为本地 basename。
- **不要预先 swap hex**：runner 在 DMA/打包时对每 16B 行做字节序反转。
- VIN/VOUT IQ 见包内说明（常见 IQ68/69/70）。

示例：

```bash
cd exp_lcs_fpga_20261125_125437
bash run_one_case.sh
# 或按现场环境调用 aiacc_u280_case_runner.py
```

---

## Exp 算法与周期（源码树）

半整数归约（与 Host 同构）：

```text
t = x·log2e → tw=2t → n=RN(tw)（魔法数）
r=(tw−n)/2 ∈ [-0.25,0.25]
scale = 2^{n>>1} · (√2 if odd)
k-flush：k≤−126→0；k≥127→+Inf
y = scale · Horner(r)
```

Merged 微码：P1 core + P2 Horner，**0/Inf flush 与 Horner 空隙并行**（不另占墙钟）。  
流程与瓶颈分析：[`exp/FLOW_ANALYSIS.md`](./exp/FLOW_ANALYSIS.md)。

---

## 依赖关系（简图）

```text
                    ┌─────────┐
                    │   Exp   │
                    └────┬────┘
               ┌─────────┼─────────┐
               ▼         ▼         ▼
          Sigmoid    Softmax    （SiLU⊃Sigmoid）
               │
               ▼
             SiLU
```

改 Exp 数值或 schedule 时，需回归 Sigmoid / Softmax / SiLU 中复用 Exp 的路径。

---

## 许可与用途

面向 AIACC / GPLPU 验证与 FPGA 交付；具体 license 以各打包环境及上游工程为准。
