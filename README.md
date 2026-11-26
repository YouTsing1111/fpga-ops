# fpga-ops

GPLPU / AIACC 激活与归一化算子：**Exp / Sigmoid / SiLU / Softmax**。

- 微码与文档 → [`operators/`](./operators)
- U280 LCS 板上包 → [`fpga/`](./fpga)

仓库：[YouTsing1111/fpga-ops](https://github.com/YouTsing1111/fpga-ops)

---

## 目录结构

```text
fpga-ops/
├── README.md
├── operators/                 # IQ 微码 + Host 金模型 + 设计文档
│   ├── exp/
│   ├── sigmoid/
│   ├── silu/
│   └── softmax/
└── fpga/                      # Load–Compute–Store FPGA 交付包
    ├── README.md
    ├── exp/
    ├── sigmoid/
    ├── silu/
    └── softmax256rg/
```

| 路径 | 用途 |
|------|------|
| [`operators/exp`](./operators/exp) | FP32 Exp（半整数 + Remez + k-flush） |
| [`operators/sigmoid`](./operators/sigmoid) | FP16 Sigmoid（复用 Exp + Newton） |
| [`operators/silu`](./operators/silu) | FP16 SiLU = \(x\cdot\sigma(x)\) |
| [`operators/softmax`](./operators/softmax) | Softmax（max → Exp → sum → Newton） |
| [`fpga/*`](./fpga) | 对应算子的板上 case / IQ / runner |

---

## 算子状态（摘要）

| 算子 | 状态 | 要点 |
|------|------|------|
| Exp | [`operators/exp/STATUS.md`](./operators/exp/STATUS.md) | 对 Host 目标 ULP=0；墙钟约 **115 cyc/word** |
| Sigmoid | [`operators/sigmoid/STATUS.md`](./operators/sigmoid/STATUS.md) | Phase0/1/2 + LSU；依赖 Exp |
| SiLU | [`operators/silu/README.md`](./operators/silu/README.md) | Sigmoid 末尾再乘 x |
| Softmax | [`operators/softmax/STATUS.md`](./operators/softmax/STATUS.md) | 复用 `ExpOp::compile_merged` |

---

## 依赖关系

```text
         Exp
    ┌─────┼─────┐
    ▼     ▼     ▼
Sigmoid Softmax  …
    │
    ▼
  SiLU
```

---

## 快速跑 FPGA 包

```bash
cd fpga/exp          # 或 sigmoid / silu / softmax256rg
bash run_one_case.sh
```

说明见 [`fpga/README.md`](./fpga/README.md)。注意：勿预 swap hex；VIN/VOUT IQ 见各包 `PACKAGE_INFO.txt`。
