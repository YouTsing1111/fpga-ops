# FPGA LCS 包

每个子目录是一份 **Load–Compute–Store** 板上交付包（U280 runner + case + IQ）。

| 目录 | 算子 | 微码文档 |
|------|------|----------|
| [`exp/`](./exp) | Exp | [`../operators/exp`](../operators/exp) |
| [`sigmoid/`](./sigmoid) | Sigmoid | [`../operators/sigmoid`](../operators/sigmoid) |
| [`silu/`](./silu) | SiLU | [`../operators/silu`](../operators/silu) |
| [`softmax256rg/`](./softmax256rg) | Softmax | [`../operators/softmax`](../operators/softmax) |

## 通用文件

| 文件 | 说明 |
|------|------|
| `PACKAGE_INFO.txt` | SMC 数、IQ id、endian、RTL 版本 |
| `PACKAGE_SHA256SUMS.txt` | 校验 |
| `case/` | `meta_v3.json`、UR/HBM 数据 |
| `aiacc_u280_case_runner.py` | 板上 runner |
| `run_one_case.sh` | 启动封装 |
| `README.md` | 本包说明 |

## 运行

```bash
cd fpga/exp    # 换成目标算子目录
bash run_one_case.sh
```

约束（各包 `PACKAGE_INFO.txt` 为准）：

- 使用 `iq_inst_*.txt`；`data_list` 为本地 basename
- **不要预先 swap hex**（runner 对每 16B 行反转）
- 常见 VIN/VOUT：IQ68 / IQ69 / IQ70
