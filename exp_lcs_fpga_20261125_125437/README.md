# Exp LCS FPGA 包

`exp_lcs_fpga_20261125_125437` — Exp 算子 **Load–Compute–Store** 板上运行包。

| 项 | 值（见 `PACKAGE_INFO.txt`） |
|----|---------------------------|
| operator | exp |
| num_smc | 8 |
| out_bank | 3 |
| packed | 2026-11-25 |
| expected_rtl_version | 0x00042200 |

## 文件

| 文件 | 说明 |
|------|------|
| `PACKAGE_INFO.txt` | 打包元数据、IQ id、endian 说明 |
| `PACKAGE_SHA256SUMS.txt` | 校验和 |
| `case/meta_v3.json` | case 元数据 |
| `aiacc_u280_case_runner.py` | U280 runner |
| `run_one_case.sh` | 启动封装 |

## 运行

```bash
cd exp_lcs_fpga_20261125_125437
bash run_one_case.sh
```

注意：

- 使用 **`iq_inst_*.txt`**；勿预 swap hex（runner 对 16B 行反转）。
- VIN/VOUT：IQ68 / IQ69 / IQ70（详见 `PACKAGE_INFO.txt`）。

微码与算法文档：[`../exp/README.md`](../exp/README.md)。
