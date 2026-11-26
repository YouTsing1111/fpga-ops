# Sigmoid 算子设计与测试文档 (8 SMC / LSU 全链路)

## 1. 概述与核心流程

Sigmoid 激活函数定义为：
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

在 GPLPU 架构上，Sigmoid 算子支持在 **8 SMC (SMC0~7)** 架构下通过解耦的 **Load-Compute-Store (LCS)** 机制流式执行。

全流程划分为 3 个阶段：
- **Stage 0 (Load / LDBAXI)**: 从 DDR/HBM 加载 Remez 拟合多项式系数、FP16 输入张量以及常量表至各个 SMC 的 Unified Register (UR) Bank。
- **Stage 1 (Compute)**: 纯计算流水线（分为 Phase 0 / Phase 1 / Phase 2 三个微流水阶段）。
- **Stage 2 (Store / STBAXI)**: 将最终生成的 FP16 结果从 UR Bank 回写至 DDR/HBM。

---

## 2. Bank 分配与内存布局设计

为解决硬件 CRU `FPtoFP` downcast（FP32 -> FP16）仅写低 16 位导致的高 16 位旧数据残留污染问题，采用**Bank 7 独立输出 + 计算间隙并行清零**的设计方案：

| Bank 编号 | 作用 | 格式 / 大小 | 说明 |
| :--- | :--- | :--- | :--- |
| **Bank 1** | Remez 多项式系数 $c_4 \sim c_0$ | FP32 / 5 words | 用于 Horner 法计算指数多项式 |
| **Bank 2** | FP16 输入 / Phase 0 输出 | FP16 / FP32 | 初始加载 FP16 输入；Phase 0 算完 $-x$ 写回此 Bank 作为 Exp 输入 |
| **Bank 3** | Exp 算子输出 $e^{-x}$ | FP32 | 存储 Phase 1 计算出的 $e^{-x}$ |
| **Bank 4** | 共享常量表 | FP32 / 15 words | Word 0~13: Exp 与 Newton 迭代常量；Word 14: $-1.0\text{f}$ (用于取负) |
| **Bank 5** | 分母 $1 + e^{-x}$ / 临时寄存器 | FP32 | Phase 0 暂存 FP32 $x$；Phase 2 暂存分母 |
| **Bank 6** | Newton-Raphson 迭代近似值 $x_n$ | FP32 | 存储倒数迭代过程中的中间结果 |
| **Bank 7** | **最终输出 Bank (Clean FP16)** | FP16 / wc words | 在 Phase 2 利用 ALU 气泡写入 `0.0f` 清零；最终接收 `FPtoFP` 结果并供 LSU 回写 |

### Bank 4 常量表详细映射 (Word 0 ~ 14)
> 布局 = Exp 共享 fp32 常数表（`exp_decompose.h::fill_preprocess_const_bank4` 的 fp32 分支，
> SSOT），Sigmoid 仅覆写 @8/@9（Exp 表中 Softmax 保留槽）并追加 @14。
> **常量必须是 fp32 位模式**——fp16 dest_low 会被按 fp32 读成 ≈0 次正规数，
> 导致 $e^{-x}$ 全零 → 分母≈0 → Newton 溢出 NaN。
- `bank4@0`: $\log_2 e$ (`0x3FB8AA3B`)
- `bank4@1`: INT 127（fp32 指数 bias）
- `bank4@2`: INT 23（fp32 尾数位）
- `bank4@3`: $0.5\text{f}$
- `bank4@4`: $1.0\text{f}$
- `bank4@5`: INT 1
- `bank4@6`: **INT 0（`0x00000000`，兼作 Bank 7 清零的读源）**
- `bank4@7`: $\sqrt{2}$ (`0x3FB504F3`)
- `bank4@8`: **$2.0\text{f}$**（Sigmoid 覆写；Newton 迭代 $x \times (2 - D x_n)$）
- `bank4@9`: **`0x7EF311C7`**（Sigmoid 覆写；Newton 魔常数，INT 减法全 32 bit 操作数）
- `bank4@10`: $-126.0\text{f}$ / `bank4@11`: $127.0\text{f}$ / `bank4@12`: $+\infty$ / `bank4@13`: RN 魔法数 `0x4B400000`
- **`bank4@14`**: **$-1.0\text{f}$** (用于 Phase 0 计算 $x \times (-1.0\text{f}) = -x$)

---

## 3. Compute 阶段微流水设计 (Stage 1)

### Phase 0: 符号反转 ($-x$)
1. 读入 `bank2@0` (FP16 $x$)，通过 `FPtoFP` upcast 转为 FP32 存入 `bank5@0`。
2. 读入 `bank5@0` (FP32 $x$) 与 `bank4@14` ($-1.0\text{f}$)，通过 `FPMUL` 执行 $x \times (-1.0\text{f})$，将结果 $-x$ 写入 `bank2@0`。

### Phase 1: 指数计算 ($e^{-x}$)
- 复用 Exp 算子微指令调度，以 `bank2@0` ($-x$) 为输入，`bank1` (Remez 系数) 与 `bank4` (常量) 为辅助，使用 Horner 多项式展开法计算 $e^{-x}$，并将结果写入 `bank3@0`。

### Phase 2: 倒数计算与 Bank 7 并行清零
1. **分母计算**:
   - 读入 `bank3@0` ($e^{-x}$) 与 `bank4@4` ($1.0\text{f}$)，通过 `FPADD` 计算 $D = 1 + e^{-x}$，写入 `bank5@0`。
2. **Newton-Raphson 倒数初值与迭代**:
   - 通过整数减法魔常数快速生成初值 $x_0 = \text{Magic} - D$，存入 `bank6@0`。
   - 迭代更新公式: $x_{n+1} = x_n \times (2.0 - D \times x_n)$。
3. **Bank 7 并行清零 (Zero-overhead Bubble Utilization)**:
   - 在 Phase 2 中 `FPMUL`/`FPADD` 执行的多周期 ALU 气泡期（`ur_w0` 写端口空闲），调度从 `bank4@6` (`int_zero = 0x00000000`) 读出并写入 `bank7` 对应 word，在不增加额外周期的前提下完成 Bank 7 的清零。
4. **输出截断（关键硬件语义）**:
   - `fptofp_array.v` 的 FP32→FP16 dest_low **并不是"只写低16位"**：其 D 线驱动整 32bit，低 16 位是本次转换结果，**高 16 位 = FPtoFP 内部 128bit 输出寄存器 `output_data_r` 的残留**（指令间保持）。同 step 内上一条 FPtoFP 恰为 Phase0 的 FP16→FP32（`output_data_r` = 本 word 的 fp32 x），若整字 `wbe=0xFFFF` 写入，x 的高半将盖掉先期清零的 0。
   - 因此最终写使用 **字节写掩码 `wbe=0x3333`**（UR `wea` 一字节一 bit，0x3333 = 每 lane 仅字节 0-1）：只将 fp16 低 16 位写进 `bank7`，高 16 位保持气泡清零的 `0x0000`。
   - 最终 `bank7 = [0x0000 | fp16(y)] per lane`，满足 strict golden。

---

## 4. 8 SMC 协同与 LSU 配置

- **SMC 并行度**: `num_smc = 8`, `smc_id_en = 7` (0x7 掩码全使能)。
- **LDBAXI (Load)**:
  - `inputs["coeff"]`: Bank 1, Offset 0, 5 words.
  - `inputs["input"]`: Bank 2, Offset 0, `wc` words.
  - `inputs["other"]`: Bank 4, Offset 0, 15 words.
- **STBAXI (Store)**:
  - `output`: Bank 7, Offset 0, `wc` words.

---

## 5. 测试与回归命令

### 1. 全链路 LSU 8 SMC 测试 (Sigmoid)
```bash
cd /home/kaicheng.yang/Desktop/ipset/1_ip/GPLPU/v1.0/3_tool/tvgen/load_store_v3
./run_rtl_test_v3.sh sigmoid
```

### 2. 基线算子回归测试 (确保公共环境与脚本未受破坏)
```bash
cd /home/kaicheng.yang/Desktop/ipset/1_ip/GPLPU/v1.0/3_tool/tvgen/load_store_v3
./run_rtl_test_v3.sh mul
./run_rtl_test_v3.sh exp
```
