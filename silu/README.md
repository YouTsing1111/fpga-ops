# SiLU 算子设计与测试文档 (8 SMC / LSU 全链路)

## 1. 概述与核心流程

SiLU（Sigmoid Linear Unit / Swish）激活函数定义为：
$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

在 GPLPU 架构上，SiLU 复用 Sigmoid 的全部基础设施（Phase0 upcast、Phase1 Exp、
Phase2 Newton 倒数），**仅在 Phase2 末尾新增一个 FPMUL 窗口**完成 `x · σ(x)`
（FP32 域单乘 → 单次 FPtoFP 降 FP16，精度最优且 golden 只需一次舍入仿真）。

全流程 3 阶段（与 Sigmoid 相同）：
- **Stage 0 (Load / LDBAXI)**: Remez 系数→bank1、FP16 输入→bank2、常量表→bank4。
- **Stage 1 (Compute)**: 三相微流水（Phase 0 / 1 / 2，见 §3）。
- **Stage 2 (Store / STBAXI)**: bank7 FP16 结果回写 DDR/HBM。

---

## 2. Bank 分配（与 Sigmoid 的唯一差异：D 与 x_fp32 换位）

UR 仅 8 个 bank（`ur_read_code` bank 字段 3 bit），Sigmoid 链路占满 0..7。
SiLU 需要 **x_fp32 活到 Phase2 末尾**，而 Sigmoid 中 x_fp32 所在的 bank5 会被
分母 D 覆写——故将 **D 挪到 bank3**（覆写已死的 e^{-x}），bank5 全程保存 x_fp32：

| Bank | 作用 | 说明 |
| :--- | :--- | :--- |
| **Bank 0** | Exp r 暂存 | Phase1 内部使用 |
| **Bank 1** | Remez 系数 c4..c0 | FP32 / 5 words |
| **Bank 2** | FP16 输入 → FP32 -x | Phase0 覆写为 Exp 输入 |
| **Bank 3** | e^{-x} → **D = 1+e^{-x}** | Phase1 输出；Phase2 `+1` 步骤同窗 read@0/write@10 覆写 |
| **Bank 4** | 共享常量表 15 words | @8=2.0 / @9=MAGIC / @14=-1.0（同 Sigmoid 覆写） |
| **Bank 5** | **x_fp32（全程存活）** | Phase0 写入后不再被写，供 Phase2 最终乘 |
| **Bank 6** | Newton x_n → σ(x) → **silu_fp32** | 最终乘覆写 σ；FPtoFP 源 |
| **Bank 7** | **最终输出 (Clean FP16)** | Newton window1 气泡刷0 + wbe=0x3333（同 Sigmoid） |

---

## 3. Compute 阶段微流水设计 (Stage 1)

- **Phase 0（复用 `SigmoidOp::compile_upcast`，零改动）**：
  FPtoFP FP16→FP32 → bank5 = x_fp32；FPMUL ×(-1 @bank4:14) → bank2 = -x。
- **Phase 1（复用 `ExpOp::compile_merged`，零改动）**：-x → e^{-x} @bank3。
- **Phase 2（`SiluOp::compile_postprocess`，本算子唯一新写调度）**：
  1. `+1`：D = e^{-x} + 1 → **bank3**（覆写 e^{-x}；Sigmoid 写 bank5）。
  2. magic x0 = `0x7EF311C7 - bits(D)`（INTADD）→ bank6；Newton×2 → σ(x) @bank6
     （window1 气泡经 ur_r1@bank4:6 刷0 bank7，同 Sigmoid）。
  3. **★ 最终乘（+15 拍/词）**：FPMUL bank5(x_fp32) × bank6(σ) → bank6，
     镜像 Newton window1 的已验证模式（S0←UR rdata0，S1←UR rdata2，DR→ur_w0）。
  4. FPtoFP FP32→FP16 dest_low → bank7（wbe=0x3333，高16位保持 0x0000）。

周期/词：P0=30 / P1=115 / **P2=140**（=Sigmoid 125 + 最终乘 15），body_total=**285**。
IQ 深度：Phase2 max_iq≈**19**（上限 `SMI_IQ_INSTQ_DEPTH`=128，余量充足）。

## 4. Golden 语义（Host 严格仿真 RTL）

```
e  = host_exp_f32(-x_h)            # Exp SSOT，含 k≤-126→0 / k≥127→+Inf
σ  = newton_recip(1.0 + e, 2)      # magic 初值 + 2 次迭代（e=+Inf 时 σ=0 防御分支）
y  = fp16( x_h * σ )               # FP32 单乘 → 单次降精度；pad lane = silu(0)=0
```

## 5. 测试与回归命令

```bash
cd /home/kaicheng.yang/Desktop/ipset/1_ip/GPLPU/v1.0/3_tool/tvgen/load_store_v3
./run_rtl_test_v3.sh silu --seed 20260712        # 全链路 8 SMC（wc=8）
LSU_WC=32 ./run_rtl_test_v3.sh silu --seed 20260712 --no-rebuild   # 多字（32 words）
```

基线回归（共用脚本零影响确认）：
```bash
./run_rtl_test_v3.sh mul --seed 20260712 --no-rebuild
./run_rtl_test_v3.sh sigmoid --seed 20260712 --no-rebuild
```
（注：run_rtl_test_v3.sh 白名单历来不含 exp，exp 回归不适用。）

## 6. 验证记录

| 日期 | 配置 | 结果 |
|------|------|------|
| 2026-09-16 | seed=20260712, wc=8 | UR 8/8 + DDR 8/8 PASS |
| 2026-09-16 | seed=20260916, wc=8 | UR 8/8 + DDR 8/8 PASS |
| 2026-09-16 | seed=20260712, LSU_WC=32 | UR 8/8 + DDR 8/8 PASS |
