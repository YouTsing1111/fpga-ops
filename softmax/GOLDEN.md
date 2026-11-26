# Softmax Host Golden 与理论误差

**版本**：2026-07-21（交付域 **x∈[-8,+8]**；pad=-1000 + Exp k-flush）  
**实现**：[`softmax_golden.h`](./softmax_golden.h)（C++）· [`softmax_host_golden.py`](./softmax_host_golden.py)（Python 扫掠）  
**Exp 段**：委托 [`../exp/exp_decompose.h`](../exp/exp_decompose.h) / [`../exp/exp_host_golden.py`](../exp/exp_host_golden.py)（勿在本目录复制 Remez）

### 交付域与 Pad

| 项 | 值 |
|----|-----|
| **logits 交付域** | **x ∈ [-8, +8]**（FP32） |
| 尾 lane 填充 | **-1000.0f**（依赖 Exp k-flush → e=0；勿用 `-inf`） |
| W | 任意；内部 pad 到 `ceil(W/4)*4` |
| Golden | 只对有效 lane `0..W-1`；pad 不参与 max/sum；输出 pad lane y=0 |
| 精度报告 | [`experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md`](./experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md)（全定义域权威；math flush 对齐） |

---

## 1. Host 黄金链路（= 预期 RTL 行为）

与芯片规划一致；**Phase0 max-sub 暂由 Host 完成**（写入 bank2 的已是 x'）。

```text
输入 x（FP32 或 FP16 round-trip）
  ① m = max(x)                         [FP32]
  ② x' = x - m                         [FP32]
  ③ e_i = halfint_exp(x')              [FP32；委托 Exp SSOT：host_exp_f32]
  ④ S = Σ e_i（按 word ADDTREE8 累加，与 RTL Phase1.5 一致）
  ⑤ inv = Newton×3(1/S)                [FP32；magic 0x7EF311C7]
  ⑥ y_i = e_i × inv                    [FP32]
  ⑦（FP16 I/O）y ← RN_to_FP16(y)       [可选]
```

**RTL 验收基准**：与 **Host 黄金** 比，目标 **ULP=0**（同 Sigmoid W=100）。  
**数学参考**（`math.exp` / float64 softmax，**默认 flush 对齐 Host Exp**）仅用于理论误差说明，**不是** RTL pass/fail 依据。  
Flush 对齐：对 `x'=x−max`，若 `host_exp_f32(x')==0` 则参考 `e=0`，否则 `e=math.exp(...)`——量算法间隙，不把「Host=0 vs 极小正数」算进主 ULP。

---

## 2. 各阶段理论误差（FP32 内部）

| 阶段 | 机制 | 相对 f32 理想值 | 备注 |
|------|------|-----------------|------|
| ① max | FP32 `max` | 0 | 正常有限输入精确 |
| ② x−m | FP32 减法 | 0～1 ULP | stable；x≈m 时 x' 更准 |
| ③ Exp | halfint Remez + √2 | **≤4 ULP/点** | 与 [`../exp/README.md`](../exp/README.md) 交付一致；Remez 多项式本身 ~3e-7 rel |
| ④ sum | FP32 ADDTREE8 按 word | ~O(words)·0.5 ULP | 与 RTL Phase1.5 同序；勿用顺序加 |
| ⑤ Newton×3 | 无除法单元 | ~0–1 ULP on inv | 标量 1/S；×3 后 all_zero ULP≈1 |
| ⑥ mul | FP32 | ~1 ULP/点 | e 已 ≤1 时 y 量级 ≤1 |

**误差叠加（粗估）**：

- 单点 y 相对 **float64 真 softmax**：FP32 路径 **≲ 1e-5 rel**（扫掠 max **~6.5×10⁻⁶**）
- 相对 **float32(真 softmax)**：ULP 可达 **10² 量级** 于最坏情形（见 §4），但 **rel 仍 ~10⁻⁶**

---

## 3. FP32 vs FP16 交付口径

| 路径 | 输入 | 内部 | 输出 | 相对 f64 真值（扫掠） | 相对 f16 格（扫掠） |
|------|------|------|------|---------------------|---------------------|
| **FP32 I/O** | FP32 | FP32 全链路 | FP32 | max rel **~6.5×10⁻⁶** | — |
| **FP16 I/O** | RN→FP16→FP32 | 同上 | RN→FP16 | max rel **~0.2%**（W=32 uniform 最坏） | max ULP **0～4** |

说明：

- **FP32**：主要损失来自 **Newton 倒数 + 顺序求和**，不是 Exp（Exp 段单独 max ULP **≤5**）。
- **FP16**：末级 RN 把 y 收到 half 格上；相对 f64 的 rel 看起来更大，但相对 **最近 f16 格** ULP 通常 **0～4**（与 Sigmoid 经验一致）。
- Attention / 交付 logits **∈ [-8, +8]**：Host vs math_f64（flush 对齐）max ULP≈**33**、mean≈**3.53**、p99≈**16**（见全定义域 REPORT）；窄域 [-5,3.3] 仅作历史对照。

---

## 4. 实测扫掠摘要

工具：

```bash
python3 op/softmax/softmax_host_golden.py --w 4 --seed 20260720
```

日志：[`experiment_logs/softmax_golden_analysis_20260720_144855/`](./experiment_logs/softmax_golden_analysis_20260720_144855/)

| 场景 | W | max ULP(y) vs math→f32 | max rel vs f64 | max ULP Exp |
|------|---|------------------------|----------------|-------------|
| smoke logits | 4 | 77 | 5.4×10⁻⁶ | 2 |
| uniform [-5,3.3] | 4 | 6 | 3.4×10⁻⁷ | 2 |
| 全相等（最坏） | 4/16/32 | **109** | **6.5×10⁻⁶** | 1 |
| dominant logit | 16 | 89 | 5.3×10⁻⁶ | 1 |
| uniform [-5,3.3] | 128 | （见 summary.json） | ~10⁻⁶ 量级 | ≤5 |

**最坏（Newton×3）**：全相等 → max ULP **1**（×2 时为 109）；端到端 max ULP **23**（130k 点扫掠，见 [`softmax_limit_20260720_152636/`](./experiment_logs/softmax_limit_20260720_152636/)）。

---

## 5. 与 PyTorch / LLAMA 的关系

- PyTorch `_softmax` 用 float32 累加 + `exp` + `/sum`（真除法）。
- 本芯片：**Exp 同 delivery 近似** + **Newton 代替除法** → 相对 PyTorch 的 gap 主要由 **⑤⑥** 决定，Exp 段可复用 Exp 的 **4 ULP** 预算。
- LLAMA attention：后续可在 `s_attn` 缩放后再 softmax；当前 golden **不含 scale**。

---

## 6. 文件用法

| 文件 | 用途 |
|------|------|
| `softmax_golden.h` | C++：`softmax_host_fp32` / `softmax_host_fp16_out` / `decompose_stable_fp32` |
| `softmax_host_golden.py` | Python 同源 + 扫掠报告 |
| `GOLDEN.md` | 本文 |

后续 `calculate_golden` / RTL smoke 应调用 **host 链路**，并单独列 **math 参考列** 作说明（同 Exp W=100 表下注释）。

---

## 7. 验收建议

| 对比 | FP32 门槛 | FP16 门槛 |
|------|-----------|-----------|
| RTL vs **Host** | **ULP=0**（目标） | **ULP=0**（目标） |
| Host vs **math f64** | rel **< 1e-5** | rel **< 1e-2** 或 f16 ULP **≤4** |
| Exp 段 alone | ULP **≤4** vs f32(exp) | — |

**一句话**：理论误差预算 **Exp ≤4 ULP** + **Newton×3/求和 ~10⁻⁶ rel**；FP32 端到端 max rel **~1.6×10⁻⁶**、max ULP **23**；FP16 以 **half 格 ULP** 为准（**≤1**），**下溢 rel=100% 属格式限制**。
