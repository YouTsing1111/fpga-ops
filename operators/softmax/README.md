# Softmax（开发中）

**状态**：Gate A/B/C/D + e2e ✅（W=4/16/100/101）；**交付域 x∈[-8,+8]**。  
**对照**：[`../sigmoid`](../sigmoid) 多 Phase + [`../exp`](../exp) halfint Exp（**Exp 数值 SSOT 解耦**，见 §2.2）。  
**进度**：[`STATUS.md`](./STATUS.md) · **全定义域权威**：[`experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md`](./experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md)  
**开发约定**：**凡改 Sum/Norm 微码或 pack，必跑 W=16**（`steps=4`）Gate C+D + e2e，再跑 W=4 回归。  
**Phase / stage / 脚本用法**：见 **§6.0**（含与 Exp 对比、等中断、IQ 深度 128）。

---

## 1. 公式

\[
y_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},\quad m = \max_j x_j
\]

数值上必须先减 max，再 Exp，再归一化；否则 Exp 易溢出。

---

## 2. 总体思路（对齐 Sigmoid）

Softmax = **自研 reduce / 加减乘 / Newton 倒数** + 中间整段复用 **`ExpOp::compile_merged`**。

| Trigger | 内容 | Bank 约定 |
|---------|------|-----------|
| **Phase0** | \(m=\max x\)，\(x'=x-m\) | 入 x@**bank2**；max 广播@**bank5**；出 x'@**bank2** |
| **Phase1** | Exp → \(e^{x'}\) | 入@2；出@**bank3**；coeff@1；const@4 |
| **Phase1.5** | \(S=\sum e\) | 入 e@3；累加@**bank5@0** |
| **Phase2** | Newton \(1/S\) → \(y=e/S\) | e@3；S@5；inv@6；y@3 |

```text
x (FP32 @bank2)
  --Phase0:  max + sub -->  x' = x - m (@bank2)
  --Phase1:  Exp         -->  e = exp(x') (@bank3)
  --Phase1.5: sum reduce -->  S = Σe (@bank5)
  --Phase2:  Newton + mul -->  y (@bank3)
```

约束：各 Phase **分次 trigger**；单 IQ 队列深度 ≤ **128**；FP32 每 word **4 lane**；Newton **×3**。

---

## 2.1 四个 Phase 一览

Softmax 片上路径拆成 **4 个独立 Phase**（Host 双/多 trigger，与 Sigmoid 对齐）：

| Phase | Gate | 功能 | Trigger 条件 | RTL smoke |
|-------|------|------|--------------|-----------|
| **Phase0a** | Gate Max | 片上 max reduce → \(m\)@bank5 | `SOFTMAX_HOST_MAX=0` | ✅ W=4/16 SKIP_RTL |
| **Phase0b** | Gate C | \(x' = x - m\)，\(m\) 广播@bank5 | `SOFTMAX_HOST_MAX=0` | ✅ W=4 ULP=0 |
| **Phase1** | Gate A | halfint Exp → \(e^{x'}\) | 总是 | ✅ W=4/100 |
| **Phase1.5** | Gate D | \(S = \sum e\)（ADDTREE8） | `SOFTMAX_HOST_SUM=0` | ✅ W=4 / **W=16** bank5 ULP=0 |
| **Phase2** | Gate B | Newton \(1/S\) + mul → \(y\) | 总是 | ✅ ULP=0 |

```text
Trigger-0 (可选)  Phase0   x@bank2  + m@bank5  →  x'@bank2
Trigger-1         Phase1   x'@bank2 + Exp      →  e@bank3
Trigger-2 (可选)  Phase1.5 e@bank3 + ADDTREE8  →  S@bank5@0
Trigger-3         Phase2   e@bank3  + S@bank5  →  y@bank3
```

### Bank 分配（Compute 路径）

| Bank | 角色 | Phase0 | Phase1 | Phase1.5 | Phase2 |
|------|------|--------|--------|----------|--------|
| **0** | — | — | — | — | — |
| **1** | Exp 多项式系数 | — | 读 | — | — |
| **2** | 原始 x / x' | 读+写 | 读（x' 入） | — | — |
| **3** | \(e^{x'}\) / 最终 y | — | 写 | 读 | 读+写 |
| **4** | 常量 | — | 读（Exp const @word0..7） | — | 读 **2.0 @word8**（Newton） |
| **5** | max 广播 / 累加和 S | 读（m） | — | 写（S 四 lane 广播） | 读（S） |
| **6** | Newton \(x_n\) / inv | — | — | 暂存 word_sum（多 word） | 读+写 |
| **7** | — | — | — | — | — |

说明：

- **bank5**：Phase1.5 用 ADDTREE `dst_id=0..3` 各写一次，**四 lane 同 S**（供 Phase2 Newton SIMD）；Gate D golden 亦四 lane 广播。
- **bank4@8=2.0**：Newton 读 2.0；Exp 常量占 word0..7，e2e 串联时不冲突（standalone Stage1 pack 亦写 word8）。
- **bank6**：Phase2 Newton；Phase1.5 多 word 时暂存 word_sum（`steps>1`，待 W>4 验证）。

### Body cycles 与 IQ 深度（W=4，steps=1，单 SMC）

实测来自 `emit_softmax_stages` + `same-code+loops` 压缩后计数（2026-07-20）：

| Phase | body (cyc/word) | max_iq（单队列） | 主要 IQ 队列 | fit≤128 |
|-------|-----------------|------------------|--------------|---------|
| Phase0 max-sub | **15** | **4** | ur-r0/r1/w0, fpadd, rtovr-5/6/35 | ✅ |
| Phase1 Exp | **142** | **24** | Exp merged P1\|P2 全套 | ✅ |
| Phase1.5 sum | **48** | **10** | ur-r0/w0, addtree8×4, rtovr-32/33/35 | ✅ |
| Phase2 Norm **@host** | **138** | **16** | `HOST_X0=1`：Host 预载 x0，Newton×3 + mul | ✅ |
| Phase2 Norm **@chip** | **153** | **17** | `HOST_X0=0`：+15 cyc magic 前导，再 Newton×3 + mul | ✅ |

多 word 时（`steps = ceil(W/4)`）：

- Phase0 / Phase1 / Phase2：body **不变**，靠外层 EMBED + `addr_inc` 逐 word 回放。
- Phase1.5 在 `steps>1` 时 body 增至 **60 cyc/word**（bank6 标量 acc + ADDTREE 累加 + 四 lane 广播），IQ 仍 fit。
- **W=16（steps=4）为 Sum/Norm 联调基准**；W=4（steps=1）仅作回归，不能代替多 word 验收。

Phase1 Exp 内部：`p1_body=64`（halfint 预处理）+ `p2=78`（Horner unroll）= 142 wall cycles/word。

### Exp 数值 SSOT 解耦

Softmax **不复制** Remez / 半整数分解；Exp 改实现后，Host golden / pack / 扫掠自动对齐。

| 层级 | SSOT | Softmax 委托方式 |
|------|------|------------------|
| **C++** | [`../exp/exp_decompose.h`](../exp/exp_decompose.h) `host_exp_f32` / `host_exp_f16_bits` | [`softmax_golden.h`](./softmax_golden.h) `#include "../exp/exp_decompose.h"` → `halfint_exp_f32()` |
| **Python** | [`../exp/exp_host_golden.py`](../exp/exp_host_golden.py) `host_exp_f32` | `pack_softmax_rtl_case.py` / `softmax_host_golden.py` / `measure_softmax_limit.py` `from exp_host_golden import host_exp_f32` |
| **RTL 微码** | [`../exp/exp_op.cpp`](../exp/exp_op.cpp) `ExpOp::compile_merged` | Softmax Phase1 **直接调用**，不改 Exp 源 |

**优化 Exp 时**：只改 `exp_decompose.h`（系数/分解）+ 必要时 Exp 微码；然后跑 Exp smoke + `W=4 bash op/softmax/run_softmax_e2e_smoke.sh`。  
RTL 仍须与 **Host Exp golden bit 一致**（e2e ULP=0）；只改性能、不改数值时 Softmax 通常无需动。

---

## 2.2 数据流：输入 x 之后（Host 写什么 / 片上算什么）

**结论**：e2e 全链路里，**x'、e、S、y 都不是 Host 灌进去的**；Host 只写**入口 x** 和**常量/辅助量**，中间 tensor 由四个 Phase **在 UR bank 里接力算完**。

### 公式与 Phase 对应

\[
m=\max(x),\quad x'_i=x_i-m,\quad e_i=\exp(x'_i),\quad S=\sum_i e_i,\quad y_i=e_i/S
\]

| 量 | 谁算 | 存哪 | e2e 是否 Host 灌 |
|----|------|------|------------------|
| **x** | 用户 / Host | bank2 | ✅ Host 写（输入） |
| **m** | Host（暂） | bank5 | ✅ Host 写（**不是** x'/e/S） |
| **x'** | **芯片 Phase0** | bank2 | ❌ 片上算，覆盖 x |
| **e** | **芯片 Phase1** | bank3 | ❌ 片上算 |
| **S** | **芯片 Phase1.5** | bank5 | ❌ 片上算（覆盖 m） |
| **x0** | Host（@host）/ **芯片 Phase2**（@chip） | bank6 | @host：Host 写 Newton 初值；@chip：Norm 开头 magic 前导 |
| **1/S → y** | **芯片 Phase2** | bank6→bank3 | ❌ 片上算；**y 最终 @bank3** |

### e2e 一次仿真里的完整路径

以 `SOFTMAX_HOST_MAX=0`、`SOFTMAX_HOST_SUM=0`、W=4 smoke 为例：`x = [-2, -1, 0, 1]`，`m=1`。

**① Host 写 UR（仿真开始前，只加载一次）**

| Bank | 内容 |
|------|------|
| bank2 | 原始 **x** |
| bank5 | **m** 四 lane 广播，例 `[1,1,1,1]` |
| bank6 | **@host**：Host 预载 **magic(x0)**；**@chip**：init=0，由 stage_3 片上算 |
| bank1 / bank4 | Exp 系数与常量；bank4 **@word8=2.0**，**@word9=MAGIC** |

Host **不写** x'、e、S、y。

**② 芯片四 stage 顺序 trigger（UR 持久，bank 不回 Host）**

```text
Host 写入:  x@bank2, m@bank5, x0@bank6, 常量@bank1/4
                │
 stage_0 Phase0  ├─► x' 写回 bank2     例 [-3,-2,-1,0]   ← 芯片算
                │
 stage_1 Phase1   ├─► e  写入 bank3     例 halfint Exp(x')
                │
 stage_2 Phase1.5 ├─► S  写入 bank5     四 lane 广播 S   ← 覆盖 m
                │
 stage_3 Phase2   └─► y  写回 bank3     覆盖 e；读 bank3 即输出
```

**③ 输出**：**bank3** = softmax(x)；e2e W=4 已与 Host `golden.bin` **ULP=0**。

### 和「分 Gate 单独测」的区别

分 Gate 是为了**单段调试**，pack 时会**模拟**「上一步 Host 已算好」——**不是** e2e 的正常数据路径：

| 测法 | 中间量谁提供 |
|------|--------------|
| **e2e@host** | Host 给 x + m + x0 + 常量；中间 x'/e/S/y 全在片上 |
| **e2e@chip** | 同上，但 **bank6 init=0**，x0 由 stage_3 magic 前导算出 |
| Gate A 单独（Exp） | Host 预写 **x'**@bank2（跳过 Phase0） |
| Gate D 单独（Sum） | Host 从 Exp dump **灌 e**@bank3 |
| Gate B@host 单独 | Host pack **e + S + x0** 再跑 Norm@host |
| Gate B@chip 单独 | Host pack **e + S**，bank6=0，跑 Norm@chip |

### 仍由 Host 参与的两项（不是中间 tensor）

1. **m = max(x)**：`SOFTMAX_HOST_MAX=0` 时由 **Phase0a** 片上 max reduce（见 [`SOFTMAX_MAX_REDUCE.md`](./SOFTMAX_MAX_REDUCE.md)）；Gate 探针仍可由 Host 写 bank5 golden。
2. **Newton x0（仅 @host 模式）**：`SOFTMAX_HOST_X0=1` 时 Host 预载 bank6；`HOST_X0=0`（@chip）已在 Norm 开头用 magic 前导由 S 推导 ✅。

除此以外，**x' → e → S → y 在 e2e 里均为片上 Phase 间 bank 传递，不是 Host 灌中间结果。**

---

这三个变量控制 **哪些步骤由 Host 代做、哪些由芯片微码跑**。  
**默认全片上（`=0`）**：Host 只写 **x + m + 常量**；x'、e、S、x0、y 均由四 Phase 微码完成。  
分 Gate 单独测时可在脚本里显式设 `HOST_*=1`（见 `run_softmax_fp32_smoke.sh`）。

### `SOFTMAX_HOST_MAX` — 谁做 max-sub？

| 值 | 含义 | bank2 里是什么 | 芯片跑什么 |
|----|------|----------------|------------|
| **`0`（默认）** | Host 只灌原始 logits | **原始 x**（pad 尾 lane） | **Phase0 微码**：读 x@bank2、读 m@bank5（四 lane 广播），算 **x − m** 写回 bank2 |
| **`1`** | Host 已做完 stable 预处理 | 已是 **x' = x − m** | **跳过 Phase0**，直接 Phase1 Exp |

**一句话**：`HOST_MAX=1` = 「bank2 里已经是减过 max 的数，芯片别管 max」；`HOST_MAX=0` = 「bank2 是原始 x，请芯片自己减 max」。

**Phase0a 片上 max reduce**：两种 Phase B 方案（Extract+Broadcast / DS Transpose）、ADDTREE 广播原理、周期对比 → **[`SOFTMAX_MAX_REDUCE.md`](./SOFTMAX_MAX_REDUCE.md)**。

**当前局限（2026-07-29）**：

- `HOST_MAX=0` 时，**Phase0a max reduce** + **Phase0b 减法**（`compile_max_reduce` / `compile_maxsub`）已 emit ✅；Gate Max W=4/16 SKIP_RTL PASS
- **Phase0a 默认方案 A**（Extract+Broadcast）已可 RTL；**方案 B**（DS Transpose）compare 微码已有，STB/DS/LDB 待接入
- e2e 串联 Phase0a→0b 时 pack 可不再 Host 写 m（进行中）

### `SOFTMAX_HOST_SUM` — 谁做 exp 求和？

| 值 | 含义 | 芯片跑什么 |
|----|------|------------|
| **`0`（默认）** | 芯片自己对 bank3 的 e 做 reduce | **Phase1.5 微码**：`FP32ADDTREE8TO1` 四字求和 + 累加到 bank5@0 |
| **`1`** | Host 算好 S，pack 时写 bank5 + bank6 magic | **跳过 Phase1.5**，直接 Phase2 Newton+mul |

### `SOFTMAX_HOST_INV` — 谁算 1/S？

| 值 | 含义 |
|----|------|
| **`0`（默认）** | Phase2 跑 Newton×3 |
| **`1`** | Host 直写 1/S@bank6，Phase2 只做 mul |

### `SOFTMAX_HOST_X0` — Newton 初值 x0 谁写 bank6？

| 值 | 模式 | 含义 | Phase2 body |
|----|------|------|-------------|
| **`0`（默认）** | **@chip** | Norm 开头插入 magic 前导：`INTADD(MAGIC − bits(S))` → bank6 | 153 cyc |
| **`1`** | **@host** | Host pack 预载 magic(x0)@bank6 | 138 cyc |

配套：`SOFTMAX_MAGIC_ONLY=1` 时只 emit/跑 magic 前导（Gate Magic 隔离测，body=15）。  
pack：`phase2_norm --no-host-x0` / `e2e --no-host-x0` → bank6 init=0。

### 组合示例

```bash
# 四 Phase e2e（默认全片上：x + m + 常量 → y）
W=4 bash op/softmax/run_softmax_e2e_smoke.sh

# 分 Gate A+B 联调（Host 预写 x'/S，Norm 仍 @chip）
bash op/softmax/run_softmax_fp32_smoke.sh

# 分 Gate 全 @host 联调（显式覆盖默认）
SOFTMAX_HOST_X0=1 bash op/softmax/run_softmax_fp32_smoke.sh

# Gate Magic 隔离（仅 magic 前导 → bank6）
W=4 bash op/softmax/run_softmax_magic_x0_smoke.sh

# Gate C/D
bash op/softmax/run_softmax_gate_cd_smoke.sh
GATE_D=0 bash op/softmax/run_softmax_gate_cd_smoke.sh   # 仅 Gate C

# 全片上微码 emit（不跑 RTL）
SKIP_RTL=1 W=4 bash op/softmax/run_softmax_e2e_smoke.sh

# 任意 W（例 W=101，尾 3 lane pad -104）
W=101 SKIP_RTL=1 bash op/softmax/run_softmax_fp32_smoke.sh
```

---

## 4. W 任意 + Pad 规格

| 项 | 约定 |
|----|------|
| 逻辑长度 | **W 任意**（不必 4 整除） |
| 硬件字数 | `words = ceil(W/4)` |
| 尾 lane 填充 | **`-104.0f`**（`x'=pad−m` ≲ −104 → FP32 `exp` 数学下溢为 0；**Exp RTL 尚不能吃 |x'|≳100**，W≠4k 的 pad 需 flush 或更温和 pad） |
| 禁止 | **`-inf`**（halfint Exp 输入会 NaN） |
| Golden / gate | 只验收有效 lane `0..W-1`；pad lane 不参与 max/sum 语义 |

实现位置：

- C++：`softmax_golden.h` → `kSoftmaxPadValue`、`pad_vector_fp32()`
- Python pack：`pack_softmax_rtl_case.py` → `pad_softmax_bin()`
- `generate_data`：Host 写 bank2 时自动 pad

pad 选 **−1000** 的原因：交付域 logits ∈ **[−8, +8]** 时 far below max，经 Exp k-flush 后 e=0，不影响 sum；又比 −inf 安全。

---

## 5. 当前实现进度（2026-07-20）

### Gate 状态

| Gate | 内容 | 状态 |
|------|------|------|
| Host Golden | halfint Exp + Newton×3 | ✅ |
| **Gate A** | Host x' → Phase1 Exp RTL | ✅ W=4 / W=100 PASS |
| **Gate B** | Phase2 Newton×3 + mul | ✅ ULP=0 |
| **Gate C** | Phase0 `x - broadcast(m)` RTL | ✅ **PASS** W=4 bank2 ULP=0 |
| **Gate D** | Phase1.5 sum reduce RTL | ✅ W=4 / **W=16** bank5 ULP=0 |
| **e2e** | P0→Exp→Sum→Norm 四 stage | ✅ W=4 ULP=0；✅ **W=16** max_ulp≤2（见 §5.2） |
| 任意 W | pad=-104 | ✅ W=100 e2e；W=101 验 pad |

Gate C/D 一键 smoke：

```bash
bash op/softmax/run_softmax_gate_cd_smoke.sh   # Gate C + Gate D 均跑
```

### RTL Smoke 结果（Gate A+B，默认 env）

| 测试 | Stage0 Exp | Stage1 Norm |
|------|------------|-------------|
| W=4 | PASS，ULP≤4 | ULP=0 4/4 |
| W=100 | PASS，ULP≤13 | ULP=0 100/100 |

### 四 Phase e2e 串联（multi_inst_stage）

**目标**：一次仿真、UR 持久、四段 IQ 顺序 trigger（对齐 Exp `multi_inst_stage` 框架）。

| stage | Phase | IQ 来源（emit） |
|-------|-------|-----------------|
| stage_0 | Phase0 max-sub | `phase0_hex/` |
| stage_1 | Phase1 Exp | `phase1_exp_hex/` |
| stage_2 | Phase1.5 sum | `phase15_sum_hex/` |
| stage_3 | Phase2 norm | `phase2_norm_hex/` |

**初始 UR**（仅输入 + 常量，无中间结果）：

- bank2 = raw x；bank5 = m 广播；bank6 = magic(x0)；bank1/4 = Exp coeff/const
- bank4 **@word8 = 2.0**（Newton；Exp 占 word0..7）

**片上数据流**（stage 间 bank 不回 Host）：

```text
P0: bank2(x) + bank5(m) → bank2(x')
P1: bank2(x') → bank3(e)        // bank4 Exp 常量保留
P2: bank3(e) → bank5(S 四 lane)
P3: bank5(S) → [@chip: magic→bank6] → Newton×3 → bank3(y)
```

**耦合要点**（2026-07-20 落地）：

1. **bank4**：Newton 2.0 从 word1 改到 **word8**，避免覆盖 Exp bias127@word1；**word9=MAGIC** 供 @chip magic 前导。
2. **bank5**：Sum 阶段 ADDTREE **dst_id=0..3 各写一次**，四 lane 广播 S（Phase2 SIMD Newton 需要）。
3. **bank6**：**e2e@host** 仍 Host 预载 magic(x0)；**e2e@chip**（`HOST_X0=0`）bank6 init=0，stage_3 片上算 x0 ✅。

**命令**：

```bash
cd aiacc/SMI_IQ
# 默认全片上 e2e
W=4 bash op/softmax/run_softmax_e2e_smoke.sh
# e2e@host 对比（显式 Host 预载 x0）
SOFTMAX_HOST_X0=1 W=4 bash op/softmax/run_softmax_e2e_smoke.sh
```

**状态**：✅ W=4 / **W=16** e2e 默认（全片上）PASS；Gate D W=16 ULP=0；e2e bank3 **max_ulp≤2**（与 multi_stage TB RELAXED 对齐，`SOFTMAX_E2E_MAX_ULP` 可改）。TB：`sim_multi_stage/run_sim_aiacc_top_f_multi_stage.sh`。

---

## 5.2 多 word Sum / Norm（2026-07-20，v4 已闭环）

> **已修复**：Phase1.5 跨 word 累加；根因是 ADDTREE **DR sticky** + bank6 **`wbe=0xFFFF`** 污染。  
> **验收**：Gate D bank5 **ULP=0**；e2e bank3 **max_ulp≤2**（默认）；改代码后 **必跑 W=16**，再跑 W=4 回归。

### 测试结果快照（W=16，`words=4`，`steps=4`）

| 测试 | 结果 | 说明 |
|------|------|------|
| Gate C Phase0 | ✅ 16/16 ULP=0 | max-sub 多 word 正常 |
| Gate D Sum | ✅ ULP=0 | bank5/bank6：`S≈2.42241`；bank6=`[S,0,0,0]` |
| e2e 默认 @chip | ✅ 16/16 max_ulp≤2 | y@bank3；实测 max_ulp=2 |
| W=4 e2e 回归 | ✅ 4/4 ULP=0 | steps=1 单 word 未破坏 |

历史失败（v3，已修）：

```text
GateD-sum-bank5-lane0: FAIL rtl=27.6180878 gold=2.42241383
# bank6 曾被污染为 (6.97, 6.88, 6.88, 6.88)
```

### 根因（已确认）

1. **FPADD 是 4-lane SIMD**：不能用 FPADD 做跨 word 标量累加。
2. **e2e 时 bank5@0 不能当 acc**：Phase0 后 bank5 仍是 **m**；acc 放 **bank6@0**。
3. **ADDTREE DR sticky**（关键）：`dst` 只写对应 32bit，**其它 lane 保留旧值**。广播 dst=0..3 后 DR=`[S,S,S,S]`；若再用 `wbe=0xFFFF` 写 bank6，下一拍 `ΣS1` 把旧 acc 加三遍 → rtl≈27.6。
4. **v4 修复**：bank6 写回 **仅 `wbe=lane0 (0x000F)`**，保持 `[acc,0,0,0]`。

### 方案演进

| 版本 | 思路 | W=16 Gate D |
|------|------|-------------|
| v1 | FPADD + bank6 | rtl≈0.09 |
| v2 | 修 fpadd 时序 | rtl≈27.6 |
| v3 | ADDTREE acc@bank6 + 广播；`wbe=FFFF` | ❌ rtl≈27.6 |
| **v4** | 同上 + bank6 **仅 wbe lane0** | ✅ ULP=0 |

### Norm 侧（e2e W=16 已验）

- `ur_r2` **`addr_inc=false`**：magic/Newton 读 bank6@0。
- `ur_r0` **`para_ur_r0_inc_from_cycle`**：仅最终 `y=e×inv` 随 word 递增。

### 必跑命令（改 Sum/Norm 后）

```bash
cd aiacc/SMI_IQ
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd

W=16 bash op/softmax/run_softmax_gate_cd_smoke.sh
W=16 bash op/softmax/run_softmax_e2e_smoke.sh
W=4  bash op/softmax/run_softmax_e2e_smoke.sh
# 更严：SOFTMAX_E2E_MAX_ULP=0 W=16 bash op/softmax/run_softmax_e2e_smoke.sh
```

### 下一步

1. 扩 W=100 / W=101 e2e。
2. 可选：Sum body 末 word 再广播（省每 word 4×ADDTREE）。

### 已知坑

- **勿 `--rebuild` multi_stage VCS**；复用已有 `simv`。
- Gate D 以 bank5 **ULP=0** 为准；e2e smoke 默认 **max_ulp≤2**（对齐 TB RELAXED）。

---

## 5.3 Gate D / ADDTREE8 调试经验（2026-07-20）

### FP32ADDTREE8TO1 规格（Softmax 用法）

CRU 8-bit：`[0]` vld，`[2:1]` dst_id（0–3 → DR 128bit 中 4 个 32bit lane），`[7:3]` smc_id_en。

```
DR_FP32ADDTREE8TO1_D[dst_id] = sum(
  S0[0..3], S1[0..3])    // 共 8 路 FP32
```

Softmax Phase1.5：

- **S0** = `e@bank3` 四字（rtovr-32 ← UR R0）
- **S1** = 0（rtovr-33 ← tied_zero）
- **dst_id=0** → `word_sum` 在 DR lane0
- 单 word：`word_sum` 经 **RtoVR-35 → UR-W0** 直写 **bank5@0**

### 根因：RtoVR 连接矩阵限制

| 路径 | 是否合法 | 结果 |
|------|----------|------|
| ADDTREE DR (in28) → **UVR-W0** (rtovr-35) | ✅ | 直写 bank5，S≈1.553 |
| ADDTREE DR (in28) → **FPADD-S0** (rtovr-5) | ❌ | FPADD-S0 仅允许 in12/13 (VIN1/VOUT0) |
| UR R0/R1 → FPADD-S0/S1 | ✅ | Phase0 / 多 word 累加走此路 |

因此 **不能** 把 ADDTREE 输出直连 FPADD。多 word 累加：**acc@bank6 读回 → 与 e 一起再进 ADDTREE（4+4）→ 写回 bank6**（见上节）。

### Phase1.5 ADDTREE 数据通路（输出位宽 / acc / 下一拍输入）

SMC 集成版 ADDTREE8（Softmax 实际跑的）：

| 项 | 规格 |
|----|------|
| 输入 S0 / S1 | 各 **128 bit** = 各 **4×FP32** |
| 语义 | `result = Σ S0[0..3] + Σ S1[0..3]`（8 个 FP32 求和） |
| 输出 DR | **128 bit** 总线；**有效结果只有 1 个 FP32（32 bit）** |
| `dst_reg_enc` | 0..3：把这 32bit **写入** DR 的对应 lane；**其余 lane 不清零（sticky）** |

```text
# RTL：o_dr 只更新 case(dst) 的那 32bit，其它 bit 保持旧值
# 连续 dst=0,1,2,3 广播后 DR 实际是 [S,S,S,S]，不是「其它 lane=0」

期望写入 UR 的布局（靠 wbe 保证，不是靠 DR 自清）:
  bank6@0:  [ acc | 0 | 0 | 0 ]   ← 写回时 wbe 只开 lane0（0x000F）
  bank5@0:  [  S  | S | S | S ]   ← 四次 dst + 每次 wbe 只开对应 lane
```

**acc 存哪**

- 标量 running sum 放 **bank6 @ word0**，布局必须是 `[acc, 0, 0, 0]`。
- pack Gate D / e2e 已对 bank6 **清零**；Sum 的 `ur_r1` / `ur_w0` **`addr_inc=false`**（acc 固定 @0）。
- bank3 的 e：`ur_r0` **`addr_inc=true`**，随 word 递增。
- **不要**用 bank5 当 acc（e2e 里 Phase0 后 bank5 仍是 m）。
- **禁止**对 bank6 用 `wbe=0xFFFF` 整字写 ADDTREE DR：广播后 DR 污染 lane1..3，下一拍 `ΣS1` 会把旧 acc 加 3 遍（W=16 曾 rtl≈27.6）。

**下一 word 怎么再进 ADDTREE**

```text
每 word（EMBED 回放）:
  读 e@bank3（本 word）、acc@bank6@0
  S0 ← e 的 4 lane
  S1 ← [acc, 0, 0, 0]     # 依赖 bank6 其它 lane 仍为 0
  → tree 输出 = acc + (e0+e1+e2+e3)  （写入 DR 的 dst=0 lane）
  写回 bank6@0：wbe=0x000F，只更新 lane0 → bank6 仍 [new_acc,0,0,0]

同 word 内再广播到 bank5（最后一次为准即可）:
  S0=[acc,0,0,0], S1=0，dst=0..3 各发一次
  每次 UR 写 bank5 用对应 lane 的 wbe（保留其它 lane 的 S）
```

单字（`steps=1`）：无 acc，直接 `S0=e, S1=0`，四次 dst 广播到 bank5。


### Golden 对齐

- 求和顺序须匹配 **8 输入树形加法**（非 Python 顺序 `sum()`），否则差 1 ULP。
- `pack_softmax_rtl_case.py` 中 `fp32_addtree8_word_sum()` 与 RTL 一致。
- **Gate D / e2e**：bank5 golden 为 **四 lane 广播 S**（ADDTREE dst_id=0..3 各写一次）。

### Phase0 同类经验

FPADD 输入必须走 **RtoVR-5/6**（不是 FPMUL 的 rtovr-3/4）。写回对齐 Phase0：`fpadd_dr + 4` 拍后再 UR-W。

---

## 6. 文件与工具

| 文件 | 作用 |
|------|------|
| [`softmax_op.h/cpp`](./softmax_op.cpp) | 编排：Phase0/1/1.5/2 微码 + `generate_data` |
| [`softmax_golden.h`](./softmax_golden.h) | Host 黄金 + pad 工具 |
| [`emit_softmax_stages_main.cpp`](./emit_softmax_stages_main.cpp) | 导出 IQ hex |
| [`run_softmax_fp32_smoke.sh`](./run_softmax_fp32_smoke.sh) | Gate A+B emit + pack + RTL |
| [`run_softmax_gate_cd_smoke.sh`](./run_softmax_gate_cd_smoke.sh) | Gate C+D smoke |
| [`run_softmax_e2e_smoke.sh`](./run_softmax_e2e_smoke.sh) | 四 Phase e2e（multi_inst_stage） |
| [`run_softmax_magic_x0_smoke.sh`](./run_softmax_magic_x0_smoke.sh) | Gate Magic：片上 magic(x0)→bank6 |
| [`pack_softmax_rtl_case.py`](./pack_softmax_rtl_case.py) | phase0 / phase1_exp / phase15_sum / phase2_norm / e2e / magic_x0 |
| [`measure_softmax_limit.py`](./measure_softmax_limit.py) | 相对 FP64 极限误差扫掠 |
| [`softmax_host_golden.py`](./softmax_host_golden.py) | Python 同源 golden |
| [`GOLDEN.md`](./GOLDEN.md) | 理论误差预算 |
| [`STATUS.md`](./STATUS.md) | 进度快照 |

### emit 输出目录（统一 Phase 命名）

```
out/softmax_*_w<W>/
├── phase0_hex/        # Phase0 max-sub（Gate C）
├── phase1_exp_hex/    # Phase1 Exp（Gate A）
├── phase15_sum_hex/   # Phase1.5 sum（Gate D）
├── phase2_norm_hex/   # Phase2 Norm（Gate B / Magic）
├── alpha.bin          # bank2 数据
├── golden.bin         # Host softmax 输出
└── ...
```

> 旧名 `stage0_hex`/`stage1_hex`/`sum_hex` 已废弃；pack 读时仍可 fallback。

---

## 6.0 Phase / stage、hex 拼装、仿真怎么跑

本节汇总：命名、e2e 多 stage 机制、与 Exp 的差异、脚本用法、IQ 深度。

### 6.0.1 命名：Phase vs stage（不要混）

算法只有 **4 个 Phase**。`stage_N` 是仿真 TB 的目录/触发序号，不是另一套算法。

| Softmax Phase（算法） | emit 目录 | 分 Gate case（`verification_v2/`） | e2e TB 目录 |
|----------------------|-----------|-------------------------------------|-------------|
| Phase0 max-sub | `phase0_hex/` | `softmax_fp32_p0_w*` | `smi_iq_inst/stage_0/` |
| Phase1 Exp | `phase1_exp_hex/` | `softmax_fp32_phase1_exp_w*` | `smi_iq_inst/stage_1/` |
| Phase1.5 Sum | `phase15_sum_hex/` | `softmax_fp32_phase15_sum_w*` | `smi_iq_inst/stage_2/` |
| Phase2 Norm | `phase2_norm_hex/` | `softmax_fp32_phase2_norm_w*` | `smi_iq_inst/stage_3/` |

```text
算法:   Phase0  →  Phase1(Exp)  →  Phase1.5(Sum)  →  Phase2(Norm)
emit:   phase0     phase1_exp        phase15_sum       phase2_norm
e2e TB: stage_0    stage_1           stage_2           stage_3
```

- **Phase**：文档/代码主用语。
- **stage_N**：`multi_inst_stage` TB **硬性**要求 `smi_iq_inst/stage_0` 起连续编号；内容来自对应 `phase*_hex`。
- 旧 emit 名 `stage0_hex`/`stage1_hex`/`sum_hex` 已废弃（pack 可读时 fallback）。

### 6.0.2 e2e 执行模型（多 stage）

```text
Host 一次加载 UR（x@bank2, m@bank5, 常量@bank1/4…）
  → load stage_0 IQ → trigger → 等完成   # Phase0 写出 x'
  → load stage_1 IQ → trigger → 等完成   # Exp（merged P1|P2）写出 e
  → load stage_2 IQ → trigger → 等完成   # Sum 写出 S
  → load stage_3 IQ → trigger → 等完成   # Norm 写出 y
  → dump UR（看 bank3）
```

要点：

1. **4 个 Phase ≈ 4 次 TB stage trigger**，不是「拆成 4 条队列」当主结构。
2. **每个 stage 内部**仍有多条 IQ 队列（`ur-r0`、`addtree8`、`rtovr-*`…），一次加载、一起跑完。
3. **段间不清 UR**：bank 中间结果接力（x'→e→S→y）。
4. Softmax **Phase1** = 直接调用 `ExpOp::compile_merged`（Exp 的 P1+P2 已合成一段），不会再拆成两个 TB stage。

### 6.0.3 「等中断」是谁写的？怎么知道写好了？

**不是 Softmax 微码里写的「等中断」**，是 multi_stage **仿真 TB** 在做；完成信号由 **芯片硬件** 置 CSR。

| 谁 | 干什么 |
|----|--------|
| Softmax Phase 微码（hex） | 只算：读 bank、算、写 bank；**不写等中断** |
| 芯片 RTL（DUT） | 本 stage 全部 IQ 跑完后，置 **INT @0x004** / **smi_iq_done @0x024** |
| 仿真 TB | `smi_wait_and_clear_intr`：轮询 CSR → ack → 再 load 下一段 |

TB 文件：`sim_multi_stage/tb_aiacc_top_v04_f_multi_stage.v`（约每 20 拍读一次 CSR，超时约 50 万拍）。

Softmax 侧只需保证每段 IQ 合法、能跑完；**不必**在微码里发完成指令。

### 6.0.4 IQ 深度上限

- 硬件 / Softmax 约定：**每个 IQ 队列**深度 ≤ **128**（`kSoftmaxIqDepthLimit` / `SMI_IQ_INSTQ_DEPTH`）。
- **每一个 stage 各自**受此约束：stage 跑完后 TB 再 load 下一段 hex，又是新一轮「各队列 ≤128」，**不是**四个 stage 共用一个 128。
- 判定看的是 **该 Phase / 该 stage 里最深那一条队列**（`max_iq`），不是所有队列加总，也不是 4 个 Phase 相加。
- W=4 量级：Phase0≈4、Phase1 Exp≈24、Phase1.5≈10、Phase2≈17，均 ≪ 128。
- Exp 单独一次 Merged trigger：单字 MAX≈**24**、多字≈**25**（同样 ≪ 128）。

### 6.0.5 和 Exp 交付路径对比

| | Exp（交付） | Softmax e2e |
|--|-------------|-------------|
| 算法步骤 | P1 预处理 + P2 Horner | Phase0 / Exp / Sum / Norm |
| 仿真触发 | **1 次**（`compile_merged`） | **4 次**（`stage_0..3`） |
| case 位置 | `verification_v2/exp_*` | `multi_inst_stage/softmax_e2e_*` |
| 仿真脚本 | `run_sim_aiacc_top_f.sh` | `run_sim_aiacc_top_f_multi_stage.sh` |
| IQ 放置 | `smc_0/iq_*.hex` 一套 | `smi_iq_inst/stage_N/` 四套 |
| 段间 UR | 不适用（单段） | **持久、不清** |

Exp 历史上试过 multi_stage（备份里有 `run_exp_e2e_multi_stage.sh`）；**当前交付已改为 Merged 单流**。Softmax 全链路才是现在 multi_stage 的主用户。

### 6.0.6 hex 怎么拼起来

```text
① g++ 编译 emit_softmax_stages（Host 工具，不是每次重编 RTL）
② emit 写出 4 个 phase*_hex/ + alpha.bin / golden.bin
③ pack_softmax_rtl_case.py
     分 Gate：对应 phase*_hex → case/smc_0/iq_*.hex
              + ur_data/bank*.hex + output_bank/*_final.hex
     e2e：    4 份 hex → smi_iq_inst/stage_0..3/
              UR 只写一次
④ generate_data_list（*_multi_stage.py）→ 每 stage 的 data_list_inst_subq0.txt
⑤ 仿真（见下）
```

**不必手搬 hex**；smoke / pack 会 `copy`。

e2e case 结构示例：

```text
test_atomic_ops/multi_inst_stage/softmax_e2e_w4_<时间戳>/
├── smc_0/
│   ├── ur_data/bank*.hex
│   ├── output_bank/bank3_final.hex
│   └── ur_result_*_bank*.hex    # 仿真后才有
└── smi_iq_inst/
    ├── stage_0/ … stage_3/      # 各 Phase 的 iq_*.hex
    └── stages.md                # TB stage ↔ Phase 对照
```

### 6.0.7 脚本怎么用

#### 一键（推荐）

```bash
cd aiacc/SMI_IQ
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd

W=4  bash op/softmax/run_softmax_e2e_smoke.sh      # 四 Phase 串联（multi_stage）
W=16 bash op/softmax/run_softmax_gate_cd_smoke.sh  # Gate C+D（单段 TB）
bash op/softmax/run_softmax_fp32_smoke.sh          # Gate A+B（单段 TB）
W=4  bash op/softmax/run_softmax_magic_x0_smoke.sh # Gate Magic

# 只 emit+pack、不跑 VCS
SKIP_RTL=1 W=4 bash op/softmax/run_softmax_e2e_smoke.sh
```

`run_softmax_e2e_smoke.sh` 内部：emit → `pack … e2e` →  
`sim_multi_stage/run_sim_aiacc_top_f_multi_stage.sh multi_inst_stage/…/smc_0 -s 0` → 对比 bank3 ULP。

#### 手动（调试）

```bash
# pack（假设 out/softmax_e2e_w4 已有 phase*_hex）
python3 op/softmax/pack_softmax_rtl_case.py e2e \
  --case-root ../../test_atomic_ops/multi_inst_stage/my_softmax_e2e \
  --tvgen-out out/softmax_e2e_w4 \
  --w 4 --words 1 --no-host-x0

# 仿真（必须在 Jianyuan-aiacc_4.21 根目录）
cd /home/YouTsing/Jianyuan-aiacc_4.21
bash sim_multi_stage/run_sim_aiacc_top_f_multi_stage.sh \
  multi_inst_stage/my_softmax_e2e/smc_0 -s 0
```

#### 相关文件一览

| 路径 | 作用 |
|------|------|
| `op/softmax/run_softmax_e2e_smoke.sh` | Softmax 四 Phase 一键 e2e |
| `op/softmax/pack_softmax_rtl_case.py` | `phase0` / `phase1_exp` / `phase15_sum` / `phase2_norm` / `e2e` / `magic_x0`（旧名 `stage0`/`stage1`/`sum` 为别名） |
| `op/softmax/emit_softmax_stages_main.cpp` | 导出四个 `phase*_hex` |
| `sim_multi_stage/run_sim_aiacc_top_f_multi_stage.sh` | multi_stage 仿真入口 |
| `sim_multi_stage/generate_data_list_coverage_multi_stage.py` | 为 `stage_*` 生成 data_list |
| `sim_multi_stage/tb_aiacc_top_v04_f_multi_stage.v` | TB：发现 stage、trigger、等 INT |
| `run_sim_aiacc_top_f.sh`（repo 根） | **单段**仿真（Gate / Exp） |

#### 注意

- `-s 0` 是 subq / IQ sid，**不是** stage 编号；stage 由目录 `stage_0..3` 自动发现。
- 勿随便 `--rebuild` multi_stage VCS（KDB 易失败）；复用已有 `sim_output_aiacc_top_f_multi_stage/simv`。
- 分 Gate 与 e2e **不要混仿真脚本**：`verification_v2` → 普通 sim；`multi_inst_stage` → multi_stage sim。

---

### 6.1 RTL smoke 前置：VCS 与文件权限

跑 Gate A/B/C/D、e2e、magic-x0 等 **含 VCS 仿真** 的脚本前，先确认下面两项；否则常见现象是 **`Permission denied`**（pack 删不掉旧 case）或 **`Cannot connect to the license server`**（仿真 0 输出 / bank dump 全 0）。

#### VCS 环境变量

`vcs` 默认不在 PATH。仿真（含复用预编译 `simv`）前需 export（本机 R730xd，`O-2018.09-SP2`，license 局域网 server）：

```bash
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
```

建议写入 `~/.bashrc` 或跑 smoke 前执行一次。各 `run_softmax_*_smoke.sh` 在 **未检测到 `VCS_HOME`** 时会自动 export 上述变量；**手动**在 repo 根目录跑 `./run_sim_aiacc_top_f.sh` 时须自己 export。

| 检查项 | 命令 / 说明 |
|--------|-------------|
| VCS 在 PATH | `which vcs` 有输出 |
| License 可达 | `ping bupt-PowerEdge-R730xd`；`27000` 端口未被防火墙挡 |
| 复用 simv | 默认 **fast**，只改微码/case **不要** 加 `--rebuild`；改 RTL 才重编 |
| 内核告警 | `Unsupported Linux kernel '5.15...'` **可忽略** |
| 仅测 emit/pack | `SKIP_RTL=1 bash op/softmax/run_softmax_fp32_smoke.sh`（不依赖 VCS） |

仿真失败时看：`sim_output_aiacc_top_f/sim_output.log`（license 失败会有 `Cannot connect to the license server` / `Failed to obtain license`）。

#### 文件权限（不要用 root 跑 smoke）

smoke 会在 **`Jianyuan-aiacc_4.21/test_atomic_ops/verification_v2/`** 下 `rm -rf` 并重建 case 目录。若曾用 **`sudo bash op/softmax/run_*.sh`** 或 root 跑 VCS，产物属主会变成 **root**，后续普通用户 pack 会：

```text
rm: cannot remove '.../softmax_*_w4/...': Permission denied
```

**正确做法：**

1. **始终以普通用户**（如 `YouTsing`）跑 smoke，不要用 `sudo bash run_softmax_*.sh`。
2. 若已出现 root 属主，一次性改回（把 `<user>` 换成你的登录名）：

```bash
sudo chown -R YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/test_atomic_ops/verification_v2/softmax_*
```

3. 检查：`ls -ld test_atomic_ops/verification_v2/softmax_fp32_s0_w4` 应为 `YouTsing YouTsing`。
4. `run_softmax_magic_x0_smoke.sh` 会 pack 到带时间戳的子目录（如 `softmax_magic_x0_w4_HHMMSS`），避免与旧 root 目录冲突；改完属主后也可正常 `rm` 标准路径 `softmax_magic_x0_w4`。

**仅 emit、不跑 RTL** 时不受 VCS/license 影响：`SKIP_RTL=1 W=4 bash op/softmax/run_softmax_magic_x0_smoke.sh`。

### 常用命令

```bash
cd aiacc/SMI_IQ

# 完整 smoke（默认 Host x' + Host S）
bash op/softmax/run_softmax_fp32_smoke.sh

# 四 Phase e2e（Host 只写 x+m+常量；x'/e/S/y 全在片上）
SOFTMAX_HOST_MAX=0 SOFTMAX_HOST_SUM=0 bash op/softmax/run_softmax_e2e_smoke.sh

# Gate Magic：片上 magic(x0) → bank6（需 VCS + 见 §6.1 权限）
W=4 bash op/softmax/run_softmax_magic_x0_smoke.sh

# 仅 emit，跳过 VCS
SKIP_RTL=1 bash op/softmax/run_softmax_fp32_smoke.sh

# 指定 W
W=100 bash op/softmax/run_softmax_fp32_smoke.sh
W=101 SKIP_RTL=1 bash op/softmax/run_softmax_fp32_smoke.sh

# FP64 极限误差扫掠
python3 op/softmax/measure_softmax_limit.py --n-sweep 2000 --seed 20260720

# Host golden 分场景
python3 op/softmax/softmax_host_golden.py --w 4 --seed 20260720
```

---

## 7. Host Golden 与误差

链路（Host = RTL 验收基准）：

```text
m = max(x) → x' = x - m → halfint Exp → sum S → Newton×3(1/S) → mul → y
（FP16 I/O 末级 RN 到 half）
```

| 层级 | 门槛 |
|------|------|
| RTL vs Host | **ULP = 0**（目标） |
| FP32 vs FP64 | max rel ~**1.6×10⁻⁶**（Newton×3，130k 点） |
| FP16 vs FP64→f16 | ULP ≤ 1；下溢单独声明 |

详细扫掠：[`experiment_logs/softmax_limit_20260720_152636/`](./experiment_logs/softmax_limit_20260720_152636/)

---

## 8. max 决策（交付 vs 联调）

| 阶段 | 谁算 max | 说明 |
|------|----------|------|
| **联调 Gate A–B** | Host 预写 x' | 最快打通 Exp + Norm |
| **Gate C** | 芯片 sub；max 暂 Host | `compile_maxsub` ✅ RTL PASS |
| **Gate D** | 芯片 ADDTREE8 sum | ✅ W=4 / **W=16** ULP=0（§5.2 v4） |
| **最终交付** | 芯片 Phase0 全做 | 需片上 max reduce |

本芯片无专用 FP-max 单元；跨 lane 需 `LOGICAL SELECT_GREAT` 树，跨 word 需串行累加。建议顺序：

1. 单字 W=4：lane 内 SELECT_GREAT 树 → 广播 m@bank5  
2. 多字：字间 `m ← max(m, max(word_k))`  
3. Sub：`x - m` → bank2（**已实现** ✅）

---

## 9. 下一步

1. 扩 W=100 / W=101 e2e；可选：Sum 仅末 word 广播 bank5。  
2. 片上 max reduce（SELECT_GREAT，m 不再 Host 写 bank5）  
3. Sum body 优化（W=4 单 word 48 cyc；多 word 60 cyc/word）  
4. W=100 / W=101 e2e（Sum PASS 后）  
5. 8SMC 均分 W / LSU 路径  

LLAMA 验证路径见 [`LLAMA算子验证路径.md`](./LLAMA算子验证路径.md)。
