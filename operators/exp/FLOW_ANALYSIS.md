# Exp 1.0.5 计算流程与时延瓶颈分析

> **基线版本**：exp1.0.5（半整数 + k-flush）  
> **备份**：[`backup_exp1.0.5_pre_flow_opt_20260723/`](./backup_exp1.0.5_pre_flow_opt_20260723/)  
> **目的**：在改流程前，厘清**模块顺序**、**RtoVR 建连传数**、**纯计算 vs 传数/等待**占比，判断受限类型与可优化空间。

---

## 0. 总览（一句话）

每字墙钟 **163 cyc = P1 97 + P2 66**（原 186=108+78；经 P2 stage 压缩 + P1 魔法数 RN）。  
硬件是 **单发射、单 FPMUL/FPADD 流水线 + UR 双读口约束**；P2 已压 stage 尾空等；P1 已去掉 floor 修正链。  
→ **受限类型：仍偏延迟/互连（UR/+6/flush），但相对 186 已明显靠近计算填充；再砍看 flush 口型或 gap 6→5。**

---

## 1. 端到端计算流程（算法 → 硬件）

### 1.1 算法（Host / RTL 同构）

```text
x (FP32)
  → t = x · log2(e)
  → tw = 2t；n = RN(tw)（**Host/RTL**：1.5·2²³ 魔法数，`bits(tw+MAGIC)−MAGIC`）
  → m = n>>1；odd = n&1；r = (tw − n)/2 ∈ [-0.25, 0.25]
  → scale0 = 2^m；scale = odd ? scale0·√2 : scale0
  → k-flush：k≤−126 → 0；k≥127 → +Inf
  → P = Horner(r; c0..c4) ≈ 2^r
  → y = scale · P
```

### 1.2 硬件两段（Merged 单流）

| 段 | 源码 | body | 产出 |
|----|------|------|------|
| **P1 Preprocess** | `exp_preprocess_op.*` | **97** | `r`→bank0；`scale`→bank1@5+ |
| **P2 Poly** | `exp_poly_kernel_op.*` | **66** | `y`→bank3 |
| **Merged** | `exp_op.*` + `emit_exp_embed_iq` | **163**/word | P1\|P2 同体 unroll |

多字：整段 `emit_exp_embed_iq(P1\|P2)` 按 word 回放；**禁止**「P1 后再嵌 embed=1 只回放 Horner」。

### 1.3 UR Bank 数据布局

| Bank | 内容 | 谁写 / 谁读 |
|------|------|-------------|
| bank2 | 输入 `x` | Host → P1 |
| bank4 | 常数（log2e/bias/√2/−126/127/+Inf…） | Host → P1 |
| bank1 @0..4 | Remez 系数 | Host → P2 |
| bank0 | `r`（+ scratch k@200、scale 暂存@201） | P1 → P2 |
| bank1 @5+ | 每 word `scale` | P1 → P2 Final |
| bank3 | 输出 `y` | P2 → Host/下游 |

交付仿真：**Host 预灌 UR**（Compute 路径）；DDR↔UR 的 LSU 是另一条线，不计入本 186。

---

## 2. RtoVR：建立连接与传数过程

### 2.1 角色

`RtoVR` = **结果总线 / 数据交叉开关**：把某一路 **源**（UR 读口、FPMUL/FPADD/LOGIC/INTADD 的 DR、tied-zero…）接到某一路 **目的**（FPMUL/FPADD 操作数口、UR 写口）。

```text
源 DR / UR-Rdata / tied-zero
        │
        ▼
   ┌─────────┐   路由寄存器 (+1 cyc)
   │  RtoVR  │ ──select 持续整窗──► 目的口（FPMUL S0/S1、FPADD S0/S1、UR-W …）
   └─────────┘
        │
        ▼ （若写 UR）
   UR-W settle (+1) → bank SRAM 可读
```

关键延迟（`exp_iq_utils.h`）：

| 项 | 拍数 |
|----|------|
| UR 读 issue → 计算 issue | **+3** |
| 浮点 A issue → 浮点 B issue（B 吃 A 的 DR） | **+6**（含 RtoVR+采样+槽对齐） |
| RtoVR 路由 | **+1** |
| RtoVR → UR-W 写回 settle | **+1** |
| FPMUL / FPADD 裸 DR（RTL） | **3** |
| LOGIC | **2**；INTADD / 类型转换 | **1** |

### 2.2 编码与「建连」方式

微码每拍对每个 RtoVR 输出口发：

```text
code = 1 | (source_id << 1)     // rtovr_code(source_id)
```

**必须整窗持续选通**（不能 one-shot）：Sigmoid/Exp Horner 经验——只打一拍 select 不够，consumer 采样会吃到旧源。

同源长窗口可被 `EXP_COMPRESS_NOP_LOOPS` 压成 `1 条 + loops=(N−1)`（不推进读地址，同码重放）。

### 2.3 常用端口映射（Exp）

| RtoVR 口 | 含义（`exp_iq_utils.h`） |
|----------|--------------------------|
| in 3 / out 3–4 | FPMUL DR → FPMUL 操作数 |
| in 4 / out 5–6 | FPADD DR → FPADD 操作数 |
| in 30+port | UR-Rdata₀.. → 计算口 |
| in 34 | **tied-zero**（硬绑 0，Horner stage0 清 ACC） |
| out 35 | → **UR-W0** 写数据 |

### 2.4 P2 Horner 典型连接（每 stage 13 拍 × 5 + Final 13）

```text
stage s:
  cyc0  UR-R1←r，UR-R2←c_s
  全程  RtoVR-3 ← (s==0 ? tied-zero : FPADD_DR)   // ACC 初值或上一级 P
        RtoVR-4 ← UR-R1(r)
        RtoVR-5 ← UR-R2(c)
        RtoVR-6 ← FPMUL_DR
  cyc3  FPMUL：  ACC·r  （或 0·r）
  cyc9  FPADD：  (ACC·r) + c
Final:
  UR-R0←scale
  RtoVR-3←FPADD_DR(P)，RtoVR-4←UR-R0(scale)
  FPMUL@3： y = P·scale
  RtoVR-35←FPMUL_DR → UR-W@10 → bank3
```

**传数路径本质**：DR → RtoVR(+1) → 下一单元口 / UR-W(+1 settle) →（再被读则 +3）。  
中间大量拍是 **保持选通 + 等管道**，不是新一次算术。

---

## 3. 模块使用顺序（按时间轴）

### 3.1 P1（97 cyc）— RN 魔法数 + √2 + k-flush

模块出场顺序（摘要，相位见 `exp_preprocess_op.cpp` 常量）：

| 相位段 | 模块（先后） | 算法步骤 |
|--------|--------------|----------|
| 0–9 | UR → **FPMUL** → **FPADD** | `t=x·log2e`；`tw=t+t` |
| 15–21 | **FPADD** → **INTADD** | `y=tw+MAGIC`；`n=bits(y)−MAGIC`（**无 FPtoINT/floor**） |
| 22–41 | FPADD / FPMUL / LOGIC / INTADD / UR-W | 恢复 `tw`；`r=(tw−n)/2`；`m=n>>1`；组装 `2^m` |
| 42–50 | FPMUL / LOGIC SELECT | odd·√2 → scale |
| 53–94 | UR scratch / INTtoFP / FPADD 比较 / **SELECT_LESS**×2 / UR-W | **uflow→0，oflow→+Inf**；写 scale@bank1 |

P1 用到的 IQ 通道：`UR-R0/1/2`、`UR-W0/1`、`FPMUL`、`FPADD`、`INTADD`、`LOGIC`、`INTtoFP`（flush/组装），以及 **RtoVR-\***；**已不再使用 FPtoINT**。

相对旧 floor 版（P1=108），魔法数 RN 去掉：
`tw+0.5 → FPtoINT → INTtoFP → check → SELECT → INTADD` 整段，**−11 cyc**（108→97）。

### 3.2 P2（66 cyc）— Horner + Final

| 段 | 模块 | 次数 |
|----|------|------|
| Horner ×5 | UR-R + **FPMUL@3** + **FPADD@9**，**stage=11** | 每 stage 11 cyc（原 13） |
| Final | UR-R(scale) + **FPMUL** + UR-W@9，**11 cyc** | 原 13 |

纯浮点算术 issue：**5×(MUL+ADD) + 1×MUL = 11 次**；  
墙钟 66（原 78）：RtoVR **短持到 consumer**，stage 尾空等砍掉 2×5+2 拍。

### 3.3 流程压缩（2026-07-23）

| 改动 | 效果 |
|------|------|
| P2 `kStageCycles` 13→**11**，`kFinalCycles` 13→**11**，写回 10→**9** | 墙钟 **78→66**（−12） |
| Horner RtoVR 短持到 consumer | 少空选通 |
| P1：`⌊tw+0.5⌋` → **RN 魔法数**（`tw+MAGIC` + `bits−MAGIC`） | **108→97**（−11）；Host/RTL 对齐 |
| P1：`x/log2e` 选通短持到 FPMUL@3 | 减无用 UR 路由窗 |
| **仍硬** | UR 读+3、浮点级联+6、写 settle+1；k-flush 两段 SELECT |

合并墙钟：**186 → 163**（P1 97 + P2 66）。

---

## 4. 时间拆分：纯计算 vs 传数 / 等待

### 4.1 口径定义

| 类别 | 含义 |
|------|------|
| **算术占用** | FPMUL/FPADD/INTADD/LOGIC/转换 **issue 拍**（单元真正开工） |
| **传数 / 互连** | RtoVR 选通保持、UR 读延迟(+3)、写 settle(+1)、DR→下一单元 gap(+6 中非裸延迟部分) |
| **结构气泡** | 同 bank 分拍、避 DR 冲突串行、flush 对齐口型、stage 固定 13 拍槽 |

墙钟 = 算术占用 ∪ 气泡（重叠在时间轴上）；下面用 **issue 次数 / 墙钟** 估负载。

### 4.2 粗算（每 word）

| 段 | 墙钟 | 粗估算术 issue | 算术占比（粗） | 其余（传数+气泡） |
|----|------|----------------|----------------|-------------------|
| P1 | **97** | ~18–28（无 floor 链转换） | **~20–30%** | **~70–80%**（flush 仍大） |
| P2 | **66** | 11（5×MUL+5×ADD+1×MUL） | **~17–45%** | **~55–80%** |
| **合计** | **163** | — | **约 ¼～⅓ 在「算」** | **flush + UR/+6 仍主气泡** |

更硬的说法（与调度规则一致）：

```text
一次「UR 读 → MUL → ADD → 结果可再被吃」≈ 3 + 6 = 9 拍量级
其中裸 FPMUL+FPADD ≈ 3+3=6，多出来的 ≈ RtoVR + 采样 + 槽对齐
写回 UR 再读：再 +1 settle +3 读 = +4
```

→ **单次依赖边的开销接近甚至超过一次浮点运算本身。**

### 4.3 与系统级传数的区分

| 路径 | 是否计入 186 | 说明 |
|------|--------------|------|
| 片上 RtoVR / UR | **是** | Compute 墙钟主体 |
| Host 灌 bank hex | **否** | 仿真预载 |
| LSU DDR↔UR | **否** | `load_store_v3`；真机端到端另算 |

真机若 LSU 带宽不足，可能变成 **片外带宽受限**；当前交付口径的 186 **只反映片上 Compute**。

---

## 5. 受限类型结论

| 层级 | 判定 | 依据 |
|------|------|------|
| **片上 Compute（本 163）** | **仍偏延迟/互连；已改善** | floor 已砍；k-flush + UR/+6 仍主导 |
| **非** | 算术吞吐打满 | 单 FPMUL/FPADD，Horner 串行 |
| **下一步** | 压缩 flush 口型 /（可选）gap 6→5 | 预期再砍数～十余拍 |

一句话：**不是算力不够，是「算一下要等好几拍才能传到下一个模块」+「避 UR/DR 冲突被迫拉长」。**

---

## 6. 优化整体实现时间的空间（按收益预期）

### 6.1 算法侧（改 186 组成）

| 方向 | 预期 | 风险 |
|------|------|------|
| 换更短 P1（如 int-k / Schraudolph / 少 flush） | P1 97→~50–70 | 精度掉档；flush 语义要另处理 |
| **P1 魔法数 RN（已做）** | **108→97** | Host/RTL 对齐 |
| Remez 降阶 / 分段短 Horner（`remez_seg4`） | P2 66→~50–60 | 系数 ROM↑；需新微码 |
| Softmax 场景放宽 vs Math | 可砍 flush 或缩域 | 仅当产品接受 |

### 6.2 微码 / 调度侧（不改算法）

| 方向 | 预期 | 说明 |
|------|------|------|
| 压缩 P1 flush 口型（少一次 UR 绕行） | 数～十余拍 | 需严格对齐 SELECT 语义 |
| P1 内更多并行（异 bank 同拍读已用） | 有限 | 同 bank 双读仍禁 |
| P2 stage 从 13 压到 10–11 | **已做 11**（−12 cyc） | 可再试 10，需 RTL 冒烟 |
| P1 RtoVR 无用长窗 | **已收 x/log2e** | 继续收其它窗不减 body |
| 恢复 ACC fold（历史路径） | 不一定更快 | 曾有字间泄漏；已弃用 unroll |

### 6.3 架构侧（大改）

| 方向 | 预期 |
|------|------|
| 双 FPMUL 或 FMA | Horner 吞吐↑，改 RtoVR 矩阵 |
| UR 同 bank 双读 | P1 分拍气泡↓ |
| 写后旁路（bypass DR→下一拍） | settle 链↓ |

### 6.4 建议优化顺序（结合你现在的目标）

1. **先画清本文件的时间账**（已完成口径）→ 定「砍 P1 还是砍 P2」。  
2. **算法选型**：用 `exp_cluster` 短名单换短路径，再估新 body。  
3. **再动微码**：在选定算法上压 RtoVR 窗 / stage 长度。  
4. 真机再补 **LSU vs Compute** 占比，看端到端是否变带宽受限。

---

## 7. 流程图（Merged 一字）

```text
Host 灌 UR
   │
   ▼
┌──────────────── P1 97 cyc ─────────────────┐
│ UR-R(x,const) ─RtoVR─► FPMUL/FPADD/INTADD  │
│      │                                      │
│      ▼                                      │
│ RN magic / odd·√2 / k-flush                 │
│      │                                      │
│      ▼                                      │
│ UR-W：r@bank0，scale@bank1                  │
└──────────────────┬─────────────────────────┘
                   │ 片上 UR（无 Host 搬运）
                   ▼
┌──────────────── P2 66 cyc ─────────────────┐
│ 5×(UR-R → RtoVR短持 → FPMUL → RtoVR → FPADD)│
│ Final：P × scale → RtoVR-35 → UR-W y@bank3 │
└────────────────────────────────────────────┘
                   │
                   ▼
              dump / 下游 Softmax
```

---

## 8. 关键文件索引

| 内容 | 路径 |
|------|------|
| 延迟常量 / RtoVR 编码 / emit | `exp_iq_utils.h` |
| P1 相位表 | `exp_preprocess_op.cpp` |
| P2 Horner 调度 | `exp_poly_kernel_op.cpp` |
| Merged 入口 | `exp_op.cpp` |
| 历史延迟专章 | `backup_exp1.0.3_20260718/README.md` §4.2 |
| 交付 README | `README.md` |
| 本分析前备份 | `backup_exp1.0.5_pre_flow_opt_20260723/` |

---

*文档用途：流程优化账本；改微码后请同步更新 §3–§4 相位与墙钟拆分（现行 **163=97+66**）。*
