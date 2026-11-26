# Softmax Phase0a：片上 Max Reduce 两种方案

**范围**：Phase0a（Gate Max）——一行 W 个 FP32 求 \(m=\max_j x_j\)，广播写 **bank5**。  
**代码**：`softmax_op.cpp` · `compile_max_reduce()` · `SoftmaxMaxReducePhaseBMode`  
**对比工具**：`compare_max_reduce_phase_b_main.cpp`  
**讲稿**：[`SOFTMAX_MAX_REDUCE_TALK.md`](./SOFTMAX_MAX_REDUCE_TALK.md)（汇报用口播稿）

Phase0a 拆为 **Phase A（列 tournament，与 W 相关）** + **Phase B（4 lane 标量 tournament，固定 4 lane）**。两种方案 **Phase A 相同**，差异仅在 **Phase B**。

---

## 1. 共同前提：Phase A

| 项 | 说明 |
|----|------|
| 输入 | raw x @ **bank2**，按 word 读（每 word 4 lane） |
| 累加器 | **bank6@0**，pack 预置 `-inf` |
| 操作 | 每 word：`FPADD(update_status)` + `LOGICAL SELECT_GREAT`，同 lane 取 max |
| 结果 | `bank6@0 = [c0, c1, c2, c3]`（4 个列 max，各占一 lane） |
| 周期 | **15 cyc/word**（`kSmMaxCompareSpan=13` + 2） |

**硬约束**：`SELECT_GREAT` 只能 **同 lane** 比较（lane i 对 lane i），不能直接 `max(c0,c1)`。

```text
Phase A 每 word:
  UR_R0 ← x@bank2[wi]
  UR_R1 ← acc@bank6@0
  FPADD(check) → status
  LOGIC SELECT_GREAT(s0=acc, s1=word) → 较大者
  UR_W0 → bank6@0
```

---

## 2. ADDTREE8TO1 硬件语义（广播/抽 lane 的基础）

模块：`cells/fp32_addtree_8to1/` · IQ：`Fp32AddTree8To1Iq`

| 项 | 说明 |
|----|------|
| 输入 | **S0**、**S1** 各 128b，拼成 **8 个 FP32** 做加法树 |
| 输出 | **一个** FP32 求和结果 |
| `dst_reg_enc` (0..3) | 和写到 DR 的 **哪一个 32b lane**（**不是**四 lane 广播） |
| 流水 DR | **sticky**：多次 issue 只更新目标 lane，其它 lane 保留（见 pipelined RTL） |

**没有 hardware broadcast 指令**。Softmax 用「8 输入求和 + 单 lane 写回」**模拟**抽 lane / 广播。

---

## 3. 方案 A：`EXTRACT_BROADCAST`（当前默认，已 Gate 验证）

**枚举**：`SoftmaxMaxReducePhaseBMode::EXTRACT_BROADCAST`

### 3.1 数据流

```text
Phase A 结束: bank6@0 = [c0, c1, c2, c3]

Phase B:
  ① extract_lane ×4（FPMUL 掩码 + ADDTREE8）
     bank6@20..23 = [c0,0,0,0], [c1,0,0,0], [c2,0,0,0], [c3,0,0,0]

  ② 每次 pairwise max 前 broadcast_scalar ×2
     [ck,0,0,0] → [ck,ck,ck,ck]（ADDTREE8 + wbe×4）

  ③ Tournament（3× max_scalar_pair）:
     max(c0,c1) → m01@25
     max(c2,c3) → m23@26
     max(m01,m23) → m@27

  ④ broadcast_m_to_bank5: m → bank5 四 lane
```

### 3.2 `emit_extract_lane`（FPMUL + ADDTREE 抽单 lane）

**目标**：`bank6@0` 的 lane k → `bank6@dst` 的 `[vk,0,0,0]`

```text
UR_R0 ← bank6@0           （[c0,c1,c2,c3]）
UR_R1 ← bank4@(10+k)      （lane k 掩码：1,0,0,0 / 0,1,0,0 / …）
FPMUL   → 掩码后仅 lane k 非零
S0 ← FPMUL 结果, S1 ← 0
ADDTREE8 dst=0            → sum = vk（其余 7 输入为 0）
UR_W0 wbe=lane0           → [vk,0,0,0]
```

**周期**：约 **12 cyc/次** × 4 = **48 cyc**

### 3.3 `emit_broadcast_scalar`（ADDTREE 伪广播）

**目标**：`[v,0,0,0]@src` → `[v,v,v,v]@dst`

```text
loop dst = 0..3:
  UR_R1 ← src word          （读 [v,0,0,0]）
  S0 ← UR_R1,  S1 ← 0       （8 输入 = [v,0,0,0,0,0,0,0]）
  ADDTREE8 dst_reg_enc=dst  → sum = v，结果写到 DR 的 lane dst
  UR_W0 wbe=lane(dst)       → 只更新 dst word 的第 dst 个 lane
```

4 次循环后 UR word = `[v,v,v,v]`。

**周期**：约 **12 cyc/lane** × 4 = **48 cyc/次 broadcast**

**为何不用 LOGIC COPY / UR 整 word copy？**  
二者都是 lane i ← lane i，`[v,0,0,0]` 拷贝后仍是 `[v,0,0,0]`，无法跨 lane 扩播。

### 3.4 `emit_max_scalar_pair`

```text
broadcast_scalar(src_a) → scratch@24 = [va,va,va,va]
broadcast_scalar(src_b) → scratch@29 = [vb,vb,vb,vb]
SELECT_GREAT(Row24, Row29) → 同 lane max → 写 lane0 @dst
```

**周期**：约 **111 cyc/次**（48+48+15）× 3 = **333 cyc**

### 3.5 Phase B 周期汇总（方案 A）

| 子阶段 | 周期 |
|--------|------|
| extract_lane ×4 | 48 |
| max_scalar_pair ×3（含 6× broadcast） | 333 |
| broadcast_m → bank5 | 48 |
| **Phase B 合计** | **429** |

### 3.6 UR Bank 布局

| 地址 | 内容 |
|------|------|
| bank6@0 | Phase A 结果 `[c0,c1,c2,c3]` |
| bank4@10..13 | lane 掩码（pack 预置） |
| bank6@20..23 | `[ck,0,0,0]` 标量 |
| bank6@24,29 | broadcast scratch A/B |
| bank6@25..27 | m01, m23, m |
| bank5@0 | 最终 `[m,m,m,m]` |

### 3.7 工程状态

| 项 | 状态 |
|----|------|
| 微码 emit | ✅ `emit_softmax_max_reduce_main.cpp` |
| Gate RTL | ✅ W=4/16 SKIP_RTL PASS |
| IQ 深度 W=4 | max_iq=74, fit=1 |

---

## 4. 方案 B：`DS_TRANSPOSE`

**枚举**：`SoftmaxMaxReducePhaseBMode::DS_TRANSPOSE`

### 4.1 数据流

```text
Phase A 结束: bank6@0 = [c0, c1, c2, c3]

Phase B:
  ① UR 复制 3 行（emit_ur_copy_word）
     @0 → @1,@2,@3，四行均为 [c0,c1,c2,c3]

  ② STB → DS 4×4 转置 → LDB 写回 bank6@0..3
     Row_i @i = [ci, ci, ci, ci]

  ③ 3× emit_max_broadcast_row_pair（直接行 max，无需再 broadcast）
     Row0 vs Row1 → m01
     Row2 vs Row3 → m23
     m01 vs m23 → m

  ④ broadcast_m → bank5
```

### 4.2 转置矩阵（4 行 × 4 lane）

**转置前**（四行相同）：

```text
        lane0  lane1  lane2  lane3
Row0:    c0     c1     c2     c3
Row1:    c0     c1     c2     c3
Row2:    c0     c1     c2     c3
Row3:    c0     c1     c2     c3
```

**DS 4×4 转置后**（第 j 列 → 第 j 行）：

```text
Row0: [c0, c0, c0, c0]
Row1: [c1, c1, c1, c1]
Row2: [c2, c2, c2, c2]
Row3: [c3, c3, c3, c3]
```

之后 **跨 row、同 lane** 做 `SELECT_GREAT` 即可求 `max(c0..c3)`。

### 4.3 `emit_ur_copy_word`（UR_R + RtoVR + UR_W）

**不是 Host，不是 LOGIC COPY**——片上微指令：

```text
UR_R0  读 bank6@src
RtoVR-35  选源 ur_rdata_0 → uvr_wdata_0
UR_W0  写 bank6@dst，wbe=0xFFFF（整 128b word）
```

**周期**：**8 cyc/次** × 3 = **24 cyc**

### 4.4 DS 微指令

```cpp
DsIq::make_transpose(DsDataType::BIT32, /*dim0=*/4, /*dim1=*/4, inst_id, smc_id_en);
// → t_dim0_cnt=3, t_dim1_cnt=3
```

路径：**STB/LDBAXI → AXI Stream → DS → LDB/STBAXI**（与 UR+SMC 计算路径不同）。

**周期模型**（待 DS RTL 标定）：

| 段 | 周期 |
|----|------|
| STB 读 4 word | 12 |
| DS 4×4 转置 | 32 |
| LDB 写 4 word | 12 |
| **合计** | **56** |

### 4.5 `emit_max_broadcast_row_pair`

行已是 `[ci,ci,ci,ci]`，直接读两 row、`SELECT_GREAT`、全 wbe 写回。

**周期**：约 **15 cyc/次** × 3 = **45 cyc**

### 4.6 Phase B 周期汇总（方案 B）

| 子阶段 | 周期 |
|--------|------|
| UR copy ×3 | 24 |
| STB+DS+LDB | 56 |
| row max ×3 | 45 |
| broadcast_m → bank5 | 48 |
| **Phase B 合计** | **173** |

### 4.7 工程状态

| 项 | 状态 |
|----|------|
| compare 微码调度 | ✅ |
| STB/DS/LDB IQ emit | ⚠️ 未接入（DS 段为周期占位） |
| Gate RTL | ❌ 未跑 |

---

## 5. 全链路周期对比

Phase B 与 W 无关；Phase A = 15 × words。

| W | words | Phase A | 方案 A total | 方案 B total | Phase B 节省 |
|---|-------|---------|--------------|--------------|--------------|
| 4 | 1 | 15 | 444 | **188** | −256 (−59.7%) |
| 16 | 4 | 52 | 489 | **233** | −256 (−59.7%) |
| 100 | 25 | 325 | 804 | **548** | −256 (−32%) |

运行对比：

```bash
cd aiacc/SMI_IQ
# 构建 compare_max_reduce_phase_b_main 后
./compare_max_reduce_phase_b   # 默认 W=4,16,100
```

---

## 6. 专题：单标量 `[c1,0,0,0]` 能否「复制四行 + DS 转置」广播？

### 6.1 算法

四行均为 `[c1,0,0,0]` 时，4×4 转置后：

```text
@0: [c1, c1, c1, c1]   ✅
@1: [ 0,  0,  0,  0]
@2: [ 0,  0,  0,  0]
@3: [ 0,  0,  0,  0]
```

算法可行，但 @1..3 浪费。

### 6.2 周期对比（单个标量）

| 方法 | 周期 | 更快 |
|------|------|------|
| ADDTREE broadcast（§3.3） | **48** | ✅ |
| UR copy×3 + DS 转置 | 24+56=**80** | |
| ADDTREE sticky DR + 1 写（未实现，可优化） | ~35–45（估） | ✅ |

**结论**：单标量 broadcast 应继续用 **ADDTREE**；DS 适合 **整行** `[c0,c1,c2,c3]` 复制四行再转置（方案 B 主路径）。

四行 `[c0,0,0,0]…[c3,0,0,0]` 做一次 DS **不能**同时得到四个 broadcast 行（转置后 `@0=[c0,c1,c2,c3]`，不是四行 `[ci,ci,ci,ci]`）。

---

## 7. 方案选型建议

| 维度 | 方案 A Extract+Broadcast | 方案 B DS Transpose |
|------|--------------------------|---------------------|
| Phase B 周期 | 429 | **173** |
| 工程可落地 | **✅ 已验证** | ⚠️ 需 LSU+DS |
| 依赖 | bank4 掩码 | STB/DS/LDB tid |
| FPMUL/ADDTREE Phase B | 多（14 次 tree 级操作） | 少（仅 m→bank5） |
| 推荐场景 | 短期 Gate / e2e | DS RTL 就绪后替换 Phase B |

**中期优化（不改 DS）**：extract 后将四标量 **常驻四 broadcast 行**，compare 用 `emit_max_broadcast_row_pair`，可省约 **96 cyc**（仍不如 DS 方案 B 的 173 cyc）。

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `softmax_op.h` | `SoftmaxMaxReducePhaseBMode`、周期统计字段 |
| `softmax_op.cpp` | `compile_max_reduce()`、emit_* 实现 |
| `emit_softmax_max_reduce_main.cpp` | 导出 phase0a_hex（默认方案 A） |
| `compare_max_reduce_phase_b_main.cpp` | 两方案周期对比 |
| `pack_softmax_rtl_case.py phase0a` | Gate Max pack/golden |
| `run_softmax_gate_max_smoke.sh` | Gate Max smoke |
| `iq/modules/ds_iq.h` | DS transpose 微指令 |
| `cells/fp32_addtree_8to1/` | ADDTREE8 RTL |

---

## 9. 编译切换 Phase B 模式

```cpp
// 方案 A（默认）
SoftmaxOp::compile_max_reduce(req, steps, elems, smc_en,
    SoftmaxMaxReducePhaseBMode::EXTRACT_BROADCAST);

// 方案 B
SoftmaxOp::compile_max_reduce(req, steps, elems, smc_en,
    SoftmaxMaxReducePhaseBMode::DS_TRANSPOSE);
```

`compile_two_stage()` 在 `SOFTMAX_HOST_MAX=0` 时仍调用方案 A。
