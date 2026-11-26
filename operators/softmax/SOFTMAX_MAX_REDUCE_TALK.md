# Softmax 片上 Max Reduce 讲稿

> 时长约 20–25 分钟 · 配套细节见 [`SOFTMAX_MAX_REDUCE.md`](./SOFTMAX_MAX_REDUCE.md)

---

## 开场

大家好。今天讲 **Softmax Phase0a：一行 FP32 在片上怎么求最大值 m，写到 bank5**。

Stable Softmax 要先算 m = max(x)，再 x' = x − m。我们单独做了 **Gate Max**，和后面的 max-sub 分开。

**今天一个数字先说清楚：Phase B 有两种做法，周期差 256 个时钟——429 cyc 对 173 cyc，差一倍以上。** 后面我会把这个数拆给你们看。

---

## 一、讲什么、不讲什么

**讲**：Phase0a 怎么从一行 x 得到 m；两种 Phase B 方案；**每个阶段多少 cyc**。

**不讲**：Exp、Sum、Newton——那些是后面 Phase。

一行必须在 **同一个 SMC** 里跑完 max → exp → sum → norm，不能拆 8 个 SMC。

---

## 二、硬约束：比较只能「同 lane 对同 lane」

片上 max 用 FPADD 出 status，再用 SELECT_GREAT 选大的。

问题是：这是 **4-lane SIMD**，lane0 只能和 lane0 比，lane1 只能和 lane1 比。

Phase A 做完，bank6 上是一个 word：**[c0, c1, c2, c3]**——四个列最大值各占一个 lane。我们要的是 **一个数** m = max(c0,c1,c2,c3)，但 **没法直接** max(c0,c1)，因为它俩不在同一 lane。

所以 Phase B 的全部工作，就是 **改数据布局**，让比较变成「同 lane 能比」——两种方案差就差在这里，**周期差 256 cyc 也出在这里**。

---

## 三、Phase A：列 max（跟 W 走）

Phase A 按 word 扫输入 x，每个 lane 自己跟自己比，更新列 max。

- **每个 word：15 cyc**
- W=4（1 个 word）：**15 cyc**
- W=16（4 个 word）：**52 cyc**
- W=100（25 个 word）：**325 cyc**

扫完得到 **[c0, c1, c2, c3]**，存在 bank6@0。

**W 越大，Phase A 越长；但 Phase B 固定，跟 W 无关。** 这是后面读周期表时要记住的。

---

## 四、周期总览（先亮结论）

| 阶段 | 方案 A | 方案 B | 备注 |
|------|--------|--------|------|
| Phase A | 15×words | 15×words | 相同 |
| **Phase B** | **429 cyc** | **173 cyc** | **差 256 cyc** |
| **W=4 总计** | **444 cyc** | **188 cyc** | |
| **W=16 总计** | **489 cyc** | **233 cyc** | |
| **W=100 总计** | **804 cyc** | **548 cyc** | |

**一句话：方案 B Phase B 只要 173 cyc，方案 A 要 429 cyc；全链路永远少 256 cyc，跟 W 无关。**

W=100 时 Phase A 占 325 cyc，Phase B 省 256 cyc，总体大约快 **32%**；W=4 时 Phase B 占比大，总体大约快 **58%**。

---

## 五、方案 A：Extract + Broadcast（429 cyc，已验证）

当前默认方案，Gate Max W=4/16 已 PASS。

### 5.1 在干什么

1. 把 [c0,c1,c2,c3] **拆成四个标量** [c0,0,0,0] … [c3,0,0,0]
2. 每次两两比 max 之前，把标量 **广播成整行** [ck,ck,ck,ck]
3. 三步 tournament 得到 m，再广播到 bank5

### 5.2 周期拆解（方案 A Phase B = 429 cyc）

| 步骤 | 做什么 | **周期** |
|------|--------|----------|
| extract ×4 | FPMUL 掩码 + ADDTREE 抽 lane | **48 cyc**（12×4） |
| broadcast ×6 | 三次 compare，每次先 broadcast 两个标量 | **288 cyc**（48×6） |
| compare ×3 | SELECT_GREAT 两两 max | **45 cyc**（15×3，含在 333 里） |
| compare 小计 | 含 6 次 broadcast | **333 cyc** |
| m → bank5 | ADDTREE 广播 m 四 lane | **48 cyc** |
| **Phase B 合计** | | **429 cyc** |

**瓶颈在哪？288 cyc 的重复 broadcast——占 Phase B 的 67%。**

### 5.3 ADDTREE 为啥能「广播」

硬件 **没有 broadcast 指令**。ADDTREE 本来是 8 个数相加，结果只写 **一个 lane**。

我们喂 [v,0,0,0,0,0,0,0]，和就是 v；然后 **连做 4 次**，每次写不同 lane，拼成 [v,v,v,v]。

- **一次标量广播：48 cyc**
- LOGIC COPY、UR 整 word 复制 **都不行**——它们不能跨 lane 扩播

---

## 六、方案 A 的小优化：Sticky DR + 1 次写

### 6.1 思路

现在每次 ADDTREE 算完就 UR 写一次 lane，广播一个标量要 **4 写**，共 **48 cyc**。

流水 ADDTREE 的 DR 是 **sticky** 的：每次只改一个 lane，其它 lane **保留**。可以在 DR 里先拼好 [v,v,v,v]，**最后 1 次整 word 写回**。

### 6.2 周期对比

| 做法 | **一次标量广播** |
|------|------------------|
| 现在：4 tree + 4 次 partial 写 | **48 cyc** |
| 优化：4 tree + 1 次整 word 写 | **约 35–45 cyc**（待 RTL 验证） |

Phase B 若 6 次 broadcast 都优化，理论上还能再省 **约 20–80 cyc**，但 **仍远不如方案 B 的 256 cyc 差距**。

注意：Sum reduce 里对 acc bank 整字写会污染数据；max 写到空 scratch 相对安全。

---

## 七、方案 B：UR 复制 + DS 转置（173 cyc）

### 7.1 思路

别反复 broadcast 标量。先把 [c0,c1,c2,c3] **复制成四行相同的行**，再 **DS 4×4 转置**，一次得到：

- 第 0 行 [c0,c0,c0,c0]
- 第 1 行 [c1,c1,c1,c1]
- …

之后 **行对行、同 lane** 直接 SELECT_GREAT，**每次只要 15 cyc**，不用再 broadcast。

### 7.2 周期拆解（方案 B Phase B = 173 cyc）

| 步骤 | 做什么 | **周期** |
|------|--------|----------|
| UR 复制 ×3 | @0 整 word 拷到 @1/@2/@3（UR 读+写） | **24 cyc**（8×3） |
| STB + DS + LDB | 4 word 进 DS 4×4 转置再写回 | **56 cyc**（模型：12+32+12） |
| 行 max ×3 | 行已是 broadcast，直接比 | **45 cyc**（15×3） |
| m → bank5 | 同方案 A | **48 cyc** |
| **Phase B 合计** | | **173 cyc** |

对比方案 A：**429 − 173 = 256 cyc**，Phase B 快 **59.7%**。

### 7.3 工程状态

compare 微码已有；STB/DS/LDB 还没接 RTL，56 cyc 是模型估计。Gate 还没跑方案 B。

---

## 八、两种方案怎么选（看周期）

| 场景 | 推荐 | **Phase B 周期** |
|------|------|------------------|
| 现在交付、Gate 验证 | 方案 A | **429 cyc** |
| DS 通路就绪后 | 方案 B | **173 cyc** |
| 只优化 broadcast、不上 DS | sticky DR 或常驻四行 | 约 **330–380 cyc**（估） |
| 单个 [c1,0,0,0] 扩播 | ADDTREE，别用 DS | **48 cyc** vs DS 路径 **80 cyc** |

补充：单标量 [c1,0,0,0] 复制四行再 DS 转置，算法上能得到 [c1,c1,c1,c1]，但要 **80 cyc**，比 ADDTREE **48 cyc** 慢。DS 赢的是 **整行 [c0,c1,c2,c3]** 那种 layout，不是 lone scalar。

---

## 九、用一张图记 Phase B 周期

```text
方案 A Phase B（429 cyc）:
  extract  48 ─────────────────────────
  broadcast+compare 333 ████████████████████████████████████  ← 瓶颈
  m→bank5  48 ─────────────────────────

方案 B Phase B（173 cyc）:
  UR copy   24 ────────
  DS        56 ──────────────
  compare   45 ─────────
  m→bank5   48 ─────────────────────────
```

**256 cyc 差在哪？主要在方案 A 那 288 cyc 的重复 broadcast；方案 B 用 24+56=80 cyc 的 layout 变换一次搞定。**

---

## 十、总结（带周期）

1. Phase A：**15 cyc/word**，得到 [c0,c1,c2,c3]
2. 硬约束：SELECT_GREAT **同 lane**，Phase B 本质是改 layout
3. **方案 A：Phase B 429 cyc**，已 Gate 验证；瓶颈 **288 cyc broadcast**
4. **方案 B：Phase B 173 cyc**，少 **256 cyc**；DS 待接入
5. ADDTREE 不是广播器，是「8 输入求和 + 单 lane 写」拼出来的；sticky DR 可略优化到 **~40 cyc/次**
6. **W=4 总计 444 vs 188 cyc；W=100 总计 804 vs 548 cyc**

---

## 十一、Q&A

**Q：为什么 Phase B 跟 W 无关？**  
A：四个 lane 的列 max 固定 4 个，Phase B 只做这 4 个数的 tournament，不随 word 数变。

**Q：256 cyc 能再省吗？**  
A：方案 A 内可 sticky DR、少重复 broadcast，大概几十 cyc；要大幅省还得方案 B 或改 ISA 加 shuffle。

**Q：56 cyc 的 DS 准吗？**  
A：模型值，RTL 标定后可能变；盈亏平衡：DS 段只要不超过约 **312 cyc**，方案 B 仍优于方案 A 的 429 cyc——目前 56 cyc 裕量很大。

**Q：文档？**  
A：细节 [`SOFTMAX_MAX_REDUCE.md`](./SOFTMAX_MAX_REDUCE.md)；周期对比跑 `compare_max_reduce_phase_b_main.cpp`。

---

## 收尾

今天核心就三个数：**15 cyc/word**、**429 vs 173 cyc**、**省 256 cyc**。  
方案 A 能交付，方案 B 周期更短，等 DS 通路接上 worth 切。

谢谢，有问题继续聊。
