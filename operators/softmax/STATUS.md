# Softmax RTL 状态

**更新**：2026-07-21

## 交付口径（重要）

| 项 | 约定 |
|----|------|
| **定义域 / 交付域** | **x ∈ [-8, +8]**（FP32；全定义域 = 该闭区间） |
| pad | 尾 lane **-1000**；不参与 max；依赖 Exp k-flush → e=0 → y=0 |
| Host / RTL | halfint Exp + Newton×3；有效 lane 验收 |
| **全定义域权威报告** | [`experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md`](./experiment_logs/softmax_fulldomain_m8p8_20260721_100419/REPORT.md) |
| 同域较早一轮（纯 math，无 flush 对齐） | `experiment_logs/softmax_fulldomain_m8p8_20260721_094752/` |
| 多域混合 Host（flush 对齐） | `experiment_logs/softmax_scale_ulp_20260721_100458/` |
| 宽域探底 [-100,+100]（非交付；flush 对齐） | [`experiment_logs/softmax_domain_m100p100_20260721_100444/REPORT.md`](./experiment_logs/softmax_domain_m100p100_20260721_100444/REPORT.md) |

**对照口径**：math = `math.exp` + **Host Exp flush 门控**（`host_exp_f32(x')==0` → 参考 e=0）。

**全定义域 Host vs math_f64（232900 lane）**：max ULP=**33**，mean≈**3.53**，p99=**16**，≤2 ULP≈**50%**。

## 当前进度

| Gate | 模式 | 内容 | 状态 |
|------|------|------|------|
| Host Golden | — | halfint Exp + Newton×3 + pad=-1000（Exp 委托 `exp_decompose.h` / `exp_host_golden.py`） | ✅ |
| **Gate A** | `@host` | Host x' → Phase1 Exp RTL | ✅ W=4 / W=100 |
| **Gate B@host** | `HOST_X0=1` | Phase2-norm@host：Newton×3 + mul | ✅ ULP=0 |
| **Gate B@chip** | `HOST_X0=0`（**Norm 默认**） | Phase2-norm@chip：magic + Newton×3 + mul | ✅ **W=4 PASS** bank3 ULP=0 4/4 |
| **Gate Magic** | `MAGIC_ONLY=1` | 仅 magic 前导 → bank6 | ✅ **W=4 PASS** ULP=0 4/4 |
| **Gate C** | 默认 | Phase0 max-sub RTL | ✅ **W=4 PASS** bank2 ULP=0 |
| **Gate D** | 默认 | Phase1.5 sum reduce | ✅ **W=4 / W=16** bank5 ULP=0（v4：bank6 wbe lane0） |
| **e2e（默认）** | 全 `@chip` | P0→Exp→Sum→Norm；Host 只写 x+m+常量 | ✅ **W=4** ULP=0；✅ **W=16** max_ulp≤2；✅ **W=100**；✅ **W=101**（pad=-1000 + Exp flush） |
| **e2e@host** | `HOST_X0=1` | 同上，但 bank6 Host 预载 x0 | ✅ **W=4 PASS** bank3 ULP=0 4/4 |
| 任意 W | — | pad=-1000；Exp k-flush → pad e=0；**e2e W=101 max_ulp≤2** | ✅ |
| 交付域精度 | Host 扫掠 | **x∈[-8,+8]**；见上表 / REPORT | ✅ |

> **默认行为（2026-07-20）**：`SOFTMAX_HOST_MAX=0`、`SOFTMAX_HOST_SUM=0`、`SOFTMAX_HOST_X0=0`  
> → Host 只提供 **x + m + 常量**；x'、e、S、x0、y 全在片上。  
> `@host` 模式需显式设对应 env=1（分 Gate 联调用）。

## 四 Phase 参数（W=4，steps=1）

| Phase | 模式 | body | max_iq | 说明 |
|-------|------|------|--------|------|
| Phase0 | 默认 | 15 | 4 | max-sub |
| Phase1 Exp | 默认 | **186** | ~25 | halfint Exp + k-flush（P1=108） |
| Phase1.5 | 默认 | **48** | **10** | sum reduce |
| **Phase2 Norm** | **@chip（默认）** | **153** | **17** | magic 前导 + Newton×3 + mul |
| **Phase2 Norm** | **@host** | 138 | 16 | Host 预载 x0@bank6 |

## e2e 默认数据流

```text
Host 写入:  x@bank2, m@bank5, 常量@bank1/4（含 bank4@8=2.0, @9=MAGIC）
stage_0(=Phase0)   bank2(x) + bank5(m) → bank2(x')
stage_1(=Phase1)   bank2(x') → bank3(e)
stage_2(=Phase1.5) bank3(e) → bank5(S)
stage_3(=Phase2)   bank5(S) → magic(x0)@bank6 → Newton×3 → bank3(y)
```

> emit 目录统一为 `phase0_hex` / `phase1_exp_hex` / `phase15_sum_hex` / `phase2_norm_hex`；  
> e2e 的 `stage_0..3` 是 multi_stage TB 目录名。
## 复现

```bash
cd aiacc/SMI_IQ

# 四 Phase e2e（默认全片上，推荐）
W=4 bash op/softmax/run_softmax_e2e_smoke.sh

# e2e@host 对比（Host 预载 x0）
SOFTMAX_HOST_X0=1 W=4 bash op/softmax/run_softmax_e2e_smoke.sh

# 分 Gate A+B 联调（脚本内 HOST_MAX=1 HOST_SUM=1）
bash op/softmax/run_softmax_fp32_smoke.sh

# Gate Magic / Gate C+D
W=4 bash op/softmax/run_softmax_magic_x0_smoke.sh
bash op/softmax/run_softmax_gate_cd_smoke.sh

# 任意 W emit
W=101 SKIP_RTL=1 bash op/softmax/run_softmax_e2e_smoke.sh
```

## 下一步

1. W>4 / 多 word：Sum FPADD 累加 + S 广播验证
2. 片上 max reduce（SELECT_GREAT，m 仍 Host 写 bank5）
3. Sum body 优化（当前 48 cyc，4×ADDTREE 广播）
4. 8SMC / LSU
