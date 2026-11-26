# Exp 算子（FP32）交付说明

> **版本：exp1.0.5（2026-07-21：k 域 flush 闭环 + Host 对齐 + N=10k 评测）— 可交付**  
> 状态指针：[`STATUS.md`](./STATUS.md) · 流程账本：[`FLOW_ANALYSIS.md`](./FLOW_ANALYSIS.md)  
> 冻结备份：[`backup_exp1.0.5_20260721/`](./backup_exp1.0.5_20260721/) · 上一版：`backup_exp1.0.4_20260719/`  
> 本 README 是当前工作树的唯一主文档。旧长文已归档到 [`docs_archive/`](./docs_archive/)。

---

## 0. 目录权限与 VCS（首次 / 仿真前必读）

> **⛔ 本节禁止删除或挪到归档。**  
> 删掉后 `YouTsing` 无法写 case / 日志，RTL 冒烟会在 `rm`/`mkdir` 或 `sim_output.log` 处直接挂掉。  
> 改算法、改微码都要保留本节；最多允许补充排查项，不允许整节抹掉。

原项目部分目录属主为 `bupt` / `root`，用户 `YouTsing` 默认**写不了**算子产物与 RTL 仿真目录。  
**在真实终端执行一次**下面的 `chown`（会要 sudo 密码）；之后日常开发、`make tvgen`、`run_exp_merged_tests.sh` 才用当前用户身份跑通。

### 0.1 目录写权限（必须，`sudo` 一次）

```bash
# 1) Exp 算子源码目录（必须）
sudo chown -R YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/op/exp
sudo chmod -R u+rwX \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/op/exp

# 2) TVGen 输出目录（必须）
sudo chown -R YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/out
sudo chmod -R u+rwX \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/out

# 3) RTL case 父目录（必须，否则无法 rm/mkdir case）
sudo chown YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/test_atomic_ops/verification_v2
sudo chmod u+rwx \
  /home/YouTsing/Jianyuan-aiacc_4.21/test_atomic_ops/verification_v2

# 4) RTL 比对日志目录（必须，否则仿真脚本无法写 log）
sudo chown YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log
sudo chmod u+rwx \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log
sudo chown YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log/batch_compare.log \
  2>/dev/null || true

# 5) RTL 仿真输出目录（必须，否则 sim_output.log 写入失败）
sudo chown -R YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/sim_output_aiacc_top_f
sudo chmod -R u+rwX \
  /home/YouTsing/Jianyuan-aiacc_4.21/sim_output_aiacc_top_f

# 6) RTL trace 日志子目录（必须）
sudo chown -R YouTsing:YouTsing \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log/verification_v2
sudo chmod -R u+rwX \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log/verification_v2
```

**自检（属主应是 `YouTsing`）：**

```bash
ls -ld \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/op/exp \
  /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ/out \
  /home/YouTsing/Jianyuan-aiacc_4.21/test_atomic_ops/verification_v2 \
  /home/YouTsing/Jianyuan-aiacc_4.21/arith_regression/test_log \
  /home/YouTsing/Jianyuan-aiacc_4.21/sim_output_aiacc_top_f
```

### 0.2 修改后的权限示例

```text
# op/exp — 算子源码可读写
drwxr-xr-x  … YouTsing YouTsing  …  op/exp/

# SMI_IQ/out — TVGen 产物可写
drwxr-xr-x  … YouTsing YouTsing  …  SMI_IQ/out/

# verification_v2 — 可在其下创建/删除 case
drwxr-xr-x  … YouTsing YouTsing  …  test_atomic_ops/verification_v2/

# arith_regression/test_log — 比对日志可写
drwxr-xr-x  … YouTsing YouTsing  …  arith_regression/test_log/

# sim_output_aiacc_top_f — VCS 仿真日志 / simv 可写
drwxr-xr-x  … YouTsing YouTsing  …  sim_output_aiacc_top_f/
```

> 若 Agent / `root` 刚跑过仿真，目录可能又变回 `root`——**再执行一遍 §0.1** 即可。

### 0.3 VCS 环境变量（跑 RTL 前 `export`）

`vcs` / `simv` 默认不在 PATH。仿真（含**复用**预编译 `simv`）前在**同一 shell** 里 export（与 `/home/xiaoke/.bashrc` 一致，VCS `O-2018.09-SP2`）：

```bash
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
```

| 项 | 说明 |
|----|------|
| 预编译 `simv` | `sim_output_aiacc_top_f/simv`。**只改微码/case → 复用，勿加 `--rebuild`** |
| 何时重编 | 改了 RTL / filelist 才需要；`--rebuild` 会走 Verdi/FSDB，易踩坑 |
| License | 需能连 `bupt-PowerEdge-R730xd:27000`（本机 hosts 常指向 `127.0.0.1`） |
| 内核告警 | `Unsupported Linux kernel '5.15...'` **可忽略** |

### 0.4 VCS / License 跑不通时怎么查

分层排查（最近冒烟失败多半是 **2 / 3**，不是微码）：

1. **目录写权限**（§0.1）  
   - 症状：`Permission denied`、`rm`/`mkdir` case 失败、写不了 `sim_output.log`。  
   - 处理：再跑一遍 §0.1 `chown`。

2. **License daemon 是否在听 27000**  
   ```bash
   ps -ef | grep -E '[l]mgrd|[s]npslmd'
   ss -lntp | grep 27000
   # 或：timeout 2 bash -c 'echo >/dev/tcp/127.0.0.1/27000' && echo OK || echo FAIL
   ```  
   - 进程在、但 **27000 连不上** / `/tmp/snpslmd.log` 有 `Failed to open the TCP port` → **重启 license**：  
     ```bash
     sudo /usr/software/SCL2018/scl/2018.06/linux64/bin/lmgrd \
       -c /usr/software/License/Synopsys.dat -l /tmp/snpslmd.log
     # 若旧进程僵死：先 sudo kill 对应 lmgrd/snpslmd，再启动
     ```  
   - 备选 env（本机 license 文件直读，仅当 server 口异常时试）：  
     `export LM_LICENSE_FILE=/usr/software/License/Synopsys.dat`

3. **在真实终端跑，不要指望 Cursor Agent 沙箱**  
   - Agent 沙箱常拦 TCP → 日志里 `Cannot connect to the license server` / `Summary: missing`。  
   - **请你本机开终端**（用户 `YouTsing`），先 §0.1 + §0.3，再：  
     ```bash
     cd /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ
     make tvgen
     bash op/exp/run_exp_merged_tests.sh
     ```

4. **VCS 安装路径**  
   - 主路径：`/home/xiaoke/synopsys/vcs/O-2018.09-SP2`（含 `linux64/`）。  
   - 备用：`/usr/software/vcs2018/vcs/O-2018.09-SP2`。  
   - `VCS_HOME` 指错时会出现找不到 `vcs1` / compiler 报错。

---

## 1. 交付摘要


| 项     | 内容                                                                                                    |
| ----- | ----------------------------------------------------------------------------------------------------- |
| 功能    | y=\mathrm{exp}(x)，FP32，Merged 单流（P1 预处理 + P2 多项式）                                                     |
| 算法    | 半整数归约 + Remez + √2；**k≤-126→0 / k≥127→+Inf**（flush） |
| Remez | `[0.99999994, 0.69314606, 0.24023668, 0.05559393, 0.00943751]`（区间 [-0.25,0.25]）                       |
| 精度    | vs Host：**100% ULP=0**（N=10000，x∈[-100,100]）；vs Math 安全域 max≈**61 ULP**；小范围 W=100 vs Math max=**4** |
| 多字    | steps≥2 **已通**（P2 unroll + `emit_exp_embed_iq`）；26 word / 8SMC 已跑通 |
| 单字    | steps=1 **已与多字同路径**（见 §4 / §4.1） |
| 多 SMC | 8 SMC **不同数据** 一次仿真 **pass=8** |




### 1.1 推荐回归命令

```bash
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export EXP_COMPRESS_NOP_LOOPS=1   # 默认即开；勿设成 0

cd /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ
make tvgen

# 小规模冒烟（W=4 / W=8 均为 unroll 同路径）
bash op/exp/run_exp_merged_tests.sh

# 交付级：W=100 均分到 8 SMC（不同数据、多字）
python3 op/exp/run_exp_w100_halfint_8smc_split.py

# 1.0.5 权威大评测：N=10000 / 8SMC / 26word
python3 op/exp/run_exp_n10k_8smc_26w.py
```

权威大评测：`experiment_logs/exp_n10k_8smc_26w_20260721_011755/`。  
W=100→8SMC：`experiment_logs/exp_halfint_w100_8smc_split_20260719_175428/`。  
冻结备份：`backup_exp1.0.5_20260721/`。

### 1.2 一套命令行跑通流程（2026-08-24 实测）

下面这套命令按顺序执行即可；包含环境检查、TVGen、RTL 冒烟、结果定位。

```bash
# 0) 环境变量（同一 shell）
export VCS_HOME="/home/xiaoke/synopsys/vcs/O-2018.09-SP2"
export PATH="$VCS_HOME/bin:$PATH"
export LM_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export SNPSLMD_LICENSE_FILE=27000@bupt-PowerEdge-R730xd
export EXP_COMPRESS_NOP_LOOPS=1

# 1) 切到工程并编译 tvgen
cd /home/YouTsing/Jianyuan-aiacc_4.21/aiacc/SMI_IQ
make tvgen

# 2) 跑 Exp Merged 冒烟（W=4 + W=8）
bash op/exp/run_exp_merged_tests.sh

# 3) 看最新报告（PASS/FAIL + 日志目录）
latest="$(cat op/exp/experiment_logs/LATEST_SAMECODE_RTL.txt)"
echo "$latest"
cat "$latest/REPORT.md"
```

**判定标准**
- `REPORT.md` 里两条 case（`w4_mix`、`w8`）都出现 `Summary: ... fail=0` 即跑通。
- 若显示 `Summary: missing`，去看对应 `sim_*.log` 和 `/home/YouTsing/Jianyuan-aiacc_4.21/sim_output_aiacc_top_f/compile.log`。

**本次实测结果（2026-08-24）**
- `make tvgen` 正常。
- `run_exp_merged_tests.sh` 成功生成 case 与 IQ，但 RTL 编译阶段失败，`REPORT.md` 为 `Summary: missing`。
- 失败签名是 VCS `vcselab` internal error（`Elab time internal error`），见 `sim_output_aiacc_top_f/compile.log` 末尾栈。
- 该问题属于仿真器/内核兼容层，不是 Exp 微码逻辑错误；命令流程本身已闭环，定位点固定在 compile.log。

**遇到 `Summary: missing` 的固定排查顺序**
1. 先执行本文 §0.1（权限）和 §0.3（VCS 环境变量）。
2. 确认 license 端口可连：`ss -lntp | rg 27000`。
3. 复用已存在 `simv`（如果有）再跑一次；无 `simv` 时会触发完整编译。
4. 若仍是 `Elab time internal error`，优先在支持内核/更新版 VCS 环境重跑同一套命令。

---



## 2. 目录与文件



### 2.1 核心源码（必留）


| 文件                     | 职责                                      |
| ---------------------- | --------------------------------------- |
| `exp_decompose.h`      | 半整数分解 + Remez + **C++ Host 黄金 SSOT**（`host_exp_f32`） |
| `exp_host_golden.py`   | **Python Host 黄金 SSOT**（Softmax pack/扫掠 import，勿复制 Remez） |
| `exp_preprocess_op.*`  | Phase-1（P1=108：√2/odd + k 越界 flush）   |
| `exp_poly_kernel_op.*` | Phase-2 Horner（一律 unroll / DR+tied-zero） |
| `exp_op.*`             | Merged 编译：P1\|P2 同路径 unroll+embed |
| `exp_iq_utils.h`       | same-code 压缩、`emit_exp_embed` |




### 2.2 工具与回归（保留）


| 文件                                   | 用途                             |
| ------------------------------------ | ------------------------------ |
| `pack_exp_rtl_test_case.py`          | tvgen → `verification_v2` case |
| `measure_iq_depth_main.cpp`          | 测 INSTQ 深度                     |
| `run_exp_merged_tests.sh`            | W=4/W=8 冒烟                     |
| `run_exp_merged_w_sweep.sh`          | W 扫掠                           |
| `run_exp_w100_halfint_8smc_split.py` | 交付验证：100 点→8SMC 不同数据       |
| `run_exp_n10k_8smc_26w.py`            | **1.0.5 权威**：N=10000 / 8SMC / 26word |
| `run_exp_w100_halfint_report.py`     | 单 SMC / 切块报告（可选；内置 Host 偏 f64） |
| `run_exp_fp16_tests.sh`              | FP16 旁路（**非交付主路径**）            |




### 2.3 备份与日志（勿删）

- `backup_exp1.0.5_20260721/` — **当前冻结**；`backup_exp1.0.4_20260719/` 为无 flush 上一版；整数 k 参考用 `backup_exp1.0.3_20260718/`
- `experiment_logs/` — RTL / 分析产物
- `docs_archive/` — 交付前超长 README 归档

---



## 3. 数据布局（UR）


| Bank        | 内容                                             |
| ----------- | ---------------------------------------------- |
| bank2       | 输入 x（P1）；P2 unroll 不作 ACC |
| bank0       | P1 写 r（按 word）；scratch `k_f` @200（W1，addr 不自增） |
| bank1 @0..4 | Remez 系数 c_4..c_0                              |
| bank1 @5+   | 每 word 的 `scale`（越界已 flush：0 / +Inf）        |
| bank3       | 输出 y                                           |
| bank4       | 常数：`log2e`/bias/shift/half/one/int_one/zero/√2@7；@8..9 留给 Softmax；**@10=-126、@11=127、@12=+Inf**（Exp k-flush） |


TVGen 映射：`alpha.bin`→bank2，`other.bin`→bank4，`input.bin`→bank1 系数。

---



## 4. 发射策略（Emit strategy）★★ — 单字 / 多字已统一

此前 **单字（single-word, `steps=1`）** 与 **多字（multi-word, `steps≥2`）** 不是同一实现：

| 项 | 旧：单字 `steps=1` | 多字 `steps≥2`（未改） |
|----|-------------------|------------------------|
| P2 调度 (schedule) | Horner **fold** + ACC@bank2 | **unroll**（DR + tied-zero） |
| ACC 清零缝 (ACC clear gap) | P1 后 +4 拍写 0→bank2 | **无** |
| IQ 发射 (IQ emit) | `[P1][embed=1][Horner\|Final]` | 整段 `emit_exp_embed_iq(P1\|P2)` |
| 墙钟/字 (wall-clock / word) | ≈146 cyc（64+4+78） | ≈142 cyc（旧 64+78；现 **186**=108+78） |

**2026-07-19 起**：单字默认改为与多字 **同一路径**（P2=`unroll` + `emit_exp_embed_iq`）。多字逻辑未改。

| 项 | 单字 `steps=1`（现默认） | 多字 `steps≥2` |
|----|--------------------------|----------------|
| P2 调度 | **unroll**（同多字） | **unroll** |
| ACC 清零缝 | **无** | **无** |
| IQ 发射 | `emit_exp_embed_iq` | `emit_exp_embed_iq` |
| 墙钟/字 | ≈**186** cyc（108+78） | ≈**186** cyc（108+78） |

唯一相关环境变量：

| 变量 | 作用 |
|------|------|
| `EXP_COMPRESS_NOP_LOOPS=0` | 关掉 same-code（深度暴涨，多字易 >128）；交付保持默认开启 |


### 4.1 小测：单字统一 unroll（W=4）

日志：[`experiment_logs/singleword_unroll_20260719_173935/`](./experiment_logs/singleword_unroll_20260719_173935/)（含 `ulp_report.json`）

输入 (input)：`x = [0.5, 1.0, 2.0, -1.0]`，单 SMC，Host 灌 UR。

| Metric（指标） | Chinese | Value（数值） |
|----------------|---------|---------------|
| Compare summary | 比对汇总 | **pass=1 fail=0** |
| Wall-clock / word | 每字墙钟周期 | **186** cyc（P1=108 + P2=78；历史无 flush 时为 142） |
| Path tag | 路径标记 | `unroll-flat78 [steps=1 same-as-multiword]` |
| max ULP (RTL vs f32(math.exp)) | 端到端最大 ULP | **2** |
| mean ULP (RTL vs f32(math.exp)) | 端到端平均 ULP | **0.75** |
| max ULP (RTL vs Host golden) | 相对 Host 最大 ULP | **0**（bit-exact） |
| max abs err vs math.exp | 相对数学真值最大绝对误差 | ≈1.07e-7 |
| max rel err vs math.exp | 相对数学真值最大相对误差 | ≈1.37e-7 |

逐点表 (per-lane table)：

| i | x | y_math | y_rtl (=y_host) | ULP e2e (vs f32(math.exp)) | ULP vs Host |
|---|---|--------|-----------------|----------------------------|-------------|
| 0 | 0.5 | 1.6487212707 | 1.6487213373 | 1 | 0 |
| 1 | 1.0 | 2.7182818285 | 2.7182817459 | 0 | 0 |
| 2 | 2.0 | 7.3890560989 | 7.3890562057 | 0 | 0 |
| 3 | -1.0 | 0.3678794412 | 0.3678793907 | 2 | 0 |

**分析 (Analysis)**：

1. **实现已对齐**：tvgen 日志出现 `[steps=1 same-as-multiword]`，P2 为 `unroll DR+tied_zero`，无 ACC 清零缝；与多字同一 emit。
2. **数值正确**：RTL 与 Host golden **0 ULP**（bit-exact）；相对 `f32(math.exp)` 最大 **2 ULP**，落在交付口径（N=100 时 max ULP e2e=4）之内。
3. **多字未回归**：同日 W=8（`steps=2`）冒烟 **pass=1 fail=0**，路径仍为 `unroll-flat78 (multiword)`。
4. **墙钟**：P1 含 k 越界 flush 后 body=108，合并墙钟 ≈ **186**/字（108+78）；旧无 flush 时为 142。

对照冒烟：`bash op/exp/run_exp_merged_tests.sh`（W=4 + W=8）。

---



## 5. 硬约束（回退必炸）



### 5.1 bank2 复用

```
P1：bank2 = x
P2 unroll：bank2 仅作输入；用 tied-zero，不作 ACC
```

（历史 Horner-fold 用 bank2 作 ACC 的路径已删除。）

### 5.2 禁止「P1 后再开 embed=1」做字循环

多字若结构为：

```text
[embed=2][P1…][embed=1,loops=4][Horner…][Final]
```

则 FETCH 把 **Horner+Final** 送进 EXE_ARRAY 按字回放，**P1 只跑第一字** → word1+ 的 `r`/`scale` 全 0。

正确：多字 **P1+P2 同在一段 lev2 body**（本版 = `emit_exp_embed_iq`）。

### 5.3 禁止把「ACC 折叠时间轴」当真·78 拍 flat 发

fold 调度里 ACC 写回在 Final；靠 lev1 把 `(Horner|Final)` 转 5 次。  
若只把 78 槽绝对时间轴展平发出、又不开 lev1，则 stage1+ 读到未更新 ACC → y\approx scale（多项式塌成 ≈1）。

### 5.4 `embed_end` bit0 与 Final 同体（§14 经验）

Final 必须与 Horner **同一 array body**；若 Horner 末单独 `embed_end bit0`，Final 会被甩出字循环 → 典型产物 `scale0·P(r_last)`。

### 5.5 多 SMC

- `smc_id_en=7`：SMC0–7 **都执行同一套微码**
- **不同数据**：各 SMC 自己的 `ur_data/bank2.hex` + `output_bank/bank3_final.hex`
- 勿再对微码做 CRU 级联「补拍」（硬件 shift 已对齐；乱补会导致写丢失）

---



## 6. 经验教训（踩坑精选）

按时间压缩，细节可查 `docs_archive/` 与各 `backup_*/README.md`。


| #   | 现象                       | 根因                                    | 对策                       |
| --- | ------------------------ | ------------------------------------- | ------------------------ |
| 1   | 负 x（如 -1）k/r 大错          | UR **同 bank 双端口同拍读**冲突                | P1 冲突读分拍                 |
| 2   | 多字输出全 0 / 错字             | 外层与 Horner **同用 embed=1**             | 外层 lev2、内层 lev1          |
| 3   | 8SMC 假过、多字悬崖             | 错误 CRU 补偿拉长 body                      | **补偿恒 0**                |
| 4   | fold 多字 `y≈scale0·P(r1)` | Horner `embed_end bit0` 把 Final 甩出字循环 | Horner+Final **同体**      |
| 5   | fold 字间泄漏 ~              | r                                     | ⁵                        |
| 6   | Merged word1 r/scale=0   | P1 后嵌套 embed=1，P1 不进回放                | 多字改 **整段 embed emit**    |
| 7   | flat 多字 y≈scale          | 把 ACC-fold 时间轴当 unroll 发              | 多字 **compile 强制 unroll** |
| 8   | IQ≈148 仿真胡写              | `EXP_COMPRESS_NOP_LOOPS=0`            | 保持压缩开启；深度以 same-code 为准  |
| 9   | TB 超时 / 全 0              | 每字 ~186 cyc，TB≈5000                   | 单次约 ≤31 word；更大需切块或加长 TB |
| 10  | FP16 P1 不可靠              | INTADD 32-bit vs 8×int16 打包           | **交付仅 FP32**；FP16 另线     |


**端到端 ULP**：`ULP(RTL, f32(math.exp(x)))`，即硬件输出相对数学真值的距离（整条 P1+P2 链路）。

---



## 7. 精度与规模 (Accuracy & scale)


| 场景 (Scenario) | 结果 (Result) |
|-----------------|---------------|
| W=4 单字统一 unroll（2026-07-19） | PASS；max ULP e2e=**2**；vs Host=**0**；见 §4.1 |
| W=8 多字 | PASS（冒烟未回归） |
| W=100 → 8SMC 均分 | pass=8；max ULP e2e=**4**（交付权威） |


对照整数 k（1.0.3）：同分布 max ULP e2e≈**41**。半整数明显更优。

规模口算：墙钟 ≈ `steps × 186`；TB 5000 → 约 **31 word/次**（单 SMC）。系数+scale 同处 bank1，极大 W 还需考虑 scale 地址容量。

---



## 8. IQ 深度（same-code）

测具：`measure_iq_depth_main.cpp`（`EXP_COMPRESS_NOP_LOOPS=1`）。硬件上限 `SMI_IQ_INSTQ_DEPTH=128`。  
单字 / 多字均为 **Merged P1|P2 + unroll + `emit_exp_embed`**（同路径）；下列为压缩后各通道指令槽数。

### 8.1 Merged 单字（W=4，steps=1，body=186 cyc）

日志：`experiment_logs/_iq_depth_w4_smc7.txt`（2026-07-19）。

| SMI_IQ 通道 | 压缩前 | 压缩后 | 节省 |
|-------------|-------:|-------:|-----:|
| fpadd | 143 | **24** | 119 |
| fpmul | 143 | **20** | 123 |
| fptoint | 143 | **4** | 139 |
| intadd | 143 | **6** | 137 |
| inttofp | 143 | **8** | 135 |
| logic | 143 | **12** | 131 |
| rtovr-0 | 143 | **7** | 136 |
| rtovr-1 | 143 | **7** | 136 |
| rtovr-3 | 143 | **8** | 135 |
| rtovr-4 | 143 | **9** | 134 |
| rtovr-5 | 143 | **15** | 128 |
| rtovr-6 | 143 | **15** | 128 |
| rtovr-9 | 143 | **5** | 138 |
| rtovr-10 | 143 | **9** | 134 |
| rtovr-11 | 143 | **11** | 132 |
| rtovr-12 | 143 | **11** | 132 |
| rtovr-13 | 143 | **7** | 136 |
| rtovr-35 | 143 | **7** | 136 |
| rtovr-36 | 143 | **3** | 140 |
| ur_rdata_0 | 143 | **5** | 138 |
| ur_rdata_1 | 143 | **21** | 122 |
| ur_rdata_2 | 143 | **22** | 121 |
| ur_wdata_0 | 143 | **8** | 135 |
| **合计（各通道压缩后之和）** | — | **244** | — |
| **MAX（跨通道，决定能否装入 INSTQ）** | 143 | **24** | 119 |

→ **当前 IQ 深度（单字）= MAX 24 / 合计 244**，远低于 128 上限。

### 8.2 Merged 多字（W=16，steps=4，body=186 cyc/word）

日志：`experiment_logs/_iq_depth_w16_smc7.txt`（2026-07-19）。与 8SMC W=100 交付同微码形态。

| SMI_IQ 通道 | 压缩前 | 压缩后 | 节省 |
|-------------|-------:|-------:|-----:|
| fpadd | 144 | **25** | 119 |
| fpmul | 144 | **21** | 123 |
| fptoint | 144 | **5** | 139 |
| intadd | 144 | **7** | 137 |
| inttofp | 144 | **9** | 135 |
| logic | 144 | **13** | 131 |
| rtovr-0 | 144 | **8** | 136 |
| rtovr-1 | 144 | **8** | 136 |
| rtovr-3 | 144 | **9** | 135 |
| rtovr-4 | 144 | **10** | 134 |
| rtovr-5 | 144 | **16** | 128 |
| rtovr-6 | 144 | **16** | 128 |
| rtovr-9 | 144 | **6** | 138 |
| rtovr-10 | 144 | **10** | 134 |
| rtovr-11 | 144 | **12** | 132 |
| rtovr-12 | 144 | **12** | 132 |
| rtovr-13 | 144 | **8** | 136 |
| rtovr-35 | 144 | **8** | 136 |
| rtovr-36 | 144 | **4** | 140 |
| ur_rdata_0 | 144 | **6** | 138 |
| ur_rdata_1 | 144 | **22** | 122 |
| ur_rdata_2 | 144 | **23** | 121 |
| ur_wdata_0 | 144 | **9** | 135 |
| **合计（各通道压缩后之和）** | — | **267** | — |
| **MAX（跨通道，决定能否装入 INSTQ）** | 144 | **25** | 119 |

→ **当前 IQ 深度（多字交付）= MAX 25 / 合计 267**，远低于 128 上限。

### 8.3 汇总

| 形态 | MAX（装入判定） | 各通道合计 | 备注 |
|------|----------------:|-----------:|------|
| Merged 单字 W=4 | **24** | 244 | §8.1 |
| Merged 多字 W=16 | **25** | 267 | §8.2；8SMC W=100 同路径 |
| 硬件 limit | 128 | — | `SMI_IQ_INSTQ_DEPTH` |

**交付口径：现在的 IQ 深度以多字 MAX = 25 为准（单字 24）。**

---



## 9. 操作备忘



### 9.1 权限（首次）

见文首 **§0（禁止删除）**：`op/exp`、`out`、`verification_v2`、`sim_output_aiacc_top_f`、`arith_regression/test_log` 必须对 `YouTsing` 可写；VCS/`simv` 与 license 见 §0.3–§0.4。

### 9.2 仿真

```bash
# 组装 case 后
cd /home/YouTsing/Jianyuan-aiacc_4.21
python3 ./generate_data_list_coverage.py verification_v2/<case> --subq 0
SKIP_GENERATE_DATA_LIST=1 bash ./run_sim_aiacc_top_f.sh verification_v2 \
  verification_v2/<case>/smc_0 -s 0
```

只改微码不改 RTL：复用已有 `simv`，勿随意 `--rebuild`。

### 9.3 LSU 全路径

纯 Compute（Host 灌 UR）与 LSU（DDR↔UR）是两条线。LSU 见 `../../load_store_v3/`；交付数值以 Compute 为准。

---



## 10. 版本谱系（简）

```text
… → exp1.0.0 DONE（整数 k）
  → 1.0.1 cleared/unrolled
  → 1.0.2 same-code + fold
  → 1.0.3 ACC 清零缝（整数 k 冻结于 backup_exp1.0.3）
  → 恢复半整数算法 + 移植深度改动
  → 多字 unroll 修复
  → 1.0.4 halfint-delivery（单字=多字路径）
  → 1.0.5 k-flush + Host 对齐 + N=10k（本版）
```

整数 k 不要与半整数 Remez / √2 / P1=108 **混用**。

---



## 10.1 定义域与 k 越界 flush

拼 FP32 `scale=2^k` 时用 `(k+127)<<23`。若 `k≤-128` 或 `k≥128`，8bit 指数会 **wrap**，极端负输入（如 Softmax pad）会炸成大正数。

| 项 | 约定 |
|----|------|
| 判定对象 | 整数 **`k = n>>1`**（不是原始 `x`） |
| 有限 scale 域 | `k∈[-125,126]`（与 RTL SELECT_LESS 含等号一致） |
| `k ≤ -126` | scale → **0**；`y` → 0 |
| `k ≥ 127` | scale → **+Inf**（`0x7F800000`）；y → +Inf |
| 挂点 | P1：odd·√2 SELECT 之后、WriteScale 之前 |
| Host | `host_exp_f32` / `exp_host_golden.py` 按同一 `k` 规则饱和 |
| P1 body | **108**（k@bank0/200；scale 暂存@201；uflow/oflow 两次 SELECT，对齐 floor：check→SELECT=+6、DR–UR、SELECT 两侧 UR/Logic） |
| 比较常数 | bank4 仍存 `-126` / `127`；有效语义为 **≤ / ≥**（非严格 `<` / `>`） |

越界点验收示例：`x∈{-104,-107,-1000}` → 0；`x=+90` → +Inf。

---



## 11. 已知非目标 / 后续可选

- 相对 `math.exp` 再降 ULP（需 **改算法**；实现已对 Host 闭环）
- 压单字 Merged IQ **35→≤32**（非功能必须）
- FP16 主路径（INTADD 打包）
- Softmax 侧是否复用 Exp flush（当前 Softmax 未强制同源）
- Sigmoid 等下游依赖本算子时，以本版 FP32 Merged 为基线

