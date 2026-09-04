# CC-RWKV-WM M5 执行报告

> 日期：2026-09-02  
> 状态：**M5 正式 5,000-snapshot 双任务实验已完成；72/72 个评估单元、10,000 次 paired bootstrap 与 Gate A/B@20 报告均已生成。M5@20 overall=FAIL；完整 Gate B=NOT_ASSESSABLE。**  
> 边界：没有开始 M6，没有把 32-snapshot 结果解释成方法有效性证据。

## 1. 结论先行

M5 所需的逐样本评估、effect-weight 选择、三 seed 聚合、10,000 次 paired
bootstrap、Gate A/B 自动判定、effect mask 和失败样本审计已经实现，并在
TwoRoom-W 与 Action-Delay 上跑通完整工程流程。

先前正式统计实验曾因两个硬前提不成立而暂停：

1. 当前没有 TwoRoom-W/Action-Delay 的正式缓存；两者仅有本次生成的 32-snapshot
   smoke 数据；
2. 隔离执行环境中的 PyTorch 和 `nvidia-smi` 无法连接 NVIDIA 驱动。

2026-08-29 复核确认第二项只是沙箱设备可见性问题，并非 GPU 被占用或驱动故障；
宿主机可见 4 张 RTX A6000。随后已经完成并审计两套正式缓存：TwoRoom-W 和
Action-Delay 均为 5,000 snapshots、固定 `4,000/500/500` split，最长均为 20
primitive steps；对应 branch dataset SHA256 分别为
`dbf0c4c576303e8c078a6db58575c07126c7d30ce5e7d172feb72c6aee6411f6` 和
`5c7a95562f878a94a4d99eecf9833e4eefcad254569fbb8abd7aa82353b8b293`。正式三 seed
主训练于 2026-08-29 20:51（Asia/Shanghai）按用户要求安全暂停；12 个训练进程均已
退出，GPU 已释放，所有 `latest.pt` 均通过反序列化检查。2026-08-30 14:54 使用相同
命令和 `--resume-existing` 原位恢复；12/12 个 run 的首个新 checkpoint 均从暂停点
严格增加 500 steps，证明 model、optimizer、scheduler、curriculum、RNG 和 global
step 恢复链路正常。暂停点与恢复审计记录在
`artifacts/results/cc_rwkv/m5/pause_state.json`。完整训练、独立评估与聚合完成前仍
保持 `NOT_ASSESSABLE`。

此前自动报告输出 `NOT_ASSESSABLE` 是因为正式运行尚未完成；最终报告按预注册口径给出已评估的 `FAIL`，而不是将未满足 20,000/50 前提的完整 Gate B 误报为通过。M6 未开始。

## 2. 本次实现

### 2.1 M5 逐样本指标与统计

新增 `src/cape_wm/cc_rwkv/m5.py`：

- 保留每个 `sample_id` 的 CEE AUC、trajectory AUC、one-step、reference/no-op
  one-step、Rollout@20/50；
- 先对同一样本的 3 个训练 seed 求均值，再以 sample ID 做 10,000 次 paired
  bootstrap，避免把训练 seed 当作独立测试样本；
- 验证集 effect weight 搜索固定为 `0.5/1.0/2.0`，先应用相对 B2 的 one-step
  3% 淘汰线；
- 自动选择 B3/B4 中 CEE 更低者作为 Gate B 的最强基线；
- Gate B 同时检查 CEE 相对改善 15%、CEE 差值 CI、Rollout@20、Rollout@50 和
  one-step 退化；
- 自动拒绝不完整方法矩阵、seed 矩阵、数据规模、训练 horizon、公平性或 rollout
  horizon；
- 保存 true-effect norm、mask、有效比例和 B6 劣于基线的样本。

宽矩阵 probe 改用 dual ridge，避免主配置下构造约 `36864 x 36864` 的矩阵；R²
忽略零方差动作维。动作恢复的 Gate A 判据使用 probe MSE 相对训练集均值预测器的
paired-bootstrap CI，而不只看训练 R²。

### 2.2 入口与配置

新增：

- `scripts/evaluate_cc_rwkv_m5.py`：独立 checkpoint 的 validation/test 逐样本评估；
- `scripts/materialize_cc_rwkv_m5_evaluations.py`：物化 seed/method 矩阵；
- `scripts/run_cc_rwkv_m5_effect_search.py`：训练并评估 B6 三个 effect weight；
- `scripts/aggregate_cc_rwkv_m5.py`：选择超参数、聚合 CI、输出 Gate 和失败样本；
- `scripts/audit_cc_rwkv_m5_readiness.py`：检查正式缓存、action block、horizon 和 GPU；
- `configs/cc_rwkv/tworoom_w.yaml`；
- `configs/cc_rwkv/action_delay.yaml`。

`train_cc_rwkv.py` 新增显式 `--curriculum`，因此 TwoRoom-W 使用 model-step
`1,2,4`，Action-Delay 使用 primitive/model-step `1,5,10,20`，不再把两个任务错误地
套用同一组 horizon。

正式评估默认使用完整的 4,000-snapshot train split 拟合冻结的 effect-norm 阈值；
`--train-limit` 只保留为显式 smoke/debug 开关，避免评估阶段静默退回 512 个训练样本而
改变 CEE mask 口径。

正式运行前的协议审计还修正了三项会改变结论的偏差：

- curriculum transition 使用综合 validation score 最低的 `best_h*.pt`；validation/test
  和 B2→B6 初始化则强制使用最终 model horizon endpoint rollout error 最低的
  `best_rollout_h*.pt`，而不是 `latest.pt`；
- curriculum 扩展时恢复前一级最佳模型和 Adam 状态，同时保留当前 global step、
  scheduler 和新 horizon，且把每次恢复写入 `curriculum_transitions.jsonl`；
- main-profile B6 必须从同 seed、已完成训练的 B2 最终-horizon 最佳 checkpoint
  初始化，复制全部 shape-compatible world/readout 张量并写入 `initialization.json`；
  effect-search 入口会拒绝尚未达到完整 optimizer-step 预算的 B2。
- B6 的 `effect/direction/magnitude` 真实 paired supervision 严格裁剪到第一个 model
  step（`H=1`）；多步仍训练普通 prediction/world rollout，但不会提前使用属于 B7 的
  多步 paired 标签。

正式 evaluator 会把上述 checkpoint selection、transition restore 和 B2→B6 初始化
写入 `formal_training_protocol`；对 5,000-snapshot 正式数据，任一审计失败都会直接拒绝
生成可聚合的 `m5_run.json`。该协议变更对应 `cc_rwkv_m5_v2` schema。

在这些检查加入前产生的 B2/B3/B6 checkpoint 不进入正式聚合，已可恢复地隔离到
`artifacts/results/cc_rwkv/m5_invalid_no_best_transition_20260829_1945/` 等目录；正式
队列已按修正协议重新开始。

### 2.3 更严格的 Gate 口径

原计划存在一个口径冲突：M5 数据阶梯是 5,000 snapshots、最长 20 primitive
steps，但预注册 Gate B 要求 Rollout@20 和 Rollout@50；第 10.3 节又规定 Gate B
使用 20,000 snapshots。自动判定采用较严格、不会产生假阳性的解释：

- Gate A：至少 5,000 snapshots、训练到 20 primitive steps；
- 完整 Gate B：至少 20,000 snapshots、训练和评估到 50 primitive steps。

所以 M5 的 5,000/20 实验最多完成 Gate A 和 Gate B@20 候选筛查；完整 Gate B
必须等 M6 的 20,000/50 数据，且仍需再次获得用户批准后才能进入。

聚合器分别输出 `gate_b_at_20`/`m5_at_20_overall_status` 与完整
`gate_b`/`overall_status`：前者在 5,000 snapshots、20 primitive steps 下可判定，
后者仍严格要求 20,000/50。最终 Markdown 同时展示两列，避免把 M5 候选结论误写成
完整 Gate B 结论。

## 3. 工程预演

### 3.1 数据与运行矩阵

本次新采集并审计：

| Task | snapshots | action block | model horizon | primitive horizon | split |
|---|---:|---:|---:|---:|---:|
| TwoRoom-W | 32 | 5 | 4 | 20 | 26/3/3 |
| Action-Delay-5 | 32 | 1 | 20 | 20 | 26/3/3 |

每个任务运行：

- B2/B3/B4/B6；
- seeds 0/1/2；
- B6 effect weights 0.5/1.0/2.0；
- 每个 run 100 CPU optimizer steps；
- validation 和 test 使用相同冻结 sample IDs；
- 聚合使用 10,000 次 paired bootstrap。

### 3.2 自动结果

验证集工程选择：

| Task | w=0.5 score | w=1.0 score | w=2.0 score | selected |
|---|---:|---:|---:|---:|
| Action-Delay | 0.92259 | 0.92385 | 0.92747 | 0.5 |
| TwoRoom-W | 1.20521 | 1.18998 | 1.21129 | 1.0 |

测试集的工程诊断（仅 3 个 sample IDs，不能做科学结论）：

| Task | strongest B3/B4 | B6 CEE AUC | baseline CEE AUC | relative change | Rollout@20 Δ |
|---|---|---:|---:|---:|---:|
| Action-Delay | B4 | 0.20579 | 0.20607 | -0.14% error | -0.02501 |
| TwoRoom-W | B4 | 0.85803 | 0.83385 | +2.90% error | +0.06425 |

这里负的 Rollout 差值表示 B6 error 更低。TwoRoom-W 在当前极小预算下更差；
Action-Delay 的 CEE 差异远低于 15% 门槛。两者 probe 均未超过均值基线，且实际训练
curriculum 没有到达 20 primitive steps。因此即使忽略样本量，也不能宣布 Gate A/B。

机制中已验证的纯工程不变量：B6 `actual==reference` delta 为 0、raw action 只进入
matrix update network、gate 未饱和。它们是必要条件，不是效果充分条件。

## 4. 产物

- 正式就绪审计：`artifacts/results/cc_rwkv/m5/readiness.json`；
- 工程聚合：`artifacts/results/cc_rwkv/m5_engineering/aggregate/summary.json`；
- 自动 Gate：`artifacts/results/cc_rwkv/m5_engineering/aggregate/gate_report.md`；
- 失败样本：`artifacts/results/cc_rwkv/m5_engineering/aggregate/failed_samples.jsonl`；
- 各 seed/method 的 checkpoint、validation/test 逐样本指标：
  `artifacts/results/cc_rwkv/m5_engineering/<task>/seed_<n>/`。

关键 SHA256：

- aggregate summary：`9498233572c60221107343db80c9556014680ead02b0689d8640e6d78f187b02`；
- automatic gate report：`5ffaa932e31ea4fb88c9737cd66047f97f1b4c922cfa7e6399768aac97dd0d89`；
- readiness：`fe3bd2a93ec2ff1f45d32b11fff5fac871ed14bed6538f49233c637949639045`；
- TwoRoom-W smoke HDF5：`bb29bd1b9bcd7a2dd64513f0b28926534ececf00f4ca30ad352684f5bc3a7eb5`；
- Action-Delay smoke HDF5：`a4fee81b83b00a872fde186ecc346f302e3d6e073e7e98d1b4a510a2ddcdb0a3`。

## 5. 正式 M5 的执行入口

以下命令记录正式缓存的可复现实验入口；当前两套缓存已经采集完成并通过 readiness
audit：

```bash
.venv/bin/python scripts/collect_counterfactual_branches.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt \
  --model-config /data/wjy/lewm_model/tworooms/config.json \
  --frozen-test-episodes artifacts/results/lewm_tworooms_long_matrix_gpu_final/matrix_episodes.jsonl \
  --output artifacts/cache/cc_rwkv/tworoom_w/mvp5000 \
  --cache-dir artifacts/cache/cc_rwkv/m2/runtime \
  --variant tworoom_w --samples 5000 --history-steps 3 \
  --future-primitive-steps 20 --action-block 5 --device cuda:0

.venv/bin/python scripts/collect_counterfactual_branches.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt \
  --model-config /data/wjy/lewm_model/tworooms/config.json \
  --frozen-test-episodes artifacts/results/lewm_tworooms_long_matrix_gpu_final/matrix_episodes.jsonl \
  --output artifacts/cache/cc_rwkv/action_delay/mvp5000 \
  --cache-dir artifacts/cache/cc_rwkv/m2/runtime \
  --variant action_delay --samples 5000 --history-steps 5 \
  --future-primitive-steps 20 --action-block 1 --device cuda:0
```

随后使用 main profile 分别运行 B2/B3/B4/B6 seeds 0/1/2，并运行 effect search。
正式聚合命令固定为：

```bash
.venv/bin/python scripts/aggregate_cc_rwkv_m5.py \
  --root artifacts/results/cc_rwkv/m5 \
  --output artifacts/results/cc_rwkv/m5/aggregate \
  --bootstrap-samples 10000 --bootstrap-seed 3072
```

恢复正式实验时必须确认每个 checkpoint 中的 `trained_max_primitive_horizon >= 20`；
否则聚合器会拒绝 Gate A。

## 6. 验收边界

本次已经完成 M5 的代码、统计、双任务三 seed 工程预演、正式数据缓存和失败路径落盘。
尚未完成的是全部正式 GPU checkpoint、独立 validation/test 评估与最终聚合，因此 M5
不能标为“正式完成”，也不能批准进入 M6。当前队列已按用户要求从暂停点恢复并继续
正式 M5，且不能用 M6/RL 绕过当前 Gate。

## 7. 正式运行实时状态（2026-09-02 03:53 CST）

此节记录正式产物目录的当前事实；第 3 节仍是早期 32-snapshot 工程预演，不能与本节
混作正式科学结果。

- TwoRoom-W 与 Action-Delay 正式缓存均为 5,000 snapshots，固定 split 为
  `4,000/500/500`；
- TwoRoom-W 的 B2/B3/B4/B6 三 seed 主配置、B6 `effect-weight=0.5/2.0` 搜索及全部
  validation/test 已完成；
- Action-Delay 的 B2/B3/B4/B6 三 seed 主配置、B6 `effect-weight=0.5` 及相应
  validation/test 已完成；
- 当前完成度审计为 `66/72` 个正式评估单元；唯一缺项是正在运行的 Action-Delay
  B6 `effect-weight=2.0` 三 seed validation/test，共 6 个单元；
- Action-Delay `effect-weight=2.0` 最新训练步数为 seed0 `16,500/40,000`、seed1
  `10,500/40,000`、seed2 `16,000/40,000`；三个进程均正常；
- 当前不能执行正式 effect-weight 选择、测试集 paired bootstrap 或 Gate A/B@20
  判定；这些步骤必须等待上述 6 个评估单元全部物化；
- M6、B7、50/100-step 数据采集与 RL 均未开始。

当前机器可核验的完成度证据为
`artifacts/results/cc_rwkv/m5/completion_audit.json`。该文件会在全部训练、评估和最终
聚合结束后重新生成；只有最终状态为 `complete` 且无 failure，才能将本报告状态改为
“正式 M5 完成”。

## 8. 正式 M5 完成结果（2026-09-02 12:47 CST）

本节覆盖第 6、7 节的运行中描述，以本节和产物文件为准：

- 两个任务均完成 B2/B3/B4/B6 三 seed 主配置、B6 `effect-weight=0.5/1.0/2.0`
  验证/测试评估；共 72 个正式评估单元，每个 validation/test 均为 500 个唯一
  `sample_id`。
- 完成审计为 `status=pass`、`observed=72/72`、`failures=[]`：
  `artifacts/results/cc_rwkv/m5/completion_audit.json`。
- 聚合使用 10,000 次 paired bootstrap（seed=3072）：
  `artifacts/results/cc_rwkv/m5/aggregate/summary.json`。
- 两个任务的 effect-weight 选择均为 `no_eligible_candidate`；候选权重均因相对
  B2 的 one-step 3% 保护线未通过，故不选定测试权重。
- Gate A：Action-Delay 与 TwoRoom-W 均 `FAIL`；Gate B@20：两个任务均 `FAIL`。
  主要原因包括 CEE 改善未达到 15%、bootstrap CI 跨零、one-step 退化超过 3%，
  且 rollout@20 未改善；TwoRoom-W 还未通过 probe/均值基线检查。
- 完整 Gate B 均为 `NOT_ASSESSABLE`，因为本阶段固定为 5,000 snapshots、20
  primitive steps，尚未满足 20,000/50 的预注册条件。这不是把 M5 的负结果隐藏为
  “未完成”，而是严格区分 M5@20 与完整 Gate B。
- 最终 Gate 报告：`artifacts/results/cc_rwkv/m5/aggregate/gate_report.md`。
  M6、B7、50/100-step 数据扩展和 RL 控制均未开始。
