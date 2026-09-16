# Per-step paired counterfactual formal_corrected_v1 训练计划

## 1. 结果边界

旧目录 `artifacts/runs/per_step_pairs/formal/f1/` 保留为历史结果，不纳入正式比较。
其中旧 B3 未调用 `DWMWorldHead`，且旧 B2/B3 使用 flat triangular reduction；不得与
本计划结果混合。新结果统一写入：

```text
artifacts/runs/per_step_pairs/formal_corrected_v1/
```

训练使用 checkpoint schema `cc_rwkv_per_step_checkpoint_v3` 和 summary schema
`cc_rwkv_per_step_training_summary_v2`。默认每 1,000 steps 原子更新 `latest.pt`，
每 5,000 steps 保留编号 checkpoint。

## 2. 冻结数据和共同设置

| 任务 | 数据 | SHA256 | Train/Val/Test | Steps | Curriculum |
|---|---|---|---|---:|---|
| TwoRoom-W | `tworoom_w/formal_v2_primitive_20000_merged/pairs.h5` | `af75f5c8…ab55` | 16000/2000/2000 | 40,000 | `1,5,10,20` |
| Action-Delay | `action_delay/formal_v2_primitive_20000_merged/pairs.h5` | `9d5c1417…a5fa` | 16000/2000/2000 | 40,000 | `1,5,10,20` |

两个任务均使用 primitive-action 时间单位：`action_block=1`、`model_horizon=20`、
`action_dim=2`。TwoRoom-W 保留 15 primitive steps 历史，Action-Delay 保留完整 FIFO
所需的 5 steps 历史。共同设置：profile `main`、seeds `0/1/2`、AdamW、learning rate
`3e-3`、batch size `8`、free-running rollout、禁止 teacher forcing、
position-balanced triangular loss。优化器固定为官方风格 RWKV-7 AdamW 参数分组：
`time_mix.w0` 使用 2 倍学习率，矩阵投影权重使用 `1e-3` weight decay，其余参数不衰减。
验证使用独立 batch size `64`，覆盖完整 2,000 样本并按样本数加权聚合；step 1 只记录
训练损失，从 step 1,000 起每 1,000 steps 执行一次完整 H20 validation。
正式启动前须在目标 GPU 对四种方法完成短跑显存/数值检查；若 batch size 或 learning
rate 需要改变，必须在任何正式任务开始前整体修改并重新冻结，不能按方法单独调整。

## 3. F0：代码和协议资格检查

- B3 在每个 factual position 从同一 RWKV state 比较真实动作与确定性 batch-permuted
  动作，训练 DWM contrastive 与 orthogonality loss；alternative state 不写回主轨迹。
- B3 world-head 参数必须获得有限且非零梯度。
- B2/B3/B4 不加载 paired-effect 标签；B6 正式组才加载。
- 三角损失先在每个 intervention position 内按 offset 平均，再平均 position/batch。
- B6 structure-only 组使用 centered architecture，但 `paired_loss=false`、
  `effect_weight=0`，确保不读取 effect 标签。
- 中断/恢复、schema、数据审计、配置不匹配拒绝恢复测试全部通过。

F0 产物：`docs/formal_corrected_v1_f0_report.md`。F0 未通过不得开始 F1。

## 4. F1：B2 与正确 B3 基线

任务配置：

- `configs/experiments/formal_corrected_v1/f1_tworoom_baselines.json`
- `configs/experiments/formal_corrected_v1/f1_action_delay_baselines.json`

共 12 runs：2 tasks × 2 methods × 3 seeds。B3 固定
`contrastive=0.3`、`orthogonality=0.5`、`temperature=0.07`。B2/B3 均只使用
factual/no-op absolute targets；B3 额外使用由 batch action permutation 构造的 DWM
训练信号，不读取真实 paired-effect 标签。

F1 产物：全部 checkpoint/history/summary、队列 status 和
`docs/formal_corrected_v1_f1_report.md`。

## 5. F2：B4 容量控制

任务配置：

- `configs/experiments/formal_corrected_v1/f2_tworoom_b4.json`
- `configs/experiments/formal_corrected_v1/f2_action_delay_b4.json`

共 6 runs。B4 具有与 B6 匹配的动作 erase/write 容量，但不进行 centered update，
也不读取 paired-effect 标签。用于区分参数容量与反事实机制贡献。

F2 产物：`docs/formal_corrected_v1_f2_report.md`。

## 6. F3：B6 effect-weight 选择

任务配置：

- `configs/experiments/formal_corrected_v1/f3_tworoom_b6_weights.json`
- `configs/experiments/formal_corrected_v1/f3_action_delay_b6_weights.json`

每个任务训练 `effect_weight ∈ {0.5, 1.0, 2.0}`、seeds `0/1/2`，共 18 runs。
每个任务只能根据 validation split 跨三个 seed 的预注册分数选权重，test split 在选择
完成前不得查看。

F3 产物：权重选择 JSON 和 `docs/formal_corrected_v1_f3_report.md`。

## 7. F4：B6 structure-only 消融

任务配置：

- `configs/experiments/formal_corrected_v1/f4_tworoom_b6_structure_only.json`
- `configs/experiments/formal_corrected_v1/f4_action_delay_b6_structure_only.json`

共 6 runs。使用 centered B6 recurrence，但关闭 paired-effect 数据加载及其 loss。该组与
B4 区分 centered recurrence 的结构贡献，与正式 B6 区分 paired supervision 的贡献。

F4 产物：`docs/formal_corrected_v1_f4_report.md`。

## 8. F5：冻结测试与汇总（无新增训练）

完成 validation 选权后，一次性评估 B2、B3、B4、选中权重 B6 和 B6 structure-only。
输出至少包含 factual/no-op RMSE、按 future offset 和 intervention position 分解的 effect
误差、direction、magnitude、Action-Delay onset、effect persistence/decay，以及三 seed
paired bootstrap CI。记录参数量、训练时间、峰值显存和监督 target 数。

F5 产物写入 `artifacts/results/cc_rwkv/per_step_pairs/formal_corrected_v1/`，并生成
`docs/formal_corrected_v1_f5_report.md`。完成 F5 后才决定是否进入 RL 控制训练。

## 9. 单 GPU 队列命令模板

```bash
.venv/bin/python scripts/run_formal_stage_queue.py \
  --device cuda:0 \
  --stage <stage_name> \
  --jobs <job_json> \
  --status artifacts/runs/per_step_pairs/formal_corrected_v1/<status>.json \
  --checkpoint-interval 1000 \
  --snapshot-interval 5000 \
  --resume-existing
```

重新执行同一命令会跳过完整且配置匹配的 v2 summary；未完成任务从 v3 `latest.pt`
恢复。旧 schema、旧归一化或配置不匹配时 fail closed，不允许静默混用。

## 10. 训练数量

| 阶段 | Runs |
|---|---:|
| F1 B2/B3 | 12 |
| F2 B4 | 6 |
| F3 B6 weights | 18 |
| F4 B6 structure-only | 6 |
| 合计 | 42 |
