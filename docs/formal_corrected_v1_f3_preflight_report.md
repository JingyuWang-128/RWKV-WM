# formal_corrected_v1 F3 B6 effect-weight 审查与 preflight 报告

状态：**PASS / 允许启动 F3**。

## 审查结论

- F2 B4 已完成 6/6 个正式运行，详见 `docs/formal_corrected_v1_f2_report.md`。
- F3 包含 TwoRoom-W、Action-Delay，`effect_weight ∈ {0.5, 1.0, 2.0}`，
  每个权重使用 seed 0/1/2，共 18 个无重复运行。
- 18 个 job 均绑定冻结数据 SHA256，显式设置 `method=b6`、`paired_loss=true`，
  使用 40,000 steps、batch size 8、learning rate 3e-3 和 `1/5/10/20` curriculum。
- B6 使用 centered actual-minus-zero-reference action parameter；factual 状态在每个位置
  持久更新。pulse-noop 从相同 factual prefix/state 分叉，在当前位置执行零动作，随后执行
  原 factual suffix；临时分支不污染 factual 状态。
- paired-effect loss 覆盖完整 `[batch, intervention_position, future_offset]` 三角 mask，
  不是只监督第一个位置或第一个 future offset。
- 数据与训练均定义 effect 为 `pulse_noop - factual`，符号一致。
- B4/B6 predictor 参数量均为 10,780,224，state-dict 结构一致；差异是 B6 centered
  subtraction 与 paired supervision，不是额外容量。
- 非有限训练 loss、组件、梯度、验证 loss、模型或 optimizer checkpoint 会立即终止并
  写入 `failure.json`。

## 静态与单元测试

- Ruff：PASS。
- Pytest：113 passed；5 个 warning 均来自既有 Transformer API。
- 新增回归测试验证每个 intervention position 的 effect target 都会改变 paired loss。
- 新增回归测试验证向量化 pulse-noop 路径与逐位置 `zero-at-t + factual suffix` 路径一致。
- F3 四个队列的并集与两个权重配置严格一致：18/18，无重复、无遗漏。

## GPU H20 preflight

主模型、batch size 8、validation batch size 64、paired loss 开启，使用 H20 两步短跑：

| 任务 | step 2 train loss | step 2 validation loss | paired effect | 结果 |
|---|---:|---:|---:|---|
| TwoRoom-W | 3.136660 | 2.991602 | 0.374110 | PASS |
| Action-Delay | 2.231527 | 2.271954 | 0.123109 | PASS |

两个 preflight 的所有 loss/gradient/checkpoint tensor 均为有限值，且 summary 明确记录
`paired_loss=true`、`effect_weight=1.0`。产物位于：

`artifacts/runs/per_step_pairs/formal_corrected_v1/f3_preflight_b6_20260912/`

## 冻结协议

F3 使用：

`configs/experiments/formal_corrected_v1/protocol_manifest_f3_b6_weights_v2.json`

manifest SHA256：

`584be25a3287f4643572ae788cef7be4205ebbdde2057fa859c9a927780d6585`

权重只能根据 validation split 的三 seed 预注册聚合分数选择；选择完成前不得访问 test
split。正式队列使用 `nohup + setsid`，每 1,000 步保存 `latest.pt`、每 5,000 步保存
编号 checkpoint，并支持 `--resume-existing`。
