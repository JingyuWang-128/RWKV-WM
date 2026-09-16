# formal_corrected_v1 F0 报告

日期：2026-09-06  
状态：**PASS / 允许启动 F1**

> 2026-09-05 复核发现旧 v1 sampler 在生成 `start_row` 时复用了上一次循环的
> episode table index，导致声明的 episode/start 与实际数据行可能错配。旧数据和基于
> 旧数据产生的训练结果全部退出正式比较。下文早期 GPU preflight 只证明模型代码可运行，
> 不再构成正式数据准入。

2026-09-06 已完成 primitive-action v2 正式数据重采集、六分片合并和全量审计。
TwoRoom-W 与 Action-Delay 均为 20,000 个唯一样本，23 项审计全部通过。

## 完成的修复

1. B3 DWM objective 已进入逐位置训练：每个 factual position 都从同一 factual
   RWKV state/latent 计算真实动作和确定性 batch-permuted 动作视图；仅 factual state
   继续递推。
2. DWM world contrastive、world/action orthogonality 被加入总 loss 和 history；默认
   权重分别为 `0.3`、`0.5`，temperature 为 `0.07`。
3. triangular suffix loss 改为 `position_balanced_v1`，先平均每个 intervention
   position 的有效 offsets，再平均 position/batch。
4. 新增 B6 structure-only 模式：`paired_loss=false`、`effect_weight=0`，数据加载阶段
   即不读取 effect 标签。
5. checkpoint schema 升至 v3，run config 保存 protocol、DWM 配置、paired-loss 开关和
   loss normalization；旧 checkpoint 不能静默恢复。
6. queue 支持任务级 batch size、learning rate、DWM 参数和 paired-loss 开关，只跳过
   summary v2 且完整配置匹配的任务。
7. 正式 checkpoint、summary 和 queue 跳过/恢复判据均绑定数据 SHA256；queue 在阶段
   启动前独立重算数据文件 SHA256，训练入口同时要求 manifest 与 PASS audit。
8. 逐位置训练使用官方风格 RWKV-7 AdamW 参数分组，而非对所有参数统一 weight decay。

## 测试结果

- 本地主测试集：131 passed，5 个既有 Transformer warning，无失败。
- 服务器目标测试：24 passed，无失败。
- 真实 TwoRoom-W HDF5、B3 smoke：world contrastive 非零，world-head 梯度非零。
- 服务器 Action-Delay H20/main/batch 8 GPU preflight：B2/B3/B4/B6 均完成 2 steps，
  无 OOM、NaN 或 Inf。
- B3 H20 step 2：contrastive `9.7852`、orthogonality `0.8197`、gradient norm
  `5.9339`；与 B2 不再相同。
- B6 H20 step 2：paired effect `0.1583`、direction `1.0020`、magnitude `9.1405`，
  相应监督项已进入训练。

- 新 v2 数据在当前 GPU 3 上完成两任务 × B2/B3/B4/B6 的 H20/main/batch 8 两步预检；
  八组最终 loss 和 gradient norm 均有限，无 OOM、NaN 或 Inf。
- 新预检中 B3 的 DWM contrastive 非零，B6 的 paired-effect 非零；B2/B4 对应项为零。
- checkpoint 中存在基础 `3e-3` 与 decay-logit `6e-3` 两类学习率组，且绑定正确数据 SHA。

## 有效数据完整性记录

- TwoRoom-W：`af75f5c81fb936951cd7bf66c524f6ebb5980bd35cc7da8ac3ef7829d7f5ab55`
- Action-Delay：`9d5c1417ecebd8d9cbacb4cb87d350f220588697cb23144e1591594d4db8a5fa`
- 两者 split 均为 16,000/2,000/2,000，`action_block=1`、`model_horizon=20`。
- 两者最大 effect alignment error 均为 `0.00390625 < 0.01`，audit status 为 PASS。

## 已作废的数据完整性记录

- TwoRoom-W SHA256：`48912d2be252a47e2c43e58f1606f7212321e26a3c3b254461c475e70bfe6b55`
- Action-Delay SHA256：`15a10c0850cf4891984d214b812494ac60558e183fb35befde2fcbbd515518e9`

两个哈希只能证明文件传输完整，旧 audit 没有保存并对照 `source_row`，因此没有发现
episode/start 错配。它们不得绑定到新的 formal protocol。

## 结果隔离

旧 `artifacts/runs/per_step_pairs/formal/f1/` 不纳入正式比较。新训练只能写入
`artifacts/runs/per_step_pairs/formal_corrected_v1/`。完整代码哈希、数据哈希和 schema
记录于 `configs/experiments/formal_corrected_v1/protocol_manifest.json`。

## F1 准入结论

F0 所有准入条件已经满足，**允许启动 F1**。F1 只能使用上述两个冻结 SHA256 对应的
primitive-action v2 数据，输出写入 `artifacts/runs/per_step_pairs/formal_corrected_v1/f1/`；
旧 `formal/f1` 结果仍不得纳入正式比较。
