# formal_corrected_v1 F2 B4 代码审查与 GPU preflight 报告

审查日期：2026-09-08

## 1. 结论

F2 B4 启动前审查与两个任务的 GPU preflight 均为 **PASS**。本次只进行了两步短跑，
没有启动 F2 的 6 个 40,000-step 正式训练。

审查中发现两个 F2 job 配置原先缺少 `data_sha256`，会被正式队列 fail closed。现已补齐
冻结数据哈希，并显式设置 `paired_loss=false`、`effect_weight=0.0`。

## 2. B4 代码与配置审查

| 要求 | 审查结果 | 证据 |
|---|---|---|
| 参数容量与 B6 匹配 | PASS | 在 `latent_dim=192`、`action_dim=2`、main profile 下，B4/B6 均为 10,780,224 个可训练参数；259 个 state-dict tensor 的名称和形状完全相同 |
| 不使用 centered update | PASS | factory 对 B4 设置 `centered=False`，对 B6 设置 `centered=True`；B4 直接使用 actual action residual，不执行 `actual-reference` |
| 不读取 paired-effect 标签 | PASS | `uses_paired_loss` 只有 B6 且未关闭时为真；B4 构造 train/validation split 时均传入 `load_effect=False` |
| 不计算 paired-effect loss | PASS | B4 调用 `per_step_loss(... allow_paired_loss=False)`；effect/direction/magnitude 分支不进入，三个分量严格为 0 |
| F2 job 配置完整 | PASS | 6/6 job 均为 B4、3 seeds、40k steps、batch 8、LR 3e-3、curriculum 1/5/10/20，显式关闭 paired loss 并绑定数据 SHA256 |

main profile 的共同结构为 `model_dim=192`、6 layers、6 heads、
`channel_mlp_dim=3728`、`action_hidden_dim=64`。B4 和 B6 共享相同的 action
decay/erase/write 网络与矩阵更新容量，唯一实验变量是是否从 actual action 参数中减去
reference/no-op action 参数。因此 B4 可以隔离“新增参数容量和未中心化动作更新”的贡献。

动态审查额外确认：

1. 用真实数据构造 `load_effect=False` 的 split 时，实际读取的 8 个 tensor 不包含
   `effect_latents`；batch 内对应占位 tensor 由零值即时构造。
2. 随机化 action residual head 后，更换 B4 的 reference action 不改变输出，证明
   reference branch 未参与 B4 更新。
3. 将 batch 中的 effect label 替换为极大值，B4 总 loss 不变；paired-effect、direction、
   magnitude 三项均为 0。
4. 相关 counterfactual、per-step training、checkpoint 与数据 artifact 测试共 19 项通过。

## 3. 配置与数据哈希复核

| 任务 | F2 配置 | 全文件 SHA256 结果 |
|---|---|---|
| TwoRoom-W | `configs/experiments/formal_corrected_v1/f2_tworoom_b4.json` | `af75f5c81fb936951cd7bf66c524f6ebb5980bd35cc7da8ac3ef7829d7f5ab55` |
| Action-Delay | `configs/experiments/formal_corrected_v1/f2_action_delay_b4.json` | `9d5c1417ecebd8d9cbacb4cb87d350f220588697cb23144e1591594d4db8a5fa` |

两项全文件 SHA256 均与 F2 配置和冻结 protocol manifest 一致。

## 4. B4 GPU preflight

产物目录：

```text
artifacts/runs/per_step_pairs/formal_corrected_v1/f2_preflight_b4_20260908/
```

两个 preflight 均使用 main profile、batch size 8、validation batch size 64、H20
free-running rollout，并完成前向、反向、梯度裁剪、AdamW 更新、H20 validation 和 v3
checkpoint 写入。为缩短资格检查时间，train/validation 分别限制为 8/64 个样本；这不改变
单个 batch 和 H20 rollout 的 GPU 计算形状。

| 任务 | GPU | steps | train loss (step 1/2) | grad norm (step 1/2) | H20 validation | 结果 |
|---|---:|---:|---|---|---:|---|
| TwoRoom-W | 2 | 2 | 1.065613 / 1.108658 | 1.713576 / 0.987676 | 1.032371 | PASS |
| Action-Delay | 3 | 2 | 1.035326 / 1.169102 | 2.669809 / 1.113886 | 1.069231 | PASS |

两项检查中所有 loss/gradient 均为有限值且梯度非零；summary 为
`paired_loss=false`、`effect_weight=0.0`、`effect_threshold=0.0`，三个 paired loss 分量
均严格为 0；最终 checkpoint 为 step 2、`status=complete`、schema v3。

## 5. F2 启动条件

代码、配置、数据哈希和目标 GPU 短跑均已满足 F2 B4 正式训练的启动条件。按照阶段审批
约束，本报告不等同于批准启动 F2；6 个正式运行须在用户明确批准后再开始。
