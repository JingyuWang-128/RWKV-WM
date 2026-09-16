# formal_corrected_v1 F1 完整性审计与三 seed 基线报告

审计日期：2026-09-08

## 1. 结论

F1 **通过完整性审计**。TwoRoom-W 与 Action-Delay 上的 B2、B3 均完成
seed 0/1/2，共 12/12 个正式运行、480,000/480,000 个优化步骤。所有运行均有完整的
`summary.json`、40,000 行连续 `history.json`、8 个每 5,000 steps 保存的编号
checkpoint，以及状态为 `complete` 的 40,000-step checkpoint。

本报告只分析 validation split 上的训练目标，不读取 test split。当前结果可用于检查基线
稳定性和决定是否继续 F2，但不能代替 F5 的冻结测试集评估，也不能直接说明长程 effect
方向、幅值或持续性性能。

## 2. 审计范围与协议一致性

正式结果目录为：

```text
artifacts/runs/per_step_pairs/formal_corrected_v1/f1/
```

旧目录 `artifacts/runs/per_step_pairs/formal/f1/` 未纳入本报告。

| 检查项 | 结果 |
|---|---|
| 预期运行 | 2 tasks × 2 methods × 3 seeds = 12 |
| 完成运行 | 12/12 |
| 总优化步骤 | 480,000/480,000 |
| summary schema | 12/12 为 `cc_rwkv_per_step_training_summary_v2` |
| checkpoint schema | 12/12 为 `cc_rwkv_per_step_checkpoint_v3` |
| 最终 checkpoint | 12/12 为 step 40,000、`status=complete` |
| checkpoint 可恢复载荷 | 12/12 含 model、optimizer、history、torch/CUDA/batch RNG state |
| 中间 checkpoint | 每个运行均为 5k、10k、15k、20k、25k、30k、35k、40k，共 8 个 |
| history | 每个运行 40,000 行，step 1–40,000 连续，无 NaN/Inf |
| curriculum | 精确为 10k×H1、10k×H5、10k×H10、10k×H20 |
| validation | step 1,000 起每 1,000 steps 一次，共 40 次完整 H20 validation |
| 共同设置 | main profile、batch 8、validation batch 64、LR 3e-3、free-running |
| 损失归一化 | 12/12 为 `position_balanced_v1` |
| paired-effect | B2/B3 均为 `paired_loss=false`、阈值 0，三个 paired 分量始终为 0 |
| B3 DWM | 三个 seed 的 DWM contrastive 分量均实际参与训练 |
| 协议绑定 | 12/12 绑定同一 protocol manifest SHA256 |

冻结数据绑定：

- TwoRoom-W：`af75f5c81fb936951cd7bf66c524f6ebb5980bd35cc7da8ac3ef7829d7f5ab55`
- Action-Delay：`9d5c1417ecebd8d9cbacb4cb87d350f220588697cb23144e1591594d4db8a5fa`
- protocol manifest：`34eb23edf857757972b464bf27f78dd34fa05c70e895beeba746c2d620d867c8`

## 3. 三 seed validation 对比

下表使用各运行 **step 40,000 的完整 H20 validation loss**；越低越好。`±` 后为三个
seed 的样本标准差。

| 任务 | 方法 | seed 0 | seed 1 | seed 2 | mean ± std | median |
|---|---|---:|---:|---:|---:|---:|
| TwoRoom-W | B2 | 0.640917 | 0.630648 | 0.633933 | **0.635166 ± 0.005244** | 0.633933 |
| TwoRoom-W | B3 | 0.645736 | 0.648293 | 0.664698 | 0.652909 ± 0.010289 | 0.648293 |
| Action-Delay | B2 | 0.326818 | 0.320263 | 0.683630 | **0.443570 ± 0.207924** | 0.326818 |
| Action-Delay | B3 | 0.554244 | 0.433078 | 0.376623 | 0.454648 ± 0.090754 | 0.433078 |

按相同 seed 配对的 `B3 − B2`：

| 任务 | seed 0 | seed 1 | seed 2 | 平均差 | B3 相对 B2 均值 |
|---|---:|---:|---:|---:|---:|
| TwoRoom-W | +0.004819 | +0.017645 | +0.030765 | +0.017743 | +2.79% |
| Action-Delay | +0.227426 | +0.112816 | -0.307007 | +0.011078 | +2.50% |

正数表示 B3 的 validation loss 更高。TwoRoom-W 上 B2 在三个 seed 中均优于 B3，且
B2 方差更小。Action-Delay 上 B2 的均值略优，但 seed 2 出现明显高损失，导致其方差很
大；B3 的均值没有超过 B2，但跨 seed 更稳定。只有三个 seed，不能据此进行强显著性
结论。

## 4. 最优 validation 记录

该表用于观察训练轨迹，不表示已经选择这些 checkpoint。部分最优 step（37k/38k/39k）
不是每 5k 保留的编号 snapshot，因此 F5 应按预注册规则统一决定使用最终 checkpoint 还是
可恢复的验证最优 snapshot，不能事后按 test 结果挑选。

| 任务 | 方法 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| TwoRoom-W | B2 | 0.632118 @ 37k | 0.614943 @ 38k | 0.633933 @ 40k |
| TwoRoom-W | B3 | 0.645736 @ 40k | 0.648293 @ 40k | 0.664698 @ 40k |
| Action-Delay | B2 | 0.326818 @ 40k | 0.315457 @ 39k | 0.683630 @ 40k |
| Action-Delay | B3 | 0.554244 @ 40k | 0.433078 @ 40k | 0.376623 @ 40k |

## 5. 当前能够与不能够得出的结论

能够得出：

1. 正式 F1 产物完整、协议字段一致，能够作为后续 B4/B6 的基线。
2. 在当前 overall H20 validation objective 上，B3 没有优于 B2。
3. Action-Delay 对初始化较敏感；B2 seed 2 是后续 F5 必须保留并报告的真实 seed 方差，
   不应删除或只报告较好 seed。

暂时不能得出：

1. 不能用该 loss 判断 B2/B3 在 effect direction、magnitude、onset 或 persistence 上的
   优劣；这些指标须在 F5 统一计算。
2. 不能说明 centered matrix update 或 paired supervision 是否有效，因为 B4/B6 尚未完成。
3. 不能查看或据 test split 选择方法、权重或 checkpoint。

## 6. F1 阶段状态

**F1：PASS / COMPLETE。** 下一阶段是 F2 B4 容量控制实验；正式启动前必须完成 B4
代码/config 审查和两个任务的目标 GPU preflight。
