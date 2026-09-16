# formal_corrected_v1 F2 B4 完整性报告

状态：**PASS / COMPLETE**。

F2 的 TwoRoom-W 与 Action-Delay B4 seed 0/1/2 均完成 40,000 optimizer steps，
共 6/6 个正式运行。所有运行均使用修复后的有界 action rank-two update、相同冻结数据、
`1/5/10/20` primitive-action curriculum、batch size 8 和 learning rate 3e-3。

| 任务 | seed 0 H20 val | seed 1 H20 val | seed 2 H20 val | 三 seed mean ± sample SD |
|---|---:|---:|---:|---:|
| TwoRoom-W | 0.773544 | 0.771012 | 0.779053 | 0.774536 ± 0.004111 |
| Action-Delay | 0.775183 | 0.713688 | 0.771884 | 0.753585 ± 0.034591 |

完整性审计：每个输出均包含 v2 `summary.json`、40,000 行 `history.json`、最终
`latest.pt` 和 step 40,000 编号 checkpoint；history 中没有 NaN/Inf，且没有任何
`failure.json`。六个 summary 均绑定修复版 manifest SHA256
`5901a6e984b8cd2aaca34907e1f6d85b00e6431321dd2a104cb16665fd1a2e4b`。

B4 与 B6 predictor 参数量均为 10,780,224，state-dict 名称和 shape 相同。B4 不执行
reference subtraction、不读取 paired-effect 标签且 paired-effect loss 为零，因此能够作为
F3 B6 的容量匹配对照。

下一阶段为 F3：在 validation split 上对 B6 的 effect weight 0.5/1.0/2.0 做三 seed
选择；权重选择完成前不得访问 test split。
