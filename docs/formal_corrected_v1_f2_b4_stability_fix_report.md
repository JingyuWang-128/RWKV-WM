# formal_corrected_v1 F2 B4 数值稳定性修复与重启审查

审查日期：2026-09-10

## 1. 结论

修复后的 F2 B4 已通过代码、单元测试、长历史 GPU 压力测试、H20 free-running
validation、checkpoint 深度审计和正式 protocol 绑定审查，可以从 step 0 统一重启
TwoRoom-W 与 Action-Delay 的 3 seeds 正式训练。

旧 F2 运行不能恢复或参与比较，已整体隔离到：

```text
artifacts/runs/per_step_pairs/formal_corrected_v1/invalid_unbounded_b4_20260910/
```

## 2. 故障现象与根因

- TwoRoom-W B4 seed 0/1 分别从 step 291/568 起产生非有限训练 loss。
- Action-Delay 的一步训练 loss 暂时有限，但从 step 1,000 起的完整 H20 validation
  已经产生 `NaN`。
- 数据张量本身全部有限，两个任务 latent 的标准差约为 1、绝对值最大约为 5.6；
  数据尺度不是根因。
- 根因是 B4 的 uncentered action erase/write residual 没有幅值约束。训练到异常前，
  intervention gate 饱和到 1，raw erase/write 绝对值达到约 50。
- TwoRoom-W 的 15 步 observed history 使矩阵最大绝对值从 13.7 逐步放大到
  `2.14e29`，随后发生 float32 溢出。Action-Delay 的 observed history 为 5 步，
  但 H20 rollout 同样会暴露该问题。

## 3. 修复内容

### 3.1 有界 action rank-two update

修改 `src/cape_wm/cc_rwkv/cell.py`：

- raw erase/write residual 先经过 `tanh`；
- 乘 intervention gate；
- 再按 `0.1 / head_dim` 缩放；
- 最坏情况下，每个 action erase/write 外积的算子范数不超过约 0.1；
- 不对最终 RWKV matrix state 做 clamp 或 `nan_to_num`，避免掩盖递推错误；
- B4/B6 使用完全相同的有界参数化，参数量和 state-dict shape 不变；
- B4 仍不做 reference subtraction，B6 仍做 actual-minus-reference，保持 F2 消融定义。

### 3.2 非有限值 fail-fast

修改 `scripts/train_per_step_pairs.py` 与
`src/cape_wm/cc_rwkv/per_step_checkpoint.py`：

- training total/component 在 backward 前检查；
- gradient norm 在 optimizer step 前检查；
- validation batch/mean 检查；
- checkpoint 保存前检查 model 和 optimizer；
- checkpoint 恢复前检查 model 和 optimizer；
- 发生 `NaN/Inf` 时写入 `failure.json` 并以非零返回码退出；
- 不保存或恢复包含非有限 tensor 的 checkpoint。

集成故障注入使用无限 learning rate：step 1 完成后，程序在 step 2 的
`training_forward` 阶段按预期退出，生成 `failure.json`，且没有生成 `latest.pt`。

## 4. 静态与单元测试审查

- Ruff：PASS。
- Python compile：PASS。
- 完整 pytest：111 passed。
- B4/B6 参数量：均为 10,780,224。
- B4/B6 state-dict 名称和 shape：完全一致。
- B4 `centered=false`，B6 `centered=true`。
- bounded outer-product operator norm 测试：PASS。
- 非有限 checkpoint 保存/恢复拒绝测试：PASS。
- loss/gradient fail-fast 测试：PASS。

## 5. GPU 稳定性压力测试

main profile、batch 8、learning rate 3e-3；每条训练 1,000 steps，并使用 64 个
validation 样本执行 H20 free-running validation。

| 任务 | seed | step 1000 train loss | H20 validation | 结果 |
|---|---:|---:|---:|---|
| TwoRoom-W | 0 | 0.374320 | 1.144364 | PASS |
| TwoRoom-W | 1 | 0.337463 | 1.151164 | PASS |
| TwoRoom-W | 2 | 0.360339 | 1.173327 | PASS |
| Action-Delay | 0 | 0.403455 | 1.144124 | PASS |
| Action-Delay | 1 | 0.667775 | 1.199597 | PASS |

共审计 5,000 条 history 记录；所有 loss/gradient 均有限。5 份 step-1000
checkpoint 的全部 model/optimizer tensor 均有限，没有 `failure.json`。

在三个 TwoRoom-W checkpoint 上额外执行 15 步 history + 20 步 free rollout，
matrix 最大绝对值分别为 174.10、141.82、36.67，全部有限；相比旧实现的
`2.14e29` 已消除指数爆炸。

## 6. 正式协议冻结

F1 原始 manifest 保持不变。F2 使用独立修复版 manifest：

```text
configs/experiments/formal_corrected_v1/protocol_manifest_f2_bounded_action_v2.json
```

manifest SHA256：

```text
5901a6e984b8cd2aaca34907e1f6d85b00e6431321dd2a104cb16665fd1a2e4b
```

两个任务均已通过正式 protocol ID、数据 SHA256、代码 SHA256、H1/H10 训练和 H20
validation 的最终 GPU preflight。

## 7. 正式重启范围

从 step 0 重训以下 6 个运行：

- TwoRoom-W：B4 seed 0/1/2；
- Action-Delay：B4 seed 0/1/2。

正式输出仍写入全新的：

```text
artifacts/runs/per_step_pairs/formal_corrected_v1/f2/
```

每 1,000 step 保存 `latest.pt`，每 5,000 step保存编号快照；任意训练、验证、梯度、
保存或恢复阶段检测到非有限值时立即停止对应队列。
