# CC-RWKV-WM M1 验收报告

> 状态：**完成（工程 Go，尚未进入反事实实验 Gate A）**  
> 完成时间：2026-08-28  
> 范围：Stateful vanilla RWKV7/B2；未实现反事实、world/action 分路、分支环境、MPC 或 RL。

## 1. M1 结论

M1 已实现一个不依赖第三方 RWKV runtime/CUDA kernel 的纯 PyTorch RWKV7 x070
predictor。它具有持久 FP32 矩阵状态，真实历史与 free-running imagination 使用同一
状态，可以把历史状态安全复制到多条动作分支，并且 rollout 期间不截断 history、
不注入真实未来 latent。

RWKV7 block 的公式、状态方向和特殊机制均通过独立 oracle 测试。M1 只建立 B2
基座；反事实中心化 decay/erase/write 和 world/action 解耦必须在 M3 单独实现，不能
提前混入本里程碑，否则无法判断差异来自 RWKV7 基座还是新递推。

## 2. 官方对齐边界

实现参考：

- [RWKV7 官方训练参考 `train_temp`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp)；
- [官方 RWKV7 RNN reference](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo_rnn.py)；
- [官方最小 NumPy 递推](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py)。

本地 block 保留以下机制：

- Pre-LN 和第 0 层 `ln0`；
- `x_r/x_w/x_k/x_v/x_a/x_g` 六路独立 token shift；
- `w0 + tanh(xw @ w1) @ w2` 动态 decay；
- `kk` 分 head 归一化以及 generalized delta-rule erase；
- `a` in-context learning rate；
- 跨层 `v_first` value residual；
- gate、per-head GroupNorm（`eps=64e-5`）和 `r*k*r_k*v` bonus；
- squared-ReLU ChannelMix 及其独立 token shift；
- 官方初始化曲线、零初始化输出/value projection；
- 大矩阵 weight decay、`w0` 2x 学习率的 optimizer parameter groups；
- matrix state 固定 FP32 累积。

连续 latent/action 输入投影和 latent prediction head 是世界模型任务适配层，不属于
语言模型的 embedding/head，因此不要求与语言模型词表层相同。M1 的“官方一致”是
指 RWKV7 block 和递推状态语义一致，不声称不同 kernel 间逐 bit 一致。

## 3. 状态与接口

计划草图曾把 shift 简写为一个 `[B,L,D]` tensor。官方 RWKV7 每层实际需要两份
前一 token 状态，所以实现细化为：

```text
matrix:       [B, L, H, Dh, Dh], float32
time_shift:   [B, L, D], network dtype
channel_shift:[B, L, D], network dtype
steps:        [B], int64
```

这不是额外模型机制，而是正确复现官方 TimeMix/ChannelMix 的必要条件。

`VanillaRWKV7WorldPredictor` 已实现：

- `init_state`；
- 单步 `step`；
- 有 padding mask 的 `forward_sequence/consume_history`；
- 不共享可写 storage 的 `clone_branches`；
- 完全 free-running 的多分支 `rollout`；
- state clone/detach/index-select/序列化；
- official-style AdamW parameter groups；
- `cc_rwkv_checkpoint_v1` 安全读写与 provenance guard。

状态时间语义与计划一致：`state_t` 尚未消费 `(z_t,u_t)`；调用 `step` 后预测
`z_{t+1}` 并返回已经消费该 pair 的 `state_{t+1}`。

## 4. 参数量审计

| 配置 | D/L/H/Dh | ChannelMix width | predictor 参数量 | 相对 B0 |
|---|---|---:|---:|---:|
| M1 机制配置 | 192/6/6/32 | 768 | 3,121,344 | -71.08% |
| B2 参数匹配配置 | 192/6/6/32 | 4096 | 10,789,056 | -0.021% |
| B0 Transformer | 192/6/-/- | 768 | 10,791,360 | reference |

参数匹配只调整 B2--B7 共同拥有的 ChannelMix width，不改变 RWKV7 矩阵递推，也不
增加任何 CC 独有层。正式公平比较应使用 width=4096；width=768 仅用于机制开发和
低成本测试。

## 5. 自动化验收

| 检查 | 结果 |
|---|---|
| 分解 recurrence vs 显式 `M @ G + U` | 通过，FP32 `atol=1e-6, rtol=1e-5` |
| TimeMix vs 独立官方公式 oracle | layer 0/1 两条 `v_first` 路径均通过 |
| sequential step vs sequence wrapper | 输出和最终状态一致 |
| padding state invariant | matrix/两份 shift/steps 均不误更新 |
| branch clone | 顺序正确，无共享可写 storage |
| 20-step FP32/bf16 | 输出和 matrix 均 finite；matrix 保持 FP32 |
| persistent vs forced reset | 输出不同，证明 persistent state 实际参与预测 |
| 20-step backward | 梯度存在且全部 finite |
| checkpoint | `weights_only=True` 回读成功，hash mismatch 默认拒绝 |
| 全仓库测试 | **80 passed**，5 个已有 PyTorch warning |
| M1 修改范围 Ruff | **通过** |

全仓库 Ruff 仍保留 M0 报告中记录的 6 个历史格式问题；M1 未修改这些无关文件。

## 6. 20-step 训练 smoke

命令：

```bash
.venv/bin/python scripts/train_b2_rwkv.py \
  --smoke-synthetic \
  --output artifacts/results/cc_rwkv/m1/smoke_overfit \
  --steps 200 --batch-size 8 --horizon 20 --device cpu
```

该 smoke 使用共享动力学但互不重叠的 24 条训练序列和 8 条 validation 序列，仅用于
检验工程可训练性，不是 TwoRoom 科学结果，也不能写入论文主表。

| 指标 | 随机初始化 | 训练 200 steps 后 | 相对下降 |
|---|---:|---:|---:|
| train 20-step free-running Smooth-L1 | 0.270750 | 0.007693 | 97.16% |
| validation 20-step free-running Smooth-L1 | 0.256438 | 0.025599 | 90.02% |

结果 finite，checkpoint 可安全回读。产物 SHA256：

- `summary.json`：`d282429a1e1c6efc0b0a4bd7b409b65eb8b5c7fd428edbe40525a04b9eb2838a`；
- `b2_m1.pt`：`0b6b4a3dbe7d550996c60245063ca68552d39e21b33b080c26eb41353f1cc9dc`。

该 checkpoint 仅用于工程 smoke，未绑定正式 encoder/data manifest，因此不能用于论文。

## 7. 真实 LeWM 接入验收

真实 checkpoint 审计命令：

```bash
.venv/bin/python scripts/check_cc_rwkv_m1_lewm.py \
  --weights /data/wjy/lewm_model/tworooms/weights.pt \
  --m0-manifest artifacts/results/cc_rwkv/tworoom/m0/protocol/open_loop_manifest.npz \
  --cache-dir artifacts/cache/cc_rwkv/m1/lewm \
  --output artifacts/results/cc_rwkv/m1/lewm_bridge.json --device cpu
```

结果：

- 权重 SHA256：`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`，与 M0 一致；
- 输入 `[1,3,224,224]` 得到有限的 `[1,1,192]` latent；
- M0 action normalizer 可将 5×2 action block 转换为有限的 `[1,10]`；
- encoder/projector trainable 参数为 0；
- 不加载或复用官方 Transformer predictor。

审计结果 `lewm_bridge.json` 的 SHA256 为
`acf9630a84fb9dd594d1fd30d8cc348b976b9970c5e67d40e21dc30ddef853d3`。

## 8. 实现产物

核心代码：

- `src/cape_wm/cc_rwkv/state.py`；
- `src/cape_wm/cc_rwkv/cell.py`；
- `src/cape_wm/cc_rwkv/predictor.py`；
- `src/cape_wm/cc_rwkv/lewm.py`；
- `src/cape_wm/cc_rwkv/training.py`；
- `src/cape_wm/cc_rwkv/checkpoint.py`；
- `scripts/train_b2_rwkv.py`；
- `scripts/check_cc_rwkv_m1_lewm.py`。

新增测试：

- `tests/test_cc_rwkv_state.py`；
- `tests/test_cc_rwkv_cell.py`；
- `tests/test_cc_rwkv_predictor.py`；
- `tests/test_cc_rwkv_lewm.py`；
- `tests/test_cc_rwkv_training.py`；
- `tests/test_cc_rwkv_checkpoint.py`。

## 9. 尚未完成且不能越界解释的内容

- 尚未加入反事实 actual/reference 两次 action network；
- 尚未做 world/action 参数分路和无动作旁路审计；
- 尚未实现 diagonal-plus-rank-two CC 更新；
- 尚未采集真实 paired simulator branches；
- 尚未在冻结的 TwoRoom train/validation split 上训练正式 B2；
- 尚未得到 B2 对 B0 的长期 rollout 结论；
- 尚未通过 Gate A--D；
- 尚未实现 MPC 或 RL。

因此 M1 的 Go 仅表示“官方对齐的 stateful B2 基座可训练、可持久递推、可接入真实
LeWM encoder”。它不构成 CC-RWKV 方法有效性的证据。

M1 到此停止。下一里程碑 M2 是分支环境与 paired 数据缓存，必须得到用户批准后才能开始。
