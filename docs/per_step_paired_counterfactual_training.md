# 逐时间位置的反事实配对监督方案

## 0. 当前决定与适用范围

本文记录一个在**现有正式 M5 完成之后**再实施的扩展方案：沿数据集中的真实行为动作轨迹推进，并在每一个时间位置从相同 factual 状态临时分叉，收集“当前动作替换为零动作”后的反事实结果。

该方案不改变正在运行的 M5：

- 当前 M5 继续使用已经冻结的 H1-only paired-supervision 协议完成训练、评估、聚合与 Gate A/B@20 报告；
- 当前 M5 的 checkpoint、验证/测试结果和 schema 不与本文方案产生的结果混用；
- 本文方案在 M5 完成后作为独立扩展实验实施，开始数据重采集或训练前仍需单独批准；
- 本文不授权进入 M6，也不修改当前后台训练进程。

## 1. 目标与核心区别

设数据集中真实执行的行为动作序列为

\[
u_0,u_1,\ldots,u_{H-1},
\]

其 factual 状态轨迹为

\[
s_0,s_1,\ldots,s_H.
\]

目标不是从 \(s_0\) 开始执行一整条全零动作序列。目标是在每一个时间位置 \(t\) 都从真实轨迹的同一个 \(s_t\) 分叉：

- factual 分支当前执行数据集动作 \(u_t\)；
- pulse-noop 分支只把当前动作替换成合法零动作 \(0\)；
- 两个分支共享相同历史、simulator snapshot、外部随机状态和后续行为动作；
- factual 主轨迹继续沿数据集状态推进，pulse-noop 分支只用于构造反事实监督，不替代下一时刻的主训练状态。

因此，本文方案是“逐时间位置的局部干预”，而不是“factual 轨迹对比全零轨迹”。

## 2. 每个时间位置应构造的配对轨迹

对于位置 \(t\)，从同一 snapshot 恢复两次：

### 2.1 Factual suffix

\[
[u_t,u_{t+1},\ldots,u_{H-1}].
\]

factual suffix 应严格重放数据集原始轨迹；其状态、图像和 latent 必须与缓存中的 factual 目标一致。

### 2.2 Pulse-noop suffix

\[
[0,u_{t+1},\ldots,u_{H-1}].
\]

只有位置 \(t\) 的动作被替换成零，后续动作与 factual suffix 完全相同。这样，在后续 horizon 上观测到的差异可以归因于位置 \(t\) 的动作干预及其传播，而不会混入“后续动作也不同”这一混杂因素。

### 2.3 监督目标

对于每个 \(t\) 和有效未来偏移 \(k\)，保存：

- factual 目标状态或 latent；
- pulse-noop 目标状态或 latent；
- effect target：两者之差；
- triangular mask：仅 \(0 \leq k < H-t\) 有效；
- snapshot、外部噪声、动作后缀和恢复一致性审计信息。

当 \(k=0\) 时得到当前动作的一步局部效应；当 \(k>0\) 时得到该动作在后续递推中的延迟、传播和消退效应。

## 3. 为什么不能只收集一步 no-op 状态

只收集 \(s_{t+1}^{0}\) 可以监督即时动作效应，但不足以支持 Action-Delay 和长期 imagination。

以 delay=5 为例，时刻 \(t\) 提交的动作不会立即执行；factual 与 pulse-noop 在前若干步可能得到完全相同的物理状态。动作差异只有在进入并离开 FIFO 后才出现。因此：

- snapshot 必须包含 FIFO 内容、pointer 和所有影响延迟执行的状态；
- pulse-noop 分支必须至少 rollout 到动作可能生效的位置；
- 正式方案使用剩余完整 suffix，而不是只保存一步结果；
- 后续动作必须保持相同，否则无法隔离 \(u_t\) 的延迟因果效应。

TwoRoom-W 中，零动作也不表示静止：drift 仍然作用于两个分支。共享 snapshot 和外部噪声后，两条轨迹的差值才主要表示动作的条件效应，而绝对目标仍保留环境自主变化。

## 4. 与 RWKV 矩阵递推的结合

训练时先沿 factual 历史得到当前矩阵状态 \(M_t\)。对每个位置 \(t\)：

1. 从相同的 \(M_t\) 和 factual latent \(z_t\) 克隆 factual/no-op 状态；
2. factual 状态使用 \(u_t\) 执行 centered decay、erase、write 和官方 RWKV world update；
3. no-op 状态使用参考动作 \(0\) 执行同一递推接口；
4. 两个分支分别预测下一 latent，并与各自 simulator target 对齐；
5. 监督预测 effect 与真实 effect；
6. 若训练长期 effect，则从两个更新后的状态继续输入相同的 factual suffix，比较 effect 如何在矩阵中传播；
7. factual 主状态继续用于位置 \(t+1\)；临时 no-op 状态在完成该位置的 paired loss 后丢弃。

关键约束：动作信息仍只能通过矩阵更新进入后续预测，不能新增绕过 RWKV state 的直接 action-to-output 通道。

## 5. 建议的数据 schema

建议新增独立 schema，例如 `cc_rwkv_per_step_pairs_v1`，不要覆盖当前 `cc_rwkv_branches_v1`。每个样本至少包含：

```text
sample_id                       scalar
history_latents                 [T_history, D]
history_actions                 [T_history, A]
history_mask                    [T_history]
factual_actions                 [H, A]
factual_latents                 [H + 1, D]
factual_states                  [H + 1, S]
pulse_noop_latents              [H, H, D]
pulse_noop_states               [H, H, S]
pulse_noop_mask                 [H, H]
source_snapshot_hashes          [H]
external_noise_hashes           [H]
restore_consistent              [H]
factual_replay_error            [H]
```

`pulse_noop_latents[t,k]` 表示在 factual 时间位置 \(t\) 将当前动作替换为零后，第 \(k+1\) 个未来目标。数组采用三角 mask；也可使用 ragged/offset 存储减少重复空间。

动作必须先在环境的原始动作空间中设为语义零动作，再经过与 factual action 相同的 scaler。不能直接把标准化后的向量置零，除非已证明它仍对应环境 no-op。

## 6. 数据采集流程

对每个基础样本：

1. 恢复样本起点 snapshot 和外部随机状态；
2. 使用数据集动作重放完整 factual trajectory，并保存每个 \(s_t\) 的完整 snapshot；
3. 审计 factual replay 与原始缓存一致；
4. 对每个位置 \(t\)：
   - 恢复 \(s_t\) snapshot 与对应外部随机状态；
   - 当前提交零动作；
   - 从 \(t+1\) 开始继续提交原 factual suffix；
   - 保存全部有效未来状态、图像和 latent；
   - 再次恢复 snapshot，确认恢复结果逐字段一致；
5. 写入三角 mask、哈希和审计结果；
6. 任何 snapshot/FIFO/noise/factual replay 审计失败的样本都不得进入正式缓存。

编码图像时应批量处理并冻结 encoder；环境状态和原始图像保留一个小比例用于 raw audit，避免仅靠 latent 一致性掩盖 simulator 恢复错误。

## 7. 训练损失

建议总损失包含以下部分：

- `factual_prediction`：真实行为轨迹的完整 free-running rollout 误差；
- `noop_prediction`：各位置 pulse-noop suffix 的绝对预测误差；
- `paired_effect`：预测 factual/no-op 差值与真实差值之间的误差；
- `effect_direction`：真实 effect 足够大时的方向一致性；
- `effect_magnitude`：真实 effect 足够大时的尺度一致性；
- `world/reference`：零当前动作下环境自主变化、drift、惯性和历史动作效应；
- `matrix_entry/probe`：证明非零 action effect 确实进入更新后的 RWKV matrix。

损失按位置和有效 future offset 使用 triangular mask。为防止短 horizon 因样本数量更多而主导训练，应明确选择以下一种归一化并预注册：

- 先对每个 intervention position 的有效 future offsets 求均值，再对位置和 batch 求均值；或
- 按绝对 future horizon 分层等权采样。

不得直接对三角数组所有元素求平均而不审计权重分布。

## 8. 数据与比较公平性

逐位置 pulse-noop 会显著增加 simulator supervision。正式比较必须区分“方法改进”和“额外数据量”带来的收益。

推荐的严格口径：

- B2/B3/B4/B6 使用相同的基础 snapshot、行为轨迹和扩展分支缓存；
- 所有方法接受相同的 factual/no-op absolute prediction targets；
- 只有预注册允许使用 paired-effect 标签的方法使用 effect/direction/magnitude loss；
- optimizer steps、batch 中有效 factual/no-op trajectory 数、encoder、normalizer、curriculum 和精度保持一致；
- 单独报告 simulator branch steps、有效 target 数和总训练 FLOPs；
- 若沿用当前 B2/B3/B4 checkpoint 而只重训 B6，结果必须标记为 data-advantaged exploratory，不能作为严格 Gate 结论。

若以本文方案替换正式方法协议，为获得严格可比结论，原则上需要在新缓存上重训 B2/B3/B4/B6，而不仅是重训 B6。

## 9. 计算量估算

只保留 factual trajectory 和每个位置的一条完整 pulse-noop suffix 时，每个样本的反事实 model-step 数为

\[
\sum_{t=0}^{H-1}(H-t)=\frac{H(H+1)}{2}.
\]

- Action-Delay，\(H=20\)：210 个 pulse-noop model steps，加 20 个 factual steps，共约 230；当前四分支缓存约为 80，因此环境 rollout 量约增至 2.9 倍；
- TwoRoom-W，\(H=4\)：10 个 pulse-noop model steps，加 4 个 factual steps，共约 14；若不增加逐位置 local-perturb 分支，与当前 16 个 branch model steps 同量级；
- 若再为每个位置增加 `pulse_local@t`，需要额外增加同样的三角 suffix 量。

正式采集前必须用实测吞吐、缓存体积和 encoder 时间更新估算，不能只按环境 step 推算总时长。

## 10. 必须通过的审计与测试

### 10.1 数据审计

- 每个位置 factual/no-op 分支的初始 snapshot 完全一致；
- 外部随机状态一致；
- factual suffix 与原始行为轨迹一致；
- pulse-noop 仅当前动作是原始空间零动作；
- pulse-noop 后缀动作与 factual 后缀逐元素一致；
- Action-Delay FIFO、pointer 和历史动作完整恢复；
- factual replay state/image/latent 误差低于预注册阈值；
- triangular mask、split、sample ID 和 episode 隔离正确；
- 数据集和 split 哈希冻结。

### 10.2 模型与损失测试

- actual action 等于 reference action 时 action residual 严格为零；
- 修改任意有效位置/未来步的 paired target 会改变 paired loss；
- 修改 mask 外 target 不改变 loss；
- 每个 intervention position 都能把梯度传到 centered decay、erase 和 write；
- 去除或重置矩阵 action update 后长期 effect 显著变化；
- 禁止 action-to-output bypass；
- free-running rollout 中不重新输入真实未来 latent；
- Action-Delay 的 effect onset 能在延迟后被检测，而不是被一步标签判为零；
- float32/bfloat16 下 loss、matrix norm 和 gradient 均有限。

## 11. 评估指标

除当前 M5 指标外，增加按 intervention position 分解的结果：

- local one-step effect error；
- effect onset delay error；
- position-conditioned CEE curve/AUC；
- effect persistence/decay error；
- factual 与 no-op absolute rollout error；
- delayed-action FIFO-aware effect error；
- matrix action-update norm 与真实 effect 的相关性；
- early/middle/late position 分层结果；
- 相对相同数据量基线的三 seed paired bootstrap CI。

长期主张应依赖多步 effect curve 与 rollout fidelity，而不能只依赖一步 paired loss 下降。

## 12. M5 完成后的实施顺序

以下每一步开始前均需重新获得批准：

1. 冻结本文协议、损失归一化、公平性口径和新 schema；
2. 用 32 个样本实现逐位置 snapshot、pulse-noop suffix 和完整审计；
3. 用 128～500 个样本做过拟合测试，确认每个位置的 effect 可学习且进入矩阵；
4. 测量 Action-Delay H20 与 TwoRoom-W H4 的真实采集/编码/存储成本；
5. 重新采集冻结的 5,000-snapshot 扩展缓存；
6. 按公平协议重训必要的方法、三个 seed 和 effect-weight 候选；
7. 在冻结的 validation/test split 上评估，并与当前 M5 结果分开聚合；
8. 只有独立完成数据、训练和协议审计后，才讨论是否替换当前方法或进入更长 horizon。

## 13. 一句话定义

> 沿真实行为轨迹推进 factual RWKV 状态，并在每个时间位置从相同 factual snapshot 和矩阵状态临时分叉，只将当前动作替换为零、保持后续动作不变，以监督该动作的即时、延迟和长期因果效应如何写入并通过 RWKV 矩阵传播。

## 14. 当前实现闭环与成本实测

已实现 `cc_rwkv_per_step_pairs_v1` 采集器、HDF5 writer、逐位置审计器和
`per_step_loss` 训练入口。TwoRoom-W 与 Action-Delay 的 32-sample 和 128-sample
缓存均通过 schema、三角 mask、raw-zero、后缀一致、effect 对齐、factual 重放和
snapshot 恢复审计。B6 小模型过拟合显示 action parameter、action erase/write 路径
均有非零梯度；teacher-forcing 诊断中 TwoRoom-W 各位置 effect loss 从约 0.54 降至
0.44--0.47，Action-Delay 的 0--4 步 effect 接近零、约第 5 步后出现有效 effect，
符合 FIFO delay 的预期。free-running 过拟合尚未稳定，不能把该诊断当作泛化结果。

GPU 8-sample 端到端 benchmark（包含模型加载和 encoder）为：TwoRoom-W 72.76 s、
Action-Delay 79.60 s。固定三角展开的每样本 model-step 数分别为 14 和 230；128
sample 缓存体积分别约 20.7 KB/sample 和 382.7 KB/sample，按当前 raw-audit 比例
线性估算 5,000 samples 约 0.10 GB 和 1.91 GB。正式采集前仍需根据实际并行度和
checkpoint/临时空间再次确认总 wall time。

## 15. 数据扩充执行记录

5,000-snapshot 首轮训练在用户要求下暂停，保留其缓存和审计结果。采样器新增
`--allow-multiple-per-episode`：episode 仍只属于一个 split，但允许在同一 episode
抽取多个不同起始位置，从而避免受唯一 episode 数限制。扩充目标为 20,000 主样本，
由四张 GPU 直接分片采集；当前统一为每个任务总计 20,000 个样本。TwoRoom-W 的
GPU0/1 各承担 10,000，Action-Delay 在重新调度后由 GPU0/1/2/3 各承担 5,000。
各 shard 固定相同 episode split，使用全局 shard 索引保证 sample-id 不重叠，完成后
通过 schema/split 校验和统一审计合并。
