# CC-RWKV-WM M2 验收报告

> 状态：**完成（数据/环境 Go，尚未进入 M3）**  
> 完成时间：2026-08-28  
> 范围：完整快照、TwoRoom/Action-Delay adapter、四分支采集、支持域过滤、冻结 encoder latent cache；未实现 CC-RWKV 递推或训练。

## 1. M2 结论

M2 已建立可用于真实 effect supervision 的同状态物理干预数据管线。每个 snapshot
恢复到完全相同的 simulator/RNG/history state 后，分别执行 reference、factual、
pulse-noop 和 pulse-local 动作分支。TwoRoom 5,000-snapshot MVP 中，所有 snapshot
恢复一致，factual 分支与官方离线轨迹逐元素重放一致，失败率为 0。

M2 的 Go 只表示 paired 标签和缓存协议可信，不表示 CC-RWKV 已学会反事实效应；
world/action/reference 参数分路要到 M3 才实现。

## 2. 完整 snapshot 契约

`TwoRoomBranchAdapter` 显式保存和恢复：

- agent/target position；
- agent/target color、radius、speed；
- wall axis、thickness、color、border color；
- door number、positions、sizes、color；
- background、render-target 和 task variation；
- wall/door runtime cache；
- elapsed primitive steps；
- environment RNG；
- TwoRoom-W episode-constant drift 和 drift RNG。

Action-Delay 另外保存：

- 5×2 action FIFO；
- FIFO 指针；
- delay 配置。

Action-Delay reset 强制要求来自真实行为历史的 5 步 action history，禁止以全零 FIFO
代替。状态向量为 agent position、按未来执行顺序排列的 FIFO 和 pointer，共 13 维。

自动化测试验证：

```text
snapshot -> action -> restore -> same action
```

得到完全相同的 observation、render、state 和 FIFO。不同 branch 之间不共享可变状态。

## 3. 四分支与支持域

分支顺序被冻结为：

| code | branch | 定义 |
|---:|---|---|
| 0 | reference | 全程提交合法零动作 |
| 1 | factual | 官方 HDF5 中的真实行为 suffix |
| 2 | pulse_noop | 第一个 5-action block 置零，之后与 factual 完全相同 |
| 3 | pulse_local | 第一个 block 作 ±0.2 局部扰动，之后与 factual 完全相同 |

支持域只使用 train episode 的 finite primitive actions 建立 `cKDTree`。score 定义为
block 内 primitive action 到训练集最近邻距离的负均值，阈值是训练行为动作
leave-one-out score 的第 5 百分位：

```text
support threshold = -0.0035922738999586636
```

所有 pulse-local 首 block 均高于阈值；validation/test 动作没有参与拟合阈值。

## 4. Split 与采样

冻结 episode split hash：

```text
6c7f353d437d193c397bcf3c7d435c65d59b82d21ad3ed5673072bf8d2e46af8
```

- 正式 M0 test episode 继续作为冻结 test 集；
- 其余 episode 按 seed 3072 固定为 train/validation；
- 5,000 MVP 抽取 4,000 train、500 validation、500 test；
- 每个 snapshot 来自不同 episode；
- snapshot start 对齐 5-action block；
- 每条样本保存 3 个历史 model steps 和 4 个未来 model steps；
- 对应 15 primitive-action history 和 20 primitive-action future。

三个样本集合的 episode ID 两两不交。

## 5. HDF5 与 latent cache

schema 为 `cc_rwkv_branches_v1`，主要字段包括：

```text
history_latents          [N,3,192]       float16
history_actions_raw      [N,3,10]        float32
branch_type              [N,4]           uint8
branch_actions_raw       [N,4,4,10]      float32
branch_latents           [N,4,5,192]     float16
branch_states            [N,4,5,Ds]      float32
action_support_score     [N,4,4]         float32
restore_consistent       [N]             bool
```

冻结 encoder 权重、模型配置和 action normalizer 与 M0 完全一致。只保存 1% raw RGB
审计集，主数据不保存全部 RGB。

CUDA 对相同图片在不同 batch 排布下存在极小数值差异，因此 latent cache 验收使用
明确的 FP16 `atol=1e-3`，而不是不合理的 bitwise equality：

- 四分支相同起点 latent 最大绝对差：`1.52587890625e-05`；
- 50 个 raw audit samples 在线重编码与缓存最大绝对差：`0.0009765625`；
- 四分支起点 raw RGB 仍逐像素相同。

## 6. 产物与结果

### TwoRoom 32-sample smoke

- 路径：`artifacts/cache/cc_rwkv/tworoom/smoke32/`；
- split：26/3/3；
- HDF5 SHA256：`00ebe2096d8eb8ce84a469eab1e025373b221415ea3cfc8c8e287631358b54ae`；
- restore failures：0；
- factual replay max error：0；
- raw audit online/cache max error：0。

### Action-Delay 32-sample smoke

- 路径：`artifacts/cache/cc_rwkv/action_delay/smoke32/`；
- split：26/3/3；
- HDF5 SHA256：`06e9928fe36f7c2eec6a66a6b4e443bb5db1c326f5e4858427396af9e329805d`；
- restore failures：0；
- FIFO snapshot/restore：通过；
- raw audit online/cache max error：0。

Action-Delay factual 分支是“把行为动作提交给延迟动力学”，因此不应与原始无延迟
HDF5 的未来位置作 equality check；manifest 将该字段明确标为 not checked，而不是伪报 0。

### TwoRoom 5,000-snapshot MVP

- 路径：`artifacts/cache/cc_rwkv/tworoom/mvp5000/`；
- HDF5 大小：59,008,343 bytes（目录约 57 MiB）；
- 分支轨迹：20,000；
- future boundary latents：100,000；
- raw audit samples：50；
- restore failures：0/5,000；
- factual replay max error：0.0；
- HDF5 SHA256：`2a9e425c762141dbe3b003fd15ffbee70c5d3bae7b73bf8788a1906d290bb925`；
- spec SHA256：`001bfacd52d3c05a577abb56fb6bd06a09edebd1871e5f7fa6f53e60027c2595`；
- manifest SHA256：`9583d7662c9844f2dd424bfa4025b3f30a71c3653e8fe4ec36dacec9189fb4dc`；
- audit SHA256：`2f01f068b0b21eff90512b7a8d336f42891fd782bad62faa7017a706e312ca5e`。

来源 provenance：

- HDF5：`129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623`；
- encoder：`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`；
- model config：`2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`。

## 7. 代码与测试

新增核心实现：

- `src/cape_wm/cc_rwkv/branches.py`；
- `src/cape_wm/cc_rwkv/branch_dataset.py`；
- `scripts/collect_counterfactual_branches.py`；
- `scripts/audit_counterfactual_dataset.py`；
- `tests/test_counterfactual_branches.py`；
- `tests/test_counterfactual_branch_dataset.py`。

最终验证：

- 全仓库：**86 passed**，5 个已有 PyTorch warning；
- M2 修改范围 Ruff：通过；
- dataset loader 会拒绝 schema mismatch、episode split leakage 或任一 restore failure；
- 独立 audit 会拒绝错误 branch suffix、零动作、support、非有限 latent 或 cache mismatch。

## 8. M2 Go/No-Go

M2 Go 条件已满足：

- snapshot/data 自动化测试通过；
- TwoRoom 和 Action-Delay 32-sample smoke 通过；
- 5,000-snapshot MVP 完成；
- restore inconsistency rate 为 0；
- TwoRoom factual replay error 为 0；
- split leakage 为 0；
- latent cache/raw audit 在预注册 FP16 容差内一致。

尚未完成：

- M3 world/action/reference 参数分路；
- counterfactual-centered rank-two RWKV update；
- B4/B6 训练；
- CEE/CED/CER 与 Gate A；
- 50/100-step 长程数据；
- RL 或 MPC。

M2 到此停止。下一里程碑 M3 必须得到用户批准后才能开始。
