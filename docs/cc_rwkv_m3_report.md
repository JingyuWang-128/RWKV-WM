# CC-RWKV-WM M3 验收报告

> 状态：**完成（log-hazard 兼容修订已验收，尚未进入 M4）**  
> 完成时间：2026-08-28  
> 范围：world/action/reference 分路、反事实中心化递推、B4/B6、机制测试、真实 paired 小数据拟合；不包含 M4 完整 loss/curriculum/指标训练器。

## 1. M3 结论

M3 已实现一个以官方 RWKV-7 x070 为 world path、只允许动作通过矩阵更新产生影响的
CC-RWKV predictor。B6 对同一 world/history 下的 actual/reference 动作使用同一个参数网络，
再做参数差；B4 使用完全相同的结构和参数量，但不做 reference subtraction。

所有关键机制约束均通过：

- `actual == reference` 时三个动作增量和完整 action matrix update 严格为 0；
- 从 B2 迁移 world path 后，零反事实增量的 prediction、matrix、TimeMix shift 和
  ChannelMix shift 与官方-style vanilla RWKV7 逐元素相同；
- raw action 不能进入 latent token、receptance、readout、ChannelMix 或 prediction head；
- actual/reference 交换时，在固定 world query/state 下三个原始动作增量严格反号；
- reference subtraction 两次调用共享同一个动作参数网络，actual/reference 梯度均非零且有限；
- B6 在 128 个 M2 真实 paired samples 上训练 200 步后，一步 effect MSE 下降 44.66%；
- intervention gate 没有整体塌缩到 0 或 1。

这只是机制 Go 和小数据可学习性证明，不是 B6 优于 B2/B3/B4 的论文结果；统一训练和
CEE/CED/CER 对比属于 M4/M5。

## 2. 模型分路

### 2.1 World path

基础 token 只读取 latent：

```text
q_t = W_z z_t
```

每层保留 M1 的官方 RWKV7 x070 world 参数化：六路 time shift、receptance、world decay、
key/value、in-context learning rate、output gate、group norm、bonus term、ChannelMix 和
Pre-LN residual 均未改成近似模块。历史依然由每层 fp32 matrix state、TimeMix shift 和
ChannelMix shift 递推保存。

### 2.2 Action/reference path

raw action 只允许进入：

```text
action_parameter_network(world_query, raw_action)
```

该共享网络输出 decay、erase、write 三组动作参数以及 intervention gate。B6 计算：

```text
delta_decay = f_decay(q, u_actual) - f_decay(q, u_reference)
delta_erase = f_erase(q, u_actual) - f_erase(q, u_reference)
delta_write = f_write(q, u_actual) - f_write(q, u_reference)
```

B4 则直接使用 actual 输出，不计算 reference 差。两者类、层数、hidden width 和所有参数
完全相同，仅 `centered` 开关不同。

动作 delta head 的 weight/bias 全零初始化，因此载入 B2 后初始行为精确退化到 vanilla
world path。gate head 的 weight 为 0、bias 为 `logit(0.1)`，防止初始化时全关或全开。

### 2.3 Rank-two 矩阵更新

代码不显式构造 transition matrix，而按分解形式计算：

```text
M_world = M diag(w_cf)
        + (M world_erase) outer world_erase_key
        + world_value outer world_write_key

M_action = (M gated_delta_erase) outer action_erase_key
         + gated_delta_write outer action_write_key

M_next = M_world + M_action
```

其中 world erase/write 与官方 x070 generalized delta rule 完全对应；两个 action key 只由
world query 生成，不读取 raw action。矩阵输入、更新和持久 state 均强制 fp32。

## 3. 官方兼容的 log-hazard double-exp

原提案曾冻结为：

```text
exp(-exp(world_decay_logit + gated_action_delta))
```

但官方 RWKV7 x070 的递推是：

```text
exp(-0.606531 * sigmoid(world_decay_logit))
```

如果把同一个官方 `world_decay_logit` 直接送入 double-exp，二者在 action delta 为 0 时
也不相等。不过该冲突只存在于直接复用同一个 logit 的参数化；把官方 decay rate 映射成
log-hazard 后，可以同时保留官方基线和 double-exp 动态范围。

修订后的默认 B4/B6 先计算：

```text
world_rate = 0.606531 * sigmoid(world_decay_logit)
world_log_hazard = log(world_rate)
```

然后执行：

```text
action_log_rate_delta = clamp(gate * delta_decay_parameter, -8, 5)
decay_cf = exp(-exp(world_log_hazard + action_log_rate_delta))
```

数值实现使用完全等价、避免显式 `log` 的形式：

```text
decay_cf = exp(-world_rate * exp(action_log_rate_delta))
```

当 `actual == reference` 时，动作 log-rate residual 为 0，故
`decay_cf=exp(-world_rate)`，与官方 x070 bitwise 相同。非零动作则可以把 decay 扩展到完整
`(0,1)` 区间，而不再受官方 sigmoid 版本约 `0.545` 的 retention 下界限制。

旧的 `official_logit_residual` 被保留为显式消融模式，不再是 B4/B6 默认值。正式 config
使用 `decay_map: official_compatible_log_hazard_double_exp`。

## 4. B4/B6 与参数公平性

MVP 机制配置继续使用 `channel_mlp_dim=768`。论文主参数匹配配置固定为：

| predictor | channel width | 参数量 | 相对 B0 |
|---|---:|---:|---:|
| B0 official Transformer | official | 10,791,360 | 0% |
| B2 vanilla RWKV7 | 4,096 | 10,789,056 | -0.021% |
| B4 rank-two uncentered | 3,728 | 10,783,296 | -0.075% |
| B6 CC-RWKV centered | 3,728 | 10,783,296 | -0.075% |

B4/B6 参数量完全相同；动作模块增加的容量由较窄的 ChannelMix 抵消。所有值均在预注册
的 B0 ±5% 范围内，并且实际上小于 0.1% 偏差。

## 5. 诊断输出

每层、每步输出以下诊断量：

```text
world_decay_logit
action_decay_delta
action_log_rate_delta
world_decay_rate
counterfactual_decay_rate
counterfactual_decay
world_erase
action_erase_delta
world_write
action_write_delta
intervention_gate
matrix_update_world_norm
matrix_update_action_norm
rank_two_action_norm
```

`matrix_update_action_norm` 包括动作对 decay diagonal 和两个 rank-two residual 的总影响；
`rank_two_action_norm` 只记录 erase/write 两个 outer-product residual，便于定位究竟是动作
衰减还是动作写入导致状态改变。

## 6. 机制测试

新增 9 组 M3 测试：

1. log-hazard 形式与独立 double-exp oracle 一致；
2. 零 log-hazard residual 与官方 x070 decay bitwise 一致；
3. log-hazard 能扩展到旧 sigmoid-logit 消融之外的 retention 范围；
4. 分解更新与独立显式 transition matrix oracle 一致；
5. centered zero delta 与迁移后的官方 vanilla RWKV7 精确一致；
6. 固定 world query/state 后交换 actual/reference，三个 delta 反号；
7. 结构审计确认 raw action 的所有 Linear consumer 都位于 action parameter network；
8. 两个 equal-action pair 即使 raw action 不同，prediction/state 仍 bitwise 相同；重新执行
   非 reference update 后，matrix 和 prediction 均改变；
9. actual/reference 输入及共享 action network 参数获得有限、非零梯度，同时检查 B4/B6
   参数公平性和 gate 初始化。

最终验证：

- M3 + M1 RWKV 定向测试：18 passed；
- 全仓库：**95 passed**，5 个已有 PyTorch warning；
- M3 修改范围 Ruff：通过。

## 7. 200-step 真实 paired 拟合

数据直接读取 M2 正式缓存：

```text
artifacts/cache/cc_rwkv/tworoom/mvp5000/branches.h5
```

固定前 128 个 train snapshots，输入 3-step 真实 history。每个样本从同一个 history matrix
state 和相同 branch initial latent 分别执行 factual 首 action block 与 pulse-noop 首 action
block。训练目标同时包含 factual/noop 普通一步 MSE 和真实 effect MSE：

```text
MSE(pred_factual, target_factual)
+ MSE(pred_noop, target_noop)
+ MSE(pred_factual - pred_noop, target_factual - target_noop)
```

这是 65,088 参数的小型工程 smoke（`D=32, L=2, H=4`），不是论文主模型。

| 指标 | step 0 | step 200 | 相对下降 |
|---|---:|---:|---:|
| ordinary one-step MSE | 1.306470 | 0.559585 | 57.17% |
| factual-noop effect MSE | 1.500765 | 0.830451 | 44.66% |
| gate mean | 0.100000 | 0.193533 | — |
| gate saturated fraction | 0 | 0 | — |

最终数值全部 finite，gate 没有任何单元进入 `<0.01` 或 `>0.99` 饱和区间。

产物：

- `artifacts/results/cc_rwkv/tworoom/m3_smoke_log_hazard/summary.json`；
- `artifacts/results/cc_rwkv/tworoom/m3_smoke_log_hazard/b6_m3_smoke.pt`；
- summary SHA256：`6f0f1c7999a9a05d16da3a1f6d9c2de5589ca8a595ad52c7687c3d0165c35acb`；
- checkpoint SHA256：`b4ad337db1c0e65c71a84757fc150bad25074e4ba11c9bb33551ad356581da40`。

旧 `m3_smoke/` 未覆盖，保留为 `official_logit_residual` 工程消融的原始记录。

## 8. M3 Go/No-Go

M3 Go 条件全部满足：

- actual/reference 严格归零：通过；
- 官方 RWKV7 world path 精确退化：通过；
- no action bypass：通过；
- rank-two 显式 oracle：通过；
- B4/B6 已实现且容量一致：通过；
- B6 可拟合真实一步 effect：通过（200 步下降 44.66%）；
- gate 不全为 0/1：通过；
- 全仓回归测试：通过。

尚未完成：

- M4 的统一训练器、完整 effect/curriculum loss、early stop 和 resume；
- CEE/CED/CER、trajectory AUC 和 matrix probe；
- B3 paper-spec DWM-RWKV；
- B2/B3/B4/B6 三 seed MVP 公平比较；
- 50/100-step 长程数据与 B7；
- RL 控制规划。

M3 到此停止。下一里程碑 M4 必须得到用户批准后才能开始。
