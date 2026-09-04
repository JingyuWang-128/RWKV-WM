# CC-RWKV-WM 详细实施计划

> 对应方法提案：[long_horizon_method_proposals.md](long_horizon_method_proposals.md)  
> 计划对象：CC-RWKV-WM（Counterfactual-Centered RWKV World Model）  
> 计划版本：v1，2026-08-28  
> 第一阶段范围：只验证世界模型的长期 imagination；不引入 RL，不以 MPC 成败作为主要证据。

当前进度：**M0、M1、M2、M3、M4 已完成；M5 工程实现与 CPU 预演完成，正式
5,000-snapshot 双任务实验正在执行；M6 未开始**。先前 CUDA 不可用仅发生在隔离执行环境；
宿主机已确认 4 张 RTX A6000 可用，正式 CUDA 作业通过宿主机权限运行。M0 的冻结协议、B0 指标、provenance
和验收结果见 [CC-RWKV-WM M0 验收报告](cc_rwkv_m0_report.md)；M1 的官方
RWKV7 对齐、状态接口、训练 smoke 和验收结果见
[CC-RWKV-WM M1 验收报告](cc_rwkv_m1_report.md)。
M2 的完整环境快照、真实四分支缓存和 5,000-snapshot MVP 结果见
[CC-RWKV-WM M2 验收报告](cc_rwkv_m2_report.md)；M3 的反事实中心化递推、B4/B6、
机制测试和真实 paired 小数据拟合见
[CC-RWKV-WM M3 验收报告](cc_rwkv_m3_report.md)。
M4 的统一训练器、指标、B3、公平性检查和四方法 smoke 见
[CC-RWKV-WM M4 验收报告](cc_rwkv_m4_report.md)。
M5 的逐样本 paired bootstrap、effect-weight 搜索、双任务三 seed 工程预演、就绪审计和
正式阻塞条件见 [CC-RWKV-WM M5 执行报告](cc_rwkv_m5_report.md)。

## 1. 目标、交付物与完成定义

### 1.1 工程目标

在不重新训练 LeWM 视觉编码器的前提下，新增一个具有持久矩阵状态的 RWKV predictor，并实现反事实中心化的 decay、erase、write 递推。用相同初始 simulator state 的真实干预分支监督该递推，验证它是否比以下两类解释更好：

1. 单纯把 Transformer predictor 换成 RWKV；
2. 只在 RWKV 输出端加入 DWM 风格反事实监督。

本阶段要回答的是“世界模型能否在没有新观察的情况下更准确地 free-run 50/100 个 primitive actions”，不是“某个控制器能否完成任务”。

### 1.2 必须产出的交付物

- 可单步、有状态、可复制分支状态的 vanilla RWKV predictor；
- CC-RWKV 的纯 PyTorch diagonal-plus-rank-two 参考实现；
- 至少 TwoRoom-W 和 Action-Delay 两个环境的同状态反事实分支采集器；
- 冻结 LeWM encoder 后的成对 latent 数据集和版本化 manifest；
- B0--B7 中至少 B0、B2、B3、B4、B6、B7 的统一训练/评估入口；
- 固定动作、无中途观察的 5/10/20/50/100 primitive-action rollout 评估；
- CEE、CED、CER、latent/state trajectory error、历史 probe 和机制审计；
- 三个训练种子的聚合结果、paired bootstrap 置信区间和 Gate A--D 报告；
- 可从配置、数据 hash 和 checkpoint 完整复现实验的命令记录。

### 1.3 第一阶段完成定义

只有同时满足以下条件，第一阶段才算完成：

- 新增测试和现有 59 个测试全部通过；
- B0/B2/B3/B4/B6/B7 使用相同 split、encoder、训练样本数和 primitive-action horizon；
- Gate A 有自动化机制报告，不能仅靠训练 loss 判断；
- Gate B 至少在 TwoRoom-W 和 Action-Delay 上完成；
- Gate C 至少在一个导航任务和一个接触/连续控制任务上完成；
- Gate D 使用显式 same-image/different-history 数据完成；
- 无论结果是否支持方法，均保存失败条件、配置和结果，而不是只保留最好运行。

RL 属于第二阶段。Gate A--D 未通过时，不实现 RL 来补救世界模型结果。

---

## 2. 当前仓库审计与约束

### 2.1 已有可复用部分

- 官方 LeWM 训练参考位于 `artifacts/assets/source/leworldmodel/`；
- 实际加载官方 checkpoint 的路径由 `stable_worldmodel.wm.utils.load_pretrained` 管理；
- TwoRoom 官方 encoder latent 维度为 192，predictor 深度为 6；
- TwoRoom action block 为 5 个 primitive actions，action encoder 输入维度为 10；
- 官方 Transformer predictor 约有 10,791,360 个参数，可作为参数匹配基准；
- `/data/wjy/lewm_data/tworoom.h5` 和 `/data/wjy/lewm_model/tworooms/weights.pt` 已存在；
- 仓库已有确定性 split、checkpoint、device、paired bootstrap 和结果落盘范式；
- 当前机器可见 4 张 RTX A6000 48 GB，但计划编写时每张仅约 8.7 GB 空闲，运行前必须重新检查资源。

### 2.2 现有实现不能直接满足的部分

- `ARPredictor/Predictor.forward(x, c)` 是无显式持久状态的 Transformer 接口；
- 官方 `JEPA.rollout` 和 `LeWorldModelAdapter.batch_rollout_tensor` 都把 latent/action 历史截断到 `history_size`，默认 3；
- 仓库没有 RWKV、Mamba 或其他 SSM predictor；
- 仓库没有 counterfactual branch schema、分支 collector、effect loss 或 CEE/CED/CER；
- TwoRoom 只有 `_set_state(agent_position)`，没有完整、通用、可验证的 simulator snapshot/restore API；
- PushT、DMC 和 OGBench 的 snapshot 完整性尚未审计，不能直接排入首轮主实验；
- 当前 `pyproject.toml` 没有 RWKV 外部依赖，因此首版不能依赖第三方 RWKV CUDA kernel。

### 2.3 实现原则

不修改 pinned upstream/vendor 文件作为主实现。所有新代码放在 `src/cape_wm/cc_rwkv/`，通过兼容包装器接入官方 LeWM encoder、projector、action normalization 和 checkpoint。这样可以：

- 保持 B0 官方 checkpoint 可原样复现；
- 避免 vendor 更新覆盖方法代码；
- 明确区分“官方复现”和“paper-spec reimplementation”；
- 让单元测试不依赖完整 stable-worldmodel 运行时。

---

## 3. 目标代码结构

计划新增以下包和入口；名称在实现阶段固定，不再临时改名。

```text
src/cape_wm/cc_rwkv/
├── state.py          # RWKVMatrixState、初始化、clone、detach、序列化
├── cell.py           # VanillaRWKVCell、CounterfactualCenteredRWKVCell
├── predictor.py      # 多层 predictor、step/consume_history/rollout
├── lewm.py           # 冻结官方 encoder/projector 的兼容包装器
├── branches.py       # snapshot 协议、分支定义、HDF5 schema 与 Dataset
├── losses.py         # 普通预测、effect、direction、magnitude、world、gate、SIGReg
├── metrics.py        # CEE/CED/CER、rollout AUC、error slope、probe 数据
├── checkpoint.py     # checkpoint v1 读写和兼容性检查
└── evaluation.py     # 固定动作 open-loop/free-running 评估

scripts/
├── collect_counterfactual_branches.py
├── cache_counterfactual_latents.py
├── train_cc_rwkv.py
├── evaluate_cc_rwkv.py
└── aggregate_cc_rwkv.py

configs/cc_rwkv/
├── base.yaml
├── tworoom.yaml
├── action_delay.yaml
└── paper_matrix.yaml

tests/
├── test_cc_rwkv_state.py
├── test_cc_rwkv_cell.py
├── test_counterfactual_branches.py
├── test_cc_rwkv_losses.py
├── test_cc_rwkv_lewm.py
└── test_cc_rwkv_e2e.py
```

`artifacts/assets/source/` 保持只读。训练产物分别写入：

```text
artifacts/cache/cc_rwkv/<dataset_id>/
artifacts/checkpoints/cc_rwkv/<method>/<seed>/
artifacts/results/cc_rwkv/<task>/<method>/<seed>/
```

---

## 4. 固定接口与张量约定

### 4.1 Horizon 口径

所有 API 和结果文件必须同时保存：

- `model_steps`：predictor 递推次数；
- `primitive_steps`：环境实际执行动作数；
- `action_block`：每次模型递推包含的 primitive actions 数；
- `observation_stride`：目标 observation/latent 的采样间隔。

TwoRoom 官方比较固定 `action_block=5`：

| model steps | primitive steps |
|---:|---:|
| 1 | 5 |
| 2 | 10 |
| 4 | 20 |
| 10 | 50 |
| 20 | 100 |

因此 TwoRoom 主表中的 `Rollout@1` 必须写成“1 model transition / 5 primitive actions”，不能写成“1 primitive step”。Action-Delay 诊断环境使用 `action_block=1`，用于补足逐 primitive-step 的机制验证。

### 4.2 状态类型

`RWKVMatrixState` 使用 dataclass，不暴露松散 tensor 列表：

```python
@dataclass
class RWKVMatrixState:
    matrix: Tensor       # [B, L, H, Dh, Dh]
    shift: Tensor        # [B, L, D]，若 cell 使用 token shift
    steps: Tensor        # [B]，已消费步数

    def clone_branches(self, branches: int) -> "RWKVMatrixState": ...
    def detach(self) -> "RWKVMatrixState": ...
    def index_select(self, indices: Tensor) -> "RWKVMatrixState": ...
```

默认 MVP 结构：`D=192, L=6, H=6, Dh=32, channel_mlp_dim=768`。该结构用于机制验证；论文主比较另运行参数匹配配置，使 predictor 参数量落在 B0 的 ±5% 内。参数匹配只允许调整 channel-mix width 或 model width，不能增加 B7 独有层。

状态的时间语义固定为：`state_t` 已经消费 `(z_0,u_0),...,(z_{t-1},u_{t-1})`，但还没有消费 `(z_t,u_t)`；调用 `step(z_t,u_t,state_t)` 后得到 `pred_z_{t+1}, state_{t+1}`。teacher forcing 只把真实 `z_t` 用作当前 step 输入；free-running 从第二个未来 step 起使用上一时刻的 `pred_z`。所有 Dataset 和测试按此对齐，禁止通过多移/少移一格让 action 与 target 错配。

### 4.3 Predictor 公共接口

```python
class StatefulWorldPredictor(nn.Module):
    def init_state(self, batch_size: int, *, device, dtype) -> RWKVMatrixState: ...

    def step(
        self,
        latent: Tensor,                 # [B, Dz]
        action: Tensor,                 # [B, Ablock]
        state: RWKVMatrixState,
        reference_action: Tensor | None = None,
        *,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, RWKVMatrixState, dict[str, Tensor]]: ...

    def consume_history(
        self,
        latents: Tensor,                # [B, Th, Dz]
        actions: Tensor,                # [B, Th, Ablock]
        mask: Tensor | None = None,
    ) -> RWKVMatrixState: ...

    def rollout(
        self,
        initial_latent: Tensor,         # [B, Dz]
        future_actions: Tensor,         # [B, K, Tf, Ablock]
        state: RWKVMatrixState,
        reference_actions: Tensor | None = None,
    ) -> dict[str, Tensor]: ...
```

`step` 返回的第一个值是下一 latent。`rollout` 先将 state clone 到 K 个 imagination branches，然后完全 free-run；未来预测 latent 会作为下一步输入，期间不注入真实 latent。

### 4.4 CC 递推内部输出

`CounterfactualCenteredRWKVCell` 必须返回以下诊断量，形状保留 layer/head 维：

- `world_decay_logit`；
- `action_decay_delta`；
- `world_erase`；
- `action_erase_delta`；
- `world_write`；
- `action_write_delta`；
- `intervention_gate`；
- `matrix_update_world_norm`；
- `matrix_update_action_norm`。

当前 raw action 只能进入共享的 `action_parameter_network(q_t, action)`。readout、receptance、channel mix、world parameter network 和 prediction head 均不得读取 raw action。代码审计和测试都要验证这一约束。

### 4.5 LeWM 兼容包装器

`FrozenLeWMEncoder` 从官方 checkpoint 复制并冻结：

- encoder；
- projector；
- 图像 transform；
- action StandardScaler 参数。

默认不复用 Transformer predictor。`pred_proj` 仅在输入/输出维度都为 192 且消融验证没有隐藏动作入口时复用初始化，否则复制结构并训练新参数。包装器必须支持：

```python
encode_images(images) -> latent
normalize_action(raw_action_block) -> normalized_action
decode_prediction(predictor_hidden) -> latent
```

训练 checkpoint 不序列化整个官方 encoder，只保存其来源路径、SHA256、模型 config SHA256 和可选的微调差分参数。

---

## 5. 反事实环境与数据协议

### 5.1 快照契约

每个进入主实验的环境必须实现：

```python
class BranchableEnv(Protocol):
    def snapshot(self) -> Mapping[str, ndarray | int | float]: ...
    def restore(self, snapshot: Mapping[str, Any]) -> None: ...
    def external_noise_state(self) -> Any: ...
    def set_external_noise_state(self, state: Any) -> None: ...
```

快照至少覆盖动力学未来所依赖的全部变量、环境几何、目标、episode step、随机数状态和 wrapper 状态。只复制当前图像或 agent position 不算完整快照。

TwoRoom v1 是确定性、位置控制环境。其首版 adapter 显式保存：agent position、target position、wall axis/position/thickness、door number/positions/sizes、agent speed/radius、variation values、elapsed steps。恢复后执行同一 action 的下一状态必须逐元素一致。

Action-Delay 和 Hidden-Velocity 必须额外保存：速度、动作 FIFO、延迟索引和外生 drift 状态。

### 5.2 分支定义

每个 snapshot 默认生成 4 个分支：

1. `reference`：全程执行语义 no-op；
2. `factual`：执行数据中的行为动作 suffix；
3. `pulse_noop`：第一 model step 改为 no-op，之后与 factual 使用完全相同 suffix；
4. `pulse_local`：第一 model step使用支持域内局部扰动，之后与 factual 使用相同 suffix。

主 CEE/CED/CER 使用 `factual - pulse_noop` 和 `pulse_local - factual`，因为两支只在第一个 model step 不同，可以检验一次动作干预是否被长期保留。`factual - reference` 作为持续动作干预的次要指标，不替代 pulse 结果。

局部扰动默认从行为动作的每维加入 `0.1 * action_range` 的有符号扰动并裁剪；如果 kNN action-support 分数低于训练集第 5 百分位，则重采样。离散环境改用同状态历史中出现过的其他合法动作。

### 5.3 Common random numbers

确定性 TwoRoom 不需要随机数配对。随机环境中，每个分支在 restore 后恢复同一个外生 RNG state，使动作成为分支间唯一系统变化。另建立 `independent_noise` 评估 split，但它不能用于核心 effect 标签。

### 5.4 数据 split

先按 episode 分组，再按 `train/validation/test = 80/10/10` 固定拆分；不能从同一 episode 的不同 snapshot 分到不同 split。正式测试 episode 复用仓库既有冻结 TwoRoom test manifest，避免与此前比较结果不一致。

训练内部再从 train episode 划出 10% 作为 model-selection；不使用 test 选择 horizon、loss 权重或 reference action。

默认种子：split `3072`，训练 `0/1/2`，局部扰动 `1000 + train_seed`，bootstrap `3072`。

### 5.5 HDF5/manifest schema

主缓存存冻结 encoder latent，而不是保存全部 224×224 分支图像：

```text
/metadata/schema_version                 "cc_rwkv_branches_v1"
/metadata/environment
/metadata/dataset_sha256
/metadata/encoder_weights_sha256
/metadata/action_scaler_mean
/metadata/action_scaler_scale
/metadata/action_block
/metadata/observation_stride
/metadata/reference_action_semantics

/samples/sample_id                       [N] string
/samples/episode_id                      [N] int64
/samples/start_step                      [N] int64
/samples/snapshot_hash                   [N] string
/samples/common_noise_id                 [N] string
/samples/history_latents                 [N, Th_max, 192] float16
/samples/history_actions_raw             [N, Th_max, Ablock] float32
/samples/history_mask                    [N, Th_max] bool
/samples/branch_type                     [N, K] uint8
/samples/branch_actions_raw              [N, K, Tf_max, Ablock] float32
/samples/branch_latents                  [N, K, Tf_max+1, 192] float16
/samples/branch_states                   [N, K, Tf_max+1, Ds] float32
/samples/branch_mask                     [N, K, Tf_max+1] bool
/samples/action_support_score            [N, K, Tf_max] float32
```

只保存 1% 的 raw RGB 分支审计集，用于验证 latent cache 与在线 encoder 一致以及制作可视化。冻结 encoder 的 Stage A--C 使用 latent cache；只有 Stage D 解冻 encoder 时才采集或复用 raw RGB 子集。

manifest 额外记录采集命令、git/source hash、Python/Torch/stable-worldmodel 版本、环境 adapter 版本、每个 split 的 episode IDs、失败/丢弃样本原因和数组 SHA256。

### 5.6 数据量阶梯

| 阶段 | snapshots | branches | 最大 primitive horizon | 用途 |
|---|---:|---:|---:|---|
| 单元/冒烟 | 32 | 4 | 10 | schema、恢复一致性 |
| MVP | 5,000 | 4 | 20 | B2/B3/B4/B6、Gate A |
| TwoRoom 完整 | 20,000 | 4 | 50 | Gate B、B7 |
| 长程扩展 | 20,000 中至少 5,000 | 4 | 100 | Gate C/D |

采集前先估算磁盘；latent float16 主缓存目标控制在 5 GB 内，raw audit 目标控制在 10 GB 内。超过预算时减少 raw audit，不减少成对 latent 样本或 horizon。

### 5.7 首轮诊断环境的固定动力学

为避免实现者自行选择一个容易得出期望结论的变体，首轮环境参数预先固定：

- `TwoRoom-W`：在现有 TwoRoom collision update 后叠加 episode-constant drift；drift 方向从 8 个等角方向采样，幅度为 `0.5 pixel/primitive-step`，图像不显示 drift。snapshot 保存 drift。普通 `TwoRoom` 同时保留为 drift=0 的对照。
- `Hidden-Velocity TwoRoom`：action 表示加速度，`v_{t+1}=0.9 v_t + 1.0 a_t`，速度逐维裁剪到 `[-10.5,10.5]`，位置按 `v_{t+1}` 更新后使用现有碰撞处理；图像只渲染位置。snapshot 保存 velocity。
- `Action-Delay`：使用普通 TwoRoom 位置动力学，但执行 5 个 primitive steps 前提交的动作，即固定 FIFO 长度 5；reset 时 FIFO 从行为轨迹历史预热，不能全零初始化。snapshot 保存 FIFO 内容和指针。
- `Door/Switch`：只作为 Gate D 补充，不进入首轮 MVP；门状态由一次性 switch action 改变且保持，局部裁剪画面不显示 switch。其精确地图在 M6 前冻结。
- `Stochastic Drift`：只在确定性 Gate A/B 通过后加入；外生噪声为每步二维零均值高斯，标准差 `0.25 pixel`，paired 主结果复用同一噪声序列。

前三个环境的 action bounds 与渲染尺度沿用现有 TwoRoom。所有数值写入 config 和 manifest；任何参数修改产生新的 dataset ID，不能覆盖已有缓存。

---

## 6. 模型实现细节

### 6.1 Vanilla RWKV 基线 B2

先实现 vanilla cell，输入为 latent 与 action embedding 的和/拼接投影，并产生标准 RWKV-7 generalized delta-rule 参数。它必须：

- 真实历史与 imagination 使用同一个 state 类型；
- `consume_history` 后不因进入 rollout 而清零 state；
- 每个候选 action sequence clone 独立 state；
- 支持 padding mask，padding 不更新 state；
- 参考 PyTorch 实现支持 fp32 和 bf16；
- 不使用 history window 截断。

B2 是结构和训练管线基线，不声称创新。

### 6.2 CC cell B6/B7

对每层每个 head 计算：

```text
q = RMSNorm(Wz(z))
world_params = world_net(q)
actual_params = action_net(q, action)
reference_params = action_net(q, reference_action)
delta = actual_params - reference_params
```

然后执行提案中的更新：

```text
world_rate = 0.606531 * sigmoid(world_decay_logit)
action_log_rate_delta = clamp(gate * delta_decay_parameter, -8, 5)
w_cf = exp(-world_rate * exp(action_log_rate_delta))
G_cf = diag(w_cf) + outer(world_erase, world_key)
       + outer(delta_action_erase, action_key)
U_cf = outer(world_value, world_write_key)
       + outer(delta_action_value, action_write_key)
M_next = M @ G_cf + U_cf
```

实现时不显式构造完整 `G_cf`，使用 einsum 分解 diagonal 和两个 rank-one 项，以降低显存；但单元测试提供显式矩阵版本作为 oracle，并在 fp32 下比较。

`action_net` 的 actual/reference 两次调用共享全部权重。action/reference head 最后一层零初始化，使 Stage B 开始时 action intervention 为零。gate 的 bias 初始化为使 sigmoid 输出约 0.1，而不是 0 或 1。

数值稳定规则固定为：

- 只把 centered action log-hazard residual clamp 到 `[-8, 5]`；官方 world logit
  不 clamp，确保 action delta 为零时与 x070 bitwise 一致；
- matrix state 默认 fp32 累积，即使其余网络使用 bf16；
- effect norm 分母使用 `epsilon=1e-6`；
- gradient norm clip 为 1.0；
- 发现 NaN/Inf 时跳过 optimizer step、记录 sample IDs，并在连续 3 次后终止运行。

### 6.3 无动作旁路

CC 版本的调用图必须满足：

```text
raw action -> action_net -> update delta -> M_next -> RWKVRead -> prediction
```

禁止：

```text
raw action -> readout/prediction head
raw action -> world_net/receptance/channel mix
raw action -> residual added directly to predicted latent
```

B3（DWM-RWKV）允许在训练输出端建立 DWM 辅助 head，但推理主 predictor 仍是 vanilla RWKV；B3 不复用 CC 的内部 delta，以免基线污染。

### 6.4 Checkpoint 格式

统一保存 `cc_rwkv_checkpoint_v1`：

```python
{
  "format": "cc_rwkv_checkpoint_v1",
  "method": "b2|b3|b4|b5|b6|b7",
  "model_config": {...},
  "training_config": {...},
  "modules": {"predictor": state_dict, "prediction_head": state_dict, ...},
  "optimizer": state_dict,
  "scheduler": state_dict,
  "epoch": int,
  "global_step": int,
  "best_metric": float,
  "encoder_provenance": {...},
  "data_manifest_sha256": str,
  "rng_states": {...},
  "history": [...]
}
```

加载时检查 latent dim、action block、action dim、encoder hash、normalizer hash 和 schema version；任何不一致默认报错，只有显式 `--allow-provenance-mismatch` 可用于诊断，且结果标记 invalid-for-paper。

---

## 7. Loss、采样和优化

### 7.1 Loss 定义

总 loss 固定为：

```text
L = λpred Lpred
  + λeffect Leffect
  + λdir Ldirection
  + λmag Lmagnitude
  + λworld Lworld
  + λgate Lgate
  + λsigreg Lsigreg
```

- `Lpred`：所有有效分支的 multi-step latent Smooth-L1；
- `Leffect`：预测 branch difference 与真实 branch difference 的 Smooth-L1；
- `Ldirection`：非零真实 effect 上的 `1 - cosine_similarity`；
- `Lmagnitude`：`abs(log((||Δpred||+eps)/(||Δtrue||+eps)))`；
- `Lworld`：reference/no-op branch 的普通预测 loss；
- `Lgate`：gate 稀疏与防饱和正则；
- `Lsigreg`：沿用 LeWM 的 representation regularization，但它只在 Stage D 解冻 encoder 时参与反向传播；Stage A--C 的 target latent 来自冻结 cache，该项固定为 0，仅可作为离线诊断记录。

方向和幅度 loss 只对 `||Δtrue||` 高于训练集第 10 百分位的样本计算，避免把近零物理效应的方向噪声作为监督。

### 7.2 默认权重与选择规则

初始配置：

```yaml
loss:
  prediction: 1.0
  effect: 1.0
  direction: 0.1
  magnitude: 0.1
  world: 0.5
  gate: 0.001
  sigreg: 0.0       # Stage A--C
  sigreg_stage_d: 0.09
```

只在 validation split 上比较以下三个 effect 权重：`0.5, 1.0, 2.0`。其余权重首轮固定。选择指标为：

```text
validation_score = CEE_AUC_1_to_20 + 0.5 * normalized_rollout_error_AUC_1_to_20
```

如果 one-step latent error 相对 B2 恶化超过 3%，该配置淘汰，不进入 test。

### 7.3 Horizon curriculum

TwoRoom 以 model transitions 训练，对应 primitive horizon：

```text
1 (5) -> 2 (10) -> 4 (20) -> 10 (50) -> 20 (100)
```

Action-Delay 以 primitive/model 一致的：

```text
1 -> 5 -> 10 -> 20 -> 50
```

每级至少训练 5,000 optimizer steps；validation score 连续 3 次评估改善小于 0.5% 时可提前进入下一级。每次扩 horizon 后从前一级最佳 checkpoint 继续，保留 25% 较短 horizon batch，防止短期能力遗忘。

### 7.4 Stage A--D

#### Stage A：B2 vanilla RWKV

- 冻结 encoder/projector；
- action encoder 可训练，但 B2--B7 初始化和训练规则一致；
- 使用普通离线轨迹和所有反事实分支作为非配对普通样本；
- 先做 teacher-forced 1-step warm-up 5,000 steps，再进入 free-running curriculum；
- 选择 `Rollout@20 primitive` validation error 最低 checkpoint。

#### Stage B：B6 单步反事实初始化

- 从 B2 加载兼容的 world/readout/channel 参数；
- 新增 action/reference delta head 零初始化；
- 前 1,000 steps 冻结 world/readout，只训练 delta/gate；
- 使用 horizon 1 的 `Lworld + Leffect + Ldirection + Lmagnitude`；
- Gate A 通过后才允许进入多步。

#### Stage C：B7 多步课程

- 解冻 predictor 全部参数；
- 使用完整 free-running，不把真实中间 latent 喂回模型；
- 按上述 horizon 课程训练；
- 每个 batch 50% pulse pairs、25% factual/reference、25% 普通非配对轨迹；
- 保存每个 horizon 的最佳 checkpoint，不只保存最终 horizon。

#### Stage D：可选 encoder 微调

- 仅在 frozen-encoder B7 已通过 Gate B 后执行；
- 使用 raw RGB 审计/微调子集；
- 只解冻 encoder 最后 2 层，encoder LR 为 predictor LR 的 0.1；
- frozen-encoder 结果始终作为论文主结果，Stage D 只能作为附加结果。

### 7.5 优化默认值

```yaml
optimizer: AdamW
learning_rate: 5.0e-5
weight_decay: 1.0e-3
warmup_steps: 2000
scheduler: cosine
precision: bf16
matrix_state_precision: fp32
batch_size: 128
gradient_accumulation: 1
gradient_clip: 1.0
```

显存不足时按顺序调整：batch 128→64→32，增加 gradient accumulation 保持 effective batch 128，再启用 activation checkpoint；不改变 horizon、state 维度或 loss 来迁就显存。

---

## 8. 基线和公平比较矩阵

| ID | 实现 | 训练数据 | 内部反事实递推 | 真实 paired loss |
|---|---|---|---:|---:|
| B0 | 官方 Transformer-LeWM | 官方普通轨迹 | 否 | 否 |
| B1 | DWM-Transformer paper-spec | 数量匹配 | 否 | 弱/输出级 |
| B2 | vanilla RWKV-LeWM | 数量匹配 | 否 | 否 |
| B3 | DWM-RWKV paper-spec | 数量匹配 | 否 | 输出级 |
| B4 | rank-two RWKV，无 reference subtraction | 数量匹配 | 否 | 否 |
| B5 | CC-RWKV，动作置换伪分支 | 数量匹配 | 是 | 否 |
| B6 | CC-RWKV，一步真实 paired | 数量匹配 | 是 | 是，H=1 |
| B7 | 完整 CC-RWKV | 数量匹配 | 是 | 是，H≤50/100 |

首个开发闭环实现 B0/B2/B3/B4/B6；B1 和 B5 可在 Gate A 后并行补齐，B7 在 Gate A/B 后训练。

公平性要求：

- 同一冻结 encoder、normalizer、episode split、latent cache 和 evaluation action sequences；
- 每个方法看到相同数量的 branch trajectories；不使用 paired loss 的模型把分支当独立普通样本；
- B2/B3/B4/B6/B7 的 predictor 参数量主配置相差不超过 5%；
- 相同 optimizer steps、effective batch、horizon curriculum 和 early-stop budget；
- 训练种子 0/1/2；测试 action sequence 和 bootstrap pairing 完全一致；
- B1/B3 明确标注 `paper_spec_reimplementation`，不能标成作者官方结果。

必须做的核心消融：

- 分别移除 decay、erase、write delta；
- 移除 reference subtraction；
- no-op 与 conditional-mean reference；
- effect@1 与 effect@1:50；
- 真实 paired 与动作置换；
- rank-one 与参数匹配 rank-two；
- full state 与每 3/10/50 步重置；
- 允许 raw action 进入 readout 的旁路版本。

---

## 9. 自动化测试与机制验收

### 9.1 State/cell 单元测试

1. `init_state` 的 shape、dtype、device 正确；
2. `clone_branches` 复制值但不共享可写 storage；
3. state checkpoint round-trip 逐元素一致；
4. padding step 不改变 state；
5. sequential step 与 sequence wrapper 输出一致；
6. 分解 einsum 更新与显式 `M @ G + U` oracle 在 fp32 下 `atol=1e-5, rtol=1e-4`；
7. bf16 20 步递推无 NaN/Inf；
8. actual action 等于 reference action 时，三类 delta 和 action update 的绝对最大值 `<1e-7`（fp32）；
9. actual/reference 交换时 delta 符号反转；
10. reference subtraction 的参数梯度非零且两次 action_net 共享权重。

### 9.2 无旁路测试

构造固定 `M_next` 和 `q_t`，只替换传给审计包装器的 raw action，prediction 必须 bitwise 相同。重新执行 update 后，非 reference action 必须能改变 prediction。另用 `torch.fx` 或 forward hook 审计 raw action tensor 的消费者，只允许 action parameter network。

### 9.3 快照/数据测试

1. snapshot→action→restore→same action 得到完全一致的 observation/state；
2. 两个 branch restore 后未执行动作时的 render hash 一致；
3. branch 之间无共享可变状态；
4. common-noise 分支的外生随机样本序列一致；
5. sample 不跨 episode；
6. train/validation/test episode 集合两两不交；
7. factual 和 pulse branch 在第一步后使用相同 suffix；
8. local action 满足 action bounds 和支持阈值；
9. latent cache 与在线 encoder 在 raw audit 子集上 `allclose`；
10. schema/version/hash 不匹配时 Dataset 拒绝加载。

### 9.4 Loss 测试

- 完全正确的 branch difference 产生零 effect/direction/magnitude loss；
- 近零真实 effect 被 mask，不产生不稳定 cosine；
- CER 对零分母稳定；
- free-running loss 的中间预测不依赖真实 future latent；
- 所有 loss 对应模块梯度存在且有限；
- B2/B4 不意外接收 paired 标签。

### 9.5 E2E 冒烟

在 32 snapshots 上训练 200 steps，要求：

- loss 有限且至少下降 20%；
- checkpoint 可恢复并继续训练；
- evaluation 生成规定字段；
- actual=reference 审计通过；
- 50-step 小批 rollout 不发生 state 串支；
- CPU 测试可运行，CUDA 可用时再做 bf16 smoke。

建议验收命令：

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python scripts/collect_counterfactual_branches.py \
  --config configs/cc_rwkv/tworoom.yaml --limit 32 --output artifacts/cache/cc_rwkv/smoke
.venv/bin/python scripts/train_cc_rwkv.py \
  --config configs/cc_rwkv/tworoom.yaml --method b6 --max-steps 200 --device cpu
.venv/bin/python scripts/evaluate_cc_rwkv.py \
  --checkpoint artifacts/checkpoints/cc_rwkv/smoke/best.pt --split validation
```

这些是实现后必须提供的入口，不是当前仓库已经存在的命令。

---

## 10. 实验协议

### 10.1 实验 0：现有 B0 锚点

复用当前官方 TwoRoom checkpoint 和冻结测试 pair manifest，补做固定 action sequences 的 open-loop rollout，而不是复用 MPC 成功率作为 imagination 指标。现有 Flat MPC 结果仅作为仓库回归锚点：offset 25/50/75/100 的成功率约为 0.817/0.567/0.270/0.103，不用于证明 CC 机制。

输出：`b0_open_loop_rollouts.npz`、`summary.json`、exact pair/action manifest。

### 10.2 实验 1：递推机制最小可证伪实验

任务：TwoRoom-W、Action-Delay。方法：B2/B3/B4/B6。Horizon：1/5/10/20 primitive（TwoRoom 同时报告对应 model steps）。

主问题：同一快照中只改变第一个 action block 后，B6 的 matrix update difference 是否比 B3/B4 更准确地保持真实 effect？

报告：

- no-op delta tolerance；
- action-bypass audit；
- update difference 的 action linear probe accuracy/R²；
- gate 分布及饱和比例；
- CEE/CED/CER 曲线；
- ordinary rollout error；
- one-step degradation。

Go：Gate A 全部通过，且 B6 的 CEE AUC 优于 B3/B4；否则停止扩展 B7，先定位结构或数据问题。

### 10.3 实验 2：TwoRoom 长程 paired rollout

在 20,000 snapshots 上训练 B7，horizon 扩到 50，随后对 5,000 个长分支扩到 100。测试期间：

- 仅起点给模型真实 history/observation；
- 环境和所有模型执行完全相同的 held-out action sequence；
- 模型中途不接收环境观察；
- 不运行 CEM/MPC/RL；
- 每个方法、seed、sample 使用同一 pair ID。

主指标：CEE AUC、Rollout@20/50 latent/state trajectory AUC。次指标：CED、CER、endpoint error、error growth slope、wall/door event accuracy。

Go：Gate B；否则核心 claim 降级为“可解释递推设计”，不进入多任务论文实验。

### 10.4 实验 3：历史记忆

构造：

- Hidden-Velocity：同位置、不同速度；
- Action-Delay：同画面、不同动作 FIFO；
- Door/Switch：同视觉局部画面、不同门状态历史；
- TwoRoom drift variant：同位置、不同 drift 参数历史。

比较 full persistent state 与每 3/10/50 步重置。训练线性 probe 从矩阵状态恢复 velocity/FIFO/door/drift，只在 train latent 上拟合，test history 上评估。

Go：Gate D；如果 full 与 history=3 无差异，不能声称长期记忆带来收益。

### 10.5 实验 4：跨任务 Gate C

任务顺序固定：

1. PushT-W（接触任务）；
2. Reacher-W 或 Ball-in-Cup（二选一，以完整 snapshot 审计先通过者为准）；
3. 另一个任务作为补充；
4. OGBench 只在前述任务 Gate C 通过后进入。

每个环境先完成 snapshot completeness test 和 reference-action semantic test。不能完整恢复 simulator state 的环境只可做普通 rollout 附录，不进入 paired causal 主表。

Go：至少一个导航和一个接触任务上，B7 相对最强 B3/B4 的 Rollout@50 trajectory AUC 下降 ≥10%，CER(50)更接近 1，Rollout@100 不更快爆炸。

### 10.6 统计与报告

- 三个独立训练 seed：0/1/2；
- 每个 seed 共享完全相同 test samples/action branches；
- 以 sample ID 配对 bootstrap 10,000 次，报告 95% CI；
- 方法差值为主要统计对象，不只报告各自均值；
- horizon 曲线同时报告均值、中位数、95% CI 和有效样本数；
- Gate B 的 15% CEE AUC 改善和 Gate C 的 10% trajectory AUC 改善均需 CI 不跨 0；
- 多任务结果同时给每任务和宏平均，不用一个简单任务掩盖失败任务。

结果文件最少包含：`metrics.jsonl`、`summary.json`、`resolved_config.yaml`、`provenance.json`、`failed_samples.jsonl` 和 `gate_report.md`。

---

## 11. 分阶段实施清单与 Go/No-Go

### M0：冻结协议与基线（2--3 工程日）

任务：

- 建立 `configs/cc_rwkv/base.yaml` 和 TwoRoom override；
- 冻结 existing test pair manifest、encoder/normalizer/data hash；
- 新增 horizon 单位校验；
- 保存 B0 固定动作 open-loop baseline；
- 记录 B0 predictor 参数量和计算量。

验收：B0 结果可重复，primitive/model horizon 无混写，现有 59 tests 全过。

### M1：Stateful vanilla RWKV（已完成）

任务：

- 实现 state、vanilla cell、多层 predictor；
- 实现 history consumption、branch clone、free-running rollout；
- 接入 frozen LeWM encoder/projection；
- 实现 B2 训练和 checkpoint；
- 完成 state/cell 单元测试和 20-step overfit。

Go：B2 one-step validation error不比随机初始化 predictor 差，20-step 可稳定运行且 persistent/full 与人为重置结果确实不同。

### M2：分支环境与数据缓存（已完成）

任务：

- 定义 snapshot protocol；
- 完成 TwoRoom 和 Action-Delay adapter；
- 实现四分支采集、support filter、latent cache 和 manifest；
- 采 32-sample smoke 与 5,000-snapshot MVP；
- 审计磁盘、恢复确定性和 split 泄漏。

Go：所有 snapshot/data 测试通过；任一恢复不一致样本率必须为 0，否则数据不可用于 effect loss。

### M3：CC 递推与机制测试（已完成）

任务：

- 实现 world/action/reference 参数分路；
- 实现 rank-two reference cell 和诊断输出；
- 实现 B4 与 B6；
- 加入 no-op、反对称、显式矩阵 oracle、无旁路测试；
- 200-step smoke 和小数据 overfit。

Go：actual=reference 严格归零、无 action bypass、B6 能拟合真实一步 effect、gate 不全为 0/1。

### M4：统一训练器和指标（已完成）

任务：

- 实现所有 loss、curriculum、early stop 和恢复训练；
- 实现 CEE/CED/CER、trajectory AUC、probe；
- 实现 B3 DWM-RWKV paper-spec 基线；
- 统一数据量/参数量/optimizer budget 检查；
- 输出自动 gate report。

Go：B2/B3/B4/B6 可用一条命令矩阵运行，公平性检查不通过时程序拒绝聚合。

### M5：Gate A/B MVP（工程实现已完成，正式实验执行中）

任务：

- 在 TwoRoom-W/Action-Delay 运行 seeds 0/1/2；
- 做 B2/B3/B4/B6 和必要消融；
- 聚合 paired CI；
- 审阅失败样本和 effect mask。

Go：按提案 Gate A/B。No-Go 时先停止 50/100 步采集与 B7 训练，并按第 13 节回退。

当前判定仍为 `NOT_ASSESSABLE`，直到正在执行的 5,000-snapshot 双任务三 seed
正式矩阵完成。32-snapshot 工程矩阵不得替代 Gate A；20,000-snapshot/50-step
Gate B 属于 M6。正式 M5 已在宿主机 CUDA 环境恢复执行；M6 未获批准、未开始。

### M6：B7 与 50/100 步（7--14 工程日 + GPU）

任务：

- 扩充 20,000 snapshots；
- 训练 1→2→4→10→20 model-step curriculum；
- 运行固定动作无观察评估；
- 完成历史 probe 和 reset-window 消融；
- 判定 Gate C/D 的 TwoRoom 部分。

Go：TwoRoom 满足 Gate B，长程没有更快爆炸，history 机制至少在一个诊断任务成立。

### M7：接触任务复验（10--20 工程日/任务 + GPU）

任务：

- 逐任务做完整 snapshot 审计；
- 实现 PushT-W adapter 和首个 Reacher/Ball-in-Cup adapter；
- 数据、B3/B4/B7 三 seed 训练和相同固定动作测试；
- 判定跨任务 Gate C。

Go：一个导航和一个接触任务满足 Gate C。否则论文主张限制到已通过的环境类别。

### M8：性能优化与后续 RL（Gate A--D 后）

先 profile PyTorch reference。只有当 recurrence 占训练 wall time >40% 或无法达到目标 batch/horizon 时才实现 Triton/CUDA kernel。优化 kernel 必须与 PyTorch oracle 前向/反向对齐。

RL 另立计划：冻结 B0/B3/B7，使用相同 goal-conditioned actor/critic、相同 imagined batch 和真实交互预算，在 Fixed-5/25/50/100 无观察执行中比较。RL 不回流修改第一阶段 test split 或方法选择。

---

## 12. 运行命令设计

实现后的标准工作流固定为：

```bash
# 0. 环境与现有测试
.venv/bin/python scripts/check_environment.py
.venv/bin/pytest -q

# 1. 采集同状态真实分支
.venv/bin/python scripts/collect_counterfactual_branches.py \
  --config configs/cc_rwkv/tworoom.yaml \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --output artifacts/cache/cc_rwkv/tworoom/raw

# 2. 用冻结 LeWM encoder 建 latent cache
.venv/bin/python scripts/cache_counterfactual_latents.py \
  --input artifacts/cache/cc_rwkv/tworoom/raw \
  --weights /data/wjy/lewm_model/tworooms/weights.pt \
  --output artifacts/cache/cc_rwkv/tworoom/latent

# 3. 训练单方法单 seed
.venv/bin/python scripts/train_cc_rwkv.py \
  --config configs/cc_rwkv/tworoom.yaml \
  --method b6 --seed 0 --device cuda:0

# 4. 固定动作、无中途观察评估
.venv/bin/python scripts/evaluate_cc_rwkv.py \
  --config configs/cc_rwkv/tworoom.yaml \
  --method b6 --seed 0 --split test --device cuda:0

# 5. 完整矩阵和聚合
.venv/bin/python scripts/train_cc_rwkv.py \
  --config configs/cc_rwkv/paper_matrix.yaml --matrix
.venv/bin/python scripts/aggregate_cc_rwkv.py \
  --root artifacts/results/cc_rwkv --bootstrap-samples 10000
```

每个长任务支持 `--resume`，以 condition 目录中的 `progress.json` 判断已完成 sample IDs。矩阵调度显式传 `--device`，不让四个进程同时抢 `cuda:0`。

---

## 13. 风险、诊断和回退路径

### 13.1 B6 一步 effect 都学不会

依次检查：snapshot 是否完整→action normalization/reference 是否正确→pulse suffix 是否相同→effect mask 是否过强→delta head 是否获得梯度→gate 是否饱和。只用 128 个样本做 overfit；overfit 失败时不扩大数据。

### 13.2 no-op 分支不是真正“不干预”

先在物理 state 上测零动作短期变化。若环境有自然漂移，no-op 仍可作为“不施加控制”参考；若零动作本身触发制动/复位，则改用预注册 hold action 或行为条件均值，并把语义写入 manifest。不得在看到 test 结果后更改 reference。

### 13.3 CC 只因 rank-two 容量获益

查看 B4。如果 B4 与 B7 持平，核心 counterfactual-centered claim 不成立；保留 rank-two 工程结果，但删除反事实递推贡献表述。

### 13.4 B3 与 B7 持平

说明输出级监督已足够，内部反事实递推没有独立收益。停止 RL 和 kernel 扩展，报告负结果或把论文缩为机制分析。

### 13.5 长程训练发散

先确认 state fp32、decay clamp 和梯度有限，再降低 LR 到 `2e-5`，最后启用 truncated BPTT（窗口 10，但跨窗口传递 detached state）。truncated BPTT 必须作为单独配置报告，不能与 full BPTT 混在主结果。

### 13.6 真实 effect 随 horizon 自然趋零

按物理 state effect 大小分层报告。若任务本身会快速吸收动作扰动，不能用 CER 下降证明记忆失败；转向 Action-Delay/Hidden-Velocity/不可逆 Door-Switch 等 effect 应持续的可辨识任务。

### 13.7 snapshot 不完整

该任务退出 paired 主实验。可保留普通 rollout，但不计算“真实反事实 effect”或宣称 causal supervision。不要用两次独立 reset 的相似状态冒充相同 snapshot。

### 13.8 计算资源不足

当前 latent-only MVP 单卡即可。先缩 batch 并做梯度累积，不缩模型 state/horizon。四卡仅用于不同 method/seed 并行，不做分布式同步。运行前要求目标 GPU 至少 12 GB 空闲；Stage D 图像微调建议至少 24 GB 空闲。

---

## 14. 资源与时间估算

以下是排期预算，不是性能承诺；M1/M3 完成后用 1,000-step profile 更新。

| 工作 | 人工时间 | 单次 GPU 预算 | 备注 |
|---|---:|---:|---|
| M0--M4 工程与测试 | 22--33 工程日 | 20--50 GPU-h | 含反复 smoke |
| Gate A/B，4 方法×3 seeds | 7--10 工程日 | 60--180 GPU-h | 可跨卡并行 |
| B7 50/100 步，3 seeds | 7--14 工程日 | 60--240 GPU-h | 先 profile |
| 每个额外接触任务 | 10--20 工程日 | 80--240 GPU-h | snapshot adapter 是主要风险 |
| 全部首阶段 | 8--12 周 | 220--710 GPU-h | 不含 OGBench 和 RL |

磁盘预留建议：TwoRoom latent/cache/checkpoint/results 50 GB，raw audit 10 GB；每个额外视觉任务先预留 100 GB，再按 smoke 数据实测修正。

---

## 15. 最终验收清单

### 机制

- [ ] actual=reference 时 action intervention 数值归零；
- [ ] 固定 `M_next` 替换 raw action 不改变 prediction；
- [ ] 不同 branch state 无共享写入；
- [ ] matrix update difference 能恢复动作信息；
- [ ] gate 非全零、非全一；
- [ ] world/no-op 预测不劣于 B2。

### 数据

- [ ] snapshot 包含全部动力学和 RNG 状态；
- [ ] same snapshot/same action 确定性 replay 100% 一致；
- [ ] split 按 episode 隔离；
- [ ] pulse branch 只在首 action block 不同；
- [ ] action support filter 和 reference 语义已冻结；
- [ ] cache、encoder、normalizer 和 dataset hash 完整。

### 公平比较

- [ ] B0/B2/B3/B4/B6/B7 horizon 口径一致；
- [ ] RWKV 主配置参数量在 ±5%；
- [ ] 所有方法训练样本数和 optimizer budget 一致；
- [ ] 三个训练 seed 和完全相同 test action sequences；
- [ ] B3/B4 是强基线，不只和 Transformer/B2 比；
- [ ] 数据量匹配基线使用全部分支但不使用 pairing。

### 结果

- [ ] CEE/CED/CER 与 ordinary rollout error 同时报；
- [ ] primitive/model horizon 同时报；
- [ ] 固定动作、无中途观察是主结果；
- [ ] paired bootstrap CI 与失败样本完整；
- [ ] Gate A--D 自动报告；
- [ ] 未通过 Gate 时按预注册规则缩小主张；
- [ ] Gate A--D 通过前没有加入 RL 结果干扰结论。

---

## 16. 第一轮立即执行顺序

1. 先实现 M0，只补协议、配置和 B0 open-loop 锚点；
2. 实现 M1 的纯 PyTorch vanilla RWKV 和持久状态，不写 CUDA kernel；
3. 用 32 个 TwoRoom/Action-Delay snapshots 完成 M2 全部恢复测试；
4. 实现 M3，强制先通过 actual=reference 与 action-bypass 测试；
5. 用 128 个 paired samples overfit B6，确认一步 effect 可学；
6. 采 5,000 snapshots，运行 B2/B3/B4/B6 的三 seed MVP；
7. 只有 Gate A/B 通过，才采 20,000 snapshots 并训练 B7 到 50/100 步；
8. 只有 TwoRoom/Action-Delay 结果成立，才审计 PushT snapshot；
9. 只有导航与接触任务 Gate C/D 都成立，才进入 kernel 优化和 RL 计划。

第一轮最小成功输出不是任务成功率，而是以下配对结论：

> 在相同历史和 simulator snapshot 下，只改变第一个 action block 后，B6/B7 对真实未来 effect 的 20/50 步预测显著优于 B3/B4，同时 ordinary one-step 预测恶化不超过 3%，且该差异只能通过更新后的 RWKV 矩阵状态进入预测。
