# CC-RWKV-WM M0 验收报告

> 状态：**完成**  
> 完成时间：2026-08-28  
> 范围：冻结实验协议和官方 B0 open-loop baseline；未开始 M1/RWKV 实现。

## 1. M0 结论

M0 已建立可复用的长期 imagination 锚点：300 个冻结 TwoRoom test episodes 从 episode 起点获得一次真实观察，随后执行完全固定的 factual action sequence，官方 Transformer-LeWM 在中途没有真实观察、没有 MPC/CEM/RL 的条件下 free-run 20 个模型 transition，即 100 个 primitive actions。

官方 B0 的 latent error 随 rollout horizon 明显增长，可作为 B2--B7 的同样本 paired comparison 基线。

| primitive actions | model transitions | latent MSE mean | latent L2 mean | cosine distance mean |
|---:|---:|---:|---:|---:|
| 5 | 1 | 0.067626 | 3.552538 | 0.029057 |
| 10 | 2 | 0.086095 | 3.991628 | 0.036812 |
| 20 | 4 | 0.129688 | 4.751797 | 0.059431 |
| 50 | 10 | 0.286264 | 6.647761 | 0.153787 |
| 100 | 20 | 0.576955 | 8.851516 | 0.313944 |

- trajectory latent L2 AUC：`6.496814`；
- trajectory latent MSE AUC：`0.308491`；
- predictor 参数量：`10,791,360`；
- official LeWM 总参数量：`18,034,478`；
- 正式评估预测 transition 数：`6,000`；
- 正式评估样本数：`300`。

这些结果只描述 B0 的 latent rollout fidelity，不是控制成功率，也不是 CC-RWKV 的有效性结论。

## 2. 冻结协议

### 2.1 Horizon

TwoRoom 官方 action block 固定为 5：

```text
primitive actions: 5, 10, 20, 50, 100
model transitions: 1,  2,  4, 10,  20
observation stride: 5
```

`HorizonSpec` 会拒绝不能整除 action block、未排序、重复或 observation stride 不一致的配置。

### 2.2 样本和动作

- frozen test episodes：1,084；
- 可提供完整 100-action suffix 的 episodes：860；
- selection seed：3,072；
- 无放回选择：300 episodes；
- 每个 episode 的 start step：0；
- history frames：1；
- 每条 action sequence：20 blocks × 5 primitive actions × 2 action dims；
- 目标 latent：起点及随后每 5 primitive steps，共 21 个 latent。

官方 TwoRoom episodes 最长只有 101 帧，因此无法同时保留三帧历史和 100 步未来。M0 使用单起点观察来隔离纯 rollout 误差；full-history 与 history-window 的比较留在 Gate D，不能用 M0 代替。

### 2.3 Normalizer

action StandardScaler 使用整个官方数据集中所有 finite primitive actions 拟合。这与 released LeWM training script 在拆分前拟合 normalizer、以及官方 evaluator 的行为一致。

M0 曾在内部审计中发现“排除 test episodes 拟合”的版本与官方 preprocessing 不一致；该中间结果已经废弃，正式 manifest 和 B0 已重新生成。后续方法必须读取正式 manifest 中的 mean/scale，不能重新拟合。

## 3. 关键 provenance

| 对象 | SHA256 |
|---|---|
| TwoRoom HDF5 | `129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623` |
| official LeWM weights | `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd` |
| official model config | `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09` |
| frozen test episode source | `694bd657b4dbfb3a1a609738abdd50d907413c5ce3d4cdc13ed3271d737f0a6a` |
| open-loop manifest NPZ | `f8f54b2165a734af7661e8766d59f0f2aac4161e79f22a1a55363130f87aaef5` |
| target latent cache | `52d5b0717a214c2d209e7f35888ab643b9aa1ade6fbc67b11f5c1d126a3d5669` |

逐样本 paired error 内容 hash：

- latent MSE：`588bf97e6a3117650f32c6ba615ffdca7334c0654237b9b476c94f5dd3e331a4`；
- latent L2：`d55f3b7e05fce5e60c7ec1da9b77154ec62028fae63a6edb72c3efb271a1d19f`；
- cosine distance：`307d353fa0359078062e955984804439302893472fe18f2e42364c968f7b65d7`。

复用 target latent cache 重新加载官方模型并评估后，上述三个内容 hash 完全一致。

## 4. 实现产物

代码和配置：

- `configs/cc_rwkv/base.yaml`；
- `configs/cc_rwkv/tworoom.yaml`；
- `src/cape_wm/cc_rwkv/protocol.py`；
- `src/cape_wm/cc_rwkv/b0.py`；
- `scripts/prepare_cc_rwkv_m0.py`；
- `scripts/evaluate_b0_open_loop.py`；
- `tests/test_cc_rwkv_protocol.py`。

运行产物：

- `artifacts/results/cc_rwkv/tworoom/m0/protocol/manifest.json`；
- `artifacts/results/cc_rwkv/tworoom/m0/protocol/provenance.json`；
- `artifacts/results/cc_rwkv/tworoom/m0/protocol/open_loop_manifest.npz`；
- `artifacts/results/cc_rwkv/tworoom/m0/b0_open_loop/target_latents.npy`；
- `artifacts/results/cc_rwkv/tworoom/m0/b0_open_loop/paired_errors.npz`；
- `artifacts/results/cc_rwkv/tworoom/m0/b0_open_loop/summary.json`。

## 5. 防泄漏和复现约束

- B0 rollout 函数只接受 observed context latent 和 future actions，不接受 future target latent；
- rollout 的每个后续 latent 都来自上一步 prediction；
- controller 字段固定为 `none`；
- `no_intermediate_observations=true`；
- manifest 同时保存 primitive/model horizon；
- target cache 校验 row fingerprint 和 encoder weight hash；
- evaluator 校验 frozen manifest 文件 hash；
- paired errors 保存 sample ID，供 B2--B7 逐样本比较；
- `--limit` 只用于独立 smoke output，不能写入正式 B0 目录作为论文结果。

## 6. M0 验收

- [x] 数据、权重、模型配置和 test episode source 已 hash；
- [x] 300 个 test episodes 和固定 factual actions 已冻结；
- [x] primitive/model horizon 口径已编码并测试；
- [x] 官方 normalizer 已对齐；
- [x] B0 5/10/20/50/100 open-loop 结果已生成；
- [x] 逐样本 paired errors 已保存；
- [x] 缓存复跑内容 hash 一致；
- [x] 无 MPC/CEM/RL 和中途观察；
- [x] predictor 参数量和 transition count 已记录；
- [x] 全量测试通过：63 passed；M0 修改范围 Ruff 通过。

全仓库 `ruff check src tests scripts` 仍报告 6 个 M0 之前已经存在的格式问题，位于
`merge_candidate_records.py`、三个旧 run wrapper、`calibration.py` 和
`comparison_eval.py`。它们与本里程碑无关，本轮没有为了制造“全绿”而改动这些文件。

M0 到此停止。下一里程碑 M1 是实现 `RWKVMatrixState`、vanilla RWKV cell、多层 stateful predictor、branch clone 和无 history truncation rollout；必须得到用户批准后才能开始。
