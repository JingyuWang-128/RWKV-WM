# CC-RWKV-WM M4 验收报告

> 状态：**完成（统一训练/指标工程 Go，尚未进入 M5）**  
> 完成时间：2026-08-28  
> 范围：B2/B3/B4/B6 统一训练器、完整 loss、课程、early-stop/resume、CEE/CED/CER/AUC、matrix probe、公平性拒绝和自动 gate report；不包含 M5 三 seed Gate A/B 比较。

## 1. M4 结论

M4 已把 M1--M3 的 predictor、真实分支缓存和反事实递推接入同一训练/评估管线。
B2、B3、B4、B6 现在可以由一条命令运行；每个方法读取相同 snapshot、四条 branch
trajectory、history、horizon 和 split。公平性聚合器会主动拒绝数据 hash、encoder、
normalizer、split、轨迹数、optimizer budget、effective batch、curriculum、paired-loss 权限
或 predictor 参数量不一致的矩阵。

32-snapshot、每方法 200-step 的 CPU smoke 中，四方法 loss 均下降超过预注册 20% 门槛，
checkpoint/resume 和独立 evaluation CLI 通过，全仓 109 tests 通过。

M4 Go 只说明“实验系统已能公平运行并拒绝不公平结果”。本次只有一个 seed、32 个训练
snapshot，不能用于判断 B6 是否优于 B3/B4，也不能宣布 Gate A/B 通过。

## 2. 统一数据与 free-running

`BranchBatch` 统一承载：

```text
history_latents    [B,Th,D]
history_actions    [B,Th,A]
history_mask       [B,Th]
branch_actions     [B,K,T,A]
branch_latents     [B,K,T+1,D]
branch_mask        [B,K,T+1]
```

所有方法先消费相同真实 history，再把 matrix/shift state 克隆到 K 个 branch。从 branch
initial latent 开始完全 free-run；`branch_latents[:,:,1:]` 只作为 loss target，绝不重新输入
predictor。reference action 从 branch 0 广播到同一步的所有 B6 branch。

B2/B3/B4 把四分支当作等量普通轨迹，不允许读取 paired effect label；只有 B6 可以使用
factual-minus-reference 配对关系。调用错误 loss 权限会直接抛出异常。

## 3. Loss 实现

实现冻结总目标：

```text
L = λpred Lpred
  + λeffect Leffect
  + λdir Ldirection
  + λmag Lmagnitude
  + λworld Lworld
  + λgate Lgate
  + λsigreg Lsigreg
```

- `Lpred`：全部有效 branch/time latent 的 Smooth-L1；
- `Leffect`：`factual-pulse_noop` 与 `pulse_local-factual` 的预测/真实 effect Smooth-L1；
- `Ldirection`：`1-cos(Δpred,Δtrue)`；
- `Lmagnitude`：`abs(log((||Δpred||+eps)/(||Δtrue||+eps)))`；
- `Lworld`：reference 和 pulse-noop 普通预测 loss；
- `Lgate`：gate mean 稀疏项加 `<0.02`/`>0.98` 防饱和 hinge；
- `Lsigreg`：Stage A--C 默认为 0，并保留 Stage D 外部 representation regularizer 接口。

direction/magnitude 只在 train split 真实 effect norm 第 10 百分位以上计算。本次 32-sample
smoke 冻结阈值为 `2.3324124813079834`。完全正确 effect 的三项 effect loss 均为 0，近零
effect mask、cosine 和 CER 的零分母路径均有自动测试。

effect weight 支持预注册的 `0.5/1.0/2.0`。选择函数先拒绝 one-step error 相对 B2 恶化
超过 3% 的候选，再最小化 validation score；所有候选都违规时拒绝选择。

## 4. Curriculum、优化与恢复

实现 model-step curriculum：

```text
1 -> 2 -> 4 -> 10 -> 20
```

- 每级默认至少 5,000 optimizer steps；
- 连续 3 次 validation 相对改善不足 0.5% 才允许升级；
- 高于第一级时，25% batch 回放较短 horizon；
- validation early-stop 默认 patience 10；
- warmup + cosine scheduler；
- gradient clip 1.0；
- matrix state 继续 fp32；
- 连续 3 个非有限 step 会终止，不静默吞掉；
- B6 Stage B 默认前 1,000 steps 只训练 `action_*` 参数，然后解冻全部 predictor。

工程 smoke profile 单独使用 `lr=3e-3`、B6 freeze 1 step，以在 200-step 小预算内验证
overfit 和 freeze/unfreeze 路径。论文 main profile 仍固定 `lr=5e-5`、freeze 1,000 steps，
没有为了 smoke 修改正式超参数。

checkpoint `cc_rwkv_checkpoint_v1` 现在统一保存 method、model/loss/training config、模型、
optimizer、scheduler、curriculum、early-stop、global step、best metric、history、torch/CUDA RNG、
encoder provenance 和 data provenance。resume 会检查 schema、latent/action dim、action block、
dataset/split/normalizer/encoder hash；默认拒绝 mismatch。

## 5. 指标定义

M4 冻结如下指标：

- CEE：`RMS(Δpred-Δtrue)`，对 latent 维做归一；
- CED：有效非零 effect 上的 `1-cos(Δpred,Δtrue)`；
- CER：有效非零 effect 上的 `||Δpred||/(||Δtrue||+eps)`，理想值为 1；
- trajectory RMSE：所有 branch 的普通 latent rollout RMSE；
- normalized trajectory RMSE：除以 held-out target latent 在该 horizon 的整体标准差；
- AUC：horizon curve 的 trapezoid AUC 除以 horizon span，一步时返回该点。

normalized rollout 不再除以单样本真实位移，因为合法 reference/no-op branch 可以有严格零
位移；这种分母会产生无意义的百万级数值。当前定义对静止 branch 稳定，并保持不同
horizon 的可比尺度。

validation score 为：

```text
CEE_AUC + 0.5 * normalized_trajectory_AUC
```

输出包含每个 horizon 的曲线、有效 effect 数、one-step error、gate 分布和
`actual==reference` 审计。

## 6. Matrix probe

统一提取同一 history/initial latent 下：

```text
flatten(M_next(factual) - M_next(pulse_noop))
```

以 frozen train features 拟合带截距 ridge linear probe，在 validation/test 上报告 action-block
回归 R²；另实现 nearest-centroid classification accuracy，供离散动作/隐藏状态任务使用。
probe 严格 train-fit/evaluation-test，不在 validation/test 上拟合。

32-sample smoke 的 validation R² 为负，不足以说明 action recovery 成立。这是小样本高维
probe 的预期风险，也意味着本次结果不能宣布 Gate A；M5 必须使用正式 train/validation
样本和三 seed。

## 7. B3 DWM-RWKV paper-spec 基线

B3 明确标记：

```text
paper_spec_reimplementation
source = arXiv:2607.18715
```

实现遵循论文公开机制：

- vanilla RWKV predictor 推理路径不变；
- 训练期增加两层 MLP + BatchNorm world head；
- 同一 history/state 下，把当前 action 做 batch permutation；
- world head 两个 view 使用 symmetric InfoNCE，temperature `0.07`；
- action component 定义为 `prediction-world`；
- orthogonality 为 `abs(cos(world,action_component))`；
- 权重固定为 `λwc=0.3`、`λorth=0.5`；
- 推理时只保留 vanilla predictor，不调用 world head。

DWM 未提供与本仓库 RWKV 相适配的官方代码，因此不能标记为作者官方复现。本实现保持
B2 的 RWKV prediction head 不变，只把论文的训练期 world-head/InfoNCE/orthogonality 机制
迁移到 RWKV，以免通过改变 B3 推理架构污染强基线。

## 8. 公平性拒绝机制

每个 `RunRecord` 冻结：

```text
method, seed, encoder_sha256, normalizer_sha256,
dataset_sha256, split_sha256, branch_trajectories,
optimizer_steps, effective_batch_size, curriculum,
predictor_parameters, paired_loss, implementation_status
```

完整矩阵必须同时包含 B2/B3/B4/B6。审计器会拒绝：

- 任一 provenance 或 budget 不同；
- B2/B3/B4 意外使用 paired loss；
- B6 未使用 paired loss；
- B3 没有 `paper_spec_reimplementation` 标签；
- 同 seed predictor 参数量跨度超过 5%；
- 缺失方法或 seed 内方法不完整。

单元测试主动构造错误 dataset hash 和 20% 参数差，均确认抛出 `FairnessViolation`。

## 9. 四方法 200-step smoke

命令：

```bash
.venv/bin/python scripts/train_cc_rwkv.py \
  --max-steps 200 --batch-size 8 \
  --train-limit 32 --validation-limit 32 \
  --evaluation-interval 20 \
  --output artifacts/results/cc_rwkv/tworoom/m4_smoke_v2 \
  --device cpu
```

所有方法读取相同 32 train snapshots，即每方法 128 branch trajectories；optimizer steps
均为 200，effective batch 均为 8，curriculum 均为 `[1,2,4]`。

| 方法 | predictor 参数 | 初始总 loss | 最终总 loss | 下降 |
|---|---:|---:|---:|---:|
| B2 | 104,256 | 0.535861 | 0.339370 | 36.67% |
| B3 | 104,256 | 1.711952 | 0.577294 | 66.28% |
| B4 | 104,000 | 0.564479 | 0.360616 | 36.12% |
| B6 | 104,000 | 2.895916 | 1.007962 | 65.19% |

参数最大差约 0.246%，远小于 5%。B3 表中 predictor 参数不包含训练期 auxiliary world head，
因为公平规则比较推理 predictor；checkpoint 同时保存 world head 训练参数。

最终工程诊断：

| 方法 | CEE AUC | trajectory AUC | one-step RMSE | gate saturated |
|---|---:|---:|---:|---:|
| B2 | 0.819244 | 1.044327 | 0.996345 | — |
| B3 | 0.821325 | 1.139221 | 1.110634 | — |
| B4 | 0.802204 | 1.035463 | 0.999315 | 0.021790 |
| B6 | 0.816467 | 1.074716 | 1.028931 | 0 |

B6 `actual==reference` 最大 delta 为 0。上述数值只检查字段、finite、尺度和数据流，不能
进行方法排序；尤其 B6 one-step 相对 B2 的小样本差异不用于正式 3% 筛选。

第一次保留的 `m4_smoke/` 使用 main LR，在 200 步下 B2/B4/B6 未达到 20% overfit 门槛，
因此不作为 M4 验收结果；它保留为 smoke-budget 诊断，没有覆盖。

## 10. 输出与验证

每个方法均生成：

```text
best.pt
latest.pt
evaluation/metrics.jsonl
evaluation/summary.json
evaluation/resolved_config.yaml
evaluation/provenance.json
evaluation/failed_samples.jsonl
evaluation/gate_report.md
```

矩阵根目录生成 `fairness.json` 和 `summary.json`。独立
`scripts/evaluate_cc_rwkv.py` 已加载 B6 `latest.pt`、恢复 optimizer/scheduler/curriculum/RNG，
并再次生成完整 evaluation artifacts。

主要产物 hash：

- matrix summary SHA256：`88605109d7df53b2938147a75f455018ba6fb98ac57f7c5f035621ab4910e613`；
- fairness SHA256：`c04ea03b5e1293673918440a32a772857db755ecba003de2f81fe3b43648e745`；
- B6 latest checkpoint SHA256：`04138abd521f6cba680968b3be1f3e864693c380f7c58716a33062d14d75ac6a`；
- B6 evaluation summary SHA256：`9eaf51f6c68de80efe605cff8332cb8f2bb581c984ecf2f3e9de226c7deaa8e3`。

测试：

- M4 新增测试：14 项；
- M4/M1 checkpoint 与 training 定向测试：17 passed；
- 全仓：**109 passed**，5 个已有 PyTorch warning；
- M4 修改范围 Ruff：通过；
- 全仓 Ruff 仍报告 6 个 M4 之前已存在、位于其他模块的格式问题，未改动无关代码。

## 11. M4 Go/No-Go

M4 Go 条件全部满足：

- 七项 loss 与 effect mask：通过；
- curriculum、short-horizon replay、early-stop：通过；
- checkpoint/resume/provenance guard：通过；
- CEE/CED/CER、trajectory AUC：通过；
- ridge/classification probe：通过；
- B3 paper-spec 输出级强基线：通过；
- B2/B3/B4/B6 单命令矩阵：通过；
- 公平性不一致拒绝：通过；
- 自动 gate report：通过；
- 200-step loss 下降至少 20%：四方法均通过；
- 全仓回归测试：通过。

尚未完成：

- M5 TwoRoom-W/Action-Delay seeds 0/1/2 正式训练；
- effect weight validation search 的正式三个候选运行；
- paired bootstrap 10,000 次与 95% CI；
- Gate A/B 方法比较和失败样本审阅；
- B7、50/100-step cache；
- RL 控制规划。

M4 到此停止。M5 必须获得用户批准后才能开始。
