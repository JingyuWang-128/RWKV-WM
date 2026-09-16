# F3/F4 mask 修复与 seed 0 稳定性准入

## 授权范围与当前状态

- 仅 TwoRoom-W、Action-Delay，seed 0；先正式 F3，再正式 F4。
- 配置 effect_weight 始终为 0.5。F4 使用 paired_loss=false，实际辅助损失系数为 0，
  不读取 effect 标签、不计算三项辅助损失；参考分支与 centered 矩阵结构保留。
- 正式训练必须等待代码审查、GPU 稳定性预试验和共同验证指标审查全部通过。
- 不自动把 smoke/preflight 完成当作正式训练的授权凭证。

2026-09-15：已停止旧 F3 队列 PID 1445570 和训练 PID 1958629；训练器在
Action-Delay / weight=2 / seed=2 的第 34869 步保存 interrupted checkpoint。
其他用户的 GPU 2/3 任务未操作；本次预试验只用空闲 GPU 0/1。

## 已修复的正确性问题

原数据存储 H=20 的三角 mask。训练 H=5/10 时简单截取它，会把没有生成预测的补零
位置也纳入损失。修复后使用 data_mask AND (position + offset < current_horizon)。
每样本 H=1/5/10/20 有效项必须分别为 1/15/55/210。

新增测试验证：短 horizon 的 padding 标签不影响任何损失或参数梯度；F4 即使配置
weight=0.5 也完全跳过辅助目标。checkpoint 记录 rollout_mask_policy，拒绝把旧 mask
checkpoint 当作新协议的精确断点恢复。旧冻结 manifest 不修改、不覆盖历史结果。

## 第一轮预试验候选（不是正式实验）

- main 模型、随机初始化、seed 0、effect_weight=0.5，保留原三项损失公式。
- H=1/5/10/20，各 600 optimizer steps，总计 2400 steps。
- 学习率 1e-4；取消 decay 参数的 2 倍学习率；前 100 步 warmup。
- 本轮不使用学习率衰减（min_lr_ratio=1），避免靠训练末尾极低学习率通过门槛。
- microbatch=8，梯度累积 4 次，有效 batch=32；每次 microbatch 损失除以 4。
- 训练使用全量 train split，预试验固定取 validation 的前 256 个样本（不访问 test）。
- 每 200 步验证并保存 latest.pt；每 600 步保存编号 checkpoint。
- 固定裁剪阈值 1.0，绝不通过扩大阈值、缩小总损失或删除数据降低裁剪率。
- 保留真实历史初始化、factual/reference 完整 BPTT，不 detach，不 teacher forcing。

## 工程准入门槛（不是数学保证）

每个 H 的前 200 次更新是明确记录的适应期，之后按每 200 次更新检查：

1. loss、所有组件、梯度以及保存的模型/optimizer 状态全部有限。
2. 固定阈值 1 下，裁剪率不超过 20%。
3. 窗口内不得出现 clip scale < 0.1 的严重缩放事件。
4. 记录窗口 grad median/max、scale median/min、裁剪率；不只记录裁剪后梯度。
5. 同时审查适应期日志与固定 H20 factual/reference 验证误差，排除始终没学会、
   effect 输出退化为零等“看似稳定”。完整 H20 验证集复核是正式启动的前提。

任一数值/窗口门槛失败：写 failure.json 和 stability_windows.jsonl，进程非零退出，
不启动正式 F3/F4。20% 是本轮预先选定的工程阈值，并非通用论文标准；本轮不事后
调高它来获得 PASS。预试验通过也不能保证未来从不出现梯度尖峰，正式阶段仍需守卫。

如果预试验失败，先记录事实并定位各项梯度，不擅自把 F3 的三项损失删去、改变权重
或换模型来凑 PASS。后续配置调整应对正式 F3/F4一致，并重新验证。

## 正式对照（尚未启动）

在新目录重新初始化 F3 两个 seed 0 任务，F3 全部完成后重新审查并启动 F4 两个
seed 0 任务。两组的数据、初始化、优化设置、课程与预算匹配，只切换 paired_loss。
不直接将新 F4 与旧 mask F3 的差异归因于辅助损失。必须保留原始空间共同预测指标、
effect 评价指标、梯度稳定性记录；单 seed 只用于探索，不作多 seed 显著性结论。

## 代码入口

- src/cape_wm/cc_rwkv/per_step_training.py：动态 horizon mask。
- src/cape_wm/cc_rwkv/stability.py：学习率调度与固定阈值质量门槛。
- scripts/train_per_step_pairs.py：累积、梯度日志、fail-closed gate、checkpoint 元数据。
- scripts/run_formal_stage_queue.py：F4 禁用开关与 mask 版本检查。
- tests/test_per_step_training.py、tests/test_per_step_stability.py：回归测试。

静态检查通过；最终全量 pytest 130 passed（5 个既有 Transformer warnings）。
包含旧 mask checkpoint 拒绝精确续训的回归测试。

## 已启动的 GPU 预试验

- TwoRoom-W：物理 GPU 0，PID 2132700。
- Action-Delay：物理 GPU 1，PID 2132702。
- 两进程使用 start_new_session=True，已核实 PPID=1、SID=PID；不依赖当前会话。
- 仅可见各自一张物理 GPU，因此训练命令内部的 cuda:0 是进程内编号。
- 输出根目录：
  `artifacts/runs/per_step_pairs/formal_corrected_v1/mask_fixed_seed0/preflight_lr1e4_acc4/`
- 每任务目录的 launch.json 保存实际命令、物理 GPU、PID 和代码 SHA256；console.log
  是运行日志，stability_windows.jsonl 是质量门槛记录；未通过时保存 failure.json。
- 2026-09-16 更新：两个预试验均已因裁剪率门槛未通过而退出；正式 F3/F4 **未启动**。
  TwoRoom-W 在 step400/H1 裁剪率 65.5%，Action-Delay 在 step1000/H5 裁剪率 22.5%。
  无 NaN/Inf 失败，不修改本轮 FAIL 判定。离线梯度诊断见
  `docs/mask_fixed_gradient_diagnosis_20260916.md`；没有重新启动训练。

启动入口（已执行，请勿重复启动）：

```bash
.venv/bin/python scripts/launch_mask_fixed_preflight.py --task tworoom_w --gpu 0
.venv/bin/python scripts/launch_mask_fixed_preflight.py --task action_delay --gpu 1
```

这只是 preflight launcher，不包含任何自动启动正式训练的逻辑。正式协议需要等预试验
完成、完整验证集审查、F4 GPU preflight 和新的代码/配置 manifest 冻结后才能运行。
