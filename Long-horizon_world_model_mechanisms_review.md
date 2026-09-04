# 世界模型解决长程任务：机制综述、LeWM 方案审计与新方法重构

> 检索截止：2026-08-27  
> 约束：LeWorldModel（LeWM）为主要 backbone；simulation / offline 为主；不研究视频生成；不以真实机器人部署为贡献。  
> 证据规则：正式会议/期刊与预印本严格分层；优先链接会议、期刊或 PMLR 官方页面。  
> 相关实验定义与现有结果的更细审计见 [long_horizon_evaluation.md](docs/long_horizon_evaluation.md)；完整创新方法提案见 [long_horizon_method_proposals.md](docs/long_horizon_method_proposals.md)。

## 0. 先给结论

1. **长程失败不是单一的累积预测误差。** 至少要区分：状态/记忆不足、递归预测漂移、规划目标几何错误、候选动作覆盖不足、离线分布外利用、时间抽象失败、远期价值传播、执行中断与恢复、风险校准失效、计算量失控。每一类已有不同的正式发表解法。

2. **只做“多尺度预测 + conformal calibration + event trigger”已经不足以构成强创新。** 多尺度/skill world model、局部不确定性决定 rollout 长度、conformal planning、反馈式 conformal、option termination、可靠子目标执行均已有很近的顶会工作。创新必须来自一个此前没有被清楚建模的接口，而不是模块拼接。

3. **当前 CRAFT 结果有效，但证明范围很窄。** 它证明的是：在 TwoRoom 中，directed temporal/reachability 指标配合 5/10 步短窗闭环，在远 goal offset 上优于 Flat+TRM；它尚未证明可变长时间抽象、长 open-loop prediction、事件触发恢复或 intrinsically long-horizon planning。

4. **当前任务不足以支撑方法的主要创新主张。** offset=100 不是实际执行 100 步；多数成功轨迹约 27 步且预算为 50；94.8% 的新规划选择 5 步 commitment。TwoRoom 应降级为机制诊断，不应作为主结果。

5. **最值得重构的创新核心是“全局几何与 controller-specific actionability 分解”。**

   - 结构可达性：环境/数据支持下，状态到目标是否存在路径、需要多长时间；
   - 控制器可执行性：给定当前低层控制器，它能否在时长 $d$ 内可靠到达；
   - 停止时间：不把 duration 当独立分类标签，而估计 controller-conditioned first-passage / hazard；
   - controller transfer：全局几何冻结，更换 controller 或 search budget 时只适配小型 actionability head；
   - 选择后校准：是需要处理的统计问题，但通用 selection-conditional conformal 已有正式期刊先例，不能作为唯一原创点；
   - RL proposal：用 offline goal-conditioned policy 或 action chunks 扩大候选覆盖，LeWM 只做短程模拟和结构/执行重排。

这一路线仍需通过严格实验才能称为新方法；目前最稳妥的定位是“待验证的创新假设”，不是已证实贡献。

---

## 1. 范围、术语与检索标准

### 1.1 本文所称“长程”

必须区分四个量：

| 量 | 定义 | 常见混淆 |
|---|---|---|
| 任务内在长度 $H^\*$ | 在统一控制频率和成功阈值下，可靠策略实际需要的最少环境步数 | 数据中的 goal offset 不等于 $H^\*$ |
| 模型想象长度 $H_m$ | 一次 world-model rollout 覆盖的 primitive steps | 短 $H_m$ 也可借助 value/hierarchy 解长任务 |
| 执行承诺长度 $H_c$ | 选定计划后，不重规划而连续执行的步数 | 小 $H_c$ 的高频 MPC 不是时间抽象 |
| 总交互预算 $B$ | 一个 episode 允许的环境步数 | offset=100、$B=50$ 不代表执行 100 步 |

本文只有在 $H^\*$ 随任务级别系统增长，并且固定反馈频率、计算量和数据覆盖后仍有收益，才把结果视为长程证据。

### 1.2 “offline”也要拆开

- **backbone offline**：LeWM 只用固定 observation-action 数据训练；
- **planner offline**：proposal、critic、risk head 和 calibration 都不用新环境交互；
- **simulator calibration**：允许在仿真中执行候选收集 residual/success；
- **online adaptation**：部署过程中继续更新模型。

当前 CAPE/CRAFT 若使用 simulator closed-loop attempts 标定风险，应写成“offline backbone + simulator calibration”，不能笼统写成 fully offline。

### 1.3 文献证据层级

- **A 级**：ICLR、ICML、NeurIPS、CoRL、L4DC 等正式 proceedings，或 Nature 等正式期刊；
- **B 级**：已公开但截至检索日只有 arXiv/OpenReview preprint；
- **C 级**：工作坊、未核验项目页或二手描述，只用于线索，不支撑优先权。

下文正式发表表只列 A 级；LeWM 及其 2026 近邻单列为 B 级，避免把“最新”误写成“已被顶会验证”。

---

## 2. 全瓶颈地图

| 编号 | 瓶颈 | 机制性诊断 | 已有主要解法 | 对 LeWM 的优先级 |
|---|---|---|---|---|
| B1 | 当前 latent 不满足 Markov 性 | 相同单帧对应不同隐状态/未来 | RSSM、SSM memory、belief state、history encoder | 部分可观测任务高 |
| B2 | one-step 递归 rollout 漂移 | open-loop error 随步数快速增大 | overshooting、多步/多尺度/prefix/chunk prediction、collocation | 中高 |
| B3 | latent objective 不可规划 | 可解码任务状态被 Euclidean cost 低权重或错序 | functional distance、reachability、quasimetric、task-aligned geometry | **最高** |
| B4 | 搜索空间随 horizon 爆炸 | oracle candidate 好但采样器找不到 | hierarchy、skill、action chunks、RL/flow proposal、graph search | **最高** |
| B5 | 远期收益或目标信号传播慢 | 短段表现正常，长 goal 价值塌缩 | terminal value、successor/contrastive value、transitive/min-plus composition | 高 |
| B6 | temporal abstraction 退化 | 学到的 duration 几乎总为最短 | learned options、adaptive abstraction、hitting-time/hazard、duration regularization | **最高** |
| B7 | 子目标存在但低层不可执行 | 高层边看似可达，执行频繁失败 | adjacency constraints、controller-conditioned reachability、frontier replay | **最高** |
| B8 | offline model exploitation | planner 选择数据外动作，模型却自信 | pessimism、ensemble uncertainty、behavior/action-chunk constraints | 高 |
| B9 | 不确定性未校准 | risk score 排序差或 nominal coverage 失真 | calibration、conformal sets/tubes、feedback conformal | 高 |
| B10 | planner 选择破坏校准 | 对许多候选优化后只执行极端候选 | selection-conditional conformal、episode-level evaluation | 应用空隙；通用统计问题已有解法 |
| B11 | 固定重规划频率不合适 | 短尺度耗算力，长尺度失控 | adaptive rollout、learned stopping、event trigger、fallback | 高 |
| B12 | 受扰后不可恢复 | nominal success 高，轻微偏差后连续失败 | closed-loop replanning、termination correction、recovery policy | 中高 |
| B13 | stochastic / multimodal future | 单点 latent 平均掉多个未来分支 | stochastic RSSM、ensemble、distributional/belief prediction | 条件性高 |
| B14 | 计算预算失控 | 成功率来自更多 model calls/replans | value bootstrap、amortized policy、sparse imagination、learned optimizer | 高 |
| B15 | 数据覆盖不足 | 未出现被误当不可达；跨轨迹拼接虚假 | coverage-aware negatives、support constraints、graph/TD stitching | **最高** |
| B16 | 评价伪长程 | offset 大但实际路径短；反馈频率不匹配 | $H^\*$ 分层、fixed-commitment、equal-feedback/equal-compute 协议 | **必须先修复** |

重要关系是：

$$
\text{task success}
\neq
\text{prediction accuracy}
\neq
\text{candidate coverage}
\neq
\text{candidate ranking}
\neq
\text{execution reliability}.
$$

因此，单一 success rate 无法判断方法解决了哪一个瓶颈。

---

## 3. 正式发表的相关工作：不只 MPC，也包括 RL

### 3.1 World model、短 rollout 与长期价值

| 工作 | 正式出处 | 解决机制 | 对本项目的含义 |
|---|---|---|---|
| [PlaNet](https://proceedings.mlr.press/v97/hafner19a.html) | ICML 2019 | RSSM + latent CEM；随机状态与确定性记忆 | 证明 latent MPC 可行，但固定短 horizon 和高计算量未解决 |
| [Calibrated Model-Based Deep RL](https://proceedings.mlr.press/v97/malik19a.html) | ICML 2019 | 明确校准 probabilistic dynamics | “模型有 uncertainty head”不等于已校准 |
| [LatCo](https://proceedings.mlr.press/v139/rybkin21b.html) | ICML 2021 | latent-state/action collocation，联合优化中间状态并约束动力学可行性 | 长程规划不必只靠递归 action rollout |
| [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html) | ICML 2022 | 短程模型 rollout + terminal value | 长期信息可以由 value 压缩，不必让 LeWM rollout 到终点 |
| [TD-MPC2](https://openreview.net/forum?id=Oxh5CstDJU) | ICLR 2024 | 可扩展 latent model、policy prior、value 与局部 planning | 强 hybrid baseline；比较时不能只放 flat CEM |
| [DreamerV3 / Mastering diverse control tasks through world models](https://doi.org/10.1038/s41586-025-08744-2) | Nature 2025 | imagined actor-critic，使用 value/policy 摊销长期决策 | 正式期刊证据；RL 是 world-model 长程解法之一 |
| [PWM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/240ea1741b205ea295721d55184ac43b-Abstract-Conference.html) | ICLR 2025 | 离线预训练 world model，用一阶梯度提取 policy | 可替代每步 CEM 的 policy-extraction baseline |
| [M3PC](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5d8201b37e165a131f5c6c6ad5230f58-Abstract-Conference.html) | ICLR 2025 | 在 model、policy、planning 之间自适应组合 | 说明“只比较 MPC”不足以覆盖当前方法空间 |

**结论。** 直接延长 LeWM rollout 只是一个选项。terminal value、imagined actor-critic、policy prior 和 policy extraction 都能让短模型承担长任务。

### 3.2 记忆、长期表示与多时间尺度 dynamics

| 工作 | 正式出处 | 解决机制 | 尚未解决 |
|---|---|---|---|
| [S4WM](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html) | NeurIPS 2023 | 比较 RNN、Transformer、S4 world-model backbone | memory 强不代表 planner geometry 正确 |
| [Mastering Memory Tasks with World Models (R2I)](https://openreview.net/forum?id=1vDArHJ68h) | ICLR 2024 | structured state-space model 延长 memory/credit | 不直接解决候选搜索 |
| [HRSSM](https://proceedings.mlr.press/v235/sun24n.html) | ICML 2024 | task-relevant robust latent，抑制视觉 distractor | 部分目标依赖 reward/bisimulation |
| [Multi Time Scale World Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/54d8aab579b5a9ed3395764c7341ebec-Abstract-Conference.html) | NeurIPS 2023 | 多时间尺度动态与不确定性 | 多尺度本身不是新颖点 |
| [THICK](https://openreview.net/forum?id=TjCDNssXKU) | ICLR 2024 | 从离散 latent dynamics 学 adaptive temporal abstractions | 与拟议 duration selection 高度相邻 |
| [Sparse Imagination](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a750d52284ff70c6d6bab8072c392d74-Abstract-Conference.html) | ICLR 2026 | 只在有价值的位置 imagination，压缩计算 | equal-compute 下应纳入比较 |

**结论。** 若任务是部分可观测，单帧 LeWM latent 可能不是充分状态；若任务完全可观测，则再加 memory 可能只增加复杂度。任务选择应先决定是否要主张 belief/memory。

### 3.3 时间抽象、skill 与 hierarchy

| 工作 | 正式出处 | 机制 | 与拟议方法的重叠 |
|---|---|---|---|
| [Director](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html) | NeurIPS 2022 | manager 在 latent space 选 goal，worker 执行 | latent subgoal + worker 已是成熟模板 |
| [SkiMo](https://proceedings.mlr.press/v205/shi23a.html) | CoRL 2022 | offline skill repertoire + skill dynamics + skill-space planning | “macro model + CEM”已有直接近邻 |
| [Learning Temporally Abstract World Models without Online Experimentation](https://proceedings.mlr.press/v202/freed23a.html) | ICML 2023 | 纯 offline 学 skill 与 skill-conditioned world model，零样本 skill planning | 与项目约束高度吻合，必须作为核心 baseline |
| [Probabilistic Subgoal Representations](https://proceedings.mlr.press/v235/wang24bx.html) | ICML 2024 | 用概率分布而非单点表示 subgoal | 对多模态/随机子目标更合适 |
| [Adjacency-Constrained Subgoals](https://proceedings.neurips.cc/paper/2020/hash/f5f3b8d720f34ebebceb7765e447268b-Abstract.html) | NeurIPS 2020 | 限制高层只提低层可达子目标 | 直接覆盖“可执行 subgoal”思想 |
| [Horizon Reduction Makes RL Scalable](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0c66099070429b9b49ad8bb19c740c22-Abstract-Conference.html) | NeurIPS 2025 | 通过 horizon reduction 缓解长期优化 | 长程贡献应与 reduced effective horizon 比较 |
| [Scalable Offline Model-Based RL with Action Chunks](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1056d17e239b7f82fbfe47170cced506-Abstract-Conference.html) | ICLR 2026 | future-state chunk model + behavior chunk rejection，减少 compounding/OOD exploitation | 与“duration-conditioned macro transition”非常接近 |

**结论。** 固定集合 $\{5,10,20,40\}$、macro transition 和 subgoal hierarchy 都不是独立创新。可争取的新意必须在“结构可达性与控制器可执行性的分离”或“planner 选择后的 stopping-time calibration”上。

### 3.4 Goal-conditioned RL、可达性、图与 quasimetric

| 工作 | 正式出处 | 主要机制 | 对 LeWM 的作用 |
|---|---|---|---|
| [SoRB](https://proceedings.neurips.cc/paper/2019/hash/5c48ff18e0a47baaf81d8b8ea51eec92-Abstract.html) | NeurIPS 2019 | replay-buffer graph + goal-conditioned value + shortest path | 处理局部 value 的“wormhole” |
| [LEAP](https://proceedings.neurips.cc/paper/2019/hash/c8cc6e90ccbff44c9cee23611711cdc4-Abstract.html) | NeurIPS 2019 | goal-conditioned policy + latent subgoal planning | 长路径分解，不依赖长 LeWM rollout |
| [MBOLD](https://openreview.net/forum?id=UcoXdfrORC) | ICLR 2021 | self-supervised functional distance | raw visual/latent proximity 的经典替代 |
| [Contrastive Learning as Goal-Conditioned RL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html) | NeurIPS 2022 | 用 contrastive objective 学 long-horizon goal value | 可作为 reward-free critic 路线 |
| [QRL](https://proceedings.mlr.press/v202/wang23al.html) | ICML 2023 | quasimetric goal-reaching value，显式方向性/三角结构 | “directed temporal metric”已有坚实前驱 |
| [HIQL](https://proceedings.neurips.cc/paper_files/paper/2023/hash/6d7c4a0727e089ed6cdd3151cbe8d8ba-Abstract-Conference.html) | NeurIPS 2023 | offline hierarchical implicit Q-learning | 强 offline RL subgoal baseline |
| [Goal-conditioned Offline Planning from Curious Exploration](https://openreview.net/forum?id=QlbZabgMdK) | NeurIPS 2023 | 图结构修正 value、拼接离线经验 | coverage/stitching 的直接近邻 |
| [OGBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd92623ac899357312aaa8915853699-Abstract-Conference.html) | ICLR 2025 | 8 类环境、85 数据集，专测 stitching、长程、随机性、pixels | 最适合作为本项目主 benchmark |
| [Flow Q-Learning](https://proceedings.mlr.press/v267/park25f.html) | ICML 2025 | flow policy + offline Q-learning | 可作多模态 action proposal |
| [CGCIVL](https://proceedings.mlr.press/v267/ke25a.html) | ICML 2025 | 对 unconnected state-goal pairs 做 conservative value penalty，并结合 quasimetric | “未连接不应被乐观拼接”的直接近邻 |
| [Offline GCRL with Quasimetric Representations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c5956164472d6d8123d574aa75cd063-Abstract-Conference.html) | NeurIPS 2025 | successor/contrastive information 与 triangle inequality 结合 | 结构可达性的强正式 baseline |
| [Option-aware Temporally Abstracted Value](https://proceedings.neurips.cc/paper_files/paper/2025/hash/90080022263cddafddd4a0726f1fb186-Abstract-Conference.html) | NeurIPS 2025 | option-aware temporal abstraction for offline GCRL | “option + long value”已有正式近邻 |
| [Transitive RL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/066ac8a48e27c78aadcf934b580a0383-Abstract-Conference.html) | ICLR 2026 | divide-and-conquer value composition，递归深度由线性降至对数 | 可替代直接 temporal-offset regression |
| [Strict Subgoal Execution](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec94195bca07a55e83968fbfa6efb8b0-Abstract-Conference.html) | ICLR 2026 | 用 frontier replay 区分 admissible/unreachable，按低层失败修正图边 | 与 controller executability 极近 |
| [RD-HRL: Generating Reliable Sub-Goals](https://proceedings.iclr.cc/paper_files/paper/2026/hash/56d5bf1977917d00c6e4dd9cf8401741-Abstract-Conference.html) | ICLR 2026 | 噪声 value 下的可靠 subgoal selection | 风险感知 subgoal 已有近邻 |
| [Multistep Quasimetric Distances](https://proceedings.iclr.cc/paper_files/paper/2026/hash/eddc0fab5a42f8d8de6eb5566cd9f1d3-Abstract-Conference.html) | ICLR 2026 | multistep quasimetric，覆盖随机、视觉、约 4000 步仿真任务 | 长程 claim 的关键强 baseline |
| [TD-JEPA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d158f054ff0cb83397367234899db07-Abstract-Conference.html) | ICLR 2026 | offline reward-free TD latent prediction + successor features/policies | 与 LeWM+RL 的最新正式近邻 |
| [Occupancy Reward Shaping](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html) | ICLR 2026 | 从 world-model occupancy geometry 提取 goal-reaching reward，改善长期 credit assignment | 从模型几何提取长期信号也已有正式近邻 |
| [Hierarchical Entity-centric RL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/de2bc3eed995c24307e8011161ac1438-Abstract-Conference.html) | ICLR 2026 | factored subgoal diffusion + value-based GCRL | 多对象长程任务的强层次化 baseline |

**结论。** trajectory offset head 不是唯一也未必最好的 reachability 学法。更有理论结构的 quasimetric、TD successor、transitive/min-plus composition 值得优先测试。

### 3.5 Offline model bias、保守性和行为支持

| 工作 | 正式出处 | 解法 | 本项目应吸收的控制 |
|---|---|---|---|
| [MOReL](https://proceedings.neurips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html) | NeurIPS 2020 | 把未知区域视为 pessimistic absorbing state | 数据外候选必须惩罚 |
| [MOPO](https://proceedings.neurips.cc/paper_files/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) | NeurIPS 2020 | model uncertainty penalty | risk head 必须与 support/OOD 对齐 |
| [COMBO](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html) | NeurIPS 2021 | conservative value on model rollouts | 乐观 planning 不是默认安全 |
| [LOMPO](https://proceedings.mlr.press/v144/rafailov21a.html) | L4DC 2021 | image latent model + pessimistic offline RL | 与视觉 LeWM 更接近 |
| [MBOP](https://openreview.net/forum?id=OMNB1G5xzd4) | ICLR 2021 | behavior prior + value + model predictive trajectories | action proposal 可由数据行为约束 |
| [Trajectory Transformer](https://proceedings.neurips.cc/paper/2021/hash/099fe6b0b444c23836c4a5d07346082b-Abstract.html) | NeurIPS 2021 | 序列模型 + beam search + return conditioning | 序列式 offline baseline |
| [MuZero Unplugged](https://papers.nips.cc/paper_files/paper/2021/hash/e8258e5140317ff36c7f8225a3bf9590-Abstract.html) | NeurIPS 2021 | 将 model-based planning 扩展到 fixed datasets | offline MBRL 的强背景 |
| [MACURA](https://proceedings.mlr.press/v235/frauenknecht24a.html) | ICML 2024 | 根据局部 uncertainty 自适应选择 model rollout length | “风险越高 duration 越短”已有直接近邻 |

### 3.6 Calibration、conformal planning 与闭环保证

| 工作 | 正式出处 | 机制 | 对 CAPE/CRAFT 的挑战 |
|---|---|---|---|
| [PlanCP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html) | NeurIPS 2023 | conformal uncertainty 用于 planning | conformal planning 不是新点 |
| [Conformal robustification of model-based controllers](https://proceedings.mlr.press/v242/chee24a.html) | L4DC 2024 | 用 conformal prediction robustify model-based controller | endpoint/tube calibration 有直接先例 |
| [Recursively feasible shrinking-horizon MPC](https://proceedings.mlr.press/v242/stamouli24a.html) | L4DC 2024 | shrinking-horizon MPC + conformal guarantees | recursive feasibility 需要明确条件 |
| [Conformal Prediction in the Loop](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee1331338cdf0e4d1055304d875548df-Abstract-Conference.html) | NeurIPS 2025 | 反馈更新 uncertainty model，用于 trajectory optimization | fixed split conformal 不是最新边界 |
| [Confidence on the Focal](https://academic.oup.com/jrsssb/article/87/4/1239/8113856) | JRSS-B 2025 | 对 top-k、optimization-based 等选择规则给出 selection-conditional coverage | “选择后 coverage”已有通用正式期刊框架 |

Conformal 保证的关键不是计算一个 quantile，而是明确随机性和 exchangeability 的单位。若在 calibration set 上对单个 candidate 建区间，之后由 CEM 从大量候选中选极端值，单候选的 marginal coverage 不自动推出：

- 被选 candidate 的 coverage；
- 整条 episode 的 coverage；
- adaptive replanning 后的 coverage；
- 新布局、新动力学或不同 controller 下的 coverage。

这仍是当前方案必须处理的问题，但不能再把 selection bias 本身写成研究空白。潜在贡献只能是：将已有 selection-conditional 统计框架正确实例化到 candidate pools 与 sequential replanning，或证明 world-model planning 中出现了现有框架未覆盖的新依赖结构。

---

## 4. 与 LeWM 最接近的 2026 前沿：只作预印本证据

| 工作 | 截至 2026-08-27 状态 | 主要结果/机制 | 对创新边界的影响 |
|---|---|---|---|
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | arXiv | reward-free observation-action 训练，JEPA latent dynamics + CEM | 项目 backbone |
| [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208) | arXiv | 多时间尺度 latent WM + hierarchy；仿真中改善 maze/push 且降低 compute | 多尺度 hierarchy 已被直接做过 |
| [RC-aux](https://arxiv.org/abs/2605.07278) | arXiv | multi-horizon prediction + budget-conditioned reachability | 与 reachability gate 高度重叠 |
| [GC-IDM](https://arxiv.org/abs/2605.08732) | arXiv | horizon-conditioned inverse dynamics 摊销 LeWM planning | 证明 CEM 并非必要 |
| [TRM](https://arxiv.org/abs/2605.22164) | arXiv | horizon-matched temporal reachability metric 替换 terminal MSE | 当前 CRAFT 的直接前驱 |
| [UWM-JEPA](https://arxiv.org/abs/2605.25313) | arXiv | belief-space latent，强调 blind rollout uncertainty | 仅在部分可观测 claim 下相关 |
| [FF-JEPA](https://arxiv.org/abs/2606.09311) | arXiv | action-free latent subgoal planner + 局部 CEM | latent subgoal 已有紧邻工作 |
| [VLWM](https://arxiv.org/abs/2606.21775) | arXiv | variable-length latent prediction + curriculum | duration-conditioned prediction 重叠明显 |
| [Fast LeWM](https://arxiv.org/abs/2606.26217) | arXiv | action-prefix 并行预测，降低递归误差和时延 | prefix/chunk predictor 不宜单独作为创新 |
| [Delta-JEPA](https://arxiv.org/abs/2606.31232) | arXiv | latent-difference action decoding，提升 action sensitivity | 适合作为表示层消融 |
| [The Objective Is the Bottleneck](https://arxiv.org/abs/2608.12959) | arXiv | 在 LeWM TwoRoom 中显示 terminal objective 而非 predictor 是主要瓶颈 | 进一步削弱“只改 predictor”的论证 |
| [SCALE](https://arxiv.org/abs/2608.16287) | arXiv | 用 task-relevant simulator state 校准 latent geometry | 若新方法使用真状态监督，必须与之比较并改变 claim |
| [Reinforced Planning](https://arxiv.org/abs/2608.18669) | arXiv | offline imagined rollouts 学 critic 与 plan optimizer | RL planner 是最新强竞争方向 |

注意：

- SCALE 使用 task-relevant state 对齐 geometry；这可能符合 simulation-only，但不再是纯 observation-action reward-free 监督。
- Hierarchical Planning 含真实机器人实验，但本文只考虑其仿真机制证据，不把实机作为项目目标。
- 预印本的实验结论不能与顶会/期刊证据等权，也不能据此声称优先权已经确定。

---

## 5. 对当前 CAPE/CRAFT 方法的可行性审计

### 5.1 文档构想与实际实现不一致

文档中的 CAPE-WM 包括：

- duration-conditioned macro model；
- composition loss；
- endpoint miss、success、residual scale 等 controller-conditioned risk；
- endpoint 和 whole-sequence conformal tube；
- progress / compute / risk 选择；
- tube、stall、risk event interrupt；
- backoff、恢复和 duration-5 fallback。

当前 Phase-B CRAFT 实现则是：

- 一个 directed hitting-time / reachability model；
- duration-stratified lower-confidence progress；
- 在共同 10-step imagined path 上比较 5/10 步 prefix；
- 只在 commitment 到期时重规划；
- 没有实现文档所述的 tube/stall/risk 中断；
- 没有使用 20/40/80/100 步在线 commitment。

因此论文不能混用 CAPE 与 CRAFT 的贡献。应选一种：

1. **如实发表 CRAFT**：定位为 calibrated directed reachability for short-window closed-loop planning；
2. **继续实现 CAPE**：补全风险头、事件触发、长 duration 与选择后校准，再重新实验。

当前证据只支持第一种。

### 5.2 当前结果真正说明了什么

确认性 TwoRoom unseen-v2、每个 offset 300 个任务：

| goal offset | CRAFT | Flat+TRM | 差值 |
|---:|---:|---:|---:|
| 25 | 99.67% | 98.67% | +1.00 pp |
| 50 | 97.67% | 93.67% | +4.00 pp |
| 75 | 94.33% | 85.67% | +8.67 pp |
| 100 | 84.67% | 67.33% | +17.33 pp |
| 75/100 合并 | 89.50% | 76.50% | +13.00 pp，95% CI [9.5, 16.5] pp |

这是统计上清楚的收益，但 gate 未通过：planning overhead ratio 为 37.3%，高于 25% 阈值。

执行行为：

- 新计划 4,721 次，action decisions 22,828 次；
- 按 action decisions 计，duration=5 有 20,403 次、duration=10 有 2,425 次（89.4% 处于 5-step plan）；
- 按新规划计，**94.8% 选择 5 步**，平均计划承诺约 5.26 步、实际执行约 4.84 步；
- planner 代码本身也记录了 progress/duration 会系统偏向最短 prefix 的退化风险。

正确结论是：

> CRAFT 的 reachability-aware short receding horizon 在同一 TwoRoom 数据/预算协议下，随着目标 offset 增大，比 Flat+TRM 更稳健。

不能写成：

> CRAFT 已经学习多尺度 temporal abstraction，能用长 commitment 解决 100-step 任务。

### 5.3 当前可行性问题

| 问题 | 严重度 | 原因 | 修复 |
|---|---:|---|---|
| temporal-offset 标签不等于最短 hitting time | 高 | 行为策略慢、绕路或 coverage 不足 | quasimetric/TD composition；同图 shortest-path audit |
| cross-trajectory negative 不等于不可达 | 高 | 数据未覆盖不代表环境不可达 | support-aware negatives；unlabeled 而非负类 |
| structural reachability 与 controller executability 混在一起 | 高 | 路存在但当前 controller 走不到 | 两个 head/两个数据源，显式分解 |
| CEM selection 破坏边际 conformal coverage | **很高** | planner 专门选最乐观的误差 | selected-candidate/episode-level calibration |
| duration 只开放 5/10，且塌缩到 5 | **很高** | 没有实际时间抽象 | 连续 hitting-time/hazard；coverage regularizer；固定 duration 对照 |
| 只在 duration 到期重规划 | 高 | 未实现 event trigger | tube/stall/risk trigger 和触发精确率 |
| risk ranking 几乎无信息 | **很高** | same-candidate audit viability AUC 约 0.509 | 先提升 risk head，再谈 guarantee |
| 长尺度候选不可用 | **很高** | audit 中 20/40 feasible rate 为 0 | 修复 macro dynamics/controller，不能隐藏这些尺度 |
| calibration 每尺度样本少 | 中高 | 条件 coverage 方差大 | 增加 calibration episodes；报告置信区间 |
| 单帧 LeWM 在 POMDP 非 Markov | 条件高 | 同观测可能对应不同隐状态 | history latent；若不主张 POMDP 则排除该任务 |
| simulator true-state label 改变监督假设 | 中高 | 不再是纯视觉 reward-free | 明确列为 privileged training variant |
| CRAFT/CAPE 命名和实现漂移 | 高 | 审稿人无法复现 claim | 冻结 method spec、config、artifact manifest |

### 5.4 创新性评分

| 组件 | 独立新颖性 | 可行性 | 判断 |
|---|---:|---:|---|
| frozen LeWM + temporal metric | 1/5 | 5/5 | TRM/MBOLD/QRL 已覆盖 |
| variable-duration macro model | 1/5 | 3/5 | MTS3、THICK、VLWM、MAC、HWM 近邻很多 |
| conformal endpoint/tube | 1–2/5 | 3/5 | PlanCP、L4DC 2024、NeurIPS 2025 已覆盖 |
| uncertainty-based duration backoff | 1/5 | 4/5 | MACURA 极近 |
| event trigger / learned termination | 1–2/5 | 3/5 | options 与 feedback MPC 已成熟 |
| controller-conditioned executability | 2–3/5 | 3/5 | adjacency/SSE/RD-HRL 接近，但与 frozen LeWM 耦合仍有空间 |
| structural vs controller reachability 分解 | 3/5 | 3/5 | 有研究价值，需证明分解带来可识别收益 |
| selection-aware conformal stopping time | 2–3/5 | 2/5 | JRSS-B 已覆盖一般 optimization-based selection；只剩 planning/sequential 实例化空间 |
| RL proposal + LeWM risk reranking | 2–3/5 | 4/5 | 组合有用但仅组合不足，需新训练/选择原则 |

---

## 6. 重构后的创新方法

首选工作名更新为 **GATE-WM：Geometry–Actionability Decomposition with Temporal Execution**。完整算法、候选方案比较、代码映射、消融和预注册门槛见 [独立方法提案](docs/long_horizon_method_proposals.md)。名称只是占位，先验证机制再命名。

与早期 DRESS-WM 构想相比，GATE-WM 有两点收敛：

- 主要贡献改为“可复用 global geometry 与 controller/budget-conditioned actionability 的分离及迁移”；
- selection-conditional calibration 降为可选统计扩展，因为 optimization-based selection 的一般 coverage 已有 JRSS-B 2025 正式工作。

### 6.1 核心对象

冻结 LeWM encoder 和短程 dynamics：

$$z_t=E(o_t), \qquad \hat z_{t+k}=F_\theta(z_t,a_{t:t+k-1}).$$

学习两个不同的量。

**A. 全局几何 / deployment-controller-independent structural distance**

$$
D_G(z,g)
\approx
\text{offline-data-supported directed transit cost},
$$

只回答离线数据支持的结构中是否有可拼接路径和还需多少进展。这里的“independent”只相对于部署 controller；它仍依赖数据收集策略和 coverage，不是真实环境的最优距离。它应满足方向性、非负性和近似 triangle/min-plus composition；跨轨迹没有连接证据时标为 unknown，而不是直接作为不可达负样本。

可选实现优先级：

1. multistep quasimetric / TD successor；
2. transitive min-plus composition；
3. temporal-offset regression 仅作简单 baseline。

**B. 控制器 actionability / controller-conditioned first-passage distribution**

$$
A_{\phi}(d\mid z,s,c,\xi)
=P(T_s\le d\mid z,s,c,\xi),
$$

其中 $c$ 显式描述 controller type、search budget、feedback mode 和 action chunk，$\xi$ 表示 candidate action/path trace。这个量由 simulator closed-loop attempts 的 censored survival loss 学习，回答“当前 candidate 与 controller/budget 能不能在 $d$ 内做到”，不能与“路径是否存在”混为一个 scalar。核心 transfer 实验是在冻结 $D_G$ 后，用少量 attempts 适配新 controller。

### 6.2 duration 作为停止时间而不是分类标签

用离散 hazard 建模命中事件和 right censoring：

$$
h_k=P(T_s=k\mid T_s\ge k,z,s,c,\xi).
$$

得到任意 commitment $d$ 的 completion probability。对 candidate prefix 定义：

$$
\mathcal D_{\mathrm{adm}}
=\{d:
\underline{\Delta}_G(d)\ge\epsilon,\ 
A_\phi(d\mid z,s,c,\xi)\ge1-\rho,\ 
S_G(z,s)\ge\tau,\ 
C(c,d)\le B_{\mathrm{remain}}\},
\qquad
d^\*=\max\mathcal D_{\mathrm{adm}}.
$$

即先满足进展、actionability、support 和 episode 剩余 compute 约束，再选最长可靠 commitment；这比把 progress/duration、risk 和 compute 混成一个 scalar 更直接地针对当前 5-step collapse。若集合为空，再在不超出总预算的前提下提高 controller budget、换 behavior-supported proposal 或缩短 duration。

### 6.3 RL proposal，不把 CEM 当唯一规划器

训练 offline goal-conditioned proposal：

$$
q_\psi(a_{t:t+d-1}\mid z_t,g,d),
$$

可以采用 HIQL/FQL/TD-JEPA 风格的 latent policy，或行为约束的 action chunks。执行流程：

1. RL/flow/chunk policy 生成多模态、数据支持内候选；
2. frozen LeWM 只做短程 rollout；
3. $D_G$ 评价长期结构进展；
4. controller executability head 评价实际可执行风险；
5. CEM 仅对 top candidates 局部 refine，或在 policy 不确定时 fallback。

这会同时覆盖用户要求的 RL 路线，并减少 flat CEM 的组合搜索。

### 6.4 选择后校准是扩展，不是首版核心

首版校准单位仍必须与部署单位一致：

- 在 calibration episodes 上运行**完整 planner**；
- 保存被选择的 candidate-duration、执行 residual、是否提前失败、是否到达；
- 对 selected candidates 报告 reliability、ECE、Brier 和 empirical coverage；
- pre-selection 与 post-selection 指标分开；
- 不完成 selection-conditional 推导时，不声称 finite-sample guarantee。

高风险扩展应直接以 JRSS-B 的 focal selection 框架为起点，研究 candidate-pool size、controller、duration 和 sequential replanning taxonomy，而不是重新使用普通 Mondrian quantile。

### 6.5 事件触发与恢复

事件只保留与两个核心对象及 support 对应的可检验条件：

- actual $D_G$ progress 低于计划 lower bound；
- actionability survival probability 相对计划时显著恶化；
- support score 进入 OOD 区域。

每个 trigger 都报告 precision、recall、平均提前量和不必要重规划率。失败时按原因采取：

- structural failure：换 subgoal / graph branch；
- controller failure：缩短 duration / 换低层 proposal；
- model OOD：退回 behavior-supported policy；
- partial-observation ambiguity：请求更多观测历史后再规划。

### 6.6 两个可执行版本

**版本 A：最小可证伪 GATE**

- frozen LeWM；
- directed global geometry；
- 单 controller 的 censored actionability hazard；
- admissibility + longest-feasible duration；
- 不声称 conformal guarantee。

优点：能在 TwoRoom candidate logs 上低成本检验分解是否真的有信息增益。

**版本 B：完整 GATE**

- 多 controller/budget actionability transfer；
- offline RL/action-chunk proposal；
- support-aware geometry；
- event-causal controller switching；
- LeWM short rollout。

优点：形成“geometry 跨 controller 复用”的清楚主张。缺点：需要 OGBench、controller-swap 和真正 $H^\*$ 长任务。

只有版本 A 的 merged-head 对照和版本 B 的 few-shot controller transfer 同时通过，才值得继续 selection-conditional calibration。

---

## 7. 任务是否足以证明创新

### 7.1 当前任务结论

| 任务/协议 | 能证明 | 不能证明 | 定位 |
|---|---|---|---|
| TwoRoom goal offset 25–100 | topology-aware metric、短窗闭环排序 | intrinsic 100-step、时间抽象、POMDP、contact、随机性 | Level 0 机制诊断 |
| PushT go50/go75 | 接触条件下 ranking 与局部执行 | 多阶段长任务、trajectory stitching | Level 1 局部诊断 |
| 当前 continuous validation，n=20 | pipeline 能运行 | 稳健性、泛化、统计可靠性 | smoke test |

所以答案是：**目前任务不足以支撑“通用长程世界模型方法”的创新性；甚至不足以证明当前方法已经学会时间抽象。**

### 7.2 推荐任务组合

#### Level 0：机制单元测试

- TwoRoom / MultiRoom：拓扑、方向性、绕墙；
- controlled action-delay / drift：动作敏感性和 world effect；
- hidden-door / hidden-velocity：只在主张 memory/belief 时使用；
- stochastic teleport：多模态未来和 calibration。

用途：定位原因，不作为 headline。

#### Level 1：OGBench 主平台

[OGBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ecd92623ac899357312aaa8915853699-Abstract-Conference.html) 正式设计就覆盖 stitching、long horizon、suboptimal data、stochasticity、pixels，比 TwoRoom 更能支撑结论。建议至少包括：

- visual PointMaze / AntMaze：路径、stitching、不同内在长度；
- visual Cube：连续控制和目标重排；
- Scene：多个对象与组合依赖；
- Puzzle：有序子任务和错误动作代价；
- stochastic teleport 变体：风险与 recalibration；
- unseen layouts / dynamics：分布迁移。

不建议只做 OGBench-Cube；单一连续操纵任务仍不足以证明结构可达性和长程分解。

#### Level 2：真正长程压力测试

构造或筛选 $H^\*\in\{100,200,500,1000\}$ 的任务，而不是只按数据 offset 分层：

- 4/8/16/24 个有依赖关系的 ordered subgoals；
- 必须跨轨迹 stitching 才能成功的数据；
- 局部最优会导致不可逆失败；
- 中途注入 dynamics shift、action noise 或障碍改变；
- 相同终点图像但不同隐状态/历史，仅在 belief claim 中使用。

### 7.3 必须实施的公平协议

1. **fixed commitment**：5/10/20/40/80 与 adaptive；
2. **equal feedback**：所有方法相同环境观测/重规划次数；
3. **equal compute**：相同 LeWM calls、candidate evaluations 和 wall-clock 档位；
4. **same candidate**：固定候选集，只比较 ranking；
5. **proposal oracle**：报告 oracle-best candidate，判断 coverage；
6. **open-loop diagnosis**：固定 20/40/80 步执行，不允许频繁纠错；
7. **coverage split**：同布局、unseen layout、unseen dynamics、low-support goal；
8. **selection-aware calibration**：pre-selection 与 post-selection coverage 分开；
9. **multi-seed confidence intervals**：任务级 paired bootstrap/McNemar，不能只报均值；
10. **failure taxonomy**：proposal、ranking、model、low-level execution、recovery 分开。

### 7.4 baseline 最小集合

**LeWM planner baselines**

- LeWM + latent L2；
- LeWM + TRM / temporal metric；
- fixed 5/10/20/40 short MPC；
- HWM / VLWM / Fast-LeWM 的公平仿真适配；
- GC-IDM 或 learned planner proposal。

**正式发表的 world-model / hierarchy baselines**

- TD-MPC2 式 policy/value/local planning；
- temporally abstract offline WM；
- SkiMo / Director 式 hierarchy；
- MAC action chunks；
- MACURA adaptive rollout。

**offline RL / GCRL baselines**

- HIQL；
- QRL / quasimetric representation；
- Flow Q-Learning；
- Transitive RL；
- OTA；
- SSE / RD-HRL；
- TD-JEPA。

不要求每篇都完整复现，但至少每个竞争机制有一个强正式 baseline。只与自行实现且明显较弱的 HWM/VLWM 适配版比较，不足以支撑 SOTA 主张。

### 7.5 指标

| 层面 | 必报指标 |
|---|---|
| 任务 | success vs $H^\*$、steps-to-goal、ordered-subgoal completion |
| prediction | open-loop latent/state error vs horizon，action sensitivity |
| proposal | oracle-best success、goal/subgoal coverage、data-support score |
| ranking | same-candidate top-1/top-k、Kendall/Spearman、selected regret |
| structural reachability | directionality、triangle/min-plus violation、path-length calibration |
| executability | AUROC/AUPRC、Brier/ECE、duration-conditioned success |
| conformal | pre/post-selection coverage、episode coverage、set/tube width |
| abstraction | duration histogram、mean commitment、high-level decisions per success |
| recovery | trigger precision/recall、recovery success、unnecessary replans |
| compute | model transitions、CEM/RL samples、latency、GPU time、memory |

---

## 8. 推荐论文叙事与可接受的 claim

### 8.1 推荐主问题

> A latent world model may know that a path exists while the deployed controller cannot reliably execute it. Can reusable global geometry be separated from controller-conditioned actionability, so that a planner can adapt its controller and temporal commitment under a fixed episode-level compute budget?

这个问题比“如何给 LeWM 加多尺度和 conformal”更清楚，也与当前 failure audit 直接对应。

### 8.2 如果只完成版本 A

可写：

> 在 frozen LeWM 和固定候选池上，global geometry 与 controller-conditioned actionability 的分离比单一 reachability/risk head 更准确地区分结构失败与执行失败，并改善 selected-candidate ranking 和局部执行校准。

此时只能定位为 planner-interface / failure-decomposition 结果；没有 controller-transfer、有效长 commitment 和真正长任务证据时，不能把它写成长程方法。

不可写：

- 首个 hierarchical LeWM；
- 首个 adaptive-duration world model；
- 首个 conformal world-model planner；
- 对任意部署分布都有安全保证。

### 8.3 如果完成版本 B

只有同时满足以下条件，才可主张 stronger novelty：

- 在至少三类任务上，dual reachability 明显优于单 head；
- 新 controller 只用少量 attempts 适配 actionability，明显优于 pooled head，并接近 full retrain；
- duration 不再塌缩，且 fixed-duration/equal-feedback 控制后仍提升；
- event triggers 对真实未来失败有预测力而不是只增加重规划；
- OGBench/真正 $H^\*$ 长任务和分布迁移均有结果，并在 equal-compute 下优于强基线；
- 与 MACURA、SSE、RD-HRL、MAC、CGCIVL、TD-JEPA 做了机制级对照。

若另做 selection-conditional calibration，才追加 PlanCP、feedback conformal 和 JRSS-B focal-selection 对照；该扩展不是完整 GATE 成立的必要条件。

---

## 9. 最终研究判断

### 可行性

- **工程可行性：中等偏高。** frozen LeWM、offline RL proposal、两个轻量 head 和短 rollout 都能在现有代码上实现。
- **actionability 适配可行性：中等。** simulator attempts 能提供 censored first-passage 标签，但 controller-transfer 是否样本高效必须实证；如用 true state 生成命中标签，必须披露 privileged evaluator supervision。
- **严格选择后校准可行性：中等偏低且非首版必要。** planner selection、adaptive replanning 与 environment shift 会破坏简单 conformal 假设，需要独立 episode-level calibration 和既有 selection-conditional 框架。
- **长尺度可执行性：当前偏低。** 20/40 尺度尚无可用候选，5-step 选择占绝对多数；应先解决这个事实，再扩展模块。

### 创新性

- 原 CAPE/CRAFT 组合：**低到中等**；
- CRAFT 当前实现：**有用的 LeWM planner 改进，但更像 TRM/reachability 的后续**；
- GATE 的 geometry/actionability 分解：**中等潜力**，创新性取决于 few-shot controller-transfer 和 merged-head 反事实消融；
- selection-aware calibrated stopping time：**低到中等独立新颖性**，optimization-based selection 已有通用统计框架；
- RL proposal + LeWM reranking：**实用性强，单独创新性中等**。

### 任务充分性

- 当前 TwoRoom + 小规模 continuous validation：**不足**；
- TwoRoom + PushT：仍不足以证明通用长程；
- OGBench pixels 多环境 + $H^\*$ 分层 + stochastic/shift + equal-compute：可形成可信主证据；
- 若再加入 ordered 8–24 subgoal 与 500–1000 步任务，才足以支撑“long-horizon”强标题。

最稳妥的下一步不是继续增加模块，而是按以下顺序推进：

1. 冻结方法定义与命名，消除 CAPE/CRAFT 漂移；
2. 在 OGBench 建立 LeWM、TRM、offline GCRL、action-chunk 的强基线；
3. 先证实 structural/execution reachability 分解确实必要；
4. 再做 hazard duration，验证不塌缩；
5. 最后才加入 selection-aware conformal 和 event-triggered recovery。

若第 3 步没有带来 same-candidate ranking、execution calibration 和最终 success 的一致提升，应停止扩展这条路线，转向更简单的 offline RL/action-chunk proposal + LeWM short-horizon reranking。
