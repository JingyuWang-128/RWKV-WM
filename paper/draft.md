# CAPE-WM: Calibrated Adaptive Planning and Execution for Long-Horizon World Models

> Working draft. Numerical results and the “first” claim are placeholders until
> the preregistered final evaluation and a camera-ready literature search.

## Abstract

Long horizons reduce high-level search depth in visual world-model control, but
their subgoals are more likely to be dynamically unreachable. Short horizons
are easier for low-level control yet cause frequent replanning and local
decisions. We propose CAPE-WM, a backbone-independent module that treats latent
subgoal, temporal commitment, and plan abandonment as one closed-loop decision.
A duration-conditioned macro model proposes subgoals at several temporal
scales. A low-level-controller-specific risk head predicts endpoint miss,
success, and a residual sequence. Held-out split conformal calibration turns
these estimates into endpoint executability bounds and simultaneous trajectory
tubes. CAPE-WM selects progress-efficient executable candidates and interrupts
a commitment on tube violation, stalled goal progress, early subgoal arrival,
or increased residual risk. Failures shorten the permitted duration, whereas
two successful commitments restore it. We preregister equal-forward-budget
experiments on Two-Room, PushT, and OGBench Cube with LeWorldModel and DINO-WM
backbones. [Results pending.] The intended contribution is calibrated dynamic
temporal commitment with event-triggered recovery, rather than multi-scale
prediction or reachability learning in isolation.

## 1. Introduction

Goal-image control asks an agent to transform its current visual observation
into a target observation using a learned dynamics model. The difficult regime
is not merely one-step prediction: errors, subgoal reachability, search depth,
and compute interact over dozens or hundreds of actions. A planner that commits
for too long may optimize an attractive but unachievable latent state. A planner
that commits for too little sees only local progress and repeatedly pays the
cost of global search.

Recent systems address parts of this tradeoff. HWM predicts on several temporal
scales, and VLWM represents variable prediction lengths. Hi-LeWM identifies
unreachable subgoals and rigid stage execution as central bottlenecks. TRM and
RC-aux improve the alignment between representation distance and physical
reachability. These ingredients still leave a control question: given the
current execution evidence, for how long should the planner trust its chosen
subgoal, and exactly when should it abandon it?

CAPE-WM makes that trust explicit. Its risk bound is calibrated on independent
closed-loop attempts of the same frozen low-level controller. Its sequence
score is the maximum standardized residual of a complete attempt, avoiding an
independence assumption across controller steps. The planner then uses the
calibrated object twice: to reject a subgoal before execution and to monitor the
commitment after every real observation.

We test three hypotheses: (H1) adaptive calibrated commitment improves success
at offsets 75–100 without materially damaging offsets 25–50; (H2) pre-plan
filtering and event-triggered recovery are complementary; and (H3) the effect
transfers across frozen visual backbones and degrades gracefully under
calibration shift.

## 2. Related work

**Visual world-model planning.** DINO-WM demonstrates planning in a compact
visual representation without pixel reconstruction. LeWorldModel focuses on
long-horizon goal-image control. We use both as frozen backbones, changing
neither encoder nor primitive dynamics.

**Temporal abstraction.** HWM, VLWM, and Hi-LeWM motivate multi-scale or
variable-length prediction. CAPE-WM uses a light duration-conditioned predictor
as infrastructure, but studies closed-loop duration choice and recovery as the
independent variable.

**Reachability-aware objectives.** TRM and RC-aux show that generic latent
distance may not represent controllability. All main-table methods therefore
share one horizon-matched TRM mixture; latent L2 is restricted to an ablation.

**Uncertainty and conformal calibration.** Split conformal prediction provides
finite-sample marginal coverage under exchangeability. We apply it to low-level
endpoint miss and to the maximum standardized residual over a complete
execution segment. We do not extend that guarantee to arbitrary distribution
shift; shifted environments are empirical stress tests.

## 3. Method

Let `z_t=E(o_t)` and `z_g=E(o_g)` be frozen visual latents. For every duration
`d` in `{5,10,20,40}`, a Transformer encodes a primitive action segment into a
macro action. A duration-conditioned transition predicts `F(z_t,u_d,d)`. A
composition loss aligns a direct duration-`d` transition with two transitions
whose durations sum to `d`. Separate CEM searches use equal total physical
lookahead, and the first predicted macro state becomes the candidate subgoal.

The executability head consumes current latent, subgoal, difference features,
and duration. It predicts endpoint miss `m_hat`, success probability, and
positive residual scales `sigma_1,...,sigma_d`. On held-out attempts, endpoint
scores are `m-m_hat`; sequence scores are `max_j r_j/sigma_j`. Conservative
finite-sample quantiles yield endpoint upper bound `U_alpha` and a simultaneous
latent tube.

Candidates with `U_0.1` above the task success threshold are rejected. CAPE-WM
selects the remainder by predicted progress per step minus registered compute
and risk penalties. It replans when an observed latent exits the tube, progress
stalls for two cycles, the subgoal is reached, the remaining endpoint bound
becomes unsafe, or duration expires. Unsafe interruptions reduce the maximum
duration by one scale; two successful completions restore one. If no candidate
survives, duration-5 low-level MPC acts directly toward the final goal.

## 4. Experiments

Phase B uses frozen LeWorldModel on Two-Room and PushT at offsets 25, 50, 75,
and 100. Baselines are Flat MPC, Flat+TRM, fixed-scale HWM, VLWM, and Hi-LeWM-C.
Each configuration has three training seeds and at least 100 paired test tasks
per seed. Population sizes are adjusted so all methods receive an equal count
of latent transitions; actual model calls, planning time, and peak GPU memory
are reported.

The preregistered expansion gate requires at least +10 success points on the
long offsets, a paired-bootstrap 95% interval above zero, no more than 25%
planning-time overhead, and no more than 3 points of short-horizon regression.
If the gate fails, an oracle same-candidate audit attributes failures to
generation, ranking, or low-level execution.

After passing, Phase C adds OGBench Cube, DINO-WM, random initializations, and
visual/occlusion/friction/mass shifts. Ablations independently remove adaptive
duration, conformal endpoint/tube calibration, risk filtering, event triggers,
and composition loss; compare L2/TRM/hybrid costs; and sweep risk and compute
budgets. Binary outcomes use paired bootstrap intervals and exact McNemar
tests. Calibration reports Brier, ECE, AUROC, endpoint coverage, and whole-
sequence coverage.

## 5. Results

To be populated only from immutable raw JSONL files after Phase A reproduction
and the final-test manifest are frozen.

## 6. Limitations and broader impact

Calibration depends on exchangeability and can fail under policy-induced or
environment shift. Risk collection adds controller interactions even though
the backbone training is offline. CEM remains computationally costly, and the
four duration choices may not be suitable for every control frequency. Tests
are simulated and do not establish robot safety. The method should be treated
as a performance and monitoring mechanism, not a safety certificate.

## References

- DINO-WM, [arXiv:2411.04983](https://arxiv.org/abs/2411.04983).
- HWM, [arXiv:2604.03208](https://arxiv.org/abs/2604.03208).
- RC-aux, [arXiv:2605.07278](https://arxiv.org/abs/2605.07278).
- TRM, [arXiv:2605.22164](https://arxiv.org/abs/2605.22164).
- VLWM, [arXiv:2606.21775](https://arxiv.org/abs/2606.21775).
- Hi-LeWM, [arXiv:2607.12547](https://arxiv.org/abs/2607.12547).
- Long-horizon planning-objective diagnosis,
  [arXiv:2608.12959](https://arxiv.org/abs/2608.12959).
