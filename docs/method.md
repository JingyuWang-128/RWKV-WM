# CAPE-WM method specification

## Scope and notation

At time `t`, a frozen visual encoder maps the current and goal images to
latents `z_t` and `z_g`.  CAPE-WM may use neither simulator state nor future
observations for planning.  A candidate consists of duration `d`, latent
subgoal `s_d`, primitive action plan, predicted latent path, and measured world
model call count.  Durations are `{5,10,20,40}` environment steps.

## Duration-conditioned macro model

For an action segment `a[t:t+d]`, a Transformer action encoder emits a macro
action `u_d`.  The predictor models

```
z_hat[t+d] = F(z_t, u_d, phi(d)).
```

The training objective contains latent prediction loss, a KL penalty that keeps
sampled macro actions compatible with CEM's Gaussian search distribution, and
cross-scale consistency.  For each admissible split `d=d1+d2`, the latter
matches a direct `d` prediction to the composition of the `d1` and `d2`
predictors.  The code handles odd durations explicitly rather than silently
dropping a primitive action.

At planning time, separate macro CEM searches use the same physical lookahead
budget.  Thus a duration-5 candidate receives more macro transitions than a
duration-40 candidate, while all see the same total future time.  Candidate
`s_d` is the first macro transition; a frozen low-level MPC constructs primitive
actions toward it.

## Reachability objective and risk model

The main comparison uses one horizon-matched TRM mixture for every method:

```
C(z, g, d) = beta * TRM(z, g, d) + (1-beta) * normalized_latent_distance(z, g).
```

Pure latent L2 is an ablation.  This separation is important: it prevents a bad
goal metric from being misreported as a temporal-planning failure.

The risk head receives `[z_t, s_d, s_d-z_t, |s_d-z_t|, phi(d)]` and predicts:

- non-negative endpoint miss distance;
- logit of successful low-level execution;
- positive latent residual scale for every executed step.

Training records must come from the frozen low-level MPC run in closed loop.
Open-loop model errors are not a substitute for executability labels.
The endpoint-miss target may be computed from evaluator-only task state, putting
its calibrated bound in the same units as the environment success threshold.
True state is never passed to the planner or risk head. Latent residuals remain
the signal for the online prediction tube.

## Split conformal calibration

For calibration examples `i=1,...,n`, endpoint scores are

```
r_i = observed_miss_i - predicted_miss_i.
```

The correction is the conservative finite-sample order statistic at
`ceil((n+1)*(1-alpha))`.  A new endpoint upper bound is

```
U_alpha(x) = predicted_miss(x) + q_endpoint.
```

For a trajectory, the score is one maximum rather than a set of independent
time-step scores:

```
R_i = max_t residual[i,t] / predicted_scale[i,t].
```

The calibrated tube at step `t` is `q_sequence * predicted_scale[t]`.  This
controls the whole segment under the same split-conformal exchangeability
assumption and avoids a false independent-time-step argument.  Calibration
examples are never used for fitting model weights or selecting hyperparameters.

## Candidate selection

A candidate is executable only if

```
U_0.1(z_t, s_d, d) <= environment_success_threshold
```

and it predicts positive progress.  Among executable candidates CAPE-WM
maximizes

```
predicted_progress / d
  - lambda_compute * measured_planning_cost
  - lambda_risk * U_0.1.
```

The two lambdas are selected once from `{0,0.01,0.05,0.1}` on the selection
split.  The planner records candidate-level scores, feasibility, bounds, and
costs so a same-candidate audit can replay selection without rerunning CEM.
The implementation additionally enforces a per-decision latent-transition
budget. It reserves 10% for the fallback, distributes the remainder across the
currently allowed scales, and resizes CEM populations before search. Reported
call counts are checked against this budget; wall time is still reported because
batched and sequential calls are not equivalent.

## Event-triggered execution

The current commitment is interrupted when any of the following first occurs:

1. the observed latent leaves the sequence-calibrated tube;
2. goal cost fails to improve for two controller cycles;
3. the latent subgoal is reached early;
4. the recalculated endpoint upper bound exceeds the executability threshold;
5. the selected duration expires.

Tube, stall, or risk failures reduce the maximum duration by one level.
Two successful subgoal completions or duration expirations restore one level.
This asymmetric hysteresis prevents rapid scale oscillation.  When every
candidate is rejected, the planner uses direct duration-5 low-level MPC and
marks the action as an uncalibrated fallback in diagnostics.

## Non-claims

CAPE-WM does not claim that multi-scale prediction, TRM reachability, or
conformal prediction is individually new.  The contribution under test is the
closed-loop coupling of calibrated temporal commitment, selection, monitoring,
and recovery.  It also does not claim conformal coverage after arbitrary OOD
distribution shift; Phase C measures degradation and recovery empirically.
