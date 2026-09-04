# CRAFT: continuous reachability for long-horizon world models

CRAFT is a planner-facing world-model module, not a topology controller. It
maps a current latent state and goal latent state to one directed first-hitting
time distribution. The module never receives positions, room IDs, walls,
doors, waypoints, or a simulator graph.

## Module

For frozen world-model latents `z` and `g`, one network predicts the location
and scale of a log-normal hitting-time distribution:

`T(z -> g) ~ LogNormal(mu(z, g), sigma(z, g))`.

The same distribution provides:

- a continuous expected arrival time for fine candidate ranking;
- monotone multi-horizon reachability `P(T <= h)` for any horizon;
- a distributional spread describing temporal ambiguity.

This replaces the original coarse categorical head. On the trajectory-isolated
validation set, expected-step MAE fell from 19.40 to 12.30; the current
Flat+TRM checkpoint has MAE 12.91. Unlike the categorical model, the continuous
head can represent near-goal times below ten steps.

## Training

Training uses only ordered latent pairs from the same recorded trajectory and
their exact temporal separation. Cross-trajectory states are not labelled
unreachable because the dataset does not prove that claim. An auxiliary
semigroup constraint enforces temporal composition on observed interior
successors:

`E[T(z_t, g)] ~= elapsed + E[T(z_(t+elapsed), g)]`.

No position or topology labels are used. The head is therefore attachable to a
world model that supplies latent trajectories; it does not encode Two-Room
geometry in its interface.

## Multi-scale imagination and calibrated risk

At planning time one primitive-action CEM search produces a shared action
sequence. CRAFT evaluates its 5-step and 10-step prefixes, rather than invoking
separate controllers. The reachability model itself is trained and queried at
horizons through 100 steps; only the world-model rollout is kept short.

Longer 20/40-step online rollouts were rejected by calibration diagnostics, not
by a topology rule: their predicted progress ceased to correlate reliably with
realized progress. This follows the useful Flat+TRM pattern of combining a
long-range temporal metric with short, reliable world-model imagination.

For each duration `d`, calibration records the error between predicted progress
and progress realized after executing the selected prefix. A task-level
Mondrian conformal quantile `q_d` gives the conservative score

`L_d = predicted_progress_d - q_d`.

The internal log-normal spread was deliberately not used to normalize this
residual: on calibration rollouts it was negatively correlated with actual
world-model error. Thus CRAFT retains distributional arrival uncertainty while
using the empirically valid duration-stratified residual as its planning-risk
certificate.

## Single-path inference invariant

The CEM optimizer ranks every candidate and duration by `L_d` and executes the
selected prefix. There is no alternate controller, Flat/TRM anchor, trusted
candidate, topology branch, or fallback. The formal evaluator fails if a CRAFT
run loads TRM, macro, risk, or topology assets, or reports a fallback decision.

The former standalone macro dynamics and executability-risk heads remain only
as ablation code. They are not part of CRAFT because the macro predictor
introduced off-manifold latent error and the independent risk filter rejected
useful candidates. Their useful ideas survive as temporal composition and
calibrated progress, respectively, inside one reachability abstraction.

## Confirmatory Two-Room result

The selected checkpoint, calibration artifact, and config were frozen before
running the confirmatory unseen-v2 matrix. Results use 300 paired tasks per
offset and the existing Flat+TRM manifests.

| Goal offset | CRAFT | Flat+TRM | Delta |
|---:|---:|---:|---:|
| 25 | 99.67% | 98.67% | +1.00pp |
| 50 | 97.67% | 93.67% | +4.00pp |
| 75 | 94.33% | 85.67% | +8.67pp |
| 100 | 84.67% | 67.33% | +17.33pp |

Across offsets 75 and 100, CRAFT obtains 89.5% versus 76.5% for Flat+TRM:
+13.0pp with a paired-bootstrap 95% interval of [+9.5, +16.5]pp. Exact paired
McNemar `p = 7.47e-13`. Long-horizon model-transition overhead is +20.54% and
measured wall-time overhead after the Torch-native rollout optimization is
+23.22%. The preregistered Phase-B gate passes.

The result establishes the module on Two-Room; cross-environment generalization
still requires the planned PushT, OGBench Cube, and alternate-backbone studies.
