# CAPE-WM

**Calibrated Adaptive Planning and Execution for World Models** is a research
implementation of closed-loop temporal commitment for goal-image control.  It
jointly chooses a latent subgoal and commitment duration, filters candidates by
a split-conformal executability bound, and interrupts execution when the
observed latent trajectory leaves its calibrated prediction tube.

The repository is deliberately backbone-agnostic.  The core planner only
depends on four operations: image encoding, latent rollout, goal cost, and
candidate generation.  A concrete LeWorldModel adapter and a generic PyTorch
adapter for DINO-WM-style models are included.  Encoders and primitive dynamics
remain frozen during CAPE-WM training.

## What is implemented

- duration set `D={5,10,20,40}` with a Transformer macro-action encoder;
- duration-conditioned macro dynamics and cross-scale composition loss;
- endpoint miss, success-probability, and per-step residual-scale risk heads;
- finite-sample split conformal endpoint bounds and sequence-max tube bounds;
- compute/risk-aware multi-scale candidate selection;
- event-triggered interruption, one-level duration backoff, hysteretic recovery,
  and a shortest-horizon MPC fallback;
- deterministic trajectory-level train/selection/calibration/test manifests;
- paired bootstrap, exact McNemar, calibration, latency, and phase-gate metrics;
- an image-only toy environment that exercises the complete closed loop.

The conformal claim is intentionally narrow: the empirical marginal coverage
applies under exchangeability with the held-out calibration distribution.  The
code does not claim coverage for arbitrary OOD closed-loop trajectories.

The proposed successor research direction, CC-RWKV-WM, studies whether
counterfactual action effects can be written directly into an RWKV matrix-state
recurrence to improve long open-loop world-model rollouts. Its method proposal
and executable engineering plan are in
[docs/long_horizon_method_proposals.md](docs/long_horizon_method_proposals.md)
and [docs/cc_rwkv_implementation_plan.md](docs/cc_rwkv_implementation_plan.md).

## Installation and runtime configuration

CAPE-WM does not require a particular GPU count. `device=auto` uses the first
available CUDA, Intel XPU, or Apple MPS device and falls back to CPU when no GPU
is usable. Python 3.10 is the reference environment:

```bash
scripts/bootstrap.sh
```

The bootstrap script selects a CUDA 12.1 wheel when an NVIDIA GPU is visible and
a CPU wheel otherwise. To reuse a preinstalled ROCm/XPU/MPS PyTorch build, run
`CAPE_WM_TORCH_BACKEND=system scripts/bootstrap.sh`.

All hardware choices live in one file: `configs/runtime.yaml`.

```yaml
runtime:
  device: auto       # auto, cpu, cuda:0, cuda:1, mps, or xpu:0
  num_workers: 0
  pin_memory: auto

training:
  macro_batch_size: 64
  risk_batch_size: 64
  trm_batch_size: 64
```

`auto` uses an available accelerator and otherwise falls back to CPU. For a
larger GPU, normally only the batch sizes need to be increased. Training
scripts and environment checks read this file by default.

Check the resolved device, then run synthetic training and the image-control
smoke test:

```bash
.venv/bin/python scripts/check_environment.py
.venv/bin/cape-wm-smoke
.venv/bin/cape-wm-toy --seed 0 --json
```

The official LeWM Two-Room reproduction uses the architecture `config.json`
stored beside `weights.pt`, while planning settings come from
`configs/lewm_tworooms_baseline.yaml` and hardware selection comes only from
`configs/runtime.yaml`:

```bash
.venv/bin/python scripts/run_lewm_tworooms_baseline.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt
```

It writes the frozen task pairs, per-task JSONL, resolved configuration, and a
summary with timing, model-call accounting, asset hashes, and device metadata
under `artifacts/results/lewm_tworooms_official/`.

Run the preregistered long-horizon matrix (offsets 25/50/75/100, three
evaluation/CEM seeds, 100 tasks per condition) with resumable per-condition
outputs:

```bash
.venv/bin/python scripts/run_lewm_tworooms_matrix.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt
```

The official repository does not provide matching Two-Room implementations or
checkpoints for every comparison method. The remaining Phase-B comparisons are
therefore explicitly recorded as `paper_spec_tworoom_adaptation`: they share
the frozen official LeWM encoder, exact test pairs, action budget, CEM budget,
and horizon-matched TRM objective, but are not presented as author-official
reproductions. Prepare their leak-free latent cache, train the auxiliary heads,
then run the resumable matrix:

```bash
.venv/bin/python scripts/prepare_tworoom_baselines.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt

.venv/bin/python scripts/train_tworoom_baselines.py

.venv/bin/python scripts/run_tworoom_comparison_matrix.py \
  --data /data/wjy/lewm_data/tworoom.h5 \
  --weights /data/wjy/lewm_model/tworooms/weights.pt

.venv/bin/python scripts/aggregate_tworoom_baselines.py
```

The comparison matrix covers Flat+TRM, fixed-scale HWM, VLWM, and Hi-LeWM-C.
Each condition stores its exact pair manifest, per-episode records, resolved
configuration, model-call accounting, timing, asset/checkpoint provenance, and
summary. The aggregate report adds long-horizon AUC, paired bootstrap intervals,
and exact paired McNemar tests against official Flat MPC.

The CPU and CUDA dependency locks remain separate under `requirements/`; they
pin software packages and are not additional hardware configuration files.

## Quick smoke test

```bash
.venv/bin/cape-wm-toy --seed 0 --json
```

The toy observation is a rendered image; neither the planner nor its risk model
receives the environment state.  State is used only by the evaluator to decide
success.

## Reproducible workflow

1. Obtain pinned upstream code and place official assets under `artifacts/`:

   ```bash
   scripts/download_official_assets.sh
   ```

   Set `DOWNLOAD_LEWM_CHECKPOINTS=1` to also fetch the three official
   Hugging Face checkpoint repositories. Exact upstream hashes are recorded in
   `configs/upstream_lock.json`; override them only in an explicitly logged run.

2. Convert each official dataset to compressed trajectory files containing
   `observations`, `actions`, and optional evaluator-only `states`, then create a
   disjoint split manifest:

   ```bash
   .venv/bin/python scripts/make_manifest.py \
     artifacts/data/trajectory_ids.txt \
     artifacts/manifests/phase_b.json --seed 0
   ```

3. Cache frozen backbone latents and train the macro model, TRM, and risk head.
   The included training entry points consume NumPy artifacts so preprocessing
   stays independent from model training:

   ```bash
   .venv/bin/python scripts/train_macro.py --help
   .venv/bin/python scripts/train_risk.py --help
   ```

4. Fit conformal corrections **only** on the calibration split, freeze all
   hyperparameters selected on the model-selection split, and evaluate every
   method on the identical final-test pair IDs and seeds.

5. Apply the preregistered gate:

   ```bash
   .venv/bin/python scripts/evaluate_gate.py \
     artifacts/results/cape_long.jsonl \
     artifacts/results/baseline_long.jsonl \
     artifacts/results/cape_short.jsonl \
     artifacts/results/baseline_short.jsonl
   ```

The exact Phase B/C/D protocol is in [docs/experiments.md](docs/experiments.md),
equations and guarantees are in [docs/method.md](docs/method.md), the proposed
next method is in
[docs/long_horizon_method_proposals.md](docs/long_horizon_method_proposals.md),
and the API contract verified against pinned upstream source is in
[docs/upstream_integration.md](docs/upstream_integration.md).

## LeWorldModel integration

`LeWorldModelAdapter` follows the official JEPA call path:
`encode(info["pixels"])`, `action_encoder(actions)`, and autoregressive
`predict(latent, action_embedding)`.  Supply the official transform and a
horizon-matched TRM callable:

```python
from cape_wm.adapters import LeWorldModelAdapter

adapter = LeWorldModelAdapter(
    model=official_jepa.eval(),
    transform=official_image_transform,
    goal_metric=trained_trm,
    action_shape=(action_dim,),
    device="cuda:0",
)
```

The planner is a pure Python object with `reset()` and
`plan(observation, goal_image) -> (action, diagnostics)`. For the official
vectorized evaluator, `StableWorldModelCAPEPolicy` implements the required
`get_action(info)` interface, consumes `info["pixels"]` and `info["goal"]`, and
keeps independent commitment state per environment. See `src/cape_wm/adapters/`
and `src/cape_wm/policy.py` for exact tensor conventions.

## Repository layout

- `src/cape_wm/`: models, planner, calibration, adapters, metrics, and I/O;
- `configs/`: preregistered Phase B/C/D defaults;
- `scripts/`: environment, data split, training, calibration, and gate tools;
- `tests/`: unit and closed-loop integration tests;
- `docs/`: method specification and experimental contract.

This code is released under the MIT license.  Upstream repositories and model
weights retain their own licenses and terms.
