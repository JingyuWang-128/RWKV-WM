# Experimental contract

This document is the executable preregistration for the first CAPE-WM paper.
Any deviation must be logged before opening final-test labels.

Future claims about intrinsic task horizon, matched feedback frequency, temporal
abstraction, or long open-loop execution are additionally governed by
[`long_horizon_evaluation.md`](long_horizon_evaluation.md).  The completed
Two-Room Phase-B gate predates that stronger contract and is classified there
as a Level-0 distant-goal result.

## Data isolation

Split at the source-trajectory level into `train`, `selection`, `calibration`,
and `test`.  All start-goal pairs derived from one trajectory remain in one
split.  The manifest stores sorted trajectory IDs, allocation ratios, the RNG
seed, and a content hash.  `selection` chooses hyperparameters and checkpoints;
`calibration` fits only conformal order statistics; `test` is opened once.

Every method receives the same stable pair ID, environment seed, start, goal,
action budget, CEM population budget, and total world-model forward-call budget.
Report actual calls and wall time in addition to nominal settings.

Two-Room Phase-B protocol amendment (registered before the first CAPE final-test
run): conformal endpoint and sequence scores are stratified by the already
registered commitment duration `{5,10,20,40}` (Mondrian split conformal, 64
calibration trajectories per stratum). A pooled pilot correction exceeded the
environment success radius because the duration-40 tail dominated all shorter
commitments, making every candidate infeasible. The duration set, alpha, split,
and per-stratum sample counts are frozen; final-test labels were not inspected
when making this correction.

## Phase A: infrastructure and reproduction

- Reference Python: 3.10. Development and smoke runs may use CPU or any
  PyTorch-supported accelerator. For the final paper profile, one model must fit
  one 24 GB RTX 4090.
- Pin upstream commits and archive the resolved Python environment.
- Reproduce official LeWorldModel results on Two-Room, PushT, and OGBench Cube.
- Store raw JSONL episode records, stdout/stderr logs, config snapshots, model
  hashes, GPU model/driver, and the split manifest.

Official reproduction must pass before interpreting CAPE-WM gains. Lack of a
GPU does not block code execution or small-scale debugging, but Phase A
paper-scale reproduction remains a required run on the designated machine.

## Phase B: mechanism gate

Environments: Two-Room and PushT.  Backbone: frozen LeWorldModel.  Goal offsets:
25, 50, 75, and 100.  Methods:

- Flat MPC;
- Flat MPC + shared TRM mixture;
- fixed-scale HWM;
- VLWM;
- Hi-LeWM-C;
- CAPE-WM.

Use three independent training seeds and at least 100 completely paired final
tasks per seed and condition.  Select the strongest equal-compute baseline
without consulting final-test outcomes.

Proceed to Phase D only when all are true:

- mean long-horizon success improves by at least 10 percentage points;
- the paired-bootstrap 95% interval for the gain excludes zero;
- planning wall-clock overhead is at most 25%;
- short-horizon success decreases by at most 3 percentage points.

`scripts/evaluate_gate.py` computes this decision from raw paired JSONL records.
If the gate fails, run the same-candidate audit, assigning failures to candidate
generation, risk ranking, or low-level execution.  Do not add modules before
that attribution.

## Phase C: full validation

Add OGBench Cube, randomized initializations in PushT and Two-Room, and a
DINO-WM backbone check on PushT and Two-Room.  Evaluate stable-worldmodel
occlusion, friction, mass, and visual shifts.  Strong reproducible baselines are
PRISM, RC-aux, HWM, Hi-LeWM-C, and VLWM.  SAGE and FF-JEPA are non-equivalent
goal interfaces and belong outside the equal-compute main table.

Required ablations:

- fixed versus adaptive duration;
- raw risk, probability calibration, endpoint conformal, and full sequence
  conformal;
- pre-plan filtering only, event triggers only, and both;
- without cross-scale consistency;
- latent L2, TRM, and mixture objectives;
- every risk level and compute budget in the registered grid.

## Measurements and statistics

Primary outcomes are success, offset-success AUC, and final true-state error.
Systems metrics are model calls, wall latency, and peak allocated CUDA memory.
Mechanism metrics are infeasible-subgoal rate, Brier, ECE, AUROC, endpoint and
sequence coverage, trigger precision, recovery rate, and mean commitment.

Use paired bootstrap intervals over shared pair IDs and exact paired McNemar
tests for binary success.  Preserve per-seed results; never treat multiple
frames from one episode as independent samples.  Report both pooled estimates
and seed dispersion.

## Phase D: online extension

Only after the Phase B gate, replace the offline risk head with RSSM/SSM
ensemble disagreement and choose imagined horizons `{5,15,30}`.  Validate on
Memory Maze and POPGym against DreamerV3, R2I, and fixed-imagination variants.
Keep online memory results separate from the offline visual-control main table.
