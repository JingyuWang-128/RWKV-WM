# Legacy CC-RWKV M0–M5 archive

This directory contains entry-point scripts and script-dependent tests from the
superseded H1/four-branch CC-RWKV M0–M5 workflow. They are retained only for
historical inspection and are not part of the `formal_corrected_v1` per-step
paired training protocol.

The active workflow uses:

- `scripts/collect_per_step_paired_counterfactual.py`
- `scripts/audit_per_step_paired_counterfactual.py`
- `scripts/merge_per_step_pair_shards.py`
- `scripts/train_per_step_pairs.py`
- `scripts/run_formal_stage_queue.py`
- `scripts/evaluate_per_step_pairs.py`

Files under `tests/` were moved with the legacy scripts because they directly
imported the old script locations. They are not collected by the active
`pyproject.toml` test path.

The historical outcome remains documented in
`docs/cc_rwkv_m5_report.md`. Do not mix artifacts produced by these archived
scripts with `artifacts/runs/per_step_pairs/formal_corrected_v1/`.
