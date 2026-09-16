#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

output_root="artifacts/runs/per_step_pairs/formal_corrected_v1"
mkdir -p "$output_root"

.venv/bin/python scripts/run_formal_stage_queue.py \
  --protocol-id formal_corrected_v1 \
  --device cuda:0 \
  --stage F1_tworoom_baselines \
  --jobs configs/experiments/formal_corrected_v1/f1_tworoom_baselines.json \
  --status "$output_root/f1_status_tworoom.json" \
  --checkpoint-interval 1000 \
  --snapshot-interval 5000 \
  --resume-existing

.venv/bin/python scripts/run_formal_stage_queue.py \
  --protocol-id formal_corrected_v1 \
  --device cuda:0 \
  --stage F1_action_delay_baselines \
  --jobs configs/experiments/formal_corrected_v1/f1_action_delay_baselines.json \
  --status "$output_root/f1_status_action_delay.json" \
  --checkpoint-interval 1000 \
  --snapshot-interval 5000 \
  --resume-existing

