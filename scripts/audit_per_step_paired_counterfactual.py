from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cape_wm.cc_rwkv.per_step_pairs import SCHEMA_VERSION, SPLIT_CODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a per-step paired counterfactual cache")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-atol", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: dict[str, bool] = {}
    failures: list[str] = []
    with h5py.File(args.dataset, "r") as handle:
        metadata = handle["metadata"].attrs
        samples = handle["samples"]
        schema = metadata.get("schema_version")
        checks["schema_version"] = schema == SCHEMA_VERSION
        n, h, h2 = samples["pulse_noop_mask"].shape
        if h != h2:
            raise ValueError("pulse_noop_mask must have square position/offset axes")
        checks["finite_factual"] = bool(
            np.isfinite(np.asarray(samples["factual_latents"])).all()
            and np.isfinite(np.asarray(samples["factual_states"])).all()
        )
        checks["finite_pulse"] = bool(
            np.isfinite(np.asarray(samples["pulse_noop_latents"])).all()
            and np.isfinite(np.asarray(samples["pulse_noop_states"])).all()
        )
        mask = np.asarray(samples["pulse_noop_mask"], dtype=bool)
        expected_mask = np.zeros_like(mask)
        for position in range(h):
            expected_mask[:, position, : h - position] = True
        checks["triangular_mask"] = bool(np.array_equal(mask, expected_mask))

        factual_actions = np.asarray(samples["factual_actions"])
        pulse_actions = np.asarray(samples["pulse_noop_actions"])
        factual = np.asarray(samples["factual_latents"])
        pulse = np.asarray(samples["pulse_noop_latents"])
        effect = np.asarray(samples["effect_latents"])
        current_zero = []
        suffix_equal = []
        effect_error = []
        for position in range(h):
            current_zero.append(np.all(pulse_actions[:, position, 0] == 0))
            length = h - position
            if length > 1:
                suffix_equal.append(
                    np.array_equal(
                        pulse_actions[:, position, 1:length], factual_actions[:, position + 1 :]
                    )
                )
            else:
                suffix_equal.append(True)
            expected = pulse[:, position, :length] - factual[:, position + 1 :]
            effect_error.append(float(np.max(np.abs(effect[:, position, :length] - expected))))
        checks["current_action_is_raw_zero"] = bool(all(current_zero))
        checks["factual_suffix_is_shared"] = bool(all(suffix_equal))
        checks["effect_target_aligned"] = bool(max(effect_error, default=0.0) <= args.effect_atol)
        checks["restore_consistent"] = bool(np.asarray(samples["restore_consistent"]).all())
        checks["factual_state_replay_exact"] = bool(
            np.max(np.asarray(samples["factual_replay_error"])) == 0.0
        )
        checks["factual_image_replay_exact"] = bool(
            np.max(np.asarray(samples["factual_replay_image_error"])) == 0.0
        )
        split = np.asarray(samples["split"])
        episodes = np.asarray(samples["episode_id"])
        groups = [set(episodes[split == code].tolist()) for code in SPLIT_CODES.values()]
        checks["split_disjoint"] = not any(
            groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)
        )
        checks["finite_actions"] = bool(
            np.isfinite(factual_actions).all() and np.isfinite(pulse_actions).all()
        )
        if "audit" in handle:
            checks["raw_audit_present"] = bool(
                handle["audit/factual_images"].shape[0] > 0
                and handle["audit/pulse_noop_images"].shape[0] > 0
            )
        else:
            checks["raw_audit_present"] = False
        for name, passed in checks.items():
            if not passed:
                failures.append(name)

    result = {
        "schema_version": "cc_rwkv_per_step_pairs_audit_v1",
        "dataset": str(args.dataset),
        "samples": int(n),
        "model_horizon": int(h),
        "effect_atol": args.effect_atol,
        "checks": checks,
        "max_effect_alignment_error": max(effect_error, default=0.0),
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError(f"per-step paired audit failed: {failures}")


if __name__ == "__main__":
    main()
