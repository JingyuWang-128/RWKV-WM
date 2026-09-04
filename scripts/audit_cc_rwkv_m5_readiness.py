from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REQUIREMENTS = {
    "tworoom_w": {
        "manifest": Path("artifacts/cache/cc_rwkv/tworoom_w/mvp5000/manifest.json"),
        "action_block": 5,
        "minimum_samples": 5000,
        "gate_a_horizon": 20,
        "gate_b_horizon": 50,
    },
    "action_delay": {
        "manifest": Path("artifacts/cache/cc_rwkv/action_delay/mvp5000/manifest.json"),
        "action_block": 1,
        "minimum_samples": 5000,
        "gate_a_horizon": 20,
        "gate_b_horizon": 50,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit formal M5 data and compute readiness")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/results/cc_rwkv/m5/readiness.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {}
    for task, requirement in REQUIREMENTS.items():
        manifest_path = requirement["manifest"]
        reasons = []
        manifest = None
        if not manifest_path.is_file():
            reasons.append("formal_manifest_missing")
        else:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("variant") != task:
                reasons.append("variant_mismatch")
            if manifest.get("samples", 0) < requirement["minimum_samples"]:
                reasons.append("dataset_below_5000_snapshots")
            if manifest.get("action_block") != requirement["action_block"]:
                reasons.append("action_block_mismatch")
            if manifest.get("primitive_horizon", 0) < requirement["gate_a_horizon"]:
                reasons.append("gate_a_horizon_missing")
        tasks[task] = {
            "status": "ready" if not reasons else "not_ready",
            "reasons": reasons,
            "manifest": str(manifest_path),
            "observed": manifest,
            "gate_b_50_available": bool(
                manifest and manifest.get("primitive_horizon", 0) >= requirement["gate_b_horizon"]
            ),
        }
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "free_bytes": free,
                    "total_bytes": total,
                    "at_least_12_gib_free": free >= 12 * 1024**3,
                }
            )
    payload = {
        "schema_version": "cc_rwkv_m5_readiness_v1",
        "tasks": tasks,
        "compute": {
            "cuda_available": torch.cuda.is_available(),
            "devices": cuda_devices,
            "formal_training_ready": any(device["at_least_12_gib_free"] for device in cuda_devices),
        },
        "overall_ready": all(item["status"] == "ready" for item in tasks.values())
        and any(device["at_least_12_gib_free"] for device in cuda_devices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
