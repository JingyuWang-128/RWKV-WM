from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from cape_wm.cc_rwkv.per_step_training import ROLLOUT_MASK_POLICY
from cape_wm.cc_rwkv.protocol import file_sha256

OPTIMIZER_POLICY = "rwkv7_official_style_adamw_v1"


def _write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal stage jobs sequentially on one GPU")
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--protocol-id", default="formal_corrected_v1")
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("configs/experiments/formal_corrected_v1/protocol_manifest.json"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--snapshot-interval", type=int, default=5000)
    parser.add_argument("--dwm-contrastive-weight", type=float, default=0.3)
    parser.add_argument("--dwm-orthogonality-weight", type=float, default=0.5)
    parser.add_argument("--dwm-temperature", type=float, default=0.07)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="resume incomplete jobs from output/latest.pt and skip completed summaries",
    )
    args = parser.parse_args()
    if (
        min(
            args.batch_size,
            args.validation_batch_size,
            args.log_interval,
            args.checkpoint_interval,
            args.snapshot_interval,
        )
        <= 0
    ):
        raise ValueError("batch size and logging/checkpoint intervals must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    jobs = json.loads(args.jobs.read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("formal job file must contain a non-empty list")
    args.status.parent.mkdir(parents=True, exist_ok=True)
    if not args.protocol_manifest.is_file():
        raise FileNotFoundError(args.protocol_manifest)
    protocol_manifest_sha256 = file_sha256(args.protocol_manifest)
    verified_data: dict[str, str] = {}
    for job in jobs:
        data = str(job["data"])
        expected_sha256 = str(job.get("data_sha256", ""))
        if not expected_sha256:
            raise ValueError(f"formal job is missing data_sha256: {data}")
        if data not in verified_data:
            actual_sha256 = file_sha256(data)
            if actual_sha256 != expected_sha256:
                raise ValueError(f"formal data SHA256 mismatch: {data}")
            verified_data[data] = actual_sha256
            print(json.dumps({"verified_data": data, "sha256": actual_sha256}), flush=True)
        elif verified_data[data] != expected_sha256:
            raise ValueError(f"formal jobs disagree on data SHA256: {data}")
    records = []
    for index, job in enumerate(jobs):
        output = Path(job["output"])
        summary_path = output / "summary.json"
        batch_size = int(job.get("batch_size", args.batch_size))
        validation_batch_size = int(
            job.get("validation_batch_size", args.validation_batch_size)
        )
        learning_rate = float(job.get("learning_rate", args.learning_rate))
        effect_weight = float(job.get("effect_weight", 1.0))
        paired_loss = bool(job.get("paired_loss", job["method"] == "b6"))
        if paired_loss and job["method"] != "b6":
            raise ValueError("only B6 jobs may enable paired loss")
        if effect_weight < 0:
            raise ValueError("effect_weight must be non-negative")
        dwm_contrastive_weight = float(
            job.get("dwm_contrastive_weight", args.dwm_contrastive_weight)
        )
        dwm_orthogonality_weight = float(
            job.get("dwm_orthogonality_weight", args.dwm_orthogonality_weight)
        )
        dwm_temperature = float(job.get("dwm_temperature", args.dwm_temperature))
        stability_options = {
            "gradient_accumulation": int(job.get("gradient_accumulation", 1)),
            "decay_lr_multiplier": float(job.get("decay_lr_multiplier", 2.0)),
            "warmup_steps": int(job.get("warmup_steps", 0)),
            "min_lr_ratio": float(job.get("min_lr_ratio", 1.0)),
            "stability_window": int(job.get("stability_window", 200)),
            "stability_grace_steps": int(job.get("stability_grace_steps", 200)),
            "max_clip_fraction": job.get("max_clip_fraction"),
        }
        resolved_job = {
            **job,
            **stability_options,
            "protocol_id": args.protocol_id,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "learning_rate": learning_rate,
            "effect_weight": effect_weight,
            "paired_loss": paired_loss,
            "effective_effect_weight": effect_weight if paired_loss else 0.0,
            "rollout_mask_policy": ROLLOUT_MASK_POLICY,
            "dwm_contrastive_weight": dwm_contrastive_weight,
            "dwm_orthogonality_weight": dwm_orthogonality_weight,
            "dwm_temperature": dwm_temperature,
            "checkpoint_interval": int(job.get("checkpoint_interval", args.checkpoint_interval)),
            "snapshot_interval": int(job.get("snapshot_interval", args.snapshot_interval)),
            "optimizer_policy": OPTIMIZER_POLICY,
            "protocol_manifest_sha256": protocol_manifest_sha256,
        }
        if args.resume_existing and summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            completed = (
                summary.get("schema_version") == "cc_rwkv_per_step_training_summary_v2"
                and summary.get("protocol_id") == args.protocol_id
                and summary.get("method") == job["method"]
                and int(summary.get("steps", -1)) == int(job["max_steps"])
                and int(summary.get("seed", -1)) == int(job["seed"])
                and int(summary.get("batch_size", -1)) == batch_size
                and int(summary.get("validation_batch_size", -1)) == validation_batch_size
                and float(summary.get("learning_rate", -1.0)) == learning_rate
                and float(summary.get("effect_weight", -1.0)) == effect_weight
                and bool(summary.get("paired_loss")) == paired_loss
                and float(summary.get("dwm_contrastive_weight", -1.0)) == dwm_contrastive_weight
                and float(summary.get("dwm_orthogonality_weight", -1.0)) == dwm_orthogonality_weight
                and float(summary.get("dwm_temperature", -1.0)) == dwm_temperature
                and summary.get("data") == job["data"]
                and summary.get("data_sha256") == job["data_sha256"]
                and list(summary.get("curriculum", []))
                == [int(item) for item in job["curriculum"].split(",")]
                and summary.get("loss_normalization") == "position_balanced_v1"
                and summary.get("rollout_mask_policy") == ROLLOUT_MASK_POLICY
                and summary.get("checkpoint_schema") == "cc_rwkv_per_step_checkpoint_v3"
                and summary.get("optimizer_policy") == OPTIMIZER_POLICY
                and summary.get("protocol_manifest_sha256") == protocol_manifest_sha256
                and all(summary.get(key) == value for key, value in stability_options.items())
            )
            if completed:
                records.append(
                    {
                        **resolved_job,
                        "device": args.device,
                        "returncode": 0,
                        "elapsed_seconds": 0.0,
                        "status": "skipped_complete",
                    }
                )
                _write_status(
                    args.status,
                    {
                        "protocol_id": args.protocol_id,
                        "stage": args.stage,
                        "device": args.device,
                        "records": records,
                    },
                )
                continue
        command = [
            ".venv/bin/python",
            "scripts/train_per_step_pairs.py",
            "--data",
            job["data"],
            "--protocol-id",
            args.protocol_id,
            "--data-sha256",
            job["data_sha256"],
            "--protocol-manifest",
            str(args.protocol_manifest),
            "--output",
            job["output"],
            "--method",
            job["method"],
            "--profile",
            "main",
            "--max-steps",
            str(job["max_steps"]),
            "--batch-size",
            str(batch_size),
            "--validation-batch-size",
            str(validation_batch_size),
            "--learning-rate",
            str(learning_rate),
            "--curriculum",
            job["curriculum"],
            "--device",
            args.device,
            "--log-interval",
            str(job.get("log_interval", args.log_interval)),
            "--seed",
            str(job["seed"]),
            "--dwm-contrastive-weight",
            str(dwm_contrastive_weight),
            "--dwm-orthogonality-weight",
            str(dwm_orthogonality_weight),
            "--dwm-temperature",
            str(dwm_temperature),
            "--checkpoint-interval",
            str(job.get("checkpoint_interval", args.checkpoint_interval)),
            "--snapshot-interval",
            str(job.get("snapshot_interval", args.snapshot_interval)),
        ]
        command.extend(("--effect-weight", str(effect_weight)))
        for key, value in stability_options.items():
            if value is not None:
                command.extend(("--" + key.replace("_", "-"), str(value)))
        if job["method"] == "b6" and not paired_loss:
            command.append("--disable-paired-loss")
        if args.resume_existing and (output / "latest.pt").is_file():
            command.append("--resume")
        started = time.time()
        print(
            json.dumps({"stage": args.stage, "job_index": index, "job": resolved_job}),
            flush=True,
        )
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        result = subprocess.run(command, env=env, check=False)
        records.append(
            {
                **resolved_job,
                "device": args.device,
                "returncode": result.returncode,
                "elapsed_seconds": time.time() - started,
            }
        )
        _write_status(
            args.status,
            {
                "protocol_id": args.protocol_id,
                "stage": args.stage,
                "device": args.device,
                "records": records,
            },
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    _write_status(
        args.status,
        {
            "protocol_id": args.protocol_id,
            "stage": args.stage,
            "device": args.device,
            "records": records,
            "status": "complete",
        },
    )


if __name__ == "__main__":
    main()
