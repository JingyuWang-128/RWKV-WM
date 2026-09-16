"""Launch only stability preflight, detached from the terminal. Never launch formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "tworoom_w": "af75f5c81fb936951cd7bf66c524f6ebb5980bd35cc7da8ac3ef7829d7f5ab55",
    "action_delay": "9d5c1417ecebd8d9cbacb4cb87d350f220588697cb23144e1591594d4db8a5fa",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DATA, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    relative = Path("artifacts/runs/per_step_pairs/formal_corrected_v1/mask_fixed_seed0")
    output = relative / "preflight_lr1e4_acc4" / args.task
    command = [
        str(ROOT / ".venv/bin/python"),
        "scripts/train_per_step_pairs.py",
        "--data",
        f"artifacts/cache/cc_rwkv/per_step_pairs/{args.task}/"
        "formal_v2_primitive_20000_merged/pairs.h5",
        "--data-sha256",
        DATA[args.task],
        "--output",
        str(output),
        "--method",
        "b6",
        "--profile",
        "main",
        "--seed",
        "0",
        "--effect-weight",
        "0.5",
        "--device",
        "cuda:0",
        "--max-steps",
        "2400",
        "--curriculum",
        "1,5,10,20",
        "--batch-size",
        "8",
        "--gradient-accumulation",
        "4",
        "--learning-rate",
        "1e-4",
        "--decay-lr-multiplier",
        "1",
        "--warmup-steps",
        "100",
        "--min-lr-ratio",
        "1",
        "--stability-window",
        "200",
        "--stability-grace-steps",
        "200",
        "--max-clip-fraction",
        "0.2",
        "--validation-limit",
        "256",
        "--validation-batch-size",
        "64",
        "--log-interval",
        "200",
        "--checkpoint-interval",
        "200",
        "--snapshot-interval",
        "600",
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        print(json.dumps({"physical_gpu": args.gpu, "command": command}, indent=2))
        return
    # Conservative idle check, do not occupy another user's active accelerator.
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={args.gpu}",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    utilization, memory = [int(value.strip()) for value in raw.strip().split(",")]
    if utilization > 10 or memory > 1024:
        raise RuntimeError(f"GPU {args.gpu} is not idle: {raw.strip()}")
    directory = ROOT / output
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and (directory / "launch.json").exists():
        raise FileExistsError("preflight already launched; inspect it before resuming")
    if args.resume and not (directory / "latest.pt").is_file():
        raise FileNotFoundError("no preflight latest.pt to resume")
    sources = [
        "scripts/train_per_step_pairs.py",
        "scripts/launch_mask_fixed_preflight.py",
        "src/cape_wm/cc_rwkv/per_step_training.py",
        "src/cape_wm/cc_rwkv/stability.py",
        "src/cape_wm/cc_rwkv/cell.py",
        "src/cape_wm/cc_rwkv/counterfactual.py",
    ]
    hashes = {
        source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in sources
    }
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    with (directory / "console.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = {
        "kind": "preflight_only",
        "formal_training_allowed": False,
        "pid": process.pid,
        "physical_gpu": args.gpu,
        "command": command,
        "code_sha256": hashes,
        "python": sys.version,
    }
    (directory / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
