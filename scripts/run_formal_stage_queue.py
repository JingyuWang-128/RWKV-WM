from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal stage jobs sequentially on one GPU")
    parser.add_argument("--device", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.status.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, job in enumerate(jobs):
        command = [
            ".venv/bin/python",
            "scripts/train_per_step_pairs.py",
            "--data", job["data"],
            "--output", job["output"],
            "--method", job["method"],
            "--profile", "main",
            "--max-steps", str(job["max_steps"]),
            "--batch-size", "8",
            "--curriculum", job["curriculum"],
            "--device", args.device,
            "--log-interval", "1000",
            "--seed", str(job["seed"]),
        ]
        started = time.time()
        print(json.dumps({"stage": args.stage, "job_index": index, "job": job}), flush=True)
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        result = subprocess.run(command, env=env, check=False)
        records.append({
            **job,
            "device": args.device,
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
        })
        args.status.write_text(json.dumps({"stage": args.stage, "device": args.device, "records": records}, indent=2) + "\n")
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    args.status.write_text(json.dumps({"stage": args.stage, "device": args.device, "records": records, "status": "complete"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
