from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from cape_wm.cc_rwkv.per_step_pairs import SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and audit formal primitive-action per-step pairs"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--frozen-test-episodes", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/per_step_pairs"),
    )
    parser.add_argument("--devices", default="cuda:1,cuda:2,cuda:3")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--raw-audit-fraction", type=float, default=0.01)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path(
            "artifacts/cache/cc_rwkv/per_step_pairs/formal_v2_primitive_collection_status.json"
        ),
    )
    return parser.parse_args()


def _valid_complete_shard(path: Path, expected: dict[str, Any]) -> bool:
    manifest = path / "manifest.json"
    pairs = path / "pairs.h5"
    if not manifest.is_file() or not pairs.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    command = payload.get("command_config", {})
    return (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("variant") == expected["variant"]
        and int(command.get("samples", -1)) == expected["samples"]
        and int(command.get("shard_index", -1)) == expected["shard_index"]
        and int(command.get("num_shards", -1)) == expected["num_shards"]
        and int(command.get("action_block", -1)) == 1
        and int(command.get("future_primitive_steps", -1)) == 20
        and int(command.get("history_steps", -1)) == expected["history_steps"]
    )


def main() -> None:
    args = parse_args()
    devices = tuple(item.strip() for item in args.devices.split(",") if item.strip())
    if not devices:
        raise ValueError("at least one CUDA device is required")
    if args.samples <= 0 or args.num_shards < len(devices):
        raise ValueError("samples must be positive and num-shards >= number of devices")
    for path in (
        args.data,
        args.weights,
        args.model_config,
        args.frozen_test_episodes,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    jobs: Queue[dict[str, Any]] = Queue()
    # Interleave tasks so every device sees both environments and no task is
    # left entirely to one GPU.
    for shard_index in range(args.num_shards):
        for variant, history_steps in (("tworoom_w", 15), ("action_delay", 5)):
            jobs.put(
                {
                    "variant": variant,
                    "history_steps": history_steps,
                    "samples": args.samples,
                    "shard_index": shard_index,
                    "num_shards": args.num_shards,
                }
            )

    records: list[dict[str, Any]] = []
    record_lock = threading.Lock()
    failed = threading.Event()
    args.status.parent.mkdir(parents=True, exist_ok=True)

    def save_status() -> None:
        payload = {
            "schema_version": "formal_v2_primitive_collection_status_v1",
            "devices": list(devices),
            "samples_per_task": args.samples,
            "num_shards": args.num_shards,
            "records": records,
            "failed": failed.is_set(),
        }
        temporary = args.status.with_suffix(args.status.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.status)

    def worker(device: str) -> None:
        while not failed.is_set():
            try:
                job = jobs.get_nowait()
            except Empty:
                return
            variant = job["variant"]
            shard = int(job["shard_index"])
            output = (
                args.root / variant / f"formal_v2_primitive_{args.samples}_shards" / f"shard{shard}"
            )
            log = output / "collection.log"
            output.mkdir(parents=True, exist_ok=True)
            started = time.time()
            if _valid_complete_shard(output, job):
                result_record = {
                    **job,
                    "device": device,
                    "output": str(output),
                    "status": "skipped_complete",
                    "elapsed_seconds": 0.0,
                }
            else:
                command = [
                    ".venv/bin/python",
                    "scripts/collect_per_step_paired_counterfactual.py",
                    "--data",
                    str(args.data),
                    "--weights",
                    str(args.weights),
                    "--model-config",
                    str(args.model_config),
                    "--frozen-test-episodes",
                    str(args.frozen_test_episodes),
                    "--output",
                    str(output),
                    "--cache-dir",
                    str(args.cache_dir),
                    "--variant",
                    variant,
                    "--samples",
                    str(args.samples),
                    "--history-steps",
                    str(job["history_steps"]),
                    "--future-primitive-steps",
                    "20",
                    "--action-block",
                    "1",
                    "--allow-multiple-per-episode",
                    "--raw-audit-fraction",
                    str(args.raw_audit_fraction),
                    "--shard-index",
                    str(shard),
                    "--num-shards",
                    str(args.num_shards),
                    "--device",
                    device,
                ]
                with log.open("w") as handle:
                    env = dict(os.environ)
                    env.update(
                        {
                            "OMP_NUM_THREADS": "8",
                            "MKL_NUM_THREADS": "8",
                            "OPENBLAS_NUM_THREADS": "8",
                            "NUMEXPR_NUM_THREADS": "8",
                        }
                    )
                    completed = subprocess.run(
                        command,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                result_record = {
                    **job,
                    "device": device,
                    "output": str(output),
                    "log": str(log),
                    "status": "complete" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.time() - started,
                }
                if completed.returncode != 0:
                    failed.set()
            with record_lock:
                records.append(result_record)
                save_status()
            jobs.task_done()

    threads = [threading.Thread(target=worker, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed.is_set() or not jobs.empty():
        raise RuntimeError(f"collection queue failed; inspect {args.status}")

    for variant in ("tworoom_w", "action_delay"):
        shard_root = args.root / variant / f"formal_v2_primitive_{args.samples}_shards"
        shards = [shard_root / f"shard{index}" / "pairs.h5" for index in range(args.num_shards)]
        merged = args.root / variant / f"formal_v2_primitive_{args.samples}_merged"
        merge_command = [
            ".venv/bin/python",
            "scripts/merge_per_step_pair_shards.py",
            "--shards",
            *(str(path) for path in shards),
            "--output",
            str(merged / "pairs.h5"),
        ]
        subprocess.run(merge_command, check=True)
        subprocess.run(
            [
                ".venv/bin/python",
                "scripts/audit_per_step_paired_counterfactual.py",
                "--dataset",
                str(merged / "pairs.h5"),
                "--source",
                str(args.data),
                "--output",
                str(merged / "audit.json"),
            ],
            check=True,
        )
    with record_lock:
        records.append({"status": "merged_and_audited"})
        save_status()


if __name__ == "__main__":
    main()
