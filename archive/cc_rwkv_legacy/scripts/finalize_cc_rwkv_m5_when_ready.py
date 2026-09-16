from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the final Action-Delay B6 w=2.0 evaluation units, then "
            "run the frozen M5 aggregate and completion audit."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tworoom-manifest", type=Path, required=True)
    parser.add_argument("--action-delay-manifest", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3072)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _evaluation_ready(directory: Path) -> tuple[bool, str]:
    required = (
        "m5_run.json",
        "sample_metrics.jsonl",
        "mechanism_audit.json",
        "summary.json",
        "provenance.json",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return False, f"missing {','.join(missing)}"
    try:
        run = _read_json(directory / "m5_run.json")
        sample_ids: list[str] = []
        with (directory / "sample_metrics.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                sample_ids.append(str(row["sample_id"]))
        _read_json(directory / "mechanism_audit.json")
        _read_json(directory / "summary.json")
        _read_json(directory / "provenance.json")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return False, f"not readable yet: {error}"
    if len(sample_ids) != 500 or len(set(sample_ids)) != 500:
        return False, f"sample rows/unique IDs are {len(sample_ids)}/{len(set(sample_ids))}"
    if int(run.get("test_samples", -1)) != 500:
        return False, f"m5_run test_samples={run.get('test_samples')}"
    return True, "ready"


def _pending_units(root: Path) -> dict[str, str]:
    pending: dict[str, str] = {}
    for seed in (0, 1, 2):
        for split in ("validation", "test"):
            relative = Path("action_delay") / f"seed_{seed}" / split / "b6_w2p0"
            ready, reason = _evaluation_ready(root / relative)
            if not ready:
                pending[str(relative)] = reason
    return pending


def _run(command: list[str]) -> None:
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    state_path = args.root / "finalization_watcher.json"
    while True:
        pending = _pending_units(args.root)
        state = {
            "schema_version": "cc_rwkv_m5_finalization_watcher_v1",
            "updated_at_utc": _utc_now(),
            "status": "waiting" if pending else "finalizing",
            "pending_units": pending,
        }
        _write_state(state_path, state)
        if not pending:
            break
        print(f"waiting for {len(pending)}/6 final evaluation units", flush=True)
        time.sleep(args.poll_seconds)

    aggregate_output = args.root / "aggregate"
    try:
        _run(
            [
                sys.executable,
                "scripts/aggregate_cc_rwkv_m5.py",
                "--root",
                str(args.root),
                "--output",
                str(aggregate_output),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
            ]
        )
        audit_path = args.root / "completion_audit.json"
        _run(
            [
                sys.executable,
                "scripts/audit_cc_rwkv_m5_completion.py",
                "--root",
                str(args.root),
                "--tworoom-manifest",
                str(args.tworoom_manifest),
                "--action-delay-manifest",
                str(args.action_delay_manifest),
                "--output",
                str(audit_path),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--bootstrap-seed",
                str(args.bootstrap_seed),
            ]
        )
        audit = _read_json(audit_path)
        # The completion-audit script uses ``pass`` for a complete audit and
        # ``incomplete`` when any required artifact/check is missing.  Accept
        # the former here; ``complete`` was an obsolete watcher-side value.
        if audit.get("status") != "pass" or audit.get("failures"):
            raise RuntimeError(
                f"completion audit did not pass: status={audit.get('status')}, "
                f"failures={len(audit.get('failures', []))}"
            )
    except Exception as error:
        _write_state(
            state_path,
            {
                "schema_version": "cc_rwkv_m5_finalization_watcher_v1",
                "updated_at_utc": _utc_now(),
                "status": "failed",
                "error": repr(error),
            },
        )
        raise

    _write_state(
        state_path,
        {
            "schema_version": "cc_rwkv_m5_finalization_watcher_v1",
            "updated_at_utc": _utc_now(),
            "status": "complete",
            "aggregate": str(aggregate_output / "summary.json"),
            "completion_audit": str(args.root / "completion_audit.json"),
        },
    )
    print("formal M5 aggregate and completion audit are complete", flush=True)


if __name__ == "__main__":
    main()
