#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List


def _sanitize_sys_path() -> None:
    """Avoid importing local ./wandb directory as python package."""
    cwd = str(Path.cwd().resolve())
    script_dir = str(Path(__file__).resolve().parent)
    filtered = []
    for item in sys.path:
        item_resolved = str(Path(item or ".").resolve())
        if item in ("", "."):
            continue
        if item_resolved in (cwd, script_dir):
            continue
        filtered.append(item)
    sys.path = filtered


_sanitize_sys_path()

from wandb.proto import wandb_internal_pb2  # noqa: E402
from wandb.sdk.internal.datastore import DataStore  # noqa: E402


def _metric_key(item: wandb_internal_pb2.HistoryItem) -> str:
    if item.nested_key:
        return "/".join(item.nested_key)
    return item.key


def _parse_value(value_json: str):
    try:
        return json.loads(value_json)
    except Exception:
        return value_json


def _keep_key(key: str, keys: List[str], contains: List[str], regex_pattern) -> bool:
    if not key:
        return False
    if keys and key not in keys:
        return False
    if contains and not any(token in key for token in contains):
        return False
    if regex_pattern and regex_pattern.search(key) is None:
        return False
    return True


def export_wandb_binary_to_csv(
    input_path: Path,
    output_path: Path,
    keys: List[str],
    contains: List[str],
    regex_pattern,
    keep_system_keys: bool,
) -> None:
    ds = DataStore()
    ds.open_for_scan(str(input_path))

    rows_by_step: Dict[int, Dict[str, object]] = {}
    no_step_rows: List[Dict[str, object]] = []
    seen_columns = set()

    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if not rec.history.item:
            continue

        row = {}
        for item in rec.history.item:
            key = _metric_key(item)
            if key == "_step":
                value = _parse_value(item.value_json)
                row[key] = value
                seen_columns.add(key)
                continue
            if not keep_system_keys and key.startswith("_"):
                continue
            if not _keep_key(key, keys, contains, regex_pattern):
                continue

            value = _parse_value(item.value_json)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            row[key] = value
            seen_columns.add(key)

        if not row:
            continue

        step_val = row.get("_step")
        if step_val is None:
            no_step_rows.append(row)
            continue

        try:
            step = int(float(step_val))
        except Exception:
            no_step_rows.append(row)
            continue

        if step not in rows_by_step:
            rows_by_step[step] = {"_step": step}
            seen_columns.add("_step")
        rows_by_step[step].update(row)

    ordered_steps = sorted(rows_by_step.keys())
    rows = [rows_by_step[s] for s in ordered_steps] + no_step_rows

    columns = sorted(seen_columns)
    if "_step" in columns:
        columns.remove("_step")
        columns = ["_step"] + columns

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(rows)}")
    print(f"Cols:   {len(columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export local .wandb binary history to CSV.")
    parser.add_argument("input_wandb", type=str, help="Path to run-xxxx.wandb file")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: same folder, same name with .csv)",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="*",
        default=[],
        help="Exact metric keys to keep, e.g. ActorCritic/adv_abs_mean",
    )
    parser.add_argument(
        "--contains",
        type=str,
        nargs="*",
        default=[],
        help="Keep keys containing any token, e.g. ActorCritic/td_error WorldModel/imag_",
    )
    parser.add_argument(
        "--regex",
        type=str,
        default=None,
        help="Regex filter on key name",
    )
    parser.add_argument(
        "--keep-system-keys",
        action="store_true",
        help="Keep _step/_runtime/_timestamp etc.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_wandb).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix != ".wandb":
        raise ValueError(f"Input file must end with .wandb: {input_path}")

    if args.output is None:
        output_path = input_path.with_suffix(".csv")
    else:
        output_path = Path(args.output).expanduser().resolve()

    regex_pattern = None
    if args.regex:
        import re

        regex_pattern = re.compile(args.regex)

    export_wandb_binary_to_csv(
        input_path=input_path,
        output_path=output_path,
        keys=args.keys,
        contains=args.contains,
        regex_pattern=regex_pattern,
        keep_system_keys=args.keep_system_keys,
    )


if __name__ == "__main__":
    main()
