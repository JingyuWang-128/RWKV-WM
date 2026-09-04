#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cape_wm.data import load_calibration_records, save_calibration_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CAPE candidate-distribution records")
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    records = []
    metadata = []
    for path in args.inputs:
        records.extend(load_calibration_records(path))
        metadata.extend(
            json.loads(line)
            for line in path.with_suffix(".metadata.jsonl").read_text().splitlines()
        )
    record_ids = [record.record_id for record in records]
    metadata_ids = [item["record_id"] for item in metadata]
    if len(record_ids) != len(set(record_ids)):
        raise SystemExit("candidate record IDs are not unique")
    if record_ids != metadata_ids:
        raise SystemExit("candidate records and metadata are not aligned")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_records(args.output, records)
    args.output.with_suffix(".metadata.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in metadata)
    )
    summary = {
        "status": "complete",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "records": len(records),
        "successes": sum(record.success for record in records),
        "by_duration": {
            str(duration): {
                "records": sum(record.duration == duration for record in records),
                "successes": sum(
                    record.duration == duration and record.success for record in records
                ),
            }
            for duration in sorted({record.duration for record in records})
        },
        "prefixes": dict(
            sorted(Counter(record_id.split(":cape-candidate:")[0] for record_id in record_ids).items())
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
