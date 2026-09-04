from __future__ import annotations

import argparse
import json
from pathlib import Path

from cape_wm.cc_rwkv.evaluation import write_evaluation_artifacts
from cape_wm.cc_rwkv.m4_data import InMemoryBranchSplit, m4_data_provenance
from cape_wm.cc_rwkv.trainer import M4Trainer
from cape_wm.cc_rwkv.training import effect_norm_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one M4 CC-RWKV checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/tworoom/mvp5000/branches.h5"),
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=Path("artifacts/cache/cc_rwkv/tworoom/mvp5000/manifest.json"),
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("artifacts/results/cc_rwkv/tworoom/m0/protocol/manifest.json"),
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_provenance, encoder_provenance = m4_data_provenance(
        args.data, args.data_manifest, args.protocol_manifest
    )
    train = InMemoryBranchSplit(args.data, split="train", limit=args.limit)
    evaluation = InMemoryBranchSplit(args.data, split=args.split, limit=args.limit)
    threshold = effect_norm_threshold(train.tensors["branch_latents"])
    expected = {**data_provenance, **encoder_provenance}
    trainer = M4Trainer.resume(
        args.checkpoint,
        device=args.device,
        effect_threshold=threshold,
        expected_provenance=expected,
    )
    available = evaluation.tensors["branch_actions_raw"].shape[2]
    summary = write_evaluation_artifacts(
        args.output,
        trainer,
        list(evaluation.batches(args.batch_size, device=args.device)),
        list(train.batches(args.batch_size, device=args.device)),
        horizons=tuple(level for level in (1, 2, 4, 10, 20) if level <= available),
        provenance={
            "data": data_provenance,
            "encoder": encoder_provenance,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
        },
        fairness_status="single-checkpoint evaluation",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
