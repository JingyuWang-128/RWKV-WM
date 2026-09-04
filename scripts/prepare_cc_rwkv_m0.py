#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cape_wm.cc_rwkv.protocol import (
    HorizonSpec,
    build_tworoom_open_loop_arrays,
    file_sha256,
    load_frozen_test_episode_ids,
)
from cape_wm.config import load_config


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the CC-RWKV M0 TwoRoom protocol")
    parser.add_argument("--config", type=Path, default=Path("configs/cc_rwkv/tworoom.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    protocol = config["protocol"]
    assets = {key: Path(value).resolve() for key, value in config["assets"].items()}
    output = (args.output or Path(config["paths"]["manifest_dir"])).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    horizon = HorizonSpec(
        primitive_horizons=tuple(map(int, protocol["primitive_horizons"])),
        action_block=int(protocol["action_block"]),
        observation_stride=int(protocol["observation_stride"]),
    )
    test_episodes = load_frozen_test_episode_ids(assets["frozen_test_episodes"])
    arrays, manifest = build_tworoom_open_loop_arrays(
        assets["data"],
        test_episodes,
        horizon,
        sample_count=int(args.samples or protocol["open_loop_samples"]),
        selection_seed=int(protocol["split_seed"]),
        history_frames=int(protocol["history_frames"]),
    )
    np.savez_compressed(output / "open_loop_manifest.npz", **arrays)

    started = time.time()
    asset_hashes = {name: file_sha256(path) for name, path in assets.items()}
    manifest.update(
        {
            "status": "complete",
            "schema_version": protocol["schema_version"],
            "experiment": config["experiment"],
            "assets": {name: str(path) for name, path in assets.items()},
            "asset_hashes": asset_hashes,
            "manifest_npz_sha256": file_sha256(output / "open_loop_manifest.npz"),
            "created_at_unix": time.time(),
        }
    )
    _write_json(output / "manifest.json", manifest)
    provenance = {
        "status": "complete",
        "schema_version": protocol["schema_version"],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "base_config_sha256": file_sha256(config_path.parent / "base.yaml"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "asset_hashes": asset_hashes,
        "horizon": horizon.as_dict(),
        "manifest_npz_sha256": manifest["manifest_npz_sha256"],
        "hash_wall_seconds": time.time() - started,
    }
    _write_json(output / "provenance.json", provenance)
    print(json.dumps({"output": str(output), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

