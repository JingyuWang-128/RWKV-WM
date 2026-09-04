#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from cape_wm.config import load_config
from cape_wm.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CAPE-WM execution host")
    parser.add_argument("--require-gpus", type=int, default=0)
    parser.add_argument("--minimum-vram-gb", type=float, default=0.0)
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--device")
    args = parser.parse_args()

    import torch

    runtime = load_config(args.runtime_config)["runtime"]
    selected_device = resolve_device(args.device or runtime["device"])

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "vram_gb": properties.total_memory / 1024**3,
                "capability": [properties.major, properties.minor],
            }
        )
    packages = {}
    for name in ("cape-wm", "torch", "numpy", "scipy", "stable-worldmodel"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "selected_device": str(selected_device),
        "gpus": devices,
        "packages": packages,
    }
    print(json.dumps(report, indent=2))

    errors = []
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        errors.append("Python must be 3.10, 3.11, or 3.12")
    if len(devices) < args.require_gpus:
        errors.append(f"required {args.require_gpus} GPUs, found {len(devices)}")
    undersized = [
        device
        for device in devices[: args.require_gpus]
        if device["vram_gb"] < args.minimum_vram_gb
    ]
    if undersized:
        errors.append(f"GPUs below {args.minimum_vram_gb} GiB: {undersized}")
    if errors:
        raise SystemExit("environment validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
