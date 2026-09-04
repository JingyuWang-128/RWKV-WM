from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cape_wm.cc_rwkv.lewm import FrozenLeWMEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 real-checkpoint LeWM bridge audit")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--m0-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import stable_worldmodel as swm

    arrays = np.load(args.m0_manifest, allow_pickle=False)
    action_mean = torch.from_numpy(np.asarray(arrays["action_mean"], dtype=np.float32))
    action_scale = torch.from_numpy(np.asarray(arrays["action_scale"], dtype=np.float32))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    official = swm.wm.utils.load_pretrained(
        str(args.weights), cache_dir=str(args.cache_dir)
    ).to(args.device)
    bridge = FrozenLeWMEncoder.from_official_model(
        official,
        action_mean=action_mean,
        action_scale=action_scale,
        source_weights=args.weights,
    ).to(args.device)
    image = torch.zeros(1, 3, args.image_size, args.image_size, device=args.device)
    latent = bridge.encode_images(image)
    normalized = bridge.normalize_action(torch.zeros(1, 10, device=args.device))
    trainable = sum(
        parameter.numel() for parameter in bridge.parameters() if parameter.requires_grad
    )
    summary = {
        "schema_version": "cc_rwkv_m1_lewm_bridge_v1",
        "weights": str(args.weights),
        "weights_sha256": bridge.source_weights_sha256,
        "latent_shape": list(latent.shape),
        "latent_finite": bool(torch.isfinite(latent).all()),
        "normalized_action_shape": list(normalized.shape),
        "normalized_action_finite": bool(torch.isfinite(normalized).all()),
        "trainable_encoder_projector_parameters": trainable,
        "action_mean": action_mean.tolist(),
        "action_scale": action_scale.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
