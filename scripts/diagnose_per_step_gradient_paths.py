"""Same forward prediction, different backward paths, diagnostics only (no updates)."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from diagnose_per_step_gradients import load_batch

from cape_wm.cc_rwkv.model_factory import build_rwkv_world_model
from cape_wm.cc_rwkv.per_step_training import per_step_loss
from cape_wm.cc_rwkv.state import RWKVMatrixState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = payload["run_config"]
    model = build_rwkv_world_model(
        cfg["method"], cfg["profile"], cfg["latent_dim"], cfg["action_dim"]
    ).eval()
    model.load_state_dict(payload["model"])
    generator = torch.Generator()
    generator.set_state(payload["rng_states"]["batch_generator"].cpu())
    with h5py.File(cfg["data"], "r") as handle:
        samples = handle["samples"]
        rows = np.flatnonzero(np.asarray(samples["split"]) == 0)
        batch = load_batch(samples, rows[torch.randint(len(rows), (8,), generator=generator)])
    original_step = model.step
    parameters = list(model.parameters())
    report = {"checkpoint": str(args.checkpoint), "batch_size": 8, "updates": 0, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for horizon in (1, 5, 20):
        reference_loss = None
        for mode in ("full", "no_latent_grad", "no_matrix_grad", "no_state_grad", "local_only"):

            def diagnostic_step(
                latent, action, state, reference_action=None, *, mode=mode, **kwargs
            ):
                if mode in ("no_latent_grad", "local_only"):
                    latent = latent.detach()
                if mode in ("no_matrix_grad", "no_state_grad", "local_only"):
                    state = RWKVMatrixState(
                        matrix=state.matrix.detach(),
                        time_shift=(
                            state.time_shift
                            if mode == "no_matrix_grad"
                            else state.time_shift.detach()
                        ),
                        channel_shift=(
                            state.channel_shift
                            if mode == "no_matrix_grad"
                            else state.channel_shift.detach()
                        ),
                        steps=state.steps,
                    )
                return original_step(latent, action, state, reference_action, **kwargs)

            model.step = diagnostic_step
            try:
                total, components = per_step_loss(
                    model,
                    batch,
                    horizon=horizon,
                    effect_threshold=payload["effect_threshold"],
                    effect_weight=cfg["effect_weight"],
                )
                base = components["factual_prediction"] + components["noop_prediction"]
                losses = (total, base)
                norms = []
                for loss in losses:
                    grads = torch.autograd.grad(
                        loss, parameters, retain_graph=True, allow_unused=True
                    )
                    norms.append(
                        float(
                            torch.sqrt(
                                sum(
                                    gradient.double().square().sum()
                                    for gradient in grads
                                    if gradient is not None
                                )
                            )
                        )
                    )
                value = float(total.detach())
                if reference_loss is None:
                    reference_loss = value
                assert abs(value - reference_loss) < 1e-6, "forward prediction changed"
                record = {
                    "horizon": horizon,
                    "mode": mode,
                    "loss": value,
                    "total_gradient": norms[0],
                    "base_gradient": norms[1],
                }
                report["results"].append(record)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
            finally:
                model.step = original_step


if __name__ == "__main__":
    main()
