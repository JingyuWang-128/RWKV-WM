from __future__ import annotations

import copy

import pytest
import torch

from cape_wm.cc_rwkv.per_step_checkpoint import (
    PER_STEP_CHECKPOINT_SCHEMA,
    restore_per_step_checkpoint,
    save_per_step_checkpoint,
)


def _run_config() -> dict:
    return {
        "method": "b2",
        "profile": "smoke",
        "data": "pairs.h5",
        "max_steps": 20,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "seed": 3,
        "effect_weight": 1.0,
        "curriculum": [1, 2],
        "teacher_forcing": False,
        "latent_dim": 4,
        "action_dim": 2,
    }


def test_periodic_checkpoint_round_trip_restores_optimizer_and_rng(tmp_path):
    torch.manual_seed(11)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    expected_model = copy.deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(1704)
    torch.rand(3, generator=generator)

    latest, numbered = save_per_step_checkpoint(
        tmp_path,
        model,
        optimizer,
        step=7,
        effect_threshold=0.25,
        history=[{"step": 7.0, "loss": float(loss.detach())}],
        batch_generator=generator,
        run_config=_run_config(),
        keep_numbered=True,
    )
    expected_random = torch.rand(4, generator=generator)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    payload = restore_per_step_checkpoint(
        latest,
        model,
        optimizer,
        generator,
        expected_run_config=_run_config(),
        expected_effect_threshold=0.25,
        map_location="cpu",
    )
    assert payload["schema_version"] == PER_STEP_CHECKPOINT_SCHEMA
    assert payload["step"] == 7
    assert numbered is not None and numbered.is_file()
    assert latest.is_file()
    assert not list(tmp_path.rglob("*.tmp-*"))
    for name, expected in expected_model.items():
        torch.testing.assert_close(model.state_dict()[name], expected)
    torch.testing.assert_close(torch.rand(4, generator=generator), expected_random)


def test_resume_rejects_incompatible_run_config(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(9)
    save_per_step_checkpoint(
        tmp_path,
        model,
        optimizer,
        step=2,
        effect_threshold=0.0,
        history=[],
        batch_generator=generator,
        run_config=_run_config(),
    )
    incompatible = {**_run_config(), "seed": 4}
    with pytest.raises(ValueError, match="seed"):
        restore_per_step_checkpoint(
            tmp_path / "latest.pt",
            model,
            optimizer,
            generator,
            expected_run_config=incompatible,
            expected_effect_threshold=0.0,
            map_location="cpu",
        )


def test_resume_rejects_old_curriculum_mask_checkpoint(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    generator = torch.Generator().manual_seed(0)
    save_per_step_checkpoint(
        tmp_path, model, optimizer, step=1, effect_threshold=0.0,
        history=[], batch_generator=generator, run_config=_run_config(),
    )
    with pytest.raises(ValueError, match="rollout_mask_policy"):
        restore_per_step_checkpoint(
            tmp_path / "latest.pt", model, optimizer, generator,
            expected_run_config={
                **_run_config(), "rollout_mask_policy": "position_plus_offset_lt_horizon_v2"
            },
            expected_effect_threshold=0.0, map_location="cpu",
        )


def test_checkpoint_save_rejects_nonfinite_model_state(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite step 2 checkpoint tensor: model"):
        save_per_step_checkpoint(
            tmp_path,
            model,
            optimizer,
            step=2,
            effect_threshold=0.0,
            history=[],
            batch_generator=generator,
            run_config=_run_config(),
        )
    assert not (tmp_path / "latest.pt").exists()


def test_checkpoint_restore_rejects_nonfinite_optimizer_state(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    generator = torch.Generator().manual_seed(9)
    latest, _ = save_per_step_checkpoint(
        tmp_path,
        model,
        optimizer,
        step=2,
        effect_threshold=0.0,
        history=[],
        batch_generator=generator,
        run_config=_run_config(),
    )
    payload = torch.load(latest, map_location="cpu", weights_only=True)
    first_state = next(iter(payload["optimizer"]["state"].values()))
    first_state["exp_avg"].reshape(-1)[0] = float("inf")
    torch.save(payload, latest)
    with pytest.raises(FloatingPointError, match="non-finite resume checkpoint tensor: optimizer"):
        restore_per_step_checkpoint(
            latest,
            model,
            optimizer,
            generator,
            expected_run_config=_run_config(),
            expected_effect_threshold=0.0,
            map_location="cpu",
        )
