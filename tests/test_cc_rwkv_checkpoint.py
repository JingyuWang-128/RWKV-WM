import torch

from cape_wm.cc_rwkv.checkpoint import (
    load_b2_checkpoint,
    restore_m4_training_state,
    save_b2_checkpoint,
)
from cape_wm.cc_rwkv.predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor


def _model():
    return VanillaRWKV7WorldPredictor(
        RWKV7WorldModelConfig(
            latent_dim=6,
            action_dim=3,
            model_dim=8,
            num_layers=2,
            num_heads=2,
            channel_mlp_dim=32,
            decay_lora_dim=4,
            aaa_lora_dim=4,
            value_lora_dim=4,
            gate_lora_dim=4,
        )
    )


def test_b2_checkpoint_round_trip_and_provenance_guard(tmp_path):
    torch.manual_seed(17)
    model = _model()
    path = tmp_path / "b2.pt"
    save_b2_checkpoint(
        path,
        model,
        global_step=12,
        encoder_provenance={"source_weights_sha256": "encoder-ok"},
        data_manifest_sha256="data-ok",
        training_config={"output": tmp_path},
    )
    restored, payload = load_b2_checkpoint(
        path,
        expected_encoder_sha256="encoder-ok",
        expected_data_manifest_sha256="data-ok",
    )
    assert payload["global_step"] == 12
    assert payload["training_config"]["output"] == str(tmp_path)
    assert not payload["invalid_for_paper"]
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    try:
        load_b2_checkpoint(path, expected_encoder_sha256="wrong")
    except ValueError as error:
        assert "provenance" in str(error)
    else:
        raise AssertionError("provenance mismatch was accepted")
    _, diagnostic = load_b2_checkpoint(
        path,
        expected_encoder_sha256="wrong",
        allow_provenance_mismatch=True,
    )
    assert diagnostic["invalid_for_paper"]


def test_m4_cuda_rng_restore_supplies_cpu_byte_tensors(monkeypatch):
    captured = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: captured.extend(states))
    restore_m4_training_state(
        {
            "rng_states": {
                "torch": torch.get_rng_state(),
                "cuda": [torch.arange(8, dtype=torch.uint8)],
            }
        }
    )
    assert len(captured) == 1
    assert captured[0].device.type == "cpu"
    assert captured[0].dtype == torch.uint8
