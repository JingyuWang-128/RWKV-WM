from types import SimpleNamespace

import torch
from torch import nn

from cape_wm.cc_rwkv.lewm import FrozenLeWMEncoder


class _FakeEncoder(nn.Module):
    def forward(self, images, interpolate_pos_encoding=True):
        assert interpolate_pos_encoding
        pooled = images.mean(dim=(-1, -2))
        return SimpleNamespace(last_hidden_state=torch.stack((pooled, pooled + 1), dim=1))


class _FakeOfficialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.projector = nn.Linear(3, 5)
        self.predictor = nn.Linear(5, 5)


def test_frozen_lewm_encoder_uses_only_encoder_projector_and_normalizes_blocks():
    official = _FakeOfficialModel()
    wrapper = FrozenLeWMEncoder.from_official_model(
        official,
        action_mean=torch.tensor([1.0, -1.0]),
        action_scale=torch.tensor([2.0, 4.0]),
    )
    images = torch.randn(2, 3, 4, 4)
    latent = wrapper.encode_images(images)
    assert latent.shape == (2, 1, 5)
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
    normalized = wrapper.normalize_action(
        torch.tensor([[1.0, 3.0, 5.0, -1.0]])
    )
    torch.testing.assert_close(normalized, torch.tensor([[0.0, 1.0, 2.0, 0.0]]))
    wrapper.train()
    assert not wrapper.encoder.training
    assert not wrapper.projector.training


def test_frozen_lewm_encoder_rejects_invalid_action_scale():
    try:
        FrozenLeWMEncoder(
            _FakeEncoder(),
            nn.Identity(),
            action_mean=torch.zeros(2),
            action_scale=torch.tensor([1.0, 0.0]),
        )
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid action scale was accepted")

