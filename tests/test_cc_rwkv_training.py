import torch

from cape_wm.cc_rwkv.predictor import RWKV7WorldModelConfig, VanillaRWKV7WorldPredictor
from cape_wm.cc_rwkv.training import free_running_b2_loss, teacher_forced_b2_loss


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


def test_b2_losses_are_finite_and_backpropagate_through_twenty_steps():
    torch.manual_seed(19)
    model = _model()
    latents = torch.randn(2, 21, 6)
    actions = torch.randn(2, 20, 3)
    loss = teacher_forced_b2_loss(model, latents, actions)
    loss = loss + free_running_b2_loss(model, latents, actions)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_teacher_forced_padding_mask_excludes_invalid_targets():
    torch.manual_seed(23)
    model = _model()
    latents = torch.randn(1, 4, 6)
    actions = torch.randn(1, 3, 3)
    mask = torch.tensor([[True, False, False]])
    base = teacher_forced_b2_loss(model, latents, actions, mask)
    changed = latents.clone()
    changed[:, 2:] = 1000
    modified = teacher_forced_b2_loss(model, changed, actions, mask)
    torch.testing.assert_close(base, modified)

