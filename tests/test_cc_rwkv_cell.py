import torch
from torch.nn import functional as F

from cape_wm.cc_rwkv.cell import (
    RWKV7_DECAY_SCALE,
    RWKV7_GROUP_NORM_EPS,
    RWKV7BlockConfig,
    RWKV7TimeMix,
    rwkv7_matrix_step,
)


def test_decomposed_rwkv7_recurrence_matches_explicit_matrix_oracle():
    torch.manual_seed(1)
    batch, heads, width = 2, 3, 4
    matrix = torch.randn(batch, heads, width, width)
    decay = torch.sigmoid(torch.randn(batch, heads, width))
    key = torch.randn(batch, heads, width)
    value = torch.randn(batch, heads, width)
    normalized_key = F.normalize(torch.randn(batch, heads, width), dim=-1)
    learning_rate = torch.sigmoid(torch.randn(batch, heads, width))

    actual = rwkv7_matrix_step(
        matrix, decay, key, value, normalized_key, learning_rate
    )
    transition = torch.diag_embed(decay) - normalized_key.unsqueeze(-1) * (
        normalized_key * learning_rate
    ).unsqueeze(-2)
    write = value.unsqueeze(-1) * key.unsqueeze(-2)
    expected = matrix @ transition + write
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def _official_time_mix_oracle(module, x, previous_x, matrix, v_first):
    """Independent transcription of the official RWKV-7 RNN reference."""

    cfg = module.config
    batch, channels = x.shape
    heads, width = cfg.num_heads, cfg.head_dim
    difference = previous_x - x
    xr = x + difference * module.x_r
    xw = x + difference * module.x_w
    xk = x + difference * module.x_k
    xv = x + difference * module.x_v
    xa = x + difference * module.x_a
    xg = x + difference * module.x_g
    receptance = F.linear(xr, module.receptance.weight)
    decay_logit = module.w0 + torch.tanh(xw @ module.w1) @ module.w2
    decay = torch.exp(-RWKV7_DECAY_SCALE * torch.sigmoid(decay_logit.float()))
    key = F.linear(xk, module.key.weight)
    value = F.linear(xv, module.value.weight)
    if cfg.layer_id == 0:
        v_first = value
    else:
        value = value + (v_first - value) * torch.sigmoid(
            module.v0 + (xv @ module.v1) @ module.v2
        )
    learning_rate = torch.sigmoid(module.a0 + (xa @ module.a1) @ module.a2)
    gate = torch.sigmoid(xg @ module.g1) @ module.g2
    normalized_key = F.normalize(
        (key * module.k_k).view(batch, heads, width), dim=-1, p=2, eps=1e-12
    )
    key = key * (1 + (learning_rate - 1) * module.k_a)
    r_h = receptance.view(batch, heads, width)
    k_h = key.view(batch, heads, width)
    v_h = value.view(batch, heads, width)
    a_h = learning_rate.view(batch, heads, width)
    decay_h = decay.view(batch, heads, width)
    transition = torch.diag_embed(decay_h) - normalized_key.unsqueeze(-1) * (
        normalized_key * a_h
    ).unsqueeze(-2)
    next_matrix = matrix @ transition + v_h.unsqueeze(-1) * k_h.unsqueeze(-2)
    output = next_matrix @ r_h.unsqueeze(-1)
    output = F.group_norm(
        output.to(x.dtype).reshape(batch, channels),
        num_groups=heads,
        weight=module.group_norm.weight,
        bias=module.group_norm.bias,
        eps=RWKV7_GROUP_NORM_EPS,
    )
    output = output + (
        (r_h * k_h * module.r_k).sum(dim=-1, keepdim=True) * v_h
    ).reshape(batch, channels)
    output = F.linear(output * gate, module.output.weight)
    return output, next_matrix, v_first


def test_time_mix_matches_official_equation_oracle_for_both_value_paths():
    torch.manual_seed(3)
    for layer_id in (0, 1):
        config = RWKV7BlockConfig(
            model_dim=8,
            num_layers=2,
            num_heads=2,
            layer_id=layer_id,
            channel_mlp_dim=32,
            decay_lora_dim=4,
            aaa_lora_dim=4,
            value_lora_dim=4,
            gate_lora_dim=4,
        )
        module = RWKV7TimeMix(config)
        module.output.weight.data.normal_(std=0.1)
        batch = 3
        x = torch.randn(batch, 8)
        previous_x = torch.randn(batch, 8)
        matrix = torch.randn(batch, 2, 4, 4)
        v_first = torch.randn(batch, 8) if layer_id else None
        actual, actual_shift, actual_matrix, actual_v_first, _ = module(
            x, previous_x, matrix, v_first
        )
        expected, expected_matrix, expected_v_first = _official_time_mix_oracle(
            module, x, previous_x, matrix, v_first
        )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual_matrix, expected_matrix, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual_shift, x)
        torch.testing.assert_close(actual_v_first, expected_v_first)


def test_time_mix_gradients_are_finite():
    torch.manual_seed(5)
    config = RWKV7BlockConfig(
        model_dim=8,
        num_layers=2,
        num_heads=2,
        layer_id=0,
        channel_mlp_dim=32,
        decay_lora_dim=4,
        aaa_lora_dim=4,
        value_lora_dim=4,
        gate_lora_dim=4,
    )
    module = RWKV7TimeMix(config)
    module.output.weight.data.normal_(std=0.1)
    output = module(
        torch.randn(2, 8, requires_grad=True),
        torch.randn(2, 8),
        torch.zeros(2, 2, 4, 4),
        None,
    )[0]
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

