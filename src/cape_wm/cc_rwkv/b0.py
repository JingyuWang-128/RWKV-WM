from __future__ import annotations

import torch
from torch import Tensor


def open_loop_rollout_latents(
    model: torch.nn.Module,
    context_latents: Tensor,
    future_action_blocks: Tensor,
    *,
    history_size: int | None = None,
) -> Tensor:
    """Run official LeWM autoregressively without intermediate observations.

    Args:
        model: Official LeWM exposing ``action_encoder`` and ``predict``.
        context_latents: Observed context with shape ``[B, H, D]``.
        future_action_blocks: Strictly future blocks ``[B, T, A]``.
        history_size: Transformer context window. Defaults to checkpoint value.

    Returns:
        Predicted future latents with shape ``[B, T, D]``. Ground-truth future
        latents are never accepted by this function and therefore cannot leak
        into the free-running recurrence.
    """

    if context_latents.ndim != 3 or future_action_blocks.ndim != 3:
        raise ValueError("context_latents and future_action_blocks must be rank-3")
    if context_latents.size(0) != future_action_blocks.size(0):
        raise ValueError("context and action batch sizes must match")
    if context_latents.size(1) <= 0 or future_action_blocks.size(1) <= 0:
        raise ValueError("context and future horizon must be non-empty")
    if history_size is None:
        history_size = int(getattr(model.predictor, "num_frames", 3))
    if history_size <= 0:
        raise ValueError("history_size must be positive")

    action_embeddings = model.action_encoder(future_action_blocks)
    latent_history = list(context_latents.unbind(dim=1))
    predictions: list[Tensor] = []
    context_length = context_latents.size(1)
    for step in range(future_action_blocks.size(1)):
        end = context_length + step
        start = max(0, end - history_size)
        latent_window = torch.stack(latent_history[start:end], dim=1)
        # action index k is the block leaving latent frame k. M0 has one
        # observed context frame, so the first future action is aligned with it.
        action_start = max(0, step + 1 - history_size)
        action_window = action_embeddings[:, action_start : step + 1]
        if action_window.size(1) != latent_window.size(1):
            raise RuntimeError("latent/action windows are misaligned")
        prediction = model.predict(latent_window, action_window)[:, -1]
        predictions.append(prediction)
        latent_history.append(prediction)
    return torch.stack(predictions, dim=1)

