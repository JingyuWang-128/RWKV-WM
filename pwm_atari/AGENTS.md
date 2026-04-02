<!-- # AGENTS.md

This repository implements RWKV-based RSSM world models.

Rules for AI agents:

## Architecture
- Do NOT change RSSM semantics.
- Do NOT modify stochastic latent logic.
- Do NOT modify imagine_data / img_step behavior.
- Deterministic dynamics = RWKV only.

## Stage Policy
- Stage 0: PyTorch RNN-mode
- Stage 1: CUDA kernel in parallel_observe only
- No kernel mask
- No padding segments

## Reset
- Use segmented reset by is_first
- No soft reset
- No mixing

## Performance
- Prioritize correctness over speed
- Do not introduce compile/fused paths unless requested

## Training
- Keep KL / unimix / ste_sample unchanged
- Keep replay buffer unchanged

When unsure, ask before changing core logic. -->
