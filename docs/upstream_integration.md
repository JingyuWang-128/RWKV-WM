# Pinned upstream integration notes

Validated on 2026-08-20 against:

- LeWorldModel `bf04d3e8c3752ac24f3692fbc5f4cf50209fa765`;
- stable-worldmodel `addbab40377da680dadbfbc90250fe749f6f57e3`.

`scripts/download_official_assets.sh` fetches only these commits at depth one
and writes the resolved hashes. It refuses to replace a modified checkout.

## LeWorldModel tensor contract

The pinned `JEPA.encode` expects `info["pixels"]` with shape `[B,T,C,H,W]` and
adds `info["emb"]` with shape `[B,T,D]`. `action_encoder` maps `[B,T,A]` to
action embeddings. `predict(emb, act_emb)` returns a prediction at every input
time. `LeWorldModelAdapter` follows these methods directly and truncates
autoregressive history to the configured size (three by default).

The newer LeWM implementation vendored in stable-worldmodel accepts strictly
future action candidates plus optional real `action_history`. CAPE-WM's minimal
research interface starts from the current latent, so its adapter implements
the equivalent `H=1` path. Experiments that add observation history constitute
a separate ablation and must give the same context to every baseline.

## stable-worldmodel policy contract

The pinned `World.set_policy` immediately calls `policy.set_env(env_pool)` and
the rollout loop calls `policy.get_action(info)`. World info is vectorized and
normally contains `pixels` and `goal` with a leading environment and time axis.
`StableWorldModelCAPEPolicy` implements both methods, keeps one stateful planner
per environment, handles `_needs_flush`, returns zero actions for terminated
environments, and accepts an optional action inverse-transform. It deliberately
does not import stable-worldmodel, keeping the core planner usable on training
or login nodes without environment extras.

Dataset-driven final evaluation should use stable-worldmodel's `World.evaluate`
with frozen `episodes_idx`, `start_steps`, `goal_offset`, and `eval_budget` from
the generated test-pair manifest. Environment state and goal state may be used
only by success/error evaluators and calibration-label collection.
