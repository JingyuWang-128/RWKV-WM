import json

import h5py
import numpy as np
import pytest
import torch

from cape_wm.cc_rwkv.b0 import open_loop_rollout_latents
from cape_wm.cc_rwkv.protocol import (
    HorizonSpec,
    build_tworoom_open_loop_arrays,
    load_frozen_test_episode_ids,
)


def test_horizon_spec_maps_primitive_actions_to_model_transitions():
    spec = HorizonSpec((5, 10, 20, 50, 100), action_block=5, observation_stride=5)
    assert spec.model_horizons == (1, 2, 4, 10, 20)
    assert spec.max_model_horizon == 20
    with pytest.raises(ValueError, match="divisible"):
        HorizonSpec((5, 11), action_block=5, observation_stride=5)
    with pytest.raises(ValueError, match="observation_stride"):
        HorizonSpec((5, 10), action_block=5, observation_stride=1)


def _write_tiny_tworoom(path):
    lengths = np.asarray([11, 11, 11, 11], dtype=np.int32)
    offsets = np.asarray([0, 11, 22, 33], dtype=np.int64)
    episode = np.repeat(np.arange(4, dtype=np.int32), lengths)
    steps = np.tile(np.arange(11, dtype=np.int64), 4)
    actions = np.stack((episode + steps / 100, -episode - steps / 100), axis=-1).astype(
        np.float32
    )
    with h5py.File(path, "w") as handle:
        handle["ep_len"] = lengths
        handle["ep_offset"] = offsets
        handle["ep_idx"] = episode
        handle["step_idx"] = steps
        handle["action"] = actions
        handle["pos_agent"] = np.stack((steps, steps + 1), axis=-1).astype(np.float32)


def test_open_loop_manifest_is_deterministic_and_matches_official_normalizer(tmp_path):
    data = tmp_path / "tiny.h5"
    _write_tiny_tworoom(data)
    frozen = tmp_path / "test.jsonl"
    frozen.write_text(
        "\n".join(json.dumps({"episode_index": item}) for item in (2, 3)) + "\n"
    )
    episodes = load_frozen_test_episode_ids(frozen)
    spec = HorizonSpec((5, 10), action_block=5, observation_stride=5)
    left, left_meta = build_tworoom_open_loop_arrays(
        data, episodes, spec, sample_count=1, selection_seed=7
    )
    right, right_meta = build_tworoom_open_loop_arrays(
        data, episodes, spec, sample_count=1, selection_seed=7
    )
    assert all(np.array_equal(left[key], right[key]) for key in left)
    assert left_meta["array_hashes"] == right_meta["array_hashes"]
    assert left["rollout_rows"].shape == (1, 3)
    assert left["action_blocks_raw"].shape == (1, 2, 10)
    with h5py.File(data, "r") as handle:
        official_actions = np.asarray(handle["action"])
    assert np.allclose(left["action_mean"], official_actions.mean(axis=0))
    assert np.allclose(left["action_scale"], official_actions.std(axis=0))


class _FakePredictor:
    num_frames = 2


class _FakeLeWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = _FakePredictor()

    def action_encoder(self, actions):
        return actions

    def predict(self, latents, actions):
        return latents + actions


def test_b0_rollout_is_free_running_and_action_aligned():
    context = torch.tensor([[[1.0]]])
    actions = torch.tensor([[[2.0], [3.0], [4.0]]])
    prediction = open_loop_rollout_latents(_FakeLeWM(), context, actions)
    assert prediction.shape == (1, 3, 1)
    assert prediction.flatten().tolist() == [3.0, 6.0, 10.0]
