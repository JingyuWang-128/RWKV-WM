from types import SimpleNamespace

import numpy as np
import torch

from cape_wm.tworoom_audit import TwoRoomSameCandidateAuditPolicy


def _value(value):
    return SimpleNamespace(value=torch.tensor(value))


def _env(wall_axis: int = 1):
    return SimpleNamespace(
        wall_axis=wall_axis,
        WALL_CENTER=128.0,
        num_doors=1,
        door_positions=torch.tensor([64.0]),
        door_sizes=torch.tensor([20.0]),
        variation_space={"agent": {"radius": _value(5.0)}},
    )


def test_oracle_path_is_direct_within_one_room():
    env = _env()
    start = np.asarray([20.0, 20.0], dtype=np.float32)
    target = np.asarray([50.0, 60.0], dtype=np.float32)

    distance = TwoRoomSameCandidateAuditPolicy._oracle_path_length(env, start, target)

    assert distance == 50.0


def test_oracle_path_crosses_registered_door():
    env = _env()
    start = np.asarray([28.0, 64.0], dtype=np.float32)
    target = np.asarray([228.0, 64.0], dtype=np.float32)

    distance = TwoRoomSameCandidateAuditPolicy._oracle_path_length(env, start, target)

    assert distance == 200.0
