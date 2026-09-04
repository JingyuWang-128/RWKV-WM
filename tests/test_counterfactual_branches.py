import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_worldmodel")

from cape_wm.cc_rwkv.branches import (  # noqa: E402
    ActionDelayTwoRoomAdapter,
    ActionSupportFilter,
    BranchType,
    TwoRoomBranchAdapter,
    build_branch_actions,
    rollout_branches,
    snapshot_sha256,
)


def _env():
    return gym.make("swm/TwoRoom-v1", render_mode="rgb_array").unwrapped


def test_tworoom_snapshot_restore_is_exact_for_render_state_and_next_step():
    adapter = TwoRoomBranchAdapter(_env(), drift=np.asarray([0.25, -0.25]))
    adapter.reset(
        seed=3,
        options={
            "state": np.asarray([50.0, 70.0], dtype=np.float32),
            "target_state": np.asarray([170.0, 150.0], dtype=np.float32),
        },
    )
    adapter.step(np.asarray([0.2, 0.4], dtype=np.float32))
    snapshot = adapter.snapshot()
    before_hash = snapshot_sha256(snapshot)
    before_render = adapter.render()
    action = np.asarray([0.7, -0.3], dtype=np.float32)
    expected = adapter.step(action)
    expected_render = adapter.render()
    expected_state = adapter.state_vector()

    adapter.step(np.asarray([-1.0, 1.0], dtype=np.float32))
    adapter.restore(snapshot)
    assert snapshot_sha256(adapter.snapshot()) == before_hash
    assert np.array_equal(adapter.render(), before_render)
    actual = adapter.step(action)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(adapter.render(), expected_render)
    assert np.array_equal(adapter.state_vector(), expected_state)
    adapter.close()


def test_action_delay_snapshot_preserves_fifo_and_executes_oldest_action():
    adapter = ActionDelayTwoRoomAdapter(_env(), delay=5)
    history = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    adapter.reset(
        seed=5,
        options={
            "state": np.asarray([50.0, 70.0], dtype=np.float32),
            "target_state": np.asarray([170.0, 150.0], dtype=np.float32),
            "action_history": history,
        },
    )
    snapshot = adapter.snapshot()
    start = adapter.env.agent_position.numpy().copy()
    adapter.step(np.asarray([-0.2, -0.2], dtype=np.float32))
    # The first submitted action does not execute; history[0] does.
    assert adapter.env.agent_position[0] > start[0]
    expected_state = adapter.state_vector().copy()
    adapter.restore(snapshot)
    adapter.step(np.asarray([-0.2, -0.2], dtype=np.float32))
    assert np.array_equal(adapter.state_vector(), expected_state)
    adapter.close()


def test_branch_definitions_share_suffix_and_local_action_stays_supported():
    rng = np.random.default_rng(7)
    behavior = rng.uniform(-1, 1, size=(2000, 2)).astype(np.float32)
    support = ActionSupportFilter(behavior)
    factual = rng.uniform(-0.7, 0.7, size=(4, 10)).astype(np.float32)
    actions, scores = build_branch_actions(factual, support, rng=rng)
    assert np.all(actions[BranchType.REFERENCE] == 0)
    assert np.array_equal(actions[BranchType.FACTUAL], factual)
    assert np.all(actions[BranchType.PULSE_NOOP, 0] == 0)
    assert np.array_equal(actions[BranchType.PULSE_NOOP, 1:], factual[1:])
    assert np.array_equal(actions[BranchType.PULSE_LOCAL, 1:], factual[1:])
    assert not np.array_equal(actions[BranchType.PULSE_LOCAL, 0], factual[0])
    assert scores[BranchType.PULSE_LOCAL, 0] >= support.threshold


def test_rollout_branches_restores_identical_initial_state_without_aliasing():
    rng = np.random.default_rng(11)
    support = ActionSupportFilter(rng.uniform(-1, 1, size=(2000, 2)))
    adapter = TwoRoomBranchAdapter(_env())
    adapter.reset(
        seed=11,
        options={
            "state": np.asarray([50.0, 70.0], dtype=np.float32),
            "target_state": np.asarray([170.0, 150.0], dtype=np.float32),
        },
    )
    factual = rng.uniform(-0.5, 0.5, size=(2, 10)).astype(np.float32)
    result = rollout_branches(adapter, factual, support, action_block=5, rng=rng)
    assert result.images.shape == (4, 3, 224, 224, 3)
    assert result.states.shape == (4, 3, 2)
    assert result.restore_consistent
    assert all(np.array_equal(result.images[0, 0], result.images[k, 0]) for k in range(4))
    assert np.array_equal(result.actions[2, 1:], result.actions[1, 1:])
    adapter.close()
