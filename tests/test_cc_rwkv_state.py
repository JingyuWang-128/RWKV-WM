import torch

from cape_wm.cc_rwkv.state import RWKVMatrixState


def _state(batch=2):
    return RWKVMatrixState(
        matrix=torch.arange(batch * 2 * 2 * 4 * 4, dtype=torch.float32).reshape(
            batch, 2, 2, 4, 4
        ),
        time_shift=torch.randn(batch, 2, 8),
        channel_shift=torch.randn(batch, 2, 8),
        steps=torch.arange(batch),
    )


def test_rwkv_state_clone_branches_is_independent_and_ordered():
    state = _state()
    branches = state.clone_branches(3)
    assert branches.matrix.shape == (6, 2, 2, 4, 4)
    assert branches.steps.tolist() == [0, 0, 0, 1, 1, 1]
    assert torch.equal(branches.matrix[0], state.matrix[0])
    branches.matrix[0, 0, 0, 0, 0] = -123
    assert state.matrix[0, 0, 0, 0, 0] != -123
    assert branches.matrix[1, 0, 0, 0, 0] != -123


def test_rwkv_state_round_trip_and_index_select():
    state = _state()
    restored = RWKVMatrixState.from_dict(state.as_dict())
    assert torch.equal(restored.matrix, state.matrix)
    selected = restored.index_select(torch.tensor([1, 0], dtype=torch.long))
    assert torch.equal(selected.steps, torch.tensor([1, 0]))
    assert torch.equal(selected.time_shift[0], state.time_shift[1])


def test_rwkv_state_rejects_non_fp32_matrix():
    try:
        RWKVMatrixState(
            matrix=torch.zeros(1, 2, 2, 4, 4, dtype=torch.bfloat16),
            time_shift=torch.zeros(1, 2, 8),
            channel_shift=torch.zeros(1, 2, 8),
            steps=torch.zeros(1, dtype=torch.long),
        )
    except ValueError as error:
        assert "float32" in str(error)
    else:
        raise AssertionError("non-fp32 matrix state was accepted")

