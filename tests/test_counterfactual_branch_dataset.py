from types import SimpleNamespace

import numpy as np

from cape_wm.cc_rwkv.branch_dataset import (
    BranchDatasetLayout,
    BranchHDF5Writer,
    CounterfactualBranchDataset,
    SnapshotSpec,
    freeze_episode_split,
    sample_snapshot_specs,
)


def test_episode_split_and_snapshot_sampling_are_deterministic_and_disjoint():
    episodes = np.arange(30)
    frozen = tuple(range(27, 30))
    left = freeze_episode_split(episodes, frozen, seed=7)
    right = freeze_episode_split(episodes, frozen, seed=7)
    assert left == right
    assert not (set(left["train"]) & set(left["validation"]))
    assert not (set(left["train"]) & set(left["test"]))
    offsets = np.arange(30) * 101
    lengths = np.full(30, 101)
    specs = sample_snapshot_specs(
        left,
        episodes,
        offsets,
        lengths,
        count=10,
        history_steps=3,
        future_steps=4,
        action_block=5,
        seed=11,
    )
    assert len(specs) == 10
    assert all(spec.start_step >= 15 and spec.start_step % 5 == 0 for spec in specs)
    assert len({spec.episode_id for spec in specs}) == 10
    offset_by_episode = {
        int(episode): int(offset) for episode, offset in zip(episodes, offsets, strict=True)
    }
    assert all(
        spec.start_row == offset_by_episode[spec.episode_id] + spec.start_step for spec in specs
    )


def test_multi_position_snapshot_sampling_uses_each_episode_offset():
    episodes = np.arange(12)
    offsets = np.arange(12) * 1000
    lengths = np.full(12, 101)
    split = {
        "train": tuple(range(8)),
        "validation": tuple(range(8, 10)),
        "test": tuple(range(10, 12)),
    }
    specs = sample_snapshot_specs(
        split,
        episodes,
        offsets,
        lengths,
        count=20,
        history_steps=3,
        future_steps=4,
        action_block=5,
        seed=17,
        allow_multiple_per_episode=True,
    )
    offset_by_episode = {
        int(episode): int(offset) for episode, offset in zip(episodes, offsets, strict=True)
    }
    assert len(specs) == 20
    assert len({spec.episode_id for spec in specs}) > 3
    assert all(
        spec.start_row == offset_by_episode[spec.episode_id] + spec.start_step for spec in specs
    )


def test_hdf5_writer_and_dataset_round_trip(tmp_path):
    path = tmp_path / "branches.h5"
    layout = BranchDatasetLayout(3, 2, 4, 2, 6, 4, 2, 1)
    rollout = SimpleNamespace(
        branch_type=np.arange(4, dtype=np.uint8),
        actions=np.zeros((4, 2, 4), dtype=np.float32),
        states=np.zeros((4, 3, 2), dtype=np.float32),
        support_score=np.zeros((4, 2), dtype=np.float32),
        restore_consistent=True,
        images=np.zeros((4, 3, 224, 224, 3), dtype=np.uint8),
    )
    with BranchHDF5Writer(path, layout, metadata={"environment": "test"}) as writer:
        for index, split in enumerate(("train", "validation", "test")):
            spec = SnapshotSpec(f"sample-{index}", index, 10, index * 100, split)
            writer.write(
                index,
                spec,
                snapshot_hash=f"hash-{index}",
                history_latents=np.ones((2, 6), dtype=np.float32) * index,
                history_actions=np.zeros((2, 4), dtype=np.float32),
                rollout=rollout,
                branch_latents=np.ones((4, 3, 6), dtype=np.float32) * index,
                audit=index == 0,
            )
    dataset = CounterfactualBranchDataset(path, expected_metadata={"environment": "test"})
    assert len(dataset) == 3
    item = dataset[1]
    assert item["sample_id"] == "sample-1"
    assert item["branch_latents"].shape == (4, 3, 6)
    dataset.close()
