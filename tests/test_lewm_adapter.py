import numpy as np
import torch

from cape_wm.adapters import LeWorldModelAdapter


class FakeJEPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode(self, info):
        pixels = info["pixels"]
        pooled = pixels.mean(dim=(-1, -2))
        info["emb"] = pooled
        return info

    def action_encoder(self, actions):
        return actions

    def predict(self, states, actions):
        delta = torch.nn.functional.pad(actions[..., :2], (0, 1))
        return states[:, -actions.shape[1] :] + delta


def test_lewm_adapter_uses_official_encode_predict_path():
    adapter = LeWorldModelAdapter(FakeJEPA(), action_shape=(2,), device="cpu")
    image = np.ones((3, 4, 4), dtype=np.float32)
    latent = adapter.encode(image)
    assert latent.shape == (3,)
    actions = np.asarray([[0.1, -0.2], [0.2, 0.1]], dtype=np.float32)
    path = adapter.rollout(latent, actions)
    assert path.shape == (2, 3)
    assert np.allclose(path[0], [1.1, 0.8, 1.0])


def test_lewm_adapter_batch_encode_matches_individual_encoding():
    adapter = LeWorldModelAdapter(FakeJEPA(), action_shape=(2,), device="cpu")
    images = np.stack(
        [
            np.ones((3, 4, 4), dtype=np.float32),
            np.full((3, 4, 4), 2.0, dtype=np.float32),
        ]
    )

    batch = adapter.batch_encode(images)

    assert batch.shape == (2, 3)
    assert np.allclose(batch[0], adapter.encode(images[0]))
    assert np.allclose(batch[1], adapter.encode(images[1]))


class BlockJEPA(FakeJEPA):
    def action_encoder(self, actions):
        assert actions.shape[-1] == 4
        return actions

    def predict(self, states, actions):
        pairs = actions.reshape(*actions.shape[:-1], 2, 2)
        delta = torch.nn.functional.pad(pairs.sum(dim=-2), (0, 1))
        return states[:, -actions.shape[1] :] + delta


def test_lewm_adapter_groups_primitive_actions_into_checkpoint_blocks():
    adapter = LeWorldModelAdapter(
        BlockJEPA(), action_shape=(2,), action_block=2, device="cpu"
    )
    latent = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    actions = np.asarray(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
    )
    path = adapter.rollout(latent, actions)
    assert path.shape == (3, 3)
    assert np.allclose(path[0], [1.4, 1.6, 1.0])
    assert np.allclose(path[1], path[0])
    assert np.allclose(path[2], [1.9, 2.2, 1.0])
    tensor_path = adapter.batch_rollout_tensor(
        torch.from_numpy(latent), torch.from_numpy(actions[None])
    )
    assert torch.equal(tensor_path, torch.from_numpy(path[None]))
