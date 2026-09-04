import pytest
import torch

from cape_wm.device import resolve_device


def test_auto_device_always_resolves_to_an_available_backend():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda", "xpu", "mps"}


def test_cpu_can_always_be_selected_explicitly():
    assert resolve_device("cpu") == torch.device("cpu")


def test_unavailable_cuda_fails_only_when_explicitly_requested():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("cuda")
