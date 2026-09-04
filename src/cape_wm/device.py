from __future__ import annotations

import torch


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve a portable execution device without requiring a GPU.

    ``auto`` prefers CUDA, then Intel XPU, then Apple MPS, and finally CPU.
    Explicit device requests fail clearly when that backend is unavailable.
    """

    name = str(requested).lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return torch.device("xpu:0")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no usable CUDA device is visible")
    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise RuntimeError("XPU was requested but no usable Intel XPU is visible")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but no usable Apple GPU is visible")
    return device
