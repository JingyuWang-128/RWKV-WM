"""Optional backbone adapters.

Imports are lazy so the NumPy-only planner and toy smoke test remain usable on
login nodes where PyTorch/CUDA are intentionally absent.
"""

__all__ = ["LeWorldModelAdapter", "TorchLatentWorldModelAdapter"]


def __getattr__(name: str):
    if name == "LeWorldModelAdapter":
        from .lewm import LeWorldModelAdapter

        return LeWorldModelAdapter
    if name == "TorchLatentWorldModelAdapter":
        from .torch_adapter import TorchLatentWorldModelAdapter

        return TorchLatentWorldModelAdapter
    raise AttributeError(name)
