import os

import torch


def configure_cuda_toolchain(cuda_arch: str | None = None):
    conda_prefix = os.environ.get("CONDA_PREFIX")
    include_candidates = []

    if conda_prefix:
        os.environ["CUDA_HOME"] = conda_prefix
        conda_bin = os.path.join(conda_prefix, "bin")
        conda_nvcc = os.path.join(conda_bin, "nvcc")
        if os.path.exists(conda_nvcc):
            os.environ["CUDACXX"] = conda_nvcc

        path_items = os.environ.get("PATH", "").split(":")
        if conda_bin not in path_items:
            os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"

        include_candidates.extend(
            [
                os.path.join(conda_prefix, "include"),
                os.path.join(conda_prefix, "targets", "x86_64-linux", "include"),
            ]
        )

    include_candidates.extend(
        [
            os.path.join("/usr/local/cuda", "include"),
            os.path.join("/usr/local/cuda", "targets", "x86_64-linux", "include"),
        ]
    )
    include_candidates = [p for p in include_candidates if os.path.isdir(p)]
    if include_candidates:
        old = os.environ.get("CPLUS_INCLUDE_PATH", "")
        merged = include_candidates + ([old] if old else [])
        os.environ["CPLUS_INCLUDE_PATH"] = ":".join(dict.fromkeys(merged))

    if cuda_arch:
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch
    elif torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    return include_candidates
