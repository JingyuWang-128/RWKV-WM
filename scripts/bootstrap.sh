#!/usr/bin/env bash
set -euo pipefail

python_bin="${CAPE_WM_PYTHON:-python3.10}"
venv_path="${CAPE_WM_VENV:-.venv}"
torch_backend="${CAPE_WM_TORCH_BACKEND:-auto}"

"${python_bin}" -m venv "${venv_path}"
"${venv_path}/bin/python" -m pip install --upgrade pip
if [[ "${torch_backend}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    torch_backend="cuda121"
  else
    torch_backend="cpu"
  fi
fi
case "${torch_backend}" in
  cuda121)
    "${venv_path}/bin/python" -m pip install torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121
    ;;
  cpu)
    "${venv_path}/bin/python" -m pip install torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  system)
    printf '%s\n' "Using the PyTorch build already installed in the environment."
    ;;
  *)
    printf '%s\n' "Unknown CAPE_WM_TORCH_BACKEND=${torch_backend}" >&2
    exit 2
    ;;
esac
"${venv_path}/bin/python" -m pip install -e '.[dev]'
mkdir -p artifacts
"${venv_path}/bin/python" -m pip freeze > artifacts/resolved-requirements.txt
"${venv_path}/bin/pytest"
"${venv_path}/bin/cape-wm-smoke"
