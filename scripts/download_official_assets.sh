#!/usr/bin/env bash
set -euo pipefail

asset_root="${CAPE_WM_ASSET_ROOT:-artifacts/assets}"
mkdir -p "${asset_root}/source" "${asset_root}/checkpoints" "${asset_root}/data"

# These refs are written to the provenance file after checkout. Set explicit
# hashes in CI/paper runs with LEWM_REF and STABLE_WM_REF.
lewm_ref="${LEWM_REF:-bf04d3e8c3752ac24f3692fbc5f4cf50209fa765}"
stable_wm_ref="${STABLE_WM_REF:-addbab40377da680dadbfbc90250fe749f6f57e3}"

fetch_locked_checkout() {
  local repository_url="$1"
  local destination="$2"
  local revision="$3"
  local label="$4"
  if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "${destination}"
    git -C "${destination}" init
    git -C "${destination}" remote add origin "${repository_url}"
  fi
  if [[ -n "$(git -C "${destination}" status --porcelain)" ]]; then
    printf '%s\n' "Refusing to replace a modified ${label} checkout." >&2
    exit 2
  fi
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach FETCH_HEAD
}

fetch_locked_checkout \
  https://github.com/hongqin/leworldmodel.git \
  "${asset_root}/source/leworldmodel" "${lewm_ref}" LeWorldModel
fetch_locked_checkout \
  https://github.com/galilai-group/stable-worldmodel.git \
  "${asset_root}/source/stable-worldmodel" "${stable_wm_ref}" stable-worldmodel

{
  git -C "${asset_root}/source/leworldmodel" rev-parse HEAD
  git -C "${asset_root}/source/stable-worldmodel" rev-parse HEAD
} > "${asset_root}/source/UPSTREAM_COMMITS.txt"

if [[ "${DOWNLOAD_LEWM_CHECKPOINTS:-0}" == "1" ]]; then
  if ! command -v hf >/dev/null 2>&1; then
    printf '%s\n' "Install huggingface_hub so the 'hf' command is available." >&2
    exit 2
  fi
  for environment in pusht cube tworooms; do
    hf download "quentinll/lewm-${environment}" \
      --local-dir "${asset_root}/checkpoints/lewm-${environment}"
  done
fi

printf '%s\n' \
  "Upstream source is ready under ${asset_root}/source." \
  "Use the official LeWorldModel download instructions for licensed checkpoints and datasets." \
  "Record every downloaded file hash before final evaluation."
