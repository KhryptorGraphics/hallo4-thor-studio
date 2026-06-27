#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${HALLO4_ENV_NAME:-hallo4-thor}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required." >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -m pip install "huggingface_hub[cli]>=0.30.2"
conda run -n "${ENV_NAME}" huggingface-cli download fudan-generative-ai/hallo4 --local-dir ./pretrained_models
conda run -n "${ENV_NAME}" python scripts/thor_preflight.py --create-checkpoint-alias
