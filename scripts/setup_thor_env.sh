#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${HALLO4_ENV_NAME:-hallo4-thor}"
PYTHON_VERSION="${HALLO4_PYTHON_VERSION:-3.12}"
PYTORCH_INDEX="${HALLO4_PYTORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "ERROR: Thor setup must run on Linux aarch64, got $(uname -m)." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required for ${ENV_NAME}." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

tmp_requirements="$(mktemp)"
trap 'rm -f "${tmp_requirements}"' EXIT

grep -Ev '^(torch|torchvision|torchaudio|triton|decord|numpy|onnxruntime-gpu)==|^(flash_attn|wan) @ ' requirements.txt > "${tmp_requirements}"

if grep -E 'x86_64|linux_x86_64|file:///cpfs01' "${tmp_requirements}" >/dev/null; then
  echo "ERROR: filtered requirements still contain host-local or x86_64 wheel references." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-11.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url "${PYTORCH_INDEX}" \
  "torch==2.10.0+cu130" \
  "torchvision==0.25.0+cu130" \
  "torchaudio==2.10.0+cu130"
conda run -n "${ENV_NAME}" python -m pip install "triton==3.6.0"
conda run -n "${ENV_NAME}" python - <<'PY'
from pathlib import Path
import importlib.metadata as md

try:
    dist = md.distribution("nvidia-cusparselt-cu13")
except md.PackageNotFoundError:
    raise SystemExit(0)

root = Path(dist.locate_file(""))
wheel = root / "nvidia_cusparselt_cu13-0.8.0.dist-info" / "WHEEL"
if wheel.exists():
    text = wheel.read_text(encoding="utf-8")
    wheel.write_text(text.replace("manylinux2014_sbsa", "manylinux2014_aarch64"), encoding="utf-8")
PY

if [[ "${HALLO4_SKIP_DECORD:-0}" != "1" ]]; then
  conda install -y -n "${ENV_NAME}" -c conda-forge "decord==0.6.0"
fi

if [[ "${HALLO4_INSTALL_FULL_REQUIREMENTS:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${tmp_requirements}"
  conda run -n "${ENV_NAME}" python -m pip install "onnxruntime-gpu==1.24.0"
else
  conda run -n "${ENV_NAME}" python -m pip install --prefer-binary \
    "fastapi==0.115.12" \
    "httpx==0.28.1" \
    "uvicorn==0.34.2" \
    "python-multipart==0.0.20" \
    "aiofiles==23.2.1"
  conda run -n "${ENV_NAME}" python -m pip install --prefer-binary \
    "easydict==1.13" \
    "einops==0.8.1" \
    "icecream==2.1.4" \
    "tqdm==4.67.1" \
    "ftfy==6.3.1" \
    "imageio==2.37.0" \
    "imageio-ffmpeg==0.6.0" \
    "moviepy==1.0.3" \
    "ml_collections==1.1.0" \
    "PyYAML==6.0.2" \
    "safetensors==0.5.3" \
    "rotary-embedding-torch==0.6.5" \
    "matplotlib==3.10.1"
  conda run -n "${ENV_NAME}" python -m pip install --prefer-binary \
    "accelerate==1.6.0" \
    "diffusers==0.33.1" \
    "huggingface-hub[cli]==0.30.2" \
    "tokenizers==0.21.1" \
    "transformers==4.51.3" \
    "timm==1.0.15" \
    "scikit-image==0.25.2" \
    "opencv-python==4.11.0.86" \
    "tifffile==2025.3.30" \
    "onnx==1.17.0" \
    "onnx2torch==1.5.15" \
    "onnxruntime-gpu==1.24.0"
  conda run -n "${ENV_NAME}" python -m pip install --only-binary=:all: "flash-attn==2.8.4"
  conda run -n "${ENV_NAME}" python -m pip install --prefer-binary \
    "dashscope==1.25.23" \
    "gradio==5.0.0" \
    "huggingface-hub[cli]==0.30.2"
  conda run -n "${ENV_NAME}" python -m pip install --prefer-binary "audio-separator==0.30.2"
  conda run -n "${ENV_NAME}" python -m pip install "numpy==1.26.4" "tifffile==2025.3.30"
  conda run -n "${ENV_NAME}" python -m pip install --no-deps "wan @ git+https://github.com/Wan-Video/Wan2.1.git"
fi

if [[ "${HALLO4_BUILD_FLASH_ATTN:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python scripts/thor_preflight.py --skip-models --skip-imports decord audio_separator onnxruntime transformers diffusers
  conda run -n "${ENV_NAME}" python -m pip install --no-build-isolation flash-attn
else
  echo "Using verified flash-attn aarch64 wheel. Set HALLO4_BUILD_FLASH_ATTN=1 only to force a source rebuild."
fi

conda run -n "${ENV_NAME}" python scripts/thor_preflight.py --skip-models

cat <<EOF

Thor environment is ready enough for dependency validation.
Activate it with:
  conda activate ${ENV_NAME}

Download model assets with:
  bash scripts/download_models.sh

Start the studio backend with:
  uvicorn studio.backend.hallo4_studio.app:app --host 127.0.0.1 --port 8000
EOF
