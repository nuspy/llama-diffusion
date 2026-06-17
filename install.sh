#!/usr/bin/env bash
# Entry point — install the DiffusionGemma engine (Linux / macOS).
# Usage: ./install.sh [MODEL_FILE] [MAX_CONTEXT]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_FILE="${1:-diffusiongemma-26B-A4B-it-Q6_K.gguf}"
MAX_CTX="${2:-32768}"
REPO="unsloth/diffusiongemma-26B-A4B-it-GGUF"

# 1. diffusion-capable llama.cpp fork (PR #24423)
if [ ! -d engine/llama.cpp/.git ]; then
  echo ">> cloning llama.cpp fork (PR #24423)..."
  git clone --depth=1 https://github.com/ggml-org/llama.cpp engine/llama.cpp
  git -C engine/llama.cpp fetch --depth=1 origin pull/24423/head:diffusion-gemma
  git -C engine/llama.cpp checkout diffusion-gemma
fi

# 1b. apply the incremental-decode fix (NOT upstream): fast long prompts + no crash at small ubatch
if [ -f patches/diffusion-gemma-incremental-decode.patch ]; then
  if git -C engine/llama.cpp apply --check patches/diffusion-gemma-incremental-decode.patch 2>/dev/null; then
    git -C engine/llama.cpp apply patches/diffusion-gemma-incremental-decode.patch
    echo ">> applied incremental-decode patch"
  else
    echo ">> incremental-decode patch already applied or N/A (skip)"
  fi
fi

# 2. build the CUDA engine
echo ">> building engine (CUDA)..."
cmake -S engine/llama.cpp -B engine/llama.cpp/build -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build engine/llama.cpp/build \
      --target llama-diffusion-gemma-visual-server llama-diffusion-cli -j"$(nproc)"

# 3. python dependencies
python3 -m pip install -r requirements.txt
python3 -m playwright install chromium || true

# 4. model download (resumable)
mkdir -p models
if [ ! -f "models/$MODEL_FILE" ]; then
  echo ">> downloading model: $MODEL_FILE"
  curl -L -C - -o "models/$MODEL_FILE" \
       "https://huggingface.co/$REPO/resolve/main/$MODEL_FILE"
fi

# 5. persist chosen model + context
printf '{"model_file": "%s", "maxtok": %s, "ubatch": 2048}\n' "$MODEL_FILE" "$MAX_CTX" > config.json
echo ">> done. start with: ./start_diffusion_engine.sh"
