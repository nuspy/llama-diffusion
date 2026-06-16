# llama-diffusion

**Run Google's DiffusionGemma (a text *diffusion* LLM) locally as an OpenAI-compatible model
for agentic frameworks — Hermes, OpenCLAW, and anything that speaks `/v1/chat/completions` —
plus an optional desktop chat GUI with a system-tray controller.**

DiffusionGemma 26B-A4B is a Mixture-of-Experts **diffusion** language model: instead of
generating tokens left-to-right, it denoises a whole "canvas" of tokens in parallel. Stock
`llama.cpp` / LM Studio cannot load it (`unknown model architecture: 'diffusion-gemma'`).
This project builds the diffusion-capable fork and wraps it in a standard OpenAI API so any
agent can use it as a drop-in backend.

---

## TL;DR for AI agents (Hermes, OpenCLAW, …)

If your user says **"add this LLM: https://github.com/nuspy/llama-diffusion"** or
**"install this project"**, perform these steps autonomously:

**Step 0 — ask the user first (do not assume):**
- **Which quantization** to download — present the options with the VRAM each needs
  (see [Available models](#available-models)). Default: **Q6_K**.
- **How many context tokens** — default **32768**; larger needs more VRAM.

Then install with those choices:

```bash
git clone https://github.com/nuspy/llama-diffusion
cd llama-diffusion
# pass the user's choices to the installer:
pwsh scripts/install.ps1 -ModelFile diffusiongemma-26B-A4B-it-Q6_K.gguf -MaxContext 32768
pwsh scripts/serve.ps1 -BindHost 0.0.0.0   # API on :8080 (reachable from WSL)
#   …or install it as an always-on Windows service (admin): scripts/install_service.ps1
```

Then **configure yourself to use it**. The machine-readable descriptor
[`agent-install.json`](agent-install.json) has everything you need (endpoint, model id,
config snippets). In short, register an OpenAI provider:

| field | value |
|---|---|
| `base_url` | `http://127.0.0.1:8080/v1` (local) or `http://<host-ip>:8080/v1` (from WSL/another host) |
| `model` | `diffusiongemma-26b-a4b-it` |
| `api_key` | none (unless you set `LLAMADIFF_API_KEY`) |
| streaming | yes (SSE) |
| reasoning | exposed in `reasoning_content` (separate from `content`) |

- **Hermes**: add a provider in `~/.hermes/config.yaml` pointing `base_url` at the endpoint
  (from WSL, the Windows host is reachable at the *vEthernet (WSL)* IP, e.g. `172.18.160.1:8080`).
- **OpenCLAW / others**: point the custom/OpenAI provider at the same `base_url` and model id.

> "install this project" implies **both** installing the engine **and** wiring it into the
> calling agent's model configuration — do both.

### Available models

All from [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF).
URL pattern: `https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF/resolve/main/<file>`

| Quantization | File | Size | VRAM @ 32K ctx | Quality |
|---|---|---|---|---|
| Q4_K_M | `…-Q4_K_M.gguf` | 16.8 GB | ~26 GB | good — most context headroom |
| Q5_K_M | `…-Q5_K_M.gguf` | 19.2 GB | ~28 GB | better |
| **Q6_K** *(default)* | `…-Q6_K.gguf` | 22.7 GB | ~31 GB | high — best that fits 32 GB @ 32K |
| Q8_0 | `…-Q8_0.gguf` | 26.9 GB | ~34 GB | very high — needs a smaller ctx on 32 GB |
| BF16 | `…-BF16.gguf` | 50.5 GB | ~58 GB | full — datacenter GPUs (H100/H200) only |

VRAM ≈ model size + ~2 GB compute buffer + KV-cache (grows with context). A **smaller context
needs much less VRAM**, so a low-VRAM card can run a bigger quant at a shorter context — that is
exactly the trade-off the agent should help the user pick in Step 0.

---

## What you get

- **Inference engine** — patched `llama.cpp` (PR #24423) running `DiffusionGemma 26B-A4B` (Q6_K GGUF, ~21 GB).
- **OpenAI-compatible API** — `GET /v1/models`, `POST /v1/chat/completions` (streaming + non-streaming),
  with diffusion telemetry (tokens/s, denoising steps, timings) in `usage.timings`.
- **Desktop GUI** *(in development)* — chat with reasoning panel, live engine parameters,
  and a model downloader (pick a DiffusionGemma quantization to fetch).
- **System-tray controller** *(in development)* — unload model, preload model, open GUI,
  open settings, quit.
- **Smart VRAM lifecycle** *(in development)* — see [VRAM coordination](#vram-coordination).

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, ≥ 24 GB VRAM recommended (developed on **RTX 5090**, 32 GB). |
| CUDA | 12.8+ (Blackwell/sm_120 needs 12.8+); developed on CUDA 13.1. |
| Compiler | Visual Studio 2022 Build Tools, **or** VS 2026 with the 14.4x toolset (see `scripts/build_engine.ps1`). |
| Python | 3.10+ |
| Node | 18+ (only for the GUI) |
| Disk | ~22 GB for the model + a few GB for the engine build. |

---

## Manual installation

```powershell
# 1. clone this repo
git clone https://github.com/nuspy/llama-diffusion
cd llama-diffusion

# 2. fetch the diffusion-capable llama.cpp fork (PR #24423) into engine/llama.cpp
git clone --depth=1 https://github.com/ggml-org/llama.cpp engine/llama.cpp
git -C engine/llama.cpp fetch --depth=1 origin pull/24423/head:diffusion-gemma
git -C engine/llama.cpp checkout diffusion-gemma

# 3. build the CUDA engine (handles the VS-toolset / _CL_ / C++17 gotchas automatically)
pwsh scripts/build_engine.ps1

# 4. python dependencies
python -m pip install -r requirements.txt
python -m playwright install chromium      # for the browser tool

# 5. download the model into models/  (Q6_K, ~21 GB)
#    e.g. via huggingface-cli, or use the GUI's model downloader:
#    huggingface-cli download unsloth/diffusiongemma-26B-A4B-it-GGUF \
#        diffusiongemma-26B-A4B-it-Q6_K.gguf --local-dir models

# 6. run the API
pwsh scripts/serve.ps1        # http://127.0.0.1:8080
```

Quick smoke test of the raw engine (no API):
```powershell
python agent/engine.py -p "Write a Python function for factorial." --blocks 2 -v
```

---

## Using it as an inference engine

The API is OpenAI-compatible, so existing clients work unchanged:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "diffusiongemma-26b-a4b-it",
  "messages": [{"role":"user","content":"Explain diffusion LLMs in one sentence."}],
  "stream": true
}'
```

Notes specific to this engine:
- **Reasoning** is returned separately in `message.reasoning_content` (and `delta.reasoning_content`
  while streaming); the clean answer is in `content`.
- `max_tokens` is mapped to diffusion *blocks* (256 tokens each). The model reasons verbosely,
  so the default leaves room for the final answer.
- `temperature`/`top_p` are accepted for compatibility; the sampler uses the model's
  entropy-bound schedule.

### Running as a Windows service (always-on)

```powershell
# from an ADMIN PowerShell:
pwsh scripts/install_service.ps1
```
Installs an auto-start service `llamadiff` on `0.0.0.0:8080` (reachable by Hermes on WSL).

---

## Desktop GUI *(in development)*

An Electron app with a Python sidecar:
- chat with the model, with a collapsible **reasoning** panel;
- live **engine parameters** (context length, response length, denoising steps, flash-attn);
- per-message **telemetry** (time-to-first-output, tokens/s);
- a **model downloader**: if no DiffusionGemma model is installed, browse available
  quantizations (Q4_K_M, Q5, Q6_K, Q8…) and download one.

## System tray *(in development)*

A hidden-icons (system tray) controller with:
**Open GUI · Preload model · Unload model · Settings · Quit.**

## VRAM coordination *(in development)*

The model occupies ~21 GB of VRAM. To play nice with other GPU apps (e.g. Hermes switching to a
different model):
- **If the GUI is not open** and another app requests the GPU / Hermes switches model →
  the app **automatically releases** the model from VRAM.
- **If the GUI is open** → the app **asks** whether to unload the model from VRAM.

---

## Performance & internals

On an RTX 5090 (Q6_K, all layers in VRAM): **32 768-token context at ~200 tok/s**.

The diffusion server normally reserves a compute buffer sized to the whole context
(`n_ubatch = n_ctx`), which on a 32 GB card forces large contexts to spill to RAM and collapse
to ~2 tok/s. This project **patches the server** (env `DG_UBATCH`) to decouple the physical
micro-batch from the context, so the buffer stays small (~2 GB) while the context stays large.
See [`PLAN.md`](PLAN.md) for the full architecture and the build/runtime gotchas.

## License

The engine is `llama.cpp` (MIT). DiffusionGemma weights are under Google's license (Apache-2.0
for the unsloth GGUF). This project's own code: MIT.
