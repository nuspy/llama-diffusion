# llama-diffusion

**Run Google's DiffusionGemma (a text *diffusion* LLM) locally as an OpenAI-compatible model
for agentic frameworks — Hermes, OpenCLAW, anything that speaks `/v1/chat/completions` — plus an
optional desktop chat GUI with a system-tray controller.**

DiffusionGemma 26B-A4B is a Mixture-of-Experts **diffusion** language model: instead of generating
tokens left-to-right, it denoises a whole 256-token "canvas" in parallel. Stock `llama.cpp` /
LM Studio can't load it (`unknown model architecture: 'diffusion-gemma'`). This project builds the
diffusion-capable fork (PR #24423) **plus a fix that makes long prompts fast**, and wraps it in a
standard OpenAI API on **port 8787**.

---

## TL;DR for AI agents (Hermes, OpenCLAW, …)

If your user says **"add this LLM: https://github.com/nuspy/llama-diffusion"** or
**"install this project"**:

**Step 0 — ask the user** (don't assume): which **quantization** to download (show the VRAM each
needs — see [Available models](#available-models); default **Q6_K**) and the **context tokens**
(default **262144** = the model max; Hermes requires ≥ 64K).

**Step 1 — install** (Windows):
```bash
git clone https://github.com/nuspy/llama-diffusion
cd llama-diffusion
pwsh scripts/install.ps1 -ModelFile diffusiongemma-26B-A4B-it-Q6_K.gguf -MaxContext 262144
```
`install.ps1` clones the engine fork, **applies the incremental-decode patch**, builds with CUDA,
installs deps, downloads the model, and writes `config.json`.

**Step 2 — run the engine** (always-on for an agent → install as a service, recommended):
```powershell
Start-Process pwsh -Verb RunAs -ArgumentList '-ExecutionPolicy','Bypass','-File','scripts/install_service.ps1'
```
…or foreground: `pwsh scripts/serve.ps1 -BindHost 0.0.0.0`. Either way it listens on `0.0.0.0:8787`.

**Step 3 — register ONLY the provider in your config.** Do **NOT** hand-register the model or its
`context_length`: the model id **and** its context are **auto-discovered** via `GET /v1/models`.

| field | value |
|---|---|
| `base_url` | `http://127.0.0.1:8787/v1` (local) · `http://<host-ip>:8787/v1` (from WSL/another host) |
| `api_key` | none (unless you set `LLAMADIFF_API_KEY`) |
| model | *auto-discovered* — don't pin it; `/v1/models` returns the id + `context_length` |

- **Hermes**: in `~/.hermes/config.yaml`, add only the provider:
  ```yaml
  providers:
    diffusiongemma:
      base_url: http://172.18.160.1:8787/v1   # WSL → Windows host vEthernet IP
      api_key: not-needed
  ```
  (No `models:` block needed — Hermes discovers it via `/v1/models`.)
- **OpenCLAW / others**: point the custom/OpenAI provider at the same `base_url`.

> "install this project" = install the engine **and** wire the provider into the calling agent.

### Available models

From [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF).
URL: `…/resolve/main/<file>`.

| Quantization | File | Size | VRAM @ 256K ctx | Quality |
|---|---|---|---|---|
| Q4_K_M | `…-Q4_K_M.gguf` | 16.8 GB | ~21 GB | good — most headroom |
| Q5_K_M | `…-Q5_K_M.gguf` | 19.2 GB | ~23 GB | better |
| **Q6_K** *(default)* | `…-Q6_K.gguf` | 22.7 GB | ~26 GB | high — best on 32 GB |
| Q8_0 | `…-Q8_0.gguf` | 26.9 GB | ~30 GB | very high |
| BF16 | `…-BF16.gguf` | 50.5 GB | n/a | datacenter GPUs only |

VRAM ≈ model + ~2 GB compute buffer + KV-cache (small here thanks to sliding-window attention).
A smaller quant leaves more room; pick with the user in Step 0.

---

## What you get

- **Inference engine** — patched `llama.cpp` (PR #24423 + our incremental-decode patch) running
  `DiffusionGemma 26B-A4B`.
- **OpenAI-compatible API** on `0.0.0.0:8787` — `GET /v1/models` (with `context_length`),
  `POST /v1/chat/completions` (streaming + non-streaming), `GET /health`, plus `POST /admin/load`
  and `POST /admin/unload`.
- **JIT model loading** (LM Studio style): the model loads into VRAM on first request; `/v1/models`
  lists every `.gguf` in `models/`.
- **Reasoning split**: the model's `<|channel>thought…<channel|>` is returned as `reasoning_content`,
  separate from the clean `content`. Per-message telemetry (tok/s, denoising steps) in `usage.timings`.
- **Desktop GUI + system tray** (Electron) — chat + reasoning panel; tray menu Open GUI · Preload ·
  Unload · Settings · Quit.

---

## Requirements

| | |
|---|---|
| GPU | NVIDIA, ≥ 24 GB recommended (developed on **RTX 5090**, 32 GB). |
| CUDA | 12.8+ (Blackwell/sm_120 needs 12.8+); developed on **CUDA 13.1**. |
| Compiler | Visual Studio 2022 Build Tools, **or** VS 2026 with the **14.44** toolset (`build_engine.ps1` forces it — CUDA 13.1 rejects the newer 14.5x). |
| Python | 3.10+ · **Node** 18+ (GUI only) |
| Disk | ~22 GB model + a few GB engine build. |

---

## Manual installation

```powershell
git clone https://github.com/nuspy/llama-diffusion
cd llama-diffusion

# 1. fetch the diffusion-capable llama.cpp fork (PR #24423)
git clone --depth=1 https://github.com/ggml-org/llama.cpp engine/llama.cpp
git -C engine/llama.cpp fetch --depth=1 origin pull/24423/head:diffusion-gemma
git -C engine/llama.cpp checkout diffusion-gemma

# 2. apply the incremental-decode fix (fast long prompts; not upstream)
git -C engine/llama.cpp apply patches/diffusion-gemma-incremental-decode.patch

# 3. build the CUDA engine (handles VS-toolset 14.44 / _CL_ / C++17 gotchas)
pwsh scripts/build_engine.ps1

# 4. python deps + browser tool
python -m pip install -r requirements.txt
python -m playwright install chromium

# 5. download the model into models/ (e.g. Q6_K, ~21 GB)
huggingface-cli download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q6_K.gguf --local-dir models

# 6. run
pwsh scripts/serve.ps1 -BindHost 0.0.0.0     # http://0.0.0.0:8787
```

Cross-platform entry-points in the repo root: `install.{ps1,bat,sh}`,
`start_diffusion_engine.{ps1,bat,sh}` (engine + tray on Windows).

Quick engine smoke test (no API): `python agent/engine.py -p "Write factorial in Python." -v`

---

## Using it as an inference engine

```bash
curl http://127.0.0.1:8787/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [{"role":"user","content":"Explain diffusion LLMs in one sentence."}],
  "stream": true
}'
```
- **model is optional**: omit it and the loaded model is used; `GET /v1/models` returns the id +
  `context_length` for discovery.
- **reasoning** in `message.reasoning_content` (and `delta.reasoning_content` while streaming);
  the answer is in `content`.
- `max_tokens` maps to diffusion *blocks* (256 tokens each); the model reasons verbosely so the
  default leaves room for the answer.
- `temperature`/`top_p` accepted for compatibility (the sampler uses its entropy-bound schedule).

### Always-on Windows service (recommended for agents)

```powershell
Start-Process pwsh -Verb RunAs -ArgumentList '-ExecutionPolicy','Bypass','-File','E:\Projects\llama-diffusion\scripts\install_service.ps1'
```
Installs auto-start service **`llamadiff`** on `0.0.0.0:8787`, **opens TCP 8787 in the firewall**
(so WSL/Hermes can reach it), loads the model on first request (JIT). Manage:
```powershell
Get-Service llamadiff | Stop-Service / Start-Service ;  nssm remove llamadiff confirm  # (admin)
```
Headless (no tray). Don't run the service and `start_diffusion_engine` together — same port.

---

## Desktop GUI + system tray

```powershell
.\start_diffusion_engine.ps1     # or double-click start_diffusion_engine.bat
```
Starts the engine **and** a tray icon (hidden-icons overflow `^` — drag onto the taskbar).
Right-click: **Open GUI · Preload model · Unload model · Settings · Quit**. The chat window opens
from the tray icon; it shows the reasoning panel and per-message telemetry. The GUI binds the
engine on `0.0.0.0:8787` (open the firewall once for WSL — the service does it automatically).

*Still in development:* in-GUI model downloader, automatic VRAM release on external model switch.

---

## Performance & internals

On an RTX 5090 (Q6_K, all layers in VRAM): short prompts ~150-200 tok/s; **long prompts (5K+ tokens,
e.g. an agent's system + tools) ~40-65 tok/s** — usable.

Key engineering, all in this repo:
- **Port 8787** everywhere (8080 was taken).
- **`maxtok=262144`, `ubatch=2048`** (`config.json`). Never use auto-size (it picks 12288 → the
  O(N²) buffer + weights spill VRAM → ~2 tok/s). `ubatch` is decoupled from context (env `DG_UBATCH`).
- **Incremental-decode patch** (`patches/diffusion-gemma-incremental-decode.patch`, applied by the
  installers): without it the engine fell back to `encode` (re-processing the whole prompt every
  denoising step → 0.3 tok/s / crash on long prompts). The patch caches the prompt KV in F16,
  decodes canvas-only, and pads the KV to a multiple of 256 so the head-dim-512 global layers use
  CUDA flash-attention. → long prompts ~133× faster, no crash at `ubatch=2048`.
- **Binary subprocess pipes** in `agent/engine.py` (Windows text-mode + `bufsize=1` caused `[Errno 22]`).
- **JIT model loading** + `reasoning_content` split + telemetry in `agent/openai_server.py`.

See [`PLAN.md`](PLAN.md) for full architecture and the build/runtime gotchas
([CUDA 13.1 ↔ VS18 → toolset 14.44]).

## License

Engine: `llama.cpp` (MIT). DiffusionGemma weights: Google's license (Apache-2.0 for the unsloth
GGUF). This project's own code: MIT.
