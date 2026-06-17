"""
OpenAI-compatible adapter in front of the DiffusionGemma engine, with **JIT model loading**
(LM Studio style):

  - GET  /v1/models       lists every .gguf in models/ (loaded or not)
  - POST /v1/chat/completions  loads the requested model into VRAM on demand (unloading the
                          previous one), then generates. Streaming + non-streaming.
  - GET  /health          status + which model is currently loaded + available model ids

The visual-server holds ONE model in VRAM at a time, so requests are serialized by a lock:
the first request for a not-yet-loaded model waits for the (minutes-long) load, exactly like
LM Studio's just-in-time loading.

Start:
  python -m agent.openai_server --host 0.0.0.0 --port 8787
Auth (optional): set LLAMADIFF_API_KEY -> requests must send 'Authorization: Bearer <key>'.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .engine import DiffusionEngine, EngineError, load_engine_config, ROOT

log = logging.getLogger("openai_server")

CANVAS = 256
MAX_BLOCKS = 64
API_KEY = os.getenv("LLAMADIFF_API_KEY", "")
MODELS_DIR = ROOT / "models"

engine: Optional[DiffusionEngine] = None
_gpu_lock = asyncio.Lock()   # one model in VRAM at a time: serialize load+generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = DiffusionEngine(load_engine_config())   # do NOT load yet: JIT on first request
    log.info("adapter ready — models dir: %s (just-in-time loading on first request)", MODELS_DIR)
    yield
    if engine and engine.running:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.stop)


app = FastAPI(title="llama-diffusion OpenAI adapter", lifespan=lifespan)


# ---- request models ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any = ""

class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None   # accepted for compatibility; sampler uses entropy-bound
    top_p: Optional[float] = None


# ---- model discovery (JIT) --------------------------------------------------

def _model_id(p: Path) -> str:
    return p.stem                                   # filename without .gguf

def list_local_models() -> list[dict]:
    if not MODELS_DIR.exists():
        return []
    return [{"id": _model_id(p), "file": p.name, "size": p.stat().st_size}
            for p in sorted(MODELS_DIR.glob("*.gguf"))]

def _resolve_model_path(req_model: str) -> Optional[Path]:
    """Map an OpenAI `model` string to a .gguf path. Empty/unknown -> config default, then first."""
    files = sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []
    if not files:
        return None
    if req_model:
        rm = req_model.lower()
        for p in files:
            if rm in (p.stem.lower(), p.name.lower()):
                return p
    cfg = load_engine_config()
    if cfg.model_path.exists():
        return cfg.model_path
    return files[0]


# ---- helpers ----------------------------------------------------------------

def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if token != API_KEY:
        raise HTTPException(status_code=401, detail={
            "error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}
        })


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content or "")


def _to_engine_messages(messages: list[ChatMessage]) -> list[dict]:
    return [{"role": m.role, "content": _flatten_content(m.content)} for m in messages]


# reasoning channel:  <|channel>thought ... <channel|> ANSWER
_CHANNEL_RE = re.compile(r"<\|channel>\s*\w*\s*(.*?)<channel\|>(.*)", re.DOTALL)

def _split_reasoning(text: str) -> tuple[str, str]:
    m = _CHANNEL_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    if "<|channel>" in text:
        r = text.split("<|channel>", 1)[1].lstrip()
        if r.startswith("thought"):
            r = r[len("thought"):]
        return r.strip(), ""
    return "", text.strip()

def _assistant_message(text: str) -> dict:
    reasoning, final = _split_reasoning(text)
    msg = {"role": "assistant", "content": final}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return msg


def _blocks_for(max_tokens: Optional[int]) -> int:
    if not max_tokens or max_tokens <= 0:
        return 8                                    # ~2048 tokens; the model reasons verbosely
    return max(1, min(MAX_BLOCKS, math.ceil(max_tokens / CANVAS)))


def _usage(stats: dict) -> dict:
    p = int(stats.get("prompt_n", 0))
    c = int(stats.get("predicted_n", 0))
    dec_s = stats.get("decode_ms", 0) / 1000.0
    return {
        "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
        "timings": {
            "prompt_prepare_ms": stats.get("prompt_prepare_ms", 0),
            "decode_ms": stats.get("decode_ms", 0),
            "tokens_per_second": round(c / dec_s, 1) if dec_s > 0 else 0.0,
            "denoising_steps": stats.get("steps", 0),
            "blocks": stats.get("blocks", 0),
            "n_ctx": stats.get("n_ctx", 0),
        },
    }


def _finish_reason(res) -> str:
    return "length" if res.stop_reason == "length" else "stop"


# ---- endpoints --------------------------------------------------------------

@app.get("/health")
async def health():
    loaded = engine.current_model.name if (engine and engine.current_model) else None
    return {
        "status": "ok",
        "loaded_model": loaded,
        "maxtok": engine.maxtok if (engine and engine.running) else 0,
        "available": [m["id"] for m in list_local_models()],
    }


@app.get("/v1/models")
async def list_models():
    # expose context_length so an agent only needs to register the PROVIDER (base_url):
    # the model id and its context are auto-discovered here, no hand-registration needed.
    ctx = load_engine_config().maxtok
    return {"object": "list", "data": [
        {"id": m["id"], "object": "model", "owned_by": "local", "created": 0,
         "context_length": ctx, "max_model_len": ctx}
        for m in list_local_models()
    ]}


@app.post("/admin/load")
async def admin_load(request: Request):
    """Precarica un modello in VRAM (tray: 'Precarica modello'). Body opzionale {\"model\": id}."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    path = _resolve_model_path(body.get("model", "") if isinstance(body, dict) else "")
    if path is None:
        raise HTTPException(status_code=404, detail="no .gguf in models/")
    loop = asyncio.get_event_loop()
    async with _gpu_lock:
        await loop.run_in_executor(None, lambda: engine.ensure_loaded(path))
    return {"status": "loaded", "model": _model_id(path)}


@app.post("/admin/unload")
async def admin_unload():
    """Scarica il modello dalla VRAM (tray: 'Scarica modello')."""
    loop = asyncio.get_event_loop()
    async with _gpu_lock:
        await loop.run_in_executor(None, engine.unload)
    return {"status": "unloaded"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    _check_auth(request)
    path = _resolve_model_path(req.model)
    if path is None:
        raise HTTPException(status_code=404, detail={
            "error": {"message": "no .gguf model found in models/", "type": "invalid_request_error"}})

    messages = _to_engine_messages(req.messages)
    n_blocks = _blocks_for(req.max_tokens)
    seed = req.seed or 0
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    model_id = _model_id(path)

    if req.stream:
        return StreamingResponse(
            _stream(cid, created, model_id, path, messages, n_blocks, seed),
            media_type="text/event-stream",
        )

    loop = asyncio.get_event_loop()
    async with _gpu_lock:                       # JIT load + generate, serialized (single GPU)
        try:
            await loop.run_in_executor(None, lambda: engine.ensure_loaded(path))
            res = await loop.run_in_executor(
                None, lambda: engine.generate(messages, seed=seed, n_blocks=n_blocks))
        except EngineError as e:
            raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})

    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": created, "model": model_id,
        "choices": [{
            "index": 0,
            "message": _assistant_message(res.text),
            "finish_reason": _finish_reason(res),
        }],
        "usage": _usage(res.stats),
    })


async def _stream(cid, created, model_id, path, messages, n_blocks, seed):
    loop = asyncio.get_event_loop()

    def chunk(delta: dict, finish: Optional[str] = None) -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async with _gpu_lock:                       # hold the GPU for the whole stream
        try:
            await loop.run_in_executor(None, lambda: engine.ensure_loaded(path))   # JIT load
        except Exception as e:  # noqa: BLE001
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        q: asyncio.Queue = asyncio.Queue()
        sent = {"reasoning": 0, "final": 0}

        def on_commit(block: int, cumulative: str):
            reasoning, final = _split_reasoning(cumulative)
            loop.call_soon_threadsafe(q.put_nowait, ("split", (reasoning, final)))

        def run():
            try:
                res = engine.generate(messages, seed=seed, n_blocks=n_blocks, on_commit=on_commit)
                loop.call_soon_threadsafe(q.put_nowait, ("done", res))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

        threading.Thread(target=run, daemon=True).start()
        yield chunk({"role": "assistant"})

        while True:
            kind, payload = await q.get()
            if kind == "split":
                reasoning, final = payload
                if len(reasoning) > sent["reasoning"]:
                    yield chunk({"reasoning_content": reasoning[sent["reasoning"]:]})
                    sent["reasoning"] = len(reasoning)
                if len(final) > sent["final"]:
                    yield chunk({"content": final[sent["final"]:]})
                    sent["final"] = len(final)
            elif kind == "done":
                final_chunk = {
                    "id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(payload)}],
                    "usage": _usage(payload.stats),
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            elif kind == "error":
                yield f"data: {json.dumps({'error': {'message': payload, 'type': 'server_error'}})}\n\n"
                yield "data: [DONE]\n\n"
                return


def _main() -> None:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="OpenAI adapter for DiffusionGemma (JIT model loading)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose to WSL/Hermes")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    _main()
