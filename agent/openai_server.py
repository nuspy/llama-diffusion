"""
Adattatore OpenAI-compatible davanti al motore DiffusionGemma.

Espone l'API che sia la GUI sia Hermes Agents si aspettano:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (streaming SSE e non-streaming)

Dietro le quinte pilota il `llama-diffusion-gemma-visual-server` tramite engine.py
(modello caricato una volta all'avvio). Il chat-template e la tokenizzazione le fa
il visual-server dal GGUF, quindi qui passiamo i `messages` cosi' come sono.

Avvio:
  python -m agent.openai_server --host 0.0.0.0 --port 8080
  (host 0.0.0.0 e' necessario perche' Hermes su WSL raggiunga l'host Windows)

Auth opzionale: se la env LLAMADIFF_API_KEY e' impostata, le richieste devono
includere `Authorization: Bearer <key>` (compatibile col flusso API_SERVER_KEY di Hermes).
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
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .engine import DiffusionEngine, EngineConfig, EngineError, load_engine_config

log = logging.getLogger("openai_server")

CANVAS = 256                 # token per blocco (dal modello)
MODEL_ID = "diffusiongemma-26b-a4b-it"
MAX_BLOCKS = 64              # tetto di sicurezza alla lunghezza risposta (64*256 = 16384 token)
API_KEY = os.getenv("LLAMADIFF_API_KEY", "")

engine: Optional[DiffusionEngine] = None


# ---- ciclo di vita: carica il modello una volta -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = DiffusionEngine(load_engine_config())   # default + override da config.json (modello/ctx scelti)
    loop = asyncio.get_event_loop()
    log.info("caricamento del modello in VRAM (puo' richiedere qualche minuto)...")
    await loop.run_in_executor(None, engine.start)   # bloccante, ma una sola volta
    log.info("modello pronto: MAXTOK=%d", engine.maxtok)
    yield
    if engine:
        await loop.run_in_executor(None, engine.stop)


app = FastAPI(title="llama-diffusion OpenAI adapter", lifespan=lifespan)


# ---- modelli di richiesta ---------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: Any = ""        # stringa, oppure lista di parti (multimodale OpenAI)


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None   # accettati per compatibilita', non usati dal sampler entropy-bound
    top_p: Optional[float] = None


# ---- helper -----------------------------------------------------------------

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
    """OpenAI ammette content come stringa o lista di parti {type,text}: lo riduciamo a testo."""
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


# Il modello marca il ragionamento con un canale:  <|channel>thought ... <channel|> RISPOSTA
_CHANNEL_RE = re.compile(r"<\|channel>\s*\w*\s*(.*?)<channel\|>(.*)", re.DOTALL)


def _split_reasoning(text: str) -> tuple[str, str]:
    """Separa (reasoning, risposta_finale). Se il canale manca, reasoning='' e tutto e' risposta.
    Se il canale e' aperto ma non chiuso (troncamento), e' tutto reasoning ancora in corso."""
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
        msg["reasoning_content"] = reasoning   # estensione (DeepSeek/o1-style): la GUI lo mostra a parte
    return msg


def _blocks_for(max_tokens: Optional[int]) -> int:
    if not max_tokens or max_tokens <= 0:
        return 8                                  # default ~2048 token: il modello ragiona in modo verboso,
                                                  # servono blocchi extra perche' la risposta finale ci stia
    return max(1, min(MAX_BLOCKS, math.ceil(max_tokens / CANVAS)))


def _usage(stats: dict) -> dict:
    p = int(stats.get("prompt_n", 0))
    c = int(stats.get("predicted_n", 0))
    dec_s = stats.get("decode_ms", 0) / 1000.0
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
        # campo non-standard: telemetria della diffusione per la GUI
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


# ---- endpoint ---------------------------------------------------------------

@app.get("/health")
async def health():
    ok = engine is not None and engine.running
    return {"status": "ok" if ok else "loading", "model": MODEL_ID,
            "maxtok": engine.maxtok if engine else 0}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "owned_by": "local", "created": 0}
    ]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    _check_auth(request)
    if engine is None or not engine.running:
        raise HTTPException(status_code=503, detail="motore non pronto")

    messages = _to_engine_messages(req.messages)
    n_blocks = _blocks_for(req.max_tokens)
    seed = req.seed or 0
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            _stream(cid, created, messages, n_blocks, seed),
            media_type="text/event-stream",
        )

    # --- non-streaming ---
    loop = asyncio.get_event_loop()
    try:
        res = await loop.run_in_executor(
            None, lambda: engine.generate(messages, seed=seed, n_blocks=n_blocks))
    except EngineError as e:
        raise HTTPException(status_code=500, detail={"error": {"message": str(e), "type": "server_error"}})

    return JSONResponse({
        "id": cid, "object": "chat.completion", "created": created, "model": req.model,
        "choices": [{
            "index": 0,
            "message": _assistant_message(res.text),
            "finish_reason": _finish_reason(res),
        }],
        "usage": _usage(res.stats),
    })


async def _stream(cid: str, created: int, messages: list[dict], n_blocks: int, seed: int):
    """Bridge sync(callback)->async(SSE): la generazione gira in un thread, i commit
    arrivano in una queue e li riemettiamo come delta OpenAI."""
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()
    sent = {"reasoning": 0, "final": 0}

    def on_commit(block: int, cumulative: str):
        reasoning, final = _split_reasoning(cumulative)
        loop.call_soon_threadsafe(q.put_nowait, ("split", (reasoning, final)))

    def run():
        try:
            res = engine.generate(messages, seed=seed, n_blocks=n_blocks, on_commit=on_commit)
            loop.call_soon_threadsafe(q.put_nowait, ("done", res))
        except Exception as e:  # noqa: BLE001 - propaghiamo come evento
            loop.call_soon_threadsafe(q.put_nowait, ("error", str(e)))

    threading.Thread(target=run, daemon=True).start()

    def chunk(delta: dict, finish: Optional[str] = None) -> str:
        payload = {
            "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # primo chunk: ruolo
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
            usage = _usage(payload.stats)
            # chunk finale con finish_reason + usage (estensione utile alla GUI)
            final = {
                "id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": _finish_reason(payload)}],
                "usage": usage,
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        elif kind == "error":
            err = {"error": {"message": payload, "type": "server_error"}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return


def _main() -> None:
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Adattatore OpenAI per DiffusionGemma")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 per esporre a WSL/Hermes")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    _main()
