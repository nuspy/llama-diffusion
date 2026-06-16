"""
Wrapper Python del `llama-diffusion-gemma-visual-server`.

Il server e' un processo persistente che carica il GGUF una volta e parla via
stdin/stdout (vedi PLAN.md "Protocollo del visual-server"):

  - su stdin gli si scrive UNA riga = path di un file JSON di richiesta
        {"seed": int, "n_blocks": int, "messages": [{"role","content"}, ...]}
  - su stdout emette uno stream di righe terminato da `DONE`:
        F <block> <step> <total> <json-string>   frame (canvas decodificato) per step
        C <block> <json-string>                   testo cumulativo committato per blocco
        STATS <key=value ...>                     riga di statistiche
        DONE                                      fine richiesta
        ERR <msg>                                 errore (es. "toolong <needed> <budget>")
  - all'avvio stampa: READY <n_vocab> <MAXTOK>
  - QUIT / EOF per chiudere.

Il chat-template e il tokenizer sono nel GGUF: il client manda solo `messages`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("engine")

# Radice del progetto: .../llama-diffusion
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SERVER = ROOT / "engine" / "llama.cpp" / "build" / "bin" / "llama-diffusion-gemma-visual-server.exe"
DEFAULT_MODEL = ROOT / "models" / "diffusiongemma-26B-A4B-it-Q6_K.gguf"
TEMP_DIR = ROOT / ".temp"


class EngineError(RuntimeError):
    """Errore restituito dal server (riga ERR) o dal wrapper."""


@dataclass
class EngineConfig:
    server_exe: Path = DEFAULT_SERVER
    model_path: Path = DEFAULT_MODEL
    ngl: int = 99          # layer su GPU (il modello ha 30 blocchi: 99 = tutto in VRAM)
    maxtok: int = 32768    # contesto. Grazie a ubatch (sotto), il compute buffer e' disaccoppiato dal contesto:
                           # 32768 ctx + ubatch 2048 -> buffer ~2 GB, ~94 tok/s (VERIFICATO su RTX 5090).
                           # NON usare 0/auto: con ubatch=ctx sceglie 12288 -> buffer 12 GB -> spilla -> ~2 tok/s.
    fa: bool = False       # flash attention (da verificare sul path di diffusione non-causale)
    ubatch: int = 2048     # DG_UBATCH: cap del micro-batch fisico. 2048 = ottimo equilibrio (copre prompt
                           # iniziali fino a ~1800 token + canvas 256, e tiene il buffer ~2 GB). La patch al
                           # visual-server rende il buffer ~lineare in ubatch, non in n_ctx.


def load_engine_config() -> "EngineConfig":
    """Crea una EngineConfig applicando gli override da config.json nella radice del progetto
    (scritto da scripts/install.ps1 o dalla GUI): {model_file, maxtok, ubatch}."""
    cfg = EngineConfig()
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cfg
        if data.get("model_file"):
            cfg.model_path = ROOT / "models" / data["model_file"]
        if data.get("maxtok"):
            cfg.maxtok = int(data["maxtok"])
        if data.get("ubatch") is not None:
            cfg.ubatch = int(data["ubatch"])
    return cfg


@dataclass
class GenerationResult:
    text: str = ""                       # testo finale (ultimo commit cumulativo)
    stop_reason: str = "done"            # "done" | "length" | "error"
    stats: dict = field(default_factory=dict)
    error: Optional[str] = None


# Callback opzionali per lo streaming verso la UI.
# on_frame(block, step, total, canvas_text): chiamata a ogni step di denoising
# on_commit(block, cumulative_text):         chiamata quando un blocco viene committato
FrameCb = Callable[[int, int, int, str], None]
CommitCb = Callable[[int, str], None]


class DiffusionEngine:
    """Gestisce il ciclo di vita del visual-server e serializza le richieste."""

    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()       # il protocollo accetta una richiesta per volta
        self._stderr_thread: Optional[threading.Thread] = None
        self._req_counter = 0
        self.n_vocab: int = 0
        self.maxtok: int = 0

    # ---- ciclo di vita -------------------------------------------------

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, ready_timeout: float = 600.0) -> None:
        """Avvia il server e attende la riga READY (il load del modello puo' richiedere minuti)."""
        if self.running:
            return
        exe = Path(self.cfg.server_exe)
        model = Path(self.cfg.model_path)
        if not exe.exists():
            raise EngineError(f"server non trovato: {exe} (compila prima il target llama-diffusion-gemma-visual-server)")
        if not model.exists():
            raise EngineError(f"modello non trovato: {model}")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["NGL"] = str(self.cfg.ngl)
        env["MAXTOK"] = str(self.cfg.maxtok)
        env["FA"] = "1" if self.cfg.fa else "0"
        env["DG_UBATCH"] = str(self.cfg.ubatch)

        log.info("avvio server: %s %s (NGL=%s MAXTOK=%s FA=%s DG_UBATCH=%s)",
                 exe.name, model.name, env["NGL"], env["MAXTOK"], env["FA"], env["DG_UBATCH"])
        self._proc = subprocess.Popen(
            [str(exe), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,                 # line-buffered
            cwd=str(ROOT),
        )

        # drena stderr in un thread (diagnostica del server: sizing contesto, ecc.)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        # attende READY <n_vocab> <MAXTOK>
        deadline = threading.Event()
        result: dict = {}

        def _wait_ready():
            assert self._proc and self._proc.stdout
            for line in self._proc.stdout:
                line = line.rstrip("\r\n")
                if line.startswith("READY"):
                    parts = line.split()
                    if len(parts) >= 3:
                        result["n_vocab"] = int(parts[1])
                        result["maxtok"] = int(parts[2])
                    deadline.set()
                    return
            deadline.set()  # stdout chiuso senza READY

        t = threading.Thread(target=_wait_ready, daemon=True)
        t.start()
        if not deadline.wait(ready_timeout):
            self.stop()
            raise EngineError(f"timeout ({ready_timeout}s) in attesa di READY dal server")
        if "n_vocab" not in result:
            code = self._proc.poll() if self._proc else None
            raise EngineError(f"il server e' terminato prima di READY (exit={code}); controlla i log stderr")
        self.n_vocab = result["n_vocab"]
        self.maxtok = result["maxtok"]
        log.info("server pronto: n_vocab=%d MAXTOK=%d", self.n_vocab, self.maxtok)

    def stop(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.poll() is None and self._proc.stdin:
                try:
                    self._proc.stdin.write("QUIT\n")
                    self._proc.stdin.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        finally:
            self._proc = None

    def _drain_stderr(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        for line in proc.stderr:
            log.debug("[server] %s", line.rstrip())

    # ---- generazione ---------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        *,
        seed: int = 0,
        n_blocks: int = 1,
        on_frame: Optional[FrameCb] = None,
        on_commit: Optional[CommitCb] = None,
    ) -> GenerationResult:
        """Genera una risposta. `messages` in formato chat OpenAI ({"role","content"}).

        n_blocks controlla la lunghezza max (ogni blocco = canvas da ~256 token).
        Bloccante; serializzato da un lock (il server gestisce una richiesta per volta).
        """
        if not self.running:
            raise EngineError("server non avviato (chiama start())")

        with self._lock:
            self._req_counter += 1
            req_path = TEMP_DIR / f"req_{os.getpid()}_{self._req_counter}.json"
            payload = {"seed": int(seed), "n_blocks": int(n_blocks), "messages": messages}
            req_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            assert self._proc and self._proc.stdin and self._proc.stdout
            try:
                self._proc.stdin.write(str(req_path) + "\n")
                self._proc.stdin.flush()
            except (OSError, ValueError) as e:
                raise EngineError(f"impossibile scrivere la richiesta al server: {e}") from e

            result = GenerationResult()
            try:
                for raw in self._proc.stdout:
                    line = raw.rstrip("\r\n")
                    if not line:
                        continue
                    tag = line[0]
                    if tag == "F" and on_frame:
                        # F <block> <step> <total> <json-string>
                        parts = line.split(" ", 4)
                        if len(parts) == 5:
                            try:
                                on_frame(int(parts[1]), int(parts[2]), int(parts[3]), json.loads(parts[4]))
                            except (ValueError, json.JSONDecodeError):
                                pass
                    elif tag == "C":
                        # C <block> <json-string>
                        parts = line.split(" ", 2)
                        if len(parts) == 3:
                            try:
                                text = json.loads(parts[2])
                            except json.JSONDecodeError:
                                continue
                            result.text = text
                            if on_commit:
                                on_commit(int(parts[1]), text)
                    elif line.startswith("STATS"):
                        result.stats = _parse_stats(line)
                    elif line == "DONE":
                        break
                    elif line.startswith("ERR"):
                        result.stop_reason = "error"
                        result.error = line[3:].strip()
                        # mantieni leggere fino a DONE se presente, ma ERR e' terminale
                        break
                    elif line.startswith("READY"):
                        continue  # eco residua, ignora
            finally:
                try:
                    req_path.unlink(missing_ok=True)
                except OSError:
                    pass

            if result.error:
                raise EngineError(result.error)
            # stop_reason: se ha generato tutti i blocchi richiesti senza eog -> probabile troncamento
            if result.stats.get("blocks") and int(result.stats.get("blocks", 0)) >= n_blocks:
                result.stop_reason = "length"
            return result


def _parse_stats(line: str) -> dict:
    out: dict = {}
    for tok in line.split()[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


# --------------------------------------------------------------------------
# CLI di smoke test:  python -m agent.engine -p "ciao" --blocks 2
# --------------------------------------------------------------------------
def _main() -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Smoke test del visual-server DiffusionGemma")
    ap.add_argument("-p", "--prompt", default="Scrivi una funzione Python che inverte una stringa.")
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--maxtok", type=int, default=0)
    ap.add_argument("--fa", action="store_true", help="abilita flash attention (evita il buffer scores O(N^2))")
    ap.add_argument("--ubatch", type=int, default=0, help="cap del micro-batch fisico (DG_UBATCH); 0=legacy")
    ap.add_argument("--server", default=str(DEFAULT_SERVER))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    eng = DiffusionEngine(EngineConfig(
        server_exe=Path(args.server), model_path=Path(args.model),
        ngl=args.ngl, maxtok=args.maxtok, fa=args.fa, ubatch=args.ubatch,
    ))
    print(">> avvio server (load del modello, puo' richiedere un po')...", file=sys.stderr)
    eng.start()
    print(f">> pronto (MAXTOK={eng.maxtok}). Genero...\n", file=sys.stderr)

    last_len = 0

    def on_frame(block, step, total, text):
        # mostra l'avanzamento del denoising in-place
        print(f"\r[blocco {block} step {step}/{total}] {len(text)} char ", end="", file=sys.stderr)

    def on_commit(block, text):
        nonlocal last_len
        sys.stdout.write(text[last_len:])
        sys.stdout.flush()
        last_len = len(text)

    try:
        res = eng.generate(
            [{"role": "user", "content": args.prompt}],
            seed=args.seed, n_blocks=args.blocks,
            on_frame=on_frame, on_commit=on_commit,
        )
        st = res.stats
        dec_s = st.get("decode_ms", 0) / 1000.0
        toks = st.get("predicted_n", 0)
        tok_s = (toks / dec_s) if dec_s > 0 else 0.0
        print("\n\n--- telemetria ---", file=sys.stderr)
        print(f"  prompt:       {st.get('prompt_n','?')} token", file=sys.stderr)
        print(f"  generati:     {toks} token ({st.get('blocks','?')} blocchi, {st.get('steps','?')} step denoising)", file=sys.stderr)
        print(f"  preparazione: {st.get('prompt_prepare_ms',0):.0f} ms", file=sys.stderr)
        print(f"  generazione:  {dec_s:.1f} s", file=sys.stderr)
        print(f"  velocita':    {tok_s:.1f} token/s", file=sys.stderr)
        print(f"  contesto:     {st.get('n_ctx','?')} | stop: {res.stop_reason}", file=sys.stderr)
        return 0
    except EngineError as e:
        print(f"\nERRORE: {e}", file=sys.stderr)
        return 1
    finally:
        eng.stop()


if __name__ == "__main__":
    raise SystemExit(_main())
