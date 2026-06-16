# llama-diffusion — Piano & Architettura

Agente AI locale "tipo Claude Code" alimentato da **DiffusionGemma 26B-A4B** (modello a
diffusione testuale), con app desktop, loop agentico e tool per filesystem, shell,
compilazione, browser headless e ricerca web.

## Decisioni prese (fisse)

| Tema | Scelta |
|---|---|
| Modello | **Solo DiffusionGemma** (`models/diffusiongemma-26B-A4B-it-Q6_K.gguf`, arch `diffusion-gemma`) |
| Engine | Fork **llama.cpp PR #24423** (`engine/llama.cpp`, branch `diffusion-gemma`) |
| Build | **Windows nativo**, CUDA 13.1, MSVC VS18, arch **sm_120** (RTX 5090) |
| Servire il modello | **`llama-diffusion-gemma-visual-server`** (già nella PR): processo persistente, modello in VRAM, decoder entropy-bound + streaming per-step |
| Trasporto col modello | **stdin/stdout** (NON HTTP) — vedi protocollo sotto |
| UI | **App desktop Electron** + **sidecar Python** (FastAPI/WebSocket) |
| Tool-calling | Parser robusto lato nostro, **attivabile/disattivabile dalla GUI** |
| Stack agente | Python 3.10 |

## Architettura (aggiornata dopo lettura del sorgente PR)

```
┌──────────────────────────────────────────────────────────────┐
│  Electron (gui/)                                               │
│   chat · file/diff viewer · pannello tool · toggle tool-call   │
└───────────────┬──────────────────────────────────────────────┘
                │ WebSocket (ws://127.0.0.1:8770)
┌───────────────▼──────────────────────────────────────────────┐
│  Agent server (agent/server.py — FastAPI + WS)                │
│   • streaming token/eventi-tool verso la GUI                  │
│   • avviato come sidecar da Electron                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ agent/core.py    loop agentico (osserva→pensa→agisci)   │  │
│  │ agent/toolcall.py  parser tool-call (attivabile)        │  │
│  │ agent/tools/   files · shell/compile · browser · search │  │
│  │ agent/engine.py  wrapper subprocess del visual-server   │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────┬──────────────────────────────────────────────┘
                │ stdin (path file richiesta) / stdout (stream F/C/STATS/DONE)
┌───────────────▼──────────────────────────────────────────────┐
│  llama-diffusion-gemma-visual-server.exe  (libllama + diff.)  │
│   • modello caricato una volta, offload CUDA (NGL)            │
│   • chat-template + tokenizer dal GGUF stesso                 │
│   • decoder entropy-bound (48 step, t 0.8→0.4, canvas 256)    │
└──────────────────────────────────────────────────────────────┘
```

## Protocollo del visual-server (reale, da sorgente)

- **avvio**: `llama-diffusion-gemma-visual-server <model.gguf>`
  - env: `NGL` (gpu layers, 0=CPU), `MAXTOK` (0=auto-size sul budget VRAM/RAM), `FA` (flash-attn 0/1)
  - alla partenza stampa su stdout: `READY <n_vocab> <MAXTOK>`
- **richiesta**: si scrive su stdin **una riga = path di un file** che contiene:
  ```json
  { "seed": 0, "n_blocks": 1, "messages": [ {"role":"user","content":"..."} ] }
  ```
  `n_blocks` = quanti canvas da 256 token concatenare (controlla la lunghezza max risposta).
- **risposta** (stream di righe su stdout, poi `DONE`):
  - `F <block> <step> <total> <json-string>`  canvas corrente decodificato (1 per step di denoising)
  - `C <block> <json-string>`                  testo cumulativo committato dopo il blocco
  - `STATS prompt_n=.. predicted_n=.. wall_ms=.. decode_ms=.. blocks=.. steps=..`
  - `DONE`  fine richiesta
  - `ERR <msg>`  (es. `ERR toolong <needed> <budget>` se prompt+canvas > MAXTOK)
- **chiusura**: `QUIT` o EOF.

> Conseguenza per l'agente: ogni richiesta è **stateless** — la storia conversazione +
> output dei tool vanno accodati nei `messages` lato Python. Limite di contesto = `MAXTOK`
> (riportato a runtime); gestire troncamento per evitare `ERR toolong`.

## Formato tool-call (parser proprietario, indipendente dal modello)

Blocco esplicito e tollerante (DiffusionGemma potrebbe non avere function-calling nativo):
```
<tool name="write_file">
  <arg name="path">src/main.py</arg>
  <arg name="content">print("hi")</arg>
</tool>
```
Accetta anche fence ```tool ... ``` e JSON `{"tool":...,"args":{...}}`.
Toggle GUI: **OFF** → chat pura; **ON** → loop agentico (esegue i tool, re-inietta i risultati).

## Tool previsti
- **files**: `read_file`, `write_file`, `edit_file`, `delete_file`, `list_dir`
- **shell**: `run`, `compile` — cwd nella sandbox di progetto
- **browser**: Playwright headless — `navigate`, `extract_text`, `screenshot`
- **search**: ricerca web → risultati testuali
- **project**: `create_project` (scaffold)

## Fasi

- [x] **F0** Toolchain verificata + modello in `models/`
- [x] **F1** Clone fork PR #24423 (branch `diffusion-gemma`)
- [x] **F2** Ispezione sampler: DiffusionGemma usa `diffusion_generate_entropy_bound`; server già esistente, protocollo stdin/stdout
- [~] **F3** Build CUDA (`llama-diffusion-gemma-visual-server` + `llama-diffusion-cli`) → smoke test
- [ ] **F4** `agent/engine.py`: wrapper subprocess (scrive req file, parsa stream)
- [ ] **F5** Agente Python: `toolcall.py`, `tools/*`, `core.py`, `server.py`
- [ ] **F6** GUI Electron + sidecar + toggle tool-calling
- [ ] **F7** Integrazione end-to-end + smoke test agentico
- [ ] **F8** Adattatore OpenAI-compatible (`/v1/chat/completions` + `/v1/models`) davanti al visual-server — base comune per GUI **e** Hermes
- [ ] **F9** Integrazione Hermes Agents: provider `custom` in `~/.hermes/config.yaml` puntato all'adattatore
- [ ] **F10** GUI: catalogo modelli DiffusionGemma su HF per quantizzazione + download in `models/`
- [ ] **F11** GUI: telemetria a fine inferenza (da `STATS`) + pannello parametri motore
- [ ] **F12** Patch C++ al visual-server: esporre per-richiesta i parametri entropy-bound (steps, soglie)

## Integrazione Hermes Agents (F8–F9)

Hermes Agent (codice in `~/.hermes/hermes-agent/` su WSL, config `~/.hermes/config.yaml`)
consuma un backend modello via **API OpenAI-compatibile**: oggi un provider `custom:lmstudio`
con `base_url: http://172.18.160.1:1234/v1` (LM Studio sull'host Windows visto da WSL).
Le chiamate rilevanti per l'inferenza sono `GET /v1/models` e `POST /v1/chat/completions`
(streaming SSE incluso); l'API key arriva da `extra.key` o env `API_SERVER_KEY`.

Per usare DiffusionGemma come motore di Hermes:
1. F8 — adattatore FastAPI che parla OpenAI-compat e dietro pilota il visual-server (engine.py).
   Deve ascoltare su `0.0.0.0` (raggiungibile da WSL), non solo `127.0.0.1`.
2. F9 — in `~/.hermes/config.yaml` aggiungere un provider che punta all'adattatore + un modello
   `diffusiongemma-26b-a4b` con `context_length`/`max_tokens` coerenti col `MAXTOK` del server.
- ⚠️ **Rischio aperto**: Hermes fa tool/function-calling. Se invia `tools[]` e attende `tool_calls`
  strutturati, va verificato come il suo provider client li costruisce (ispezionare il codice del
  provider su WSL) e se DiffusionGemma li produce. Mitigazione: il nostro parser tool-call lato adattatore.

## Telemetria e parametri motore (F11–F12)

**Telemetria** — il visual-server emette gia' una riga `STATS`:
`prompt_prepare_ms`, `wall_ms`, `decode_ms`, `predicted_n`, `blocks`, `steps`, `canvas`, `n_ctx`.
Da qui: tempo-prima-dell'output (al primo record `C`), e **tok/s = predicted_n / (decode_ms/1000)**.
Il modello marca il reasoning con `<|channel>`: lo separiamo dalla risposta finale.

**Parametri** — mappatura onesta su cosa il motore espone davvero (≠ LM Studio, che e' autoregressivo):

| Parametro GUI | Dove agisce | Come |
|---|---|---|
| Max token risposta ("quantità reasoning") | per-richiesta | `n_blocks` (×256 token canvas) — gia' supportato |
| Max token contesto | avvio server | env `MAXTOK` (0=auto) — richiede restart del motore |
| Flash attention on/off | avvio server | env `FA` — richiede restart |
| GPU layers | avvio server | env `NGL` |
| Livello reasoning (n. step denoising, soglie) | per-richiesta | `eb_max_steps`, `entropy_bound`, `confidence_threshold`, `t_min/t_max` — oggi letti dai metadati GGUF, **F12: patch C++** per accettarli nel JSON di richiesta |
| Seed | per-richiesta | gia' supportato |

**Non applicabile / da verificare (onestà):**
- ❌ **TeaCache**: è una cache per modelli di **diffusione di immagini/video** (riuso dei timestep
  embedding); non esiste in llama.cpp né si applica alla diffusione testuale. Non promesso.
- ⚠️ **Quantizzazione K/V cache** (stile LM Studio f16/q8/q4): il path di diffusione è non-causale e
  ridenoise l'intero canvas; il visual-server usa un prefix-KV-cache booleano (`kv_cache`), non la
  cache-type quant standard. Da **verificare** se `--cache-type-k/v` è applicabile prima di esporlo.

## Rischi noti (onestà tecnica)
1. **Qualità agentica del modello**: sperimentale; tool-calling possibilmente debole.
   Mitigazione: parser tollerante + system prompt rigido. Se insufficiente → rivalutare.
2. **Build del target server**: il suo CMakeLists linka `llama-diffusion llama`; se il link
   fallisce per simboli `common_*`/`chat`, aggiungere `common` ai `target_link_libraries`.
3. **Blackwell sm_120**: richiede CUDA ≥12.8; abbiamo 13.1 → atteso ok, da verificare al build.
4. **Latenza**: ogni turno ridenoising dell'intero canvas; `n_blocks` alto = risposte lunghe ma più lente.
