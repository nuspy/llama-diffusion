<#
  serve.ps1 — avvia l'adattatore OpenAI di DiffusionGemma (foreground).
  Carica il modello in VRAM una volta, poi serve l'API su http://<host>:<port>.

  Uso:
    pwsh scripts/serve.ps1                          # 127.0.0.1:8787 (solo locale)
    pwsh scripts/serve.ps1 -BindHost 0.0.0.0        # raggiungibile da WSL/LAN (per Hermes/OpenCLAW)
    pwsh scripts/serve.ps1 -Port 9000
#>
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8787
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = $root
Write-Host ">> avvio API DiffusionGemma su http://$BindHost`:$Port  (caricamento ~21 GB in VRAM)..." -ForegroundColor Cyan
python -m agent.openai_server --host $BindHost --port $Port
