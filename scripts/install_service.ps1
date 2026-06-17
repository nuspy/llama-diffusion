<#
  install_service.ps1 — installa l'adattatore OpenAI di DiffusionGemma come servizio Windows (NSSM).

  Config scelta: host 0.0.0.0, porta 8787, NESSUNA auth, avvio AUTOMATICO al boot.
  Hermes (WSL) lo raggiungera' a  http://172.18.160.1:8787/v1
  GUI locale a                    http://127.0.0.1:8787/v1

  ⚠️ ESEGUIRE COME AMMINISTRATORE (l'installazione di un servizio scrive in HKLM).
     Tasto destro su PowerShell -> "Esegui come amministratore", poi:
       pwsh -ExecutionPolicy Bypass -File E:\Projects\llama-diffusion\scripts\install_service.ps1
#>
$ErrorActionPreference = "Stop"

$svc  = "llamadiff"
$root = "E:\Projects\llama-diffusion"
$py   = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { throw "python non trovato nel PATH" }
$nssm = (Get-Command nssm -ErrorAction SilentlyContinue).Source
if (-not $nssm) { throw "nssm non trovato nel PATH" }

# verifica privilegi admin
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $admin) { throw "Questo script va eseguito come AMMINISTRATORE." }

New-Item -ItemType Directory -Force -Path "$root\.temp" | Out-Null

# rimuovi un'eventuale installazione precedente (idempotente)
& $nssm stop $svc 2>$null | Out-Null
& $nssm remove $svc confirm 2>$null | Out-Null

& $nssm install $svc $py "-m" "agent.openai_server" "--host" "0.0.0.0" "--port" "8787"
& $nssm set $svc AppDirectory $root
& $nssm set $svc AppEnvironmentExtra "PYTHONUTF8=1" "PYTHONPATH=$root"
& $nssm set $svc Start SERVICE_AUTO_START
& $nssm set $svc DisplayName "DiffusionGemma OpenAI API"
& $nssm set $svc Description "Adattatore OpenAI per il motore DiffusionGemma a diffusione (host 0.0.0.0 porta 8787)"
& $nssm set $svc AppStdout "$root\.temp\service.out.log"
& $nssm set $svc AppStderr "$root\.temp\service.err.log"
& $nssm set $svc AppRotateFiles 1
& $nssm set $svc AppExit Default Restart      # riavvia se crasha

# apri la porta 8787 nel firewall (per Hermes su WSL e per la LAN)
if (-not (Get-NetFirewallRule -DisplayName "llama-diffusion 8787" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "llama-diffusion 8787" -Direction Inbound -LocalPort 8787 -Protocol TCP -Action Allow | Out-Null
}

Write-Host ">> servizio '$svc' installato (auto-start). Avvio (il modello si carica alla prima richiesta, JIT)..." -ForegroundColor Green
& $nssm start $svc
Start-Sleep -Seconds 3
& $nssm status $svc
Write-Host ">> log: $root\.temp\service.err.log  |  health: curl http://127.0.0.1:8787/health" -ForegroundColor Cyan
