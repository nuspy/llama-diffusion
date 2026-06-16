<#
  install.ps1 — automated install of the DiffusionGemma engine.

  Designed for agents too (Hermes/OpenCLAW): after the agent has asked the user WHICH model
  quantization and HOW MANY context tokens, it calls this script with those parameters. The
  chosen model file + context are persisted to config.json and picked up by the engine.

  Examples:
    pwsh scripts/install.ps1                                        # defaults: Q6_K, 32768 ctx
    pwsh scripts/install.ps1 -ModelFile diffusiongemma-26B-A4B-it-Q4_K_M.gguf -MaxContext 16384
    pwsh scripts/install.ps1 -SkipModel                            # everything but the download
#>
param(
    [string]$ModelRepo  = "unsloth/diffusiongemma-26B-A4B-it-GGUF",
    [string]$ModelFile  = "diffusiongemma-26B-A4B-it-Q6_K.gguf",
    [int]$MaxContext    = 32768,
    [int]$Ubatch        = 2048,
    [switch]$SkipModel,
    [switch]$SkipBuild
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# 1. diffusion-capable llama.cpp fork (PR #24423)
$engine = Join-Path $root "engine\llama.cpp"
if (-not (Test-Path (Join-Path $engine ".git"))) {
    Write-Host ">> cloning llama.cpp fork (PR #24423)..." -ForegroundColor Cyan
    git clone --depth=1 https://github.com/ggml-org/llama.cpp $engine
    git -C $engine fetch --depth=1 origin pull/24423/head:diffusion-gemma
    git -C $engine checkout diffusion-gemma
}

# 2. build the CUDA engine (handles VS-toolset / _CL_ / C++17 gotchas)
if (-not $SkipBuild) { & (Join-Path $PSScriptRoot "build_engine.ps1") }

# 3. python dependencies
python -m pip install -r (Join-Path $root "requirements.txt")
try { python -m playwright install chromium } catch { Write-Warning "playwright install skipped: $_" }

# 4. model download
$dest = Join-Path $root "models"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$target = Join-Path $dest $ModelFile
if (-not $SkipModel -and -not (Test-Path $target)) {
    $url = "https://huggingface.co/$ModelRepo/resolve/main/$ModelFile"
    Write-Host ">> downloading model (resumable): $url" -ForegroundColor Cyan
    curl.exe -L -C - -o $target $url
}

# 5. persist the chosen model + context so the engine uses them
$cfg = [ordered]@{ model_file = $ModelFile; maxtok = $MaxContext; ubatch = $Ubatch }
$cfg | ConvertTo-Json | Set-Content -Path (Join-Path $root "config.json") -Encoding UTF8
Write-Host ">> config.json written: model=$ModelFile maxtok=$MaxContext ubatch=$Ubatch" -ForegroundColor Green
Write-Host ">> start with:  pwsh scripts/serve.ps1   (or scripts/install_service.ps1 as admin)" -ForegroundColor Green
