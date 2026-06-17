<#  Entry point — install the DiffusionGemma engine (Windows / PowerShell).
    Delegates to scripts/install.ps1. Forwards all args, e.g.:
      .\install.ps1 -ModelFile diffusiongemma-26B-A4B-it-Q4_K_M.gguf -MaxContext 16384
#>
& (Join-Path $PSScriptRoot "scripts\install.ps1") @args
