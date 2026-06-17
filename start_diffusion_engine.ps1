<#  Start DiffusionGemma: the inference ENGINE + the system-tray icon (Electron app).
    The app starts the engine (OpenAI API on 8787) as a sidecar and shows a tray icon.
    The chat WINDOW opens from the tray icon (left-click, or "Open GUI" in the menu).
    For a pure headless engine (no tray — e.g. as a Windows service) use scripts/serve.ps1.  #>
$here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Push-Location (Join-Path $here 'gui')
try { npm start } finally { Pop-Location }
