@echo off
REM Entry point - install the DiffusionGemma engine (Windows / double-click or cmd).
REM Forwards all args to scripts\install.ps1.
pwsh -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
if errorlevel 1 pause
