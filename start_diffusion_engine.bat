@echo off
REM Start DiffusionGemma: engine + system-tray icon (Electron app).
REM The chat window opens from the tray icon (left-click or "Open GUI").
REM For a pure headless engine (no tray) use scripts\serve.ps1 or the service.
cd /d "%~dp0gui"
npm start
if errorlevel 1 pause
