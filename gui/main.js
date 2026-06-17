// Electron main process: system-tray controller + chat window + Python sidecar management
// for the DiffusionGemma engine (OpenAI adapter on http://127.0.0.1:8787).
//
// Tray menu (hidden icons): Open GUI · Preload model · Unload model · Settings · Quit.
// The adapter (agent/openai_server.py) is started as a sidecar ONLY if it isn't already
// running (service / serve.ps1) — we never spawn a second one (would double VRAM / clash port).

const { app, Tray, Menu, BrowserWindow, nativeImage, Notification } = require('electron');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const ROOT = path.join(__dirname, '..');            // project root (parent of gui/)
const HOST = '127.0.0.1';     // the GUI client (renderer) connects locally
const BIND = '0.0.0.0';       // the sidecar binds ALL interfaces so WSL/Hermes can reach it
const PORT = 8787;
const BASE = `http://${HOST}:${PORT}`;

let tray = null;
let win = null;
let sidecar = null;             // child process if we started it
let sidecarStartedByMe = false;

// ---- tiny HTTP helpers (no deps) -------------------------------------------
function httpGet(pathname, timeout = 3000) {
  return new Promise((resolve, reject) => {
    const req = http.get(BASE + pathname, { timeout }, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => resolve({ status: res.statusCode, body: data }));
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error('timeout')); });
  });
}
function httpPost(pathname, obj) {
  return new Promise((resolve, reject) => {
    const payload = Buffer.from(JSON.stringify(obj || {}));
    const req = http.request(BASE + pathname, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': payload.length },
      timeout: 600000,
    }, (res) => {
      let data = '';
      res.on('data', (c) => (data += c));
      res.on('end', () => resolve({ status: res.statusCode, body: data }));
    });
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

async function isUp() {
  try { const r = await httpGet('/health', 2000); return r.status === 200; }
  catch { return false; }
}

function notify(title, body) {
  try { new Notification({ title, body }).show(); } catch { /* ignore */ }
}

// ---- sidecar ---------------------------------------------------------------
async function ensureSidecar() {
  if (await isUp()) return;                          // already running (service / serve.ps1)
  const py = process.platform === 'win32' ? 'python' : 'python3';
  sidecar = spawn(py, ['-m', 'agent.openai_server', '--host', BIND, '--port', String(PORT)], {
    cwd: ROOT,
    env: { ...process.env, PYTHONPATH: ROOT, PYTHONUTF8: '1' },
    windowsHide: true,
  });
  sidecarStartedByMe = true;
  sidecar.stderr.on('data', (d) => process.stdout.write(`[sidecar] ${d}`));
}

function stopSidecar() {
  if (sidecar && sidecarStartedByMe) { try { sidecar.kill(); } catch { /* ignore */ } }
}

// ---- window ----------------------------------------------------------------
function openWindow(hash = '') {
  if (win) { win.show(); win.focus(); if (hash) win.webContents.send('navigate', hash); return; }
  win = new BrowserWindow({
    width: 960, height: 720, title: 'DiffusionGemma',
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  win.on('closed', () => { win = null; });
}

// ---- tray ------------------------------------------------------------------
function trayIcon() {
  const file = process.platform === 'win32' ? 'icon.ico' : 'icon.png';
  let img = nativeImage.createFromPath(path.join(__dirname, 'assets', file));
  if (img.isEmpty()) img = nativeImage.createFromPath(path.join(__dirname, 'assets', 'icon.png'));
  // Windows system tray expects a small icon: a 256x256 image renders blank.
  if (!img.isEmpty()) img = img.resize({ width: 16, height: 16 });
  return img;
}

function buildMenu() {
  return Menu.buildFromTemplate([
    { label: 'Open GUI', click: () => openWindow() },
    { type: 'separator' },
    {
      label: 'Preload model', click: async () => {
        notify('DiffusionGemma', 'Loading model into VRAM…');
        try { await httpPost('/admin/load', {}); notify('DiffusionGemma', 'Model loaded.'); }
        catch (e) { notify('DiffusionGemma', 'Load failed: ' + e.message); }
      }
    },
    {
      label: 'Unload model (free VRAM)', click: async () => {
        try { await httpPost('/admin/unload', {}); notify('DiffusionGemma', 'Model unloaded.'); }
        catch (e) { notify('DiffusionGemma', 'Unload failed: ' + e.message); }
      }
    },
    { type: 'separator' },
    { label: 'Settings', click: () => openWindow('#settings') },
    { type: 'separator' },
    { label: 'Quit', click: () => { stopSidecar(); app.quit(); } },
  ]);
}

app.whenReady().then(async () => {
  try {
    const ic = trayIcon();
    console.log('[main] tray icon: empty=' + ic.isEmpty() + ' size=' + JSON.stringify(ic.getSize()));
    tray = new Tray(ic);
    tray.setToolTip('DiffusionGemma');
    tray.setContextMenu(buildMenu());
    tray.on('click', () => openWindow());
    console.log('[main] tray created OK');
  } catch (e) {
    console.error('[main] tray creation FAILED:', e);
  }
  await ensureSidecar();
  notify('DiffusionGemma', 'In esecuzione nel tray. Click sull\'icona per aprire la chat.');
});

app.on('window-all-closed', (e) => { e.preventDefault(); });   // keep running in tray
app.on('before-quit', stopSidecar);
