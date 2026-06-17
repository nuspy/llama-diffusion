const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('appapi', {
  base: 'http://127.0.0.1:8787',
  onNavigate: (cb) => ipcRenderer.on('navigate', (_e, hash) => cb(hash)),
});
