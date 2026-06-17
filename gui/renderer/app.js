const BASE = window.appapi.base;
const $ = (id) => document.getElementById(id);

const chatEl = $('chat');
const inputEl = $('input');
const modelSel = $('model');
const statusEl = $('status');

let history = [];   // {role, content}

// ---- bootstrap ----
async function refresh() {
  try {
    const models = await (await fetch(`${BASE}/v1/models`)).json();
    modelSel.innerHTML = '';
    for (const m of models.data) {
      const o = document.createElement('option');
      o.value = m.id; o.textContent = m.id;
      modelSel.appendChild(o);
    }
    const h = await (await fetch(`${BASE}/health`)).json();
    statusEl.textContent = h.loaded_model ? `caricato: ${h.loaded_model}` : 'nessun modello in VRAM';
    statusEl.className = 'status ' + (h.loaded_model ? 'ok' : 'idle');
  } catch (e) {
    statusEl.textContent = 'motore non raggiungibile';
    statusEl.className = 'status err';
  }
}

$('load').onclick = async () => {
  statusEl.textContent = 'caricamento…';
  try {
    await fetch(`${BASE}/admin/load`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelSel.value }),
    });
  } catch {}
  refresh();
};

$('settingsBtn').onclick = () => $('settings').classList.toggle('hidden');
window.appapi.onNavigate((hash) => { if (hash === '#settings') $('settings').classList.remove('hidden'); });

// ---- chat ----
function addMsg(role) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const reason = document.createElement('details');
  reason.className = 'reasoning hidden';
  reason.innerHTML = '<summary>ragionamento</summary><pre></pre>';
  const body = document.createElement('div');
  body.className = 'body';
  const meta = document.createElement('div');
  meta.className = 'meta';
  wrap.append(reason, body, meta);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { reasonPre: reason.querySelector('pre'), reason, body, meta };
}

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg('user').body.textContent = text;
  history.push({ role: 'user', content: text });

  const ui = addMsg('assistant');
  ui.body.textContent = '…';
  let content = '', reasoning = '';

  try {
    const resp = await fetch(`${BASE}/v1/chat/completions`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelSel.value, stream: true,
        max_tokens: parseInt($('maxtok').value || '1024', 10),
        messages: history,
      }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop();
      for (const p of parts) {
        const line = p.replace(/^data:\s?/, '').trim();
        if (!line || line === '[DONE]') continue;
        let j; try { j = JSON.parse(line); } catch { continue; }
        const d = j.choices && j.choices[0] && j.choices[0].delta;
        if (d && d.reasoning_content) {
          reasoning += d.reasoning_content;
          ui.reason.classList.remove('hidden');
          ui.reasonPre.textContent = reasoning;
        }
        if (d && d.content) {
          content += d.content;
          ui.body.textContent = content;
        }
        if (j.usage && j.usage.timings) {
          const t = j.usage.timings;
          ui.meta.textContent = `${t.tokens_per_second} tok/s · ${(t.decode_ms/1000).toFixed(1)}s · ${t.denoising_steps} step`;
        }
        chatEl.scrollTop = chatEl.scrollHeight;
      }
    }
    if (!content) ui.body.textContent = '(nessuna risposta finale — prova ad aumentare i max token)';
    history.push({ role: 'assistant', content });
  } catch (e) {
    ui.body.textContent = 'errore: ' + e.message;
  }
  refresh();
}

$('send').onclick = send;
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

refresh();
setInterval(refresh, 15000);
