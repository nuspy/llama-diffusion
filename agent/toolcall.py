"""Parser dei tool-call emessi dal modello — tollerante e indipendente dal modello.

Riconosce tre formati (un modello a diffusione potrebbe non avere function-calling
nativo, quindi accettiamo piu' sintassi):

1) XML-like (formato primario insegnato nel system prompt):
   <tool name="write_file">
     <arg name="path">src/main.py</arg>
     <arg name="content">print("hi")</arg>
   </tool>

2) Fence ```tool con JSON:
   ```tool
   {"name": "write_file", "args": {"path": "src/main.py", "content": "..."}}
   ```

3) JSON inline: {"tool": "write_file", "args": {...}}

`parse_tool_calls` restituisce (testo_ripulito, [ToolCall...]) in ordine di apparizione.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_TOOL_XML = re.compile(r'<tool\s+name="([^"]+)"\s*>(.*?)</tool>', re.DOTALL)
_ARG_XML = re.compile(r'<arg\s+name="([^"]+)"\s*>(.*?)</arg>', re.DOTALL)
_FENCE = re.compile(r'```tool\s*\n(.*?)```', re.DOTALL)
_INLINE_JSON = re.compile(r'\{[^{}]*"(?:tool|name)"\s*:\s*"[^"]+"[^{}]*\}', re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict
    start: int          # posizione nel testo originale (per ordinamento/ripulitura)
    end: int


def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    calls: list[ToolCall] = []

    # 1) XML
    for m in _TOOL_XML.finditer(text):
        args = {k.strip(): v.strip() for k, v in _ARG_XML.findall(m.group(2))}
        calls.append(ToolCall(m.group(1).strip(), args, m.start(), m.end()))

    # 2) fenced json
    for m in _FENCE.finditer(text):
        obj = _try_json(m.group(1))
        if obj:
            name = obj.get("name") or obj.get("tool")
            if name:
                args = obj.get("args") or obj.get("arguments") or {}
                calls.append(ToolCall(name, args, m.start(), m.end()))

    # 3) inline json (solo se non gia' coperto da un fence)
    covered = [(c.start, c.end) for c in calls]
    for m in _INLINE_JSON.finditer(text):
        if any(s <= m.start() < e for s, e in covered):
            continue
        obj = _try_json(m.group(0))
        if obj:
            name = obj.get("name") or obj.get("tool")
            if name:
                args = obj.get("args") or obj.get("arguments") or {}
                calls.append(ToolCall(name, args, m.start(), m.end()))

    calls.sort(key=lambda c: c.start)

    # rimuovi i blocchi tool dal testo per ottenere la prosa "pulita" da mostrare
    clean = text
    for c in sorted(calls, key=lambda c: c.start, reverse=True):
        clean = clean[:c.start] + clean[c.end:]
    return clean.strip(), calls


def _try_json(s: str) -> dict | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None
