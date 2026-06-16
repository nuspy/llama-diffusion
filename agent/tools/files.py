"""Tool su filesystem, confinati nella workspace sandbox.

Ogni path e' risolto rispetto a ctx.workspace e validato: niente accesso fuori
dalla radice (protezione da path traversal).
"""

from __future__ import annotations

from pathlib import Path

from .registry import Tool, ToolContext, ToolRegistry, ToolResult

MAX_READ_BYTES = 200_000      # evita di rovesciare file enormi nel contesto del modello


def _resolve(ctx: ToolContext, rel: str) -> Path:
    """Risolve `rel` dentro la workspace; solleva se esce dalla sandbox."""
    root = ctx.workspace.resolve()
    target = (root / rel).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path fuori dalla workspace: {rel}")
    return target


def _read_file(ctx: ToolContext, args: dict) -> ToolResult:
    p = _resolve(ctx, args["path"])
    if not p.exists():
        return ToolResult(False, error=f"file inesistente: {args['path']}")
    if not p.is_file():
        return ToolResult(False, error=f"non e' un file: {args['path']}")
    data = p.read_bytes()
    truncated = len(data) > MAX_READ_BYTES
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    if truncated:
        text += f"\n... [troncato a {MAX_READ_BYTES} byte di {len(data)}]"
    return ToolResult(True, output=text)


def _write_file(ctx: ToolContext, args: dict) -> ToolResult:
    p = _resolve(ctx, args["path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    content = args.get("content", "")
    p.write_text(content, encoding="utf-8")
    return ToolResult(True, output=f"scritti {len(content)} caratteri in {args['path']}")


def _edit_file(ctx: ToolContext, args: dict) -> ToolResult:
    """Sostituisce la prima (o tutte le) occorrenza di `old` con `new`."""
    p = _resolve(ctx, args["path"])
    if not p.exists():
        return ToolResult(False, error=f"file inesistente: {args['path']}")
    text = p.read_text(encoding="utf-8", errors="replace")
    old = args["old"]
    new = args.get("new", "")
    count = text.count(old)
    if count == 0:
        return ToolResult(False, error="stringa 'old' non trovata nel file")
    replace_all = str(args.get("all", "")).lower() in ("1", "true", "yes", "si")
    if count > 1 and not replace_all:
        return ToolResult(False, error=f"'old' compare {count} volte; passa all=true o rendi la stringa univoca")
    p.write_text(text.replace(old, new), encoding="utf-8")
    return ToolResult(True, output=f"sostituite {count if replace_all else 1} occorrenze in {args['path']}")


def _delete_file(ctx: ToolContext, args: dict) -> ToolResult:
    p = _resolve(ctx, args["path"])
    if not p.exists():
        return ToolResult(False, error=f"path inesistente: {args['path']}")
    if p.is_dir():
        import shutil
        shutil.rmtree(p)
        return ToolResult(True, output=f"cartella eliminata: {args['path']}")
    p.unlink()
    return ToolResult(True, output=f"file eliminato: {args['path']}")


def _list_dir(ctx: ToolContext, args: dict) -> ToolResult:
    p = _resolve(ctx, args.get("path", "."))
    if not p.exists():
        return ToolResult(False, error=f"cartella inesistente: {args.get('path', '.')}")
    entries = []
    for child in sorted(p.iterdir()):
        kind = "d" if child.is_dir() else "f"
        size = child.stat().st_size if child.is_file() else 0
        entries.append(f"{kind} {size:>10} {child.name}")
    return ToolResult(True, output="\n".join(entries) or "(vuota)")


def register_file_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        "read_file", "Legge il contenuto di un file di testo.",
        {"path": "percorso relativo alla workspace"},
        _read_file, required=["path"],
    ))
    reg.register(Tool(
        "write_file", "Crea o sovrascrive un file con il contenuto dato.",
        {"path": "percorso relativo", "content": "contenuto completo del file"},
        _write_file, required=["path"],
    ))
    reg.register(Tool(
        "edit_file", "Sostituisce una stringa esatta dentro un file esistente.",
        {"path": "percorso relativo", "old": "stringa da cercare", "new": "stringa sostitutiva",
         "all": "true per sostituire tutte le occorrenze (default false)"},
        _edit_file, required=["path", "old"],
    ))
    reg.register(Tool(
        "delete_file", "Elimina un file o una cartella (ricorsivo).",
        {"path": "percorso relativo"},
        _delete_file, required=["path"],
    ))
    reg.register(Tool(
        "list_dir", "Elenca i file di una cartella della workspace.",
        {"path": "percorso relativo (default: radice)"},
        _list_dir,
    ))
