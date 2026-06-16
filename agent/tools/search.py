"""Tool di ricerca web via DuckDuckGo (libreria `ddgs`, import lazy)."""

from __future__ import annotations

from .registry import Tool, ToolContext, ToolRegistry, ToolResult

_INSTALL_HINT = "Libreria di ricerca non disponibile. Installa con:\n  pip install ddgs"


def _search(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # nome storico
        except ImportError:
            return ToolResult(False, error=_INSTALL_HINT)

    query = args["query"]
    n = int(args.get("max_results", 5))
    results = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=n), 1):
            title = r.get("title", "")
            href = r.get("href") or r.get("url", "")
            body = r.get("body", "")
            results.append(f"{i}. {title}\n   {href}\n   {body}")
    if not results:
        return ToolResult(True, output="(nessun risultato)")
    return ToolResult(True, output="\n".join(results))


def register_search_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        "search", "Cerca sul web e restituisce i primi risultati (titolo, URL, snippet).",
        {"query": "stringa di ricerca", "max_results": "numero di risultati (default 5)"},
        _search, required=["query"],
    ))
