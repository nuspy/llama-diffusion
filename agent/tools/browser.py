"""Tool browser headless via Playwright (import lazy).

navigate: carica una URL e ne estrae il testo (opzionalmente uno screenshot).
Gli handler sono sincroni: vanno eseguiti in un thread separato dall'event loop
del server (il core usa run_in_executor), perche' l'API sync di Playwright non puo'
girare dentro un loop asyncio.
"""

from __future__ import annotations

from .registry import Tool, ToolContext, ToolRegistry, ToolResult

MAX_TEXT = 20_000
_INSTALL_HINT = ("Playwright non disponibile. Installa con:\n"
                 "  pip install playwright beautifulsoup4\n"
                 "  python -m playwright install chromium")


def _navigate(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return ToolResult(False, error=_INSTALL_HINT)

    url = args["url"]
    want_shot = str(args.get("screenshot", "")).lower() in ("1", "true", "yes", "si")
    shot_path = None

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(500)
            title = page.title()
            html = page.content()
            if want_shot:
                shot_path = ctx.workspace / "browser_screenshot.png"
                page.screenshot(path=str(shot_path), full_page=False)
        finally:
            browser.close()

    text = _html_to_text(html)
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + f"\n... [troncato a {MAX_TEXT} caratteri]"
    out = f"# {title}\nURL: {url}\n\n{text}"
    if shot_path:
        out += f"\n\n[screenshot salvato in {shot_path.name}]"
    return ToolResult(True, output=out)


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html  # meglio l'HTML grezzo che niente
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def register_browser_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        "navigate", "Apre una pagina web in un browser headless e ne estrae il testo.",
        {"url": "URL completo da aprire",
         "screenshot": "true per salvare uno screenshot nella workspace"},
        _navigate, required=["url"],
    ))
