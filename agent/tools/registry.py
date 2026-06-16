"""Modello dei tool e registry.

Un Tool ha nome, descrizione, schema parametri (per il system prompt) e un handler.
Gli handler ricevono un ToolContext (la workspace sandbox) e un dict di argomenti.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    """Stato condiviso passato a ogni handler."""
    workspace: Path                     # radice sandbox: i tool file non escono da qui
    shell_timeout: int = 120            # timeout default per comandi shell (s)


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""

    def render(self) -> str:
        """Testo da re-iniettare nel modello come risultato del tool."""
        if self.ok:
            return self.output if self.output else "(ok, nessun output)"
        return f"ERRORE: {self.error}"


Handler = Callable[[ToolContext, dict], ToolResult]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # nome_param -> descrizione (per il prompt)
    handler: Handler
    required: list[str] = field(default_factory=list)

    def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        missing = [p for p in self.required if p not in args or args[p] is None]
        if missing:
            return ToolResult(False, error=f"argomenti mancanti per {self.name}: {', '.join(missing)}")
        try:
            return self.handler(ctx, args)
        except Exception as e:  # un tool non deve mai abbattere il loop
            return ToolResult(False, error=f"{type(e).__name__}: {e}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def render_for_prompt(self) -> str:
        """Genera la sezione del system prompt che documenta i tool disponibili."""
        lines: list[str] = []
        for t in self._tools.values():
            params = ", ".join(
                f"{p}{'*' if p in t.required else ''}" for p in t.parameters
            ) or "(nessuno)"
            lines.append(f"- {t.name}({params}): {t.description}")
            for p, desc in t.parameters.items():
                req = " [obbligatorio]" if p in t.required else ""
                lines.append(f"    · {p}{req}: {desc}")
        return "\n".join(lines)
