"""Tool di esecuzione comandi: run (shell) e compile, con cwd nella workspace."""

from __future__ import annotations

import subprocess

from .registry import Tool, ToolContext, ToolRegistry, ToolResult

MAX_OUTPUT = 30_000


def _truncate(s: str) -> str:
    if len(s) > MAX_OUTPUT:
        return s[:MAX_OUTPUT] + f"\n... [output troncato a {MAX_OUTPUT} caratteri]"
    return s


def _run_cmd(ctx: ToolContext, command: str, timeout: int) -> ToolResult:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(ctx.workspace),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, error=f"comando andato in timeout dopo {timeout}s: {command}")
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    out = _truncate(out.strip())
    if proc.returncode != 0:
        return ToolResult(False, error=f"exit code {proc.returncode}\n{out}")
    return ToolResult(True, output=out or f"(exit 0, nessun output)")


def _run(ctx: ToolContext, args: dict) -> ToolResult:
    timeout = int(args.get("timeout", ctx.shell_timeout))
    return _run_cmd(ctx, args["command"], timeout)


def _compile(ctx: ToolContext, args: dict) -> ToolResult:
    # alias semantico di run con timeout piu' lungo per build/compilazioni
    timeout = int(args.get("timeout", max(ctx.shell_timeout, 600)))
    return _run_cmd(ctx, args["command"], timeout)


def register_shell_tools(reg: ToolRegistry) -> None:
    reg.register(Tool(
        "run", "Esegue un comando di shell nella workspace e ne restituisce l'output.",
        {"command": "comando da eseguire (cmd.exe su Windows)",
         "timeout": "secondi prima del timeout (default 120)"},
        _run, required=["command"],
    ))
    reg.register(Tool(
        "compile", "Esegue un comando di build/compilazione (timeout esteso a 600s).",
        {"command": "comando di build (es. 'npm run build', 'cargo build', 'cmake --build .')",
         "timeout": "secondi prima del timeout (default 600)"},
        _compile, required=["command"],
    ))
