"""Assemblaggio del registry dei tool di default."""

from __future__ import annotations

from pathlib import Path

from .registry import Tool, ToolContext, ToolRegistry, ToolResult
from .files import register_file_tools
from .shell import register_shell_tools
from .browser import register_browser_tools
from .search import register_search_tools

__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "build_default_registry"]


def build_default_registry(enable_browser: bool = True, enable_search: bool = True) -> ToolRegistry:
    reg = ToolRegistry()
    register_file_tools(reg)
    register_shell_tools(reg)
    if enable_browser:
        register_browser_tools(reg)
    if enable_search:
        register_search_tools(reg)
    return reg
