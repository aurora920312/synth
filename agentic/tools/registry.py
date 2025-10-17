from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from ..config import Config


@dataclass
class ToolResult:
    name: str
    ok: bool
    result: Any
    stderr: str | None = None


class Tool(Protocol):
    name: str

    def describe(self) -> str: ...

    def run(self, tool_input: Dict[str, Any]) -> ToolResult: ...


class PythonTool:
    name = "python"

    def describe(self) -> str:
        return (
            "python(code:str, timeout?:int) -> Execute sandboxed Python code and return stdout."
        )

    def run(self, tool_input: Dict[str, Any]) -> ToolResult:
        code = tool_input.get("code", "")
        timeout = int(tool_input.get("timeout", 5))
        if not code:
            return ToolResult(name=self.name, ok=False, result="missing code")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                text=True,
            )
            ok = proc.returncode == 0
            return ToolResult(name=self.name, ok=ok, result=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            return ToolResult(name=self.name, ok=False, result="timeout")


class FilesystemTool:
    name = "fs"

    def __init__(self, root: str):
        self.root = root

    def _abs(self, path: str) -> str:
        safe = os.path.abspath(os.path.join(self.root, path))
        if not safe.startswith(os.path.abspath(self.root)):
            raise ValueError("Path escapes workspace root")
        return safe

    def describe(self) -> str:
        return (
            "fs(op:read|write|list, path:str, data?:str) -> filesystem operations scoped to workspace root"
        )

    def run(self, tool_input: Dict[str, Any]) -> ToolResult:
        op = tool_input.get("op")
        path = tool_input.get("path", "")
        try:
            if op == "read":
                abs_path = self._abs(path)
                if not os.path.exists(abs_path):
                    return ToolResult(name=self.name, ok=False, result="not found")
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    return ToolResult(name=self.name, ok=True, result=f.read())
            if op == "write":
                abs_path = self._abs(path)
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                data = tool_input.get("data", "")
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(data)
                return ToolResult(name=self.name, ok=True, result="written")
            if op == "list":
                abs_path = self._abs(path or ".")
                if not os.path.isdir(abs_path):
                    return ToolResult(name=self.name, ok=False, result="not a directory")
                return ToolResult(name=self.name, ok=True, result=os.listdir(abs_path))
        except Exception as e:
            return ToolResult(name=self.name, ok=False, result=str(e))
        return ToolResult(name=self.name, ok=False, result="unsupported op")


class WebTool:
    name = "web"

    def describe(self) -> str:
        return (
            "web(op:search|get, query?:str, url?:str) -> performs web search (duckduckgo) or HTTP GET"
        )

    def run(self, tool_input: Dict[str, Any]) -> ToolResult:
        import json as _json

        op = tool_input.get("op")
        try:
            if op == "search":
                query = tool_input.get("query", "")
                if not query:
                    return ToolResult(name=self.name, ok=False, result="missing query")
                try:
                    import duckduckgo_search as dds  # type: ignore

                    results = dds.DDGS().text(query, max_results=5)
                    return ToolResult(name=self.name, ok=True, result=list(results))
                except Exception as e:  # pragma: no cover - depends on network
                    return ToolResult(name=self.name, ok=False, result=str(e))
            if op == "get":
                url = tool_input.get("url", "")
                if not url:
                    return ToolResult(name=self.name, ok=False, result="missing url")
                import requests  # type: ignore

                resp = requests.get(url, timeout=10)
                return ToolResult(name=self.name, ok=True, result={
                    "status": resp.status_code,
                    "text": resp.text[:20000],
                    "headers": dict(resp.headers),
                })
        except Exception as e:
            return ToolResult(name=self.name, ok=False, result=str(e))
        return ToolResult(name=self.name, ok=False, result="unsupported op")


def load_tools(cfg: Config) -> Dict[str, Tool]:
    tools: Dict[str, Tool] = {}
    for t in cfg.enabled_tools:
        tname = t.strip()
        if not tname:
            continue
        if tname == "python":
            tools[tname] = PythonTool()
        elif tname == "fs":
            tools[tname] = FilesystemTool(cfg.workspace_root)
        elif tname == "web":
            tools[tname] = WebTool()
    return tools
