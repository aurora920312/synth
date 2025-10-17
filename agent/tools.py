import fnmatch
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Root directory will be locked to the current working directory at runtime
WORKSPACE_ROOT = os.getcwd()


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: Any
    error: Optional[str] = None


class Tool:
    name: str = "tool"
    description: str = ""
    arg_schema: Dict[str, Any] = {}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError


# ---------------------------- helper functions ---------------------------- #

def resolve_path(path: str) -> str:
    candidate = os.path.abspath(os.path.join(WORKSPACE_ROOT, path))
    if not candidate.startswith(WORKSPACE_ROOT):
        raise ToolError("Path escapes workspace root")
    return candidate


def truncate_text(text: str, max_chars: int = 10_000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n... [truncated {len(text) - max_chars} chars] ...\n\n{tail}"


DEFAULT_IGNORES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    ".cursor",
}


# --------------------------------- tools --------------------------------- #

class ListDirTool(Tool):
    name = "list_dir"
    description = "List files and directories. Args: path (str, default '.'), recursive (bool), include_hidden (bool)."
    arg_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean"},
            "include_hidden": {"type": "boolean"},
        },
        "additionalProperties": False,
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        recursive = bool(args.get("recursive", False))
        include_hidden = bool(args.get("include_hidden", False))
        root = resolve_path(path)
        results: List[Dict[str, Any]] = []
        if not recursive:
            try:
                for entry in sorted(os.listdir(root)):
                    if not include_hidden and entry.startswith("."):
                        continue
                    full = os.path.join(root, entry)
                    stat = os.stat(full)
                    results.append({
                        "name": entry,
                        "path": os.path.relpath(full, WORKSPACE_ROOT),
                        "is_dir": os.path.isdir(full),
                        "size": stat.st_size,
                    })
            except FileNotFoundError:
                return ToolResult(ok=False, output=None, error=f"Path not found: {path}")
            return ToolResult(ok=True, output=results)
        # recursive walk
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = os.path.relpath(dirpath, WORKSPACE_ROOT)
            # apply ignores
            dirnames[:] = [d for d in dirnames if (include_hidden or not d.startswith(".")) and d not in DEFAULT_IGNORES]
            for fname in sorted(filenames):
                if not include_hidden and fname.startswith("."):
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    stat = os.stat(full)
                except FileNotFoundError:
                    continue
                results.append({
                    "name": fname,
                    "path": os.path.join(rel_dir, fname),
                    "is_dir": False,
                    "size": stat.st_size,
                })
        return ToolResult(ok=True, output=results)


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file. Args: path (str), offset (int, optional), limit (int, optional)."
    arg_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path = resolve_path(args["path"])
        offset = int(args.get("offset", 0))
        limit = args.get("limit")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if offset:
                    for _ in range(offset):
                        f.readline()
                if limit is not None:
                    lines = []
                    for _ in range(int(limit)):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
            return ToolResult(ok=True, output={
                "path": os.path.relpath(path, WORKSPACE_ROOT),
                "content": content,
            })
        except FileNotFoundError:
            return ToolResult(ok=False, output=None, error=f"File not found: {args['path']}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write or append a file. Args: path (str), content (str), mode ('overwrite'|'append')."
    arg_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "mode": {"type": "string", "enum": ["overwrite", "append"]},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path = resolve_path(args["path"])
        content = str(args["content"])  # enforce string
        mode = args.get("mode", "overwrite")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            if mode == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            return ToolResult(ok=True, output={"path": os.path.relpath(path, WORKSPACE_ROOT), "bytes": len(content)})
        except Exception as e:
            return ToolResult(ok=False, output=None, error=f"Write failed: {e}")


class SearchTextTool(Tool):
    name = "search_text"
    description = "Search text across files. Args: pattern (str), regex (bool), include (list[str]), exclude (list[str]), ignore_case (bool)."
    arg_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "regex": {"type": "boolean"},
            "include": {"type": "array", "items": {"type": "string"}},
            "exclude": {"type": "array", "items": {"type": "string"}},
            "ignore_case": {"type": "boolean"},
            "max_results": {"type": "integer"},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"]) 
        use_regex = bool(args.get("regex", False))
        includes: List[str] = list(args.get("include", []) or [])
        excludes: List[str] = list(args.get("exclude", []) or [])
        ignore_case = bool(args.get("ignore_case", True))
        max_results = int(args.get("max_results", 500))

        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags) if use_regex else None

        results: List[Dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(WORKSPACE_ROOT):
            # ignore noisy dirs
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORES]
            for fname in filenames:
                # skip hidden
                if fname.startswith('.'):
                    continue
                rel_path = os.path.relpath(os.path.join(dirpath, fname), WORKSPACE_ROOT)
                if includes and not any(fnmatch.fnmatch(rel_path, pat) for pat in includes):
                    continue
                if excludes and any(fnmatch.fnmatch(rel_path, pat) for pat in excludes):
                    continue
                try:
                    with open(os.path.join(dirpath, fname), 'r', encoding='utf-8', errors='ignore') as f:
                        for idx, line in enumerate(f, start=1):
                            found = False
                            if use_regex:
                                if regex.search(line):
                                    found = True
                            else:
                                hay = line.lower() if ignore_case else line
                                needle = pattern.lower() if ignore_case else pattern
                                if needle in hay:
                                    found = True
                            if found:
                                results.append({
                                    "path": rel_path,
                                    "line": idx,
                                    "text": line.rstrip("\n"),
                                })
                                if len(results) >= max_results:
                                    return ToolResult(ok=True, output=results)
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
        return ToolResult(ok=True, output=results)


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command. Args: command (str), timeout_sec (int), cwd (str). Captures stdout/stderr/exit_code."
    arg_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_sec": {"type": "integer"},
            "cwd": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        command = str(args["command"]) 
        timeout_sec = int(args.get("timeout_sec", 30))
        cwd = args.get("cwd")
        if cwd:
            cwd = resolve_path(cwd)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd or WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                executable="/bin/bash",
            )
            return ToolResult(ok=True, output={
                "exit_code": proc.returncode,
                "stdout": proc.stdout.decode("utf-8", errors="replace"),
                "stderr": proc.stderr.decode("utf-8", errors="replace"),
            })
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, output=None, error="Command timed out")
        except Exception as e:
            return ToolResult(ok=False, output=None, error=f"Shell error: {e}")


class GitTool(Tool):
    name = "git"
    description = "Run basic git commands. Args: subcommand (str), args (list[str]). Allowed: status, diff, add, commit, log."
    arg_schema = {
        "type": "object",
        "properties": {
            "subcommand": {"type": "string"},
            "args": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["subcommand"],
        "additionalProperties": False,
    }

    ALLOWED = {"status", "diff", "add", "commit", "log"}

    def run(self, args: Dict[str, Any]) -> ToolResult:
        sub = str(args["subcommand"]) 
        if sub not in self.ALLOWED:
            return ToolResult(ok=False, output=None, error=f"Unsupported git subcommand: {sub}")
        extra_args = list(args.get("args", []) or [])
        cmd = ["git", sub] + extra_args
        try:
            proc = subprocess.run(
                cmd,
                cwd=WORKSPACE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return ToolResult(ok=True, output={
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            })
        except Exception as e:
            return ToolResult(ok=False, output=None, error=f"Git error: {e}")


# ------------------------------- registry ------------------------------- #

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for name, tool in sorted(self._tools.items()):
            specs.append({
                "name": name,
                "description": tool.description,
                "args": tool.arg_schema,
            })
        return specs


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ListDirTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(SearchTextTool())
    registry.register(ShellTool())
    registry.register(GitTool())
    return registry
