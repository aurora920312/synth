from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


DEFAULT_ENABLED_TOOLS: List[str] = [
    "python",
    "fs",
    "web",
]


@dataclass
class Config:
    model: str = os.getenv("AGENTIC_MODEL", os.getenv("OPENAI_MODEL", "local"))
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    max_steps: int = int(os.getenv("AGENTIC_MAX_STEPS", "12"))
    enabled_tools: List[str] = tuple(
        (os.getenv("AGENTIC_TOOLS", ",".join(DEFAULT_ENABLED_TOOLS))).split(",")
    )  # type: ignore[assignment]
    workspace_root: str = os.getenv("AGENTIC_WORKSPACE_ROOT", os.getcwd())
    runs_dir: str = os.getenv("AGENTIC_RUNS_DIR", os.path.join(workspace_root, "runs"))

    @staticmethod
    def from_env() -> "Config":
        return Config()
