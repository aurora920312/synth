from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RunLogger:
    runs_dir: str

    def _ts(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")

    def _run_file(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, f"{run_id}.jsonl")

    def start_run(self, prompt: str, tools_summary: str) -> str:
        os.makedirs(self.runs_dir, exist_ok=True)
        run_id = self._ts()
        with open(self._run_file(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "start", "prompt": prompt, "tools": tools_summary}) + "\n")
        return run_id

    def log_step(self, run_id: str, step: int, messages: Any, assistant_text: str, control: Dict[str, Any]) -> None:
        with open(self._run_file(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "step",
                "step": step,
                "messages": messages,
                "assistant": assistant_text,
                "control": control,
            }) + "\n")

    def log_observation(self, run_id: str, step: int, observation: Dict[str, Any]) -> None:
        with open(self._run_file(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "observation",
                "step": step,
                "observation": observation,
            }) + "\n")

    def end_run(self, run_id: str, summary: Dict[str, Any]) -> None:
        with open(self._run_file(run_id), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event": "end",
                "summary": summary,
            }) + "\n")

    def get_run_path(self, run_id: str) -> str:
        return self._run_file(run_id)
