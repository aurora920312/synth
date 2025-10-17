from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..config import Config
from ..tools.registry import Tool, ToolResult
from ..memory.run_logger import RunLogger
from ..llm.router import LLMRouter


THOUGHT_SCHEMA = {
    "required": ["thought", "tool", "tool_input"],
}


def build_tools_summary(tools: Dict[str, Tool]) -> str:
    if not tools:
        return "No tools are available. Respond directly."
    lines = [
        "Available tools:",
    ]
    for name, tool in tools.items():
        lines.append(f"- {name}: {tool.describe()}")
    lines.append(
        "Respond with ONLY JSON of shape {thought:str, tool:str, tool_input:object, confidence:number}."
    )
    lines.append("Use tool 'final' to finish with {final_answer: string}.")
    return "\n".join(lines)


class Agent:
    def __init__(self, llm: LLMRouter, tools: Dict[str, Tool], logger: RunLogger, config: Config):
        self.llm = llm
        self.tools = tools
        self.logger = logger
        self.config = config

    def _parse_control(self, text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            # Basic shape validation
            for key in THOUGHT_SCHEMA["required"]:
                if key not in data:
                    raise ValueError(f"missing {key}")
            return data
        except Exception as e:
            return {
                "thought": "Parser error; will finalize with raw text.",
                "tool": "final",
                "tool_input": {"final_answer": text},
                "confidence": 0.1,
            }

    def _run_tool(self, name: str, tool_input: Dict[str, Any]) -> Tuple[ToolResult, str]:
        if name == "final":
            # Virtual tool for finalization
            return ToolResult(name="final", ok=True, result=tool_input), ""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(name=name, ok=False, result="unknown tool"), ""
        res = tool.run(tool_input)
        return res, ""

    def run(self, user_prompt: str) -> Dict[str, Any]:
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": user_prompt}
        ]
        tools_summary = build_tools_summary(self.tools)
        run_id = self.logger.start_run(user_prompt, tools_summary)

        for step in range(self.config.max_steps):
            assistant_text = self.llm.complete(messages, tools_summary)
            control = self._parse_control(assistant_text)
            self.logger.log_step(run_id, step=step, messages=messages, assistant_text=assistant_text, control=control)

            tool_name: str = control.get("tool", "final")
            tool_input: Dict[str, Any] = control.get("tool_input", {})

            result, _ = self._run_tool(tool_name, tool_input)
            obs = {"ok": result.ok, "result": result.result, "stderr": result.stderr}
            messages.append({"role": "assistant", "content": json.dumps(control)})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs) }"})
            self.logger.log_observation(run_id, step=step, observation=obs)

            if tool_name == "final":
                final_answer = tool_input.get("final_answer", "") if isinstance(tool_input, dict) else tool_input
                out = {
                    "run_id": run_id,
                    "final_answer": final_answer,
                    "transcript": self.logger.get_run_path(run_id),
                }
                self.logger.end_run(run_id, out)
                return out

        # Max steps reached
        out = {
            "run_id": run_id,
            "final_answer": "Max steps reached without finalization.",
            "transcript": self.logger.get_run_path(run_id),
        }
        self.logger.end_run(run_id, out)
        return out
