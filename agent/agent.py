import json
import os
import traceback
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .tools import ToolRegistry, ToolResult, default_registry, truncate_text
from .prompt import build_system_prompt


class ProgrammingExpertAgent:
    def __init__(
        self,
        client: LLMClient,
        tools: Optional[ToolRegistry] = None,
        max_steps: int = 20,
        observation_char_limit: int = 24_000,
    ) -> None:
        self.client = client
        self.tools = tools or default_registry()
        self.max_steps = int(max_steps)
        self.observation_char_limit = int(observation_char_limit)

    def run(self, task: str, verbose: bool = True) -> str:
        messages: List[Dict[str, Any]] = []
        # system
        system_prompt = build_system_prompt(self.tools.list_specs())
        messages.append({"role": "system", "content": system_prompt})
        # task
        messages.append({"role": "user", "content": task})

        final_answer: Optional[str] = None

        for step in range(1, self.max_steps + 1):
            assistant_text, _raw = self.client.complete(messages)
            if verbose:
                print(f"\n[assistant raw] {truncate_text(assistant_text, 1500)}\n")
            action = self._parse_action_json(assistant_text)
            if action is None:
                # ask the model to respond with valid JSON next
                messages.append({
                    "role": "user",
                    "content": "Your last reply was not valid JSON per the action schema. Respond again with strictly valid JSON only.",
                })
                continue

            if action.get("type") == "final":
                final_answer = self._extract_final_content(assistant_text, action)
                break

            if action.get("type") == "tool":
                tool_name = action.get("name")
                tool_args = action.get("args", {})
                # include the assistant action as part of conversation for transparency
                messages.append({"role": "assistant", "content": json.dumps(action)})
                obs = self._invoke_tool(tool_name, tool_args)
                obs_str = self._format_observation(tool_name, obs)
                messages.append({"role": "user", "content": obs_str})
                continue

            # unknown action type; nudge
            messages.append({
                "role": "user",
                "content": "Invalid action.type. Use 'tool' or 'final' per schema.",
            })

        if final_answer is None:
            final_answer = "Stopped without a final answer (max steps reached)."
        return final_answer

    # ------------------------------ internals ------------------------------ #

    @staticmethod
    def _parse_action_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            # Some models wrap JSON in markdown fences. Try to extract if so.
            stripped = text.strip()
            if stripped.startswith("```"):
                # find the last fence
                first = stripped.find("\n")
                last = stripped.rfind("```")
                if first != -1 and last != -1 and last > first:
                    candidate = stripped[first + 1 : last]
                else:
                    candidate = stripped
            else:
                candidate = stripped
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def _extract_final_content(text: str, action: Dict[str, Any]) -> str:
        # If the action JSON includes a 'final' field, use it; otherwise, use the raw text
        if "final" in action:
            try:
                # allow either string or object
                final_value = action["final"]
                if isinstance(final_value, str):
                    return final_value
                return json.dumps(final_value, ensure_ascii=False)
            except Exception:
                pass
        return text

    def _invoke_tool(self, name: Optional[str], args: Dict[str, Any]) -> ToolResult:
        if not name:
            return ToolResult(ok=False, output=None, error="Missing tool name")
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(ok=False, output=None, error=f"Unknown tool: {name}")
        try:
            return tool.run(args)
        except Exception as e:
            traceback.print_exc()
            return ToolResult(ok=False, output=None, error=f"Tool raised: {e}")

    def _format_observation(self, tool_name: str, result: ToolResult) -> str:
        payload = {
            "tool": tool_name,
            "ok": result.ok,
            "output": result.output if isinstance(result.output, (dict, list)) else str(result.output),
            "error": result.error,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"Observation from tool {tool_name}:\n{truncate_text(text, self.observation_char_limit)}"
