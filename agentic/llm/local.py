from __future__ import annotations

import json
import re
from typing import Dict, List


class LocalLLM:
    """A heuristic local model for offline/demo and tests.

    It tries to output a JSON control object with fields: thought, tool, tool_input, confidence.
    """

    def chat(self, messages: List[Dict[str, str]], tools_summary: str) -> str:
        # Inspect the conversation
        merged = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break

        # If we see an Observation with a numeric result, finalize
        observation_match = re.search(r"Observation:.*?\{.*?\"result\"\s*:\s*(\-?\d+(?:\.\d+)?)", merged, re.S)
        if observation_match:
            val = observation_match.group(1)
            return json.dumps({
                "thought": "I have the computed result; returning final answer.",
                "tool": "final",
                "tool_input": {"final_answer": val},
                "confidence": 0.85,
            })

        # Primitive arithmetic detection
        expr_match = re.search(r"(\d+\s*[+\-*/]\s*\d+(?:\s*[+\-*/]\s*\d+)*)", last_user)
        if expr_match:
            expr = expr_match.group(1)
            code = f"result = {expr}\nprint(result)"
            return json.dumps({
                "thought": f"Compute {expr} using the python tool.",
                "tool": "python",
                "tool_input": {"code": code, "timeout": 2},
                "confidence": 0.7,
            })

        # If they ask to read a file
        if any(k in merged.lower() for k in ["read file", "open ", "load file", "fs:"]):
            return json.dumps({
                "thought": "Use filesystem tool to read a file.",
                "tool": "fs",
                "tool_input": {"op": "read", "path": "README.md"},
                "confidence": 0.6,
            })

        # If they ask to search the web
        if any(k in merged.lower() for k in ["web", "search", "google", "duckduckgo"]):
            return json.dumps({
                "thought": "Search the web for relevant information.",
                "tool": "web",
                "tool_input": {"op": "search", "query": last_user[:120]},
                "confidence": 0.55,
            })

        # Default: finalize echoing intent
        return json.dumps({
            "thought": "No tool is necessary; respond directly.",
            "tool": "final",
            "tool_input": {"final_answer": last_user},
            "confidence": 0.5,
        })
