from __future__ import annotations

from typing import Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - library may be absent in CI
    OpenAI = None  # type: ignore


class OpenAIClient:
    def __init__(self, api_key: Optional[str], model: str):
        if OpenAI is None:
            raise RuntimeError("openai library not installed")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], tools_summary: str) -> str:
        system_prefix = {
            "role": "system",
            "content": (
                "You are a precise agent. Always respond with ONLY a JSON object.\n"
                + tools_summary
            ),
        }
        payload = [system_prefix] + messages
        resp = self.client.chat.completions.create(model=self.model, messages=payload, temperature=0.2)
        return resp.choices[0].message.content or "{}"
