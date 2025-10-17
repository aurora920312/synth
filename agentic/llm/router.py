from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import Config
from .openai_client import OpenAIClient
from .local import LocalLLM


class LLMRouter:
    def __init__(self, model: str, openai_client: Optional[OpenAIClient], local: LocalLLM):
        self.model = model
        self.openai_client = openai_client
        self.local = local

    @staticmethod
    def from_config(cfg: Config) -> "LLMRouter":
        use_openai = cfg.openai_api_key is not None and cfg.model.lower() != "local"
        openai_client = OpenAIClient(cfg.openai_api_key, cfg.model) if use_openai else None
        local = LocalLLM()
        return LLMRouter(model=cfg.model, openai_client=openai_client, local=local)

    def complete(self, messages: List[Dict[str, str]], tools_summary: str) -> str:
        """Return assistant text. The agent ensures JSON formatting externally.
        """
        if self.openai_client is not None:
            try:
                return self.openai_client.chat(messages, tools_summary=tools_summary)
            except Exception as e:
                # Fallback to local if OpenAI fails
                return self.local.chat(messages, tools_summary=tools_summary)
        return self.local.chat(messages, tools_summary=tools_summary)
