import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class LLMClient:
    """Minimal HTTP client for POSTing chat messages to /api/inference.

    The endpoint is expected to accept JSON with keys: model, messages, temperature.
    The response parsing is resilient to a few common formats.
    """

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        temperature: float = 0.2,
        default_headers: Optional[Dict[str, str]] = None,
        request_timeout_sec: float = 120.0,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.5,
    ) -> None:
        if not endpoint_url:
            raise ValueError("endpoint_url must be provided")
        self.endpoint_url = endpoint_url
        self.model = model
        self.temperature = float(temperature)
        self.request_timeout_sec = float(request_timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_backoff_sec = float(retry_backoff_sec)
        self.default_headers = {
            "content-type": "application/json",
            **(default_headers or {}),
        }

    def complete(self, messages: List[Dict[str, Any]], temperature: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        """Send messages to the inference API and return (text, raw_response).

        The returned text is extracted from a best-effort set of common response shapes.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": float(self.temperature if temperature is None else temperature),
        }
        raw = self._post_json(self.endpoint_url, payload)
        text = self._extract_text_from_response(raw)
        return text, raw

    # ----------------------------- internals ----------------------------- #

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                req = Request(url=url, data=data, headers=self.default_headers, method="POST")
                with urlopen(req, timeout=self.request_timeout_sec) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as err:  # type: ignore[name-defined]
                last_err = err
                if attempt > self.max_retries:
                    break
                time.sleep(self.retry_backoff_sec * attempt)
        # If we got here, we failed
        raise RuntimeError(f"/api/inference request failed after retries: {last_err}")

    @staticmethod
    def _extract_text_from_response(raw: Dict[str, Any]) -> str:
        """Try a few common shapes to extract the assistant text."""
        # 1) OpenAI-compatible
        try:
            choices = raw.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"]) or ""
        except Exception:
            pass
        # 2) Simple {"output": "..."}
        if isinstance(raw.get("output"), str):
            return str(raw["output"]) or ""
        # 3) Claude-like {"content":[{"type":"text","text":"..."}]}
        try:
            content = raw.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                if content[0].get("type") == "text" and "text" in content[0]:
                    return str(content[0]["text"]) or ""
        except Exception:
            pass
        # 4) Fall back to stringifying
        return json.dumps(raw)
