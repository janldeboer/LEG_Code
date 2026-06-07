import re
import time

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

from .base import BaseProvider

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OpenRouterProvider(BaseProvider):
    """OpenRouter :free provider via the OpenAI-compatible API.

    Logs the x-request-id response header for traceability (free models
    can be de-listed without notice).
    """

    def __init__(self, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 600):
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._last_request_id: str | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def last_request_id(self) -> str | None:
        return self._last_request_id

    def generate(self, prompt: str) -> dict:
        start = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            elapsed = round(time.time() - start, 2)
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            text = _THINK_RE.sub("", text).strip()
            try:
                self._last_request_id = resp._request_id or None
            except Exception:
                self._last_request_id = None
            return {
                "status":            "ok",
                "response":          text,
                "finish_reason":     choice.finish_reason or "stop",
                "truncated":         choice.finish_reason == "length",
                "tokens_prompt":     (resp.usage.prompt_tokens or 0) if resp.usage else 0,
                "tokens_completion": (resp.usage.completion_tokens or 0) if resp.usage else 0,
                "duration_seconds":  elapsed,
                "error":             None,
            }
        except Exception as e:
            return {
                "status":            "error",
                "response":          None,
                "finish_reason":     None,
                "truncated":         False,
                "tokens_prompt":     0,
                "tokens_completion": 0,
                "duration_seconds":  round(time.time() - start, 2),
                "error":             str(e),
            }
