import re
import time

try:
    from google import genai
except ImportError:
    raise SystemExit("pip install google-genai")

from .base import BaseProvider

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class GoogleGenAIProvider(BaseProvider):
    """Google AI Studio (Gemini) provider. Strips <think>…</think> blocks."""

    def __init__(self, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 600):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> dict:
        start = time.time()
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            elapsed = round(time.time() - start, 2)
            text = (resp.text or "").strip()
            text = _THINK_RE.sub("", text).strip()
            finish = None
            usage_in = usage_out = 0
            try:
                if resp.usage_metadata:
                    usage_in = resp.usage_metadata.prompt_token_count or 0
                    usage_out = resp.usage_metadata.candidates_token_count or 0
            except Exception:
                pass
            try:
                if resp.candidates and resp.candidates[0].finish_reason:
                    finish = str(resp.candidates[0].finish_reason)
            except Exception:
                pass
            return {
                "status":            "ok",
                "response":          text,
                "finish_reason":     finish or "stop",
                "truncated":         finish == "MAX_TOKENS",
                "tokens_prompt":     usage_in,
                "tokens_completion": usage_out,
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
