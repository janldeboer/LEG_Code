import time

try:
    from groq import Groq
except ImportError:
    raise SystemExit("pip install groq")

from .base import BaseProvider


class GroqProvider(BaseProvider):
    def __init__(self, api_key: str, model: str, temperature: float = 1.0, max_tokens: int = 600):
        self._client = Groq(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> dict:
        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            elapsed = round(time.time() - start, 2)
            choice = response.choices[0]
            return {
                "status":            "ok",
                "response":          choice.message.content,
                "finish_reason":     choice.finish_reason,
                "truncated":         choice.finish_reason == "length",
                "tokens_prompt":     response.usage.prompt_tokens,
                "tokens_completion": response.usage.completion_tokens,
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
