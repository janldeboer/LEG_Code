import re
import time

from .base import BaseProvider

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaProvider(BaseProvider):
    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 600):
        try:
            import ollama as _ollama
            self._ollama = _ollama
        except ImportError:
            raise SystemExit("pip install ollama")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def generate(self, prompt: str) -> dict:
        start = time.time()
        try:
            resp = self._ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self._temperature, "num_predict": self._max_tokens},
            )
            duration = round(time.time() - start, 2)
            finish = resp.done_reason or "stop"
            text = _THINK_RE.sub("", resp.message.content).strip()
            return {
                "status":            "ok",
                "response":          text,
                "finish_reason":     finish,
                "truncated":         finish == "length",
                "tokens_prompt":     resp.prompt_eval_count or 0,
                "tokens_completion": resp.eval_count or 0,
                "duration_seconds":  duration,
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
