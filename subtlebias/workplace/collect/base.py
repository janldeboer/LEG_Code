from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """LLM provider that generates a response for a given prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> dict:
        """Return a result dict with keys: status, response, finish_reason,
        truncated, tokens_prompt, tokens_completion, duration_seconds, error."""
        ...
