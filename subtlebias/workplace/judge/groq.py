import re

try:
    from groq import Groq
except ImportError:
    raise SystemExit("pip install groq")

from ..config import StageConfig, PAIRWISE_TRUNCATE, RANKING_TRUNCATE
from ..prompts import NEUTRALIZE_PROMPT, PAIRWISE_PROMPT, RANKING_PROMPT
from ..utils import call_with_retry
from .base import Neutralizer, PairwiseJudge, RankingJudge


class GroqNeutralizer(Neutralizer):
    def __init__(self, cfg: StageConfig, api_key: str):
        self._client = Groq(api_key=api_key)
        self._cfg = cfg

    def neutralize(self, text: str, name: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": NEUTRALIZE_PROMPT.format(name=name, text=text)}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"\n    Warning: neutralization failed ({e}), using regex fallback")
            return re.sub(re.escape(name), "[ACCUSED]", text, flags=re.IGNORECASE)


class GroqPairwiseJudge(PairwiseJudge):
    def __init__(self, cfg: StageConfig, api_key: str, max_retries: int, retry_delay: float):
        self._client = Groq(api_key=api_key)
        self._cfg = cfg
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def judge(self, resp_a: str, resp_b: str, label_a: str, label_b: str) -> dict:
        prompt = PAIRWISE_PROMPT.format(
            label_a=label_a, label_b=label_b,
            response_a=resp_a[:PAIRWISE_TRUNCATE],
            response_b=resp_b[:PAIRWISE_TRUNCATE],
        )
        return call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
            ).choices[0].message.content,
            error_result={"favored": "equal"},
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )


class GroqRankingJudge(RankingJudge):
    def __init__(self, cfg: StageConfig, api_key: str, max_retries: int, retry_delay: float):
        self._client = Groq(api_key=api_key)
        self._cfg = cfg
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def rank(self, responses: dict[str, str]) -> dict:
        block = "\n\n---\n\n".join(
            f"Response {label} :\n{text[:RANKING_TRUNCATE]}"
            for label, text in responses.items()
        )
        return call_with_retry(
            lambda: self._client.chat.completions.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": RANKING_PROMPT.format(responses_block=block)}],
            ).choices[0].message.content,
            error_result={"ranking": []},
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )
