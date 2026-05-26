import re

try:
    import anthropic
except ImportError:
    raise SystemExit("pip install anthropic")

from ..config import StageConfig, PAIRWISE_TRUNCATE, RANKING_TRUNCATE
from ..prompts import NEUTRALIZE_PROMPT, PAIRWISE_PROMPT, RANKING_PROMPT
from ..utils import call_with_retry
from .base import Neutralizer, PairwiseJudge, RankingJudge


class AnthropicNeutralizer(Neutralizer):
    def __init__(self, cfg: StageConfig):
        self._client = anthropic.Anthropic()
        self._cfg = cfg

    def neutralize(self, text: str, name: str) -> str:
        try:
            msg = self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": NEUTRALIZE_PROMPT.format(name=name, text=text)}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            print(f"\n    Warning: neutralization failed ({e}), using regex fallback")
            return re.sub(re.escape(name), "[ACCUSED]", text, flags=re.IGNORECASE)


class AnthropicPairwiseJudge(PairwiseJudge):
    def __init__(self, cfg: StageConfig, max_retries: int, retry_delay: float):
        self._client = anthropic.Anthropic()
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
            lambda: self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
            ).content[0].text,
            error_result={"favored": "equal"},
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )


class AnthropicRankingJudge(RankingJudge):
    def __init__(self, cfg: StageConfig, max_retries: int, retry_delay: float):
        self._client = anthropic.Anthropic()
        self._cfg = cfg
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    def rank(self, responses: dict[str, str]) -> dict:
        block = "\n\n---\n\n".join(
            f"Response {label} :\n{text[:RANKING_TRUNCATE]}"
            for label, text in responses.items()
        )
        return call_with_retry(
            lambda: self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                messages=[{"role": "user", "content": RANKING_PROMPT.format(responses_block=block)}],
            ).content[0].text,
            error_result={"ranking": []},
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )
