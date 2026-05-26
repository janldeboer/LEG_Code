from abc import ABC, abstractmethod


class Neutralizer(ABC):
    """Rewrites a response to remove identity markers."""

    @abstractmethod
    def neutralize(self, text: str, name: str) -> str: ...


class PairwiseJudge(ABC):
    """Compares two responses and picks which treats misconduct more seriously."""

    @abstractmethod
    def judge(self, resp_a: str, resp_b: str, label_a: str, label_b: str) -> dict: ...


class RankingJudge(ABC):
    """Ranks four responses from most to least serious."""

    @abstractmethod
    def rank(self, responses: dict[str, str]) -> dict: ...
