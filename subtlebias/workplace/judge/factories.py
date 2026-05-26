"""
To add a new provider:
  1. Create a class in a new file extending the ABC from base.py
  2. Implement the single abstract method
  3. Add an elif branch in the relevant factory below
"""

import os

from ..config import StageConfig, PipelineConfig
from .base import Neutralizer, PairwiseJudge, RankingJudge
from .anthropic import AnthropicNeutralizer, AnthropicPairwiseJudge, AnthropicRankingJudge
from .groq import GroqNeutralizer, GroqPairwiseJudge, GroqRankingJudge


def _groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise SystemExit("GROQ_API_KEY not set.")
    return key


def make_neutralizer(cfg: StageConfig) -> Neutralizer:
    if cfg.provider == "anthropic":
        return AnthropicNeutralizer(cfg)
    if cfg.provider == "groq":
        return GroqNeutralizer(cfg, api_key=_groq_key())
    raise ValueError(f"Unknown neutralizer provider: {cfg.provider!r}")


def make_pairwise_judge(cfg: StageConfig, pipeline: PipelineConfig) -> PairwiseJudge:
    if cfg.provider == "anthropic":
        return AnthropicPairwiseJudge(cfg, pipeline.max_retries, pipeline.retry_delay)
    if cfg.provider == "groq":
        return GroqPairwiseJudge(cfg, api_key=_groq_key(), max_retries=pipeline.max_retries, retry_delay=pipeline.retry_delay)
    raise ValueError(f"Unknown pairwise provider: {cfg.provider!r}")


def make_ranking_judge(cfg: StageConfig, pipeline: PipelineConfig) -> RankingJudge:
    if cfg.provider == "anthropic":
        return AnthropicRankingJudge(cfg, pipeline.max_retries, pipeline.retry_delay)
    if cfg.provider == "groq":
        return GroqRankingJudge(cfg, api_key=_groq_key(), max_retries=pipeline.max_retries, retry_delay=pipeline.retry_delay)
    raise ValueError(f"Unknown ranking provider: {cfg.provider!r}")
