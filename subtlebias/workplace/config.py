from dataclasses import dataclass, field

IDENTITY_KEYS     = ["male/white", "male/arabic", "female/white", "female/arabic"]
PAIRWISE_TRUNCATE = 1500
RANKING_TRUNCATE  = 1200


@dataclass
class StageConfig:
    provider:    str   = "anthropic"
    model:       str   = ""
    temperature: float = 0.0
    max_tokens:  int   = 300


@dataclass
class PipelineConfig:
    neutralization: StageConfig = field(default_factory=lambda: StageConfig(
        model="claude-haiku-4-5-20251001", max_tokens=1000
    ))
    pairwise: StageConfig = field(default_factory=lambda: StageConfig(
        model="claude-sonnet-4-20250514", max_tokens=300
    ))
    ranking: StageConfig = field(default_factory=lambda: StageConfig(
        model="claude-sonnet-4-20250514", max_tokens=400
    ))
    max_workers: int   = 10
    max_retries: int   = 3
    retry_delay: float = 2.0
    delay:       float = 0.5
    M:           int   = 1    # people per identity per scenario (zip rounds, not Cartesian product)
