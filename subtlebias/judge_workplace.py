"""
judge_workplace.py
------------------
Runs the workplace bias judge pipeline.
All parameters are read from config.toml — edit that file to change them.

Usage:
    pip install groq anthropic
    export GROQ_API_KEY="your_key"        # if using groq provider
    python judge_workplace.py
    python judge_workplace.py --config other_config.toml
"""

import json
import os
import argparse
from pathlib import Path

from workplace.settings import load as load_config
from workplace.config import StageConfig, PipelineConfig
from workplace.judge.factories import make_neutralizer, make_pairwise_judge, make_ranking_judge
from workplace.judge.pipeline import JudgePipeline
from workplace.utils import print_section, save_json, save_csv


def _safe_filename(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.toml (default: ./config.toml)")
    args = parser.parse_args()

    raw = load_config(args.config).get("judge", {})

    input_path    = raw.get("input",             "audit_workplace_full.json")
    output_dir    = Path(raw.get("output_dir",   "."))
    cache_name    = raw.get("neutralized_cache", "neutralized_cache.json")
    M             = raw.get("M",                 1)
    test_models   = raw.get("test_models",       None)
    skip_neut     = raw.get("skip_neutralization", False)
    skip_pairwise = raw.get("skip_pairwise",       False)
    skip_ranking  = raw.get("skip_ranking",        False)
    force_neut    = raw.get("force_neutralization", False)

    neut_raw = raw.get("neutralization", {})
    pair_raw = raw.get("pairwise",       {})
    rank_raw = raw.get("ranking",        {})
    pipe_raw = raw.get("pipeline",       {})

    cfg = PipelineConfig(
        neutralization=StageConfig(
            provider=neut_raw.get("provider", "groq"),
            model=neut_raw.get("model", "llama-3.1-8b-instant"),
            temperature=0.0, max_tokens=1000,
        ),
        pairwise=StageConfig(
            provider=pair_raw.get("provider", "groq"),
            model=pair_raw.get("model", "llama-3.3-70b-versatile"),
            temperature=0.0, max_tokens=300,
        ),
        ranking=StageConfig(
            provider=rank_raw.get("provider", "groq"),
            model=rank_raw.get("model", "llama-3.3-70b-versatile"),
            temperature=0.0, max_tokens=400,
        ),
        max_workers=pipe_raw.get("max_workers", 10),
        max_retries=pipe_raw.get("max_retries", 3),
        retry_delay=pipe_raw.get("retry_delay", 2.0),
        delay=pipe_raw.get("delay", 0.5),
        M=M,
    )

    providers = {cfg.neutralization.provider, cfg.pairwise.provider, cfg.ranking.provider}
    if "anthropic" in providers and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    if "groq" in providers and not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("LOADING DATA")
    groups = JudgePipeline.load_and_group(input_path)
    print(f"  Scenarios : {len(groups)}")

    # ── Neutralization (runs once, shared across all test models) ──
    cache_path = output_dir / cache_name
    if not force_neut and cache_path.exists():
        print(f"\n  Loading neutralized cache : {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            neutralized = json.load(f)
    elif skip_neut and cache_path.exists():
        # Backward-compatible behavior: allow "skip_neutralization" to mean
        # "use cache if present". If cache is missing, we'll fall through to recompute.
        print(f"\n  Loading neutralized cache : {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            neutralized = json.load(f)
    else:
        print_section(f"NEUTRALIZATION  [{cfg.neutralization.provider} / {cfg.neutralization.model}]")
        pipeline = JudgePipeline(
            neutralizer=make_neutralizer(cfg.neutralization),
            pairwise_judge=make_pairwise_judge(cfg.pairwise, cfg),
            ranking_judge=make_ranking_judge(cfg.ranking, cfg),
            cfg=cfg,
        )
        neutralized = pipeline.neutralize_all(groups)
        save_json(cache_path, neutralized)
        print(f"\n  Cache saved : {cache_path}")

    pipeline = JudgePipeline(
        neutralizer=make_neutralizer(cfg.neutralization),
        pairwise_judge=make_pairwise_judge(cfg.pairwise, cfg),
        ranking_judge=make_ranking_judge(cfg.ranking, cfg),
        cfg=cfg,
    )

    for test_model in (test_models or [None]):
        if test_model:
            tag  = _safe_filename(test_model)
            data = JudgePipeline.filter_by_model(neutralized, test_model)
            total_recs = sum(len(recs) for ids in data.values() for recs in ids.values())
            print_section(f"MODEL : {test_model}  ({total_recs} records)")
            if total_recs == 0:
                print(f"  No records found — skipping.")
                continue
        else:
            tag  = "all"
            data = neutralized

        if not skip_pairwise:
            print_section(f"PAIRWISE  [{cfg.pairwise.provider} / {cfg.pairwise.model}]  M={M}")
            pairwise = pipeline.run_pairwise(groups, data)
            failures = [r for r in pairwise if not r.get("success", True)]
            print(f"\n  Results: {len(pairwise) - len(failures)}/{len(pairwise)} successful")
            if failures:
                save_json(output_dir / f"pairwise_failures_{tag}.json", failures)
            save_csv(output_dir / f"pairwise_{tag}.csv", pairwise)
            print(f"  Saved : pairwise_{tag}.csv")

        if not skip_ranking:
            print_section(f"RANKING  [{cfg.ranking.provider} / {cfg.ranking.model}]  M={M}")
            ranking = pipeline.run_ranking(groups, data)
            failures = [r for r in ranking if not r.get("success", True)]
            print(f"\n  Results: {len(ranking) - len(failures)}/{len(ranking)} successful")
            if failures:
                save_json(output_dir / f"ranking_failures_{tag}.json", failures)
            save_json(output_dir / f"ranking_{tag}.json", ranking)
            print(f"  Saved : ranking_{tag}.json")

    print_section("Done")


if __name__ == "__main__":
    main()
