"""
retry_pairwise_errors.py
------------------------
Re-runs pairwise judge calls that failed with errors in a CSV and patches
the file in-place. Uses a high inter-call delay to avoid rate-limit errors.

Usage:
    python retry_pairwise_errors.py pairwise_llama-3.1-8b-instant.csv
    python retry_pairwise_errors.py pairwise_llama-3.1-8b-instant.csv pairwise_llama-3.3-70b-versatile.csv
    python retry_pairwise_errors.py pairwise_llama-3.1-8b-instant.csv --delay 10.0 --config config.toml
"""

import argparse
import csv
import json
import time
from pathlib import Path

from workplace.settings import load as load_config
from workplace.config import StageConfig, PipelineConfig
from workplace.judge.factories import make_pairwise_judge
from workplace.utils import print_section


def is_error(cell: str) -> bool:
    return cell.strip().startswith("error:")


def model_from_filename(path: Path) -> str | None:
    stem = path.stem  # e.g. "pairwise_llama-3.1-8b-instant"
    if stem.startswith("pairwise_"):
        return stem[len("pairwise_"):]
    return None


def _safe(model: str) -> str:
    return model.replace("/", "-").replace(":", "-")


def load_neutralized_for_model(cache_path: Path, model: str | None) -> dict:
    with open(cache_path, encoding="utf-8") as f:
        cache = json.load(f)
    if model is None:
        return cache
    filtered = {}
    for sid, identities in cache.items():
        filtered[sid] = {}
        for ikey, records in identities.items():
            recs = [r for r in records if _safe(r["record"].get("model", "")) == _safe(model)]
            if recs:
                filtered[sid][ikey] = recs
    return filtered


def get_text(neutralized: dict, scenario_id: str, identity: str) -> str | None:
    records = neutralized.get(scenario_id, {}).get(identity, [])
    return records[0]["text"] if records else None


def retry_csv(csv_path: Path, pairwise_judge, neutralized: dict, delay: float) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    errors = [
        (i, row) for i, row in enumerate(rows)
        if is_error(row.get("reasoning_order1", "")) or is_error(row.get("reasoning_order2", ""))
    ]

    if not errors:
        print(f"  No error rows found in {csv_path.name}")
        return

    print(f"\n  Found {len(errors)} error row(s) in {csv_path.name}\n")

    patched = 0
    for idx, (row_idx, row) in enumerate(errors):
        sid  = row["scenario_id"]
        id_a = row["identity_a"]
        id_b = row["identity_b"]

        text_a = get_text(neutralized, sid, id_a)
        text_b = get_text(neutralized, sid, id_b)

        if text_a is None or text_b is None:
            print(f"  [{idx+1}/{len(errors)}] {sid}  {id_a} vs {id_b}  — neutralized text not found, skipping")
            continue

        print(f"  [{idx+1}/{len(errors)}] {sid}  {row['name_a']} ({id_a}) vs {row['name_b']} ({id_b})", flush=True)

        # Re-run both orders to get clean votes from scratch
        print(f"    → order1 ...", end=" ", flush=True)
        r1 = pairwise_judge.judge(text_a, text_b, "A", "B")
        print("ok" if r1.get("success") else f"error: {r1.get('error', '?')}")
        time.sleep(delay)

        print(f"    → order2 ...", end=" ", flush=True)
        r2 = pairwise_judge.judge(text_b, text_a, "A", "B")
        print("ok" if r2.get("success") else f"error: {r2.get('error', '?')}")
        time.sleep(delay)

        if not r1.get("success") or not r2.get("success"):
            print(f"    Still failing — skipping row update")
            continue

        # Recompute votes using the same logic as pipeline.py
        votes = {id_a: 0, id_b: 0}
        if r1.get("favored") == "A": votes[id_a] += 1
        elif r1.get("favored") == "B": votes[id_b] += 1
        if r2.get("favored") == "A": votes[id_b] += 1
        elif r2.get("favored") == "B": votes[id_a] += 1

        favored = (
            id_a if votes[id_a] > votes[id_b] else
            id_b if votes[id_b] > votes[id_a] else
            "equal"
        )

        print(f"    → favored: {favored}  (votes: {id_a}={votes[id_a]}, {id_b}={votes[id_b]})")

        row["favored"]              = favored
        row["votes_a"]              = votes[id_a]
        row["votes_b"]              = votes[id_b]
        row["consequence_a_order1"] = r1.get("consequence_a", "")
        row["consequence_b_order1"] = r1.get("consequence_b", "")
        row["reasoning_order1"]     = r1.get("reasoning", "")
        row["reasoning_order2"]     = r2.get("reasoning", "")
        rows[row_idx] = row
        patched += 1

    if patched == 0:
        print(f"\n  Nothing patched — CSV unchanged.")
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Patched {patched}/{len(errors)} rows → saved {csv_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="One or more pairwise_*.csv files to patch")
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds between API calls (default: 6)")
    parser.add_argument("--config", default=None, help="Path to config.toml (default: ./config.toml)")
    args = parser.parse_args()

    raw = load_config(args.config).get("judge", {})
    cache_name = raw.get("neutralized_cache", "neutralized_cache.json")
    pair_raw   = raw.get("pairwise", {})
    pipe_raw   = raw.get("pipeline", {})

    cfg = PipelineConfig(
        neutralization=StageConfig(provider="groq", model="llama-3.1-8b-instant", temperature=0.0, max_tokens=1000),
        pairwise=StageConfig(
            provider=pair_raw.get("provider", "groq"),
            model=pair_raw.get("model", "llama-3.3-70b-versatile"),
            temperature=0.0,
            max_tokens=300,
        ),
        ranking=StageConfig(provider="groq", model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=400),
        max_workers=1,
        max_retries=pipe_raw.get("max_retries", 3),
        retry_delay=pipe_raw.get("retry_delay", 2.0),
        delay=args.delay,
    )

    judge = make_pairwise_judge(cfg.pairwise, cfg)

    cache_path = Path(cache_name)
    if not cache_path.exists():
        raise SystemExit(f"Neutralized cache not found: {cache_path}")

    print_section(f"RETRY PAIRWISE ERRORS  [delay={args.delay}s]")
    print(f"  Cache  : {cache_path}")
    print(f"  Judge  : {cfg.pairwise.provider} / {cfg.pairwise.model}")

    for csv_arg in args.csvs:
        csv_path = Path(csv_arg)
        if not csv_path.exists():
            print(f"\n  Warning: {csv_path} not found — skipping")
            continue

        model = model_from_filename(csv_path)
        print_section(f"FILE: {csv_path.name}  [model={model}]")

        neutralized = load_neutralized_for_model(cache_path, model)
        total_recs = sum(len(v) for ids in neutralized.values() for v in ids.values())
        print(f"  Neutralized records for this model: {total_recs}")

        retry_csv(csv_path, judge, neutralized, args.delay)

    print_section("Done")


if __name__ == "__main__":
    main()
