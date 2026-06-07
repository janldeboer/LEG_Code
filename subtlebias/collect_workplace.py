"""
collect_workplace.py
--------------------
Collects LLM responses for workplace misconduct scenarios.
All parameters are read from config.toml — edit that file to change them.

Usage:
    pip install groq
    export GROQ_API_KEY="your_key"
    python collect_workplace.py
    python collect_workplace.py --config other_config.toml

    # For Ollama (no API key needed, Ollama must be running locally):
    pip install ollama
    ollama pull mistral
    # set provider = "ollama" and model = "mistral" in config.toml
    python collect_workplace.py
"""

import json
import os
import argparse
from pathlib import Path

from workplace.settings import DEFAULT_CONFIG, load as load_config
from workplace.collect.collector import DataCollector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to config.toml (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    cfg = load_config(args.config).get("collect", {})

    input_path    = cfg.get("input",    "workplace_scenarios.json")
    output        = cfg.get("output",   "audit_workplace_full.json")
    model         = cfg.get("model",    "llama-3.1-8b-instant")
    provider_name = cfg.get("provider", "groq")
    resume        = cfg.get("resume",   False)
    M             = int(cfg.get("M", 1))
    delay         = float(cfg.get("delay", 2.0))

    if provider_name == "ollama":
        from workplace.collect.ollama import OllamaProvider
        provider = OllamaProvider(model=model, temperature=1.0, max_tokens=2000)
    elif provider_name == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("GROQ_API_KEY not set.")
        from workplace.collect.groq import GroqProvider
        provider = GroqProvider(api_key=api_key, model=model, temperature=1.0, max_tokens=600)
    elif provider_name == "google_genai":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY not set.")
        from workplace.collect.google_genai import GoogleGenAIProvider
        provider = GoogleGenAIProvider(api_key=api_key, model=model, temperature=1.0, max_tokens=600)
    elif provider_name == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set.")
        from workplace.collect.openrouter import OpenRouterProvider
        provider = OpenRouterProvider(api_key=api_key, model=model, temperature=1.0, max_tokens=600)
    else:
        raise SystemExit(f"Unknown collect provider: {provider_name!r}  (choices: groq, ollama, google_genai, openrouter)")

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    collector = DataCollector(provider=provider, delay=delay)
    jobs      = collector.build_jobs(data["scenarios"], data["identities"], M=M)

    print(f"\n{'─' * 55}")
    print(f"  Config   : {config_path}")
    print(f"  Model    : {model}")
    print(f"  Scenarios: {len(data['scenarios'])}")
    print(f"  Jobs     : {len(jobs)}")
    print(f"  M        : {M}  (per gender/ethnicity group per scenario)")
    print(f"  Output   : {output}")
    print(f"{'─' * 55}\n")

    existing = []
    if resume and Path(output).exists():
        existing = DataCollector.load(output)
        done = sum(1 for r in existing if r.get("status") == "ok")
        print(f"  Resuming — {done} already done\n")

    results = collector.collect(jobs, existing_results=existing, model=model, output_path=output)
    DataCollector.save(results, output)

    ok     = sum(1 for r in results if r["status"] == "ok")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\n{'─' * 55}")
    print(f"  Done — {ok}/{len(jobs)} successful, {errors} errors")
    print(f"  {output}")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    main()
