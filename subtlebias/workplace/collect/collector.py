import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from .base import BaseProvider


class DataCollector:
    def __init__(self, provider: BaseProvider, delay: float = 2.0, save_every: int = 5):
        self._provider = provider
        self._delay = delay
        self._save_every = save_every

    def build_jobs(self, scenarios: list[dict], identities: list[dict], M: int = 1) -> list[dict]:
        if M < 1:
            raise ValueError(f"M must be >= 1 (got {M})")

        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for identity in identities:
            groups[(identity["gender"], identity["ethnicity"])].append(identity)

        jobs = []
        group_keys = sorted(groups.keys())
        group_sizes = {k: len(groups[k]) for k in group_keys}

        for s_idx, scenario in enumerate(scenarios):
            for k in group_keys:
                size = group_sizes[k]
                if size == 0:
                    continue

                start = (s_idx * M) % size
                for j in range(M):
                    identity = groups[k][(start + j) % size]

                    prompt = scenario["prompt"]
                    for key, val in identity.items():
                        if key not in ("gender", "ethnicity", "scenario_id"):
                            if isinstance(val, str):
                                prompt = prompt.replace("{" + key + "}", val)
                    jobs.append({
                        "run_id":           str(uuid.uuid4()),
                        "scenario_id":      scenario["id"],
                        "scenario_context": scenario["context"],
                        "name":             identity["NAME"],
                        "gender":           identity["gender"],
                        "ethnicity":        identity["ethnicity"],
                        "prompt":           prompt,
                    })
        return jobs

    def collect(
        self,
        jobs: list[dict],
        existing_results: list[dict] | None = None,
        model: str = "",
        output_path: str | Path | None = None,
    ) -> list[dict]:
        results = list(existing_results or [])
        done_ids = {r["run_id"] for r in results if r.get("status") == "ok"}
        total = len(jobs)

        for i, job in enumerate(jobs, 1):
            if job["run_id"] in done_ids:
                continue

            print(
                f"  [{i:>3}/{total}]  {job['scenario_id']}  "
                f"{job['gender'][:1].upper()}/{job['ethnicity'][:3]}  "
                f"{job['name']:<12}",
                end=" ", flush=True,
            )

            result = self._provider.generate(job["prompt"])
            record = {
                "run_id":           job["run_id"],
                "scenario_id":      job["scenario_id"],
                "scenario_context": job["scenario_context"],
                "name":             job["name"],
                "gender":           job["gender"],
                "ethnicity":        job["ethnicity"],
                "model":            model,
                "prompt":           job["prompt"],
                **result,
            }
            results.append(record)

            icon  = "✓" if result["status"] == "ok" else "✗"
            trunc = " [TRUNCATED]" if result["truncated"] else ""
            print(f"{icon} {result['duration_seconds']}s ({result['tokens_completion']} tok){trunc}")

            if result["status"] == "error":
                print(f"       ↳ {result['error']}")

            if output_path and len(results) % self._save_every == 0:
                self.save(results, output_path)

            time.sleep(self._delay)

        return results

    @staticmethod
    def save(results: list[dict], path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
