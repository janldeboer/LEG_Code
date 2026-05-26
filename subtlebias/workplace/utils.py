import csv
import json
import re
import threading
import time
import random
from pathlib import Path
from typing import Callable


def parse_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def call_with_retry(
    api_fn: Callable[[], str],
    error_result: dict,
    max_retries: int,
    retry_delay: float,
) -> dict:
    """
    Call an API function that returns a JSON-ish string and retry on failure.
    Uses exponential backoff with jitter (important for 429/rate limits).
    """
    for attempt in range(max_retries):
        try:
            return {"success": True, **parse_json(api_fn())}
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff + jitter to avoid thundering herd under rate limiting.
                backoff = retry_delay * (2 ** attempt)
                # Cap to something reasonable so a single job doesn't stall forever.
                backoff = min(backoff, 60.0)
                jitter = random.uniform(0.0, min(1.0, backoff * 0.1))
                time.sleep(backoff + jitter)
            else:
                return {"success": False, **error_result, "reasoning": f"error: {e}", "error": str(e)}


def retry_failed(tasks, results, process_fn, label_fn, retry_delay: float) -> None:
    failed = [(i, t) for i, (t, r) in enumerate(zip(tasks, results)) if not r.get("success", True)]
    if not failed:
        return
    print(f"\n  Warning: retrying {len(failed)} failed tasks sequentially...\n")
    for idx, task in failed:
        print(f"  Retry [{idx + 1}] {label_fn(task)}", end=" ")
        result = process_fn(task)
        results[idx] = result
        print("✓" if result.get("success", True) else "✗")
        time.sleep(retry_delay)


class ThreadSafeCounter:
    def __init__(self):
        self._n = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._n += 1
            return self._n


def print_section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def save_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
