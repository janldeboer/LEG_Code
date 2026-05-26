import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from ..config import IDENTITY_KEYS, PipelineConfig
from ..utils import ThreadSafeCounter, retry_failed
from .base import Neutralizer, PairwiseJudge, RankingJudge


class JudgePipeline:
    def __init__(
        self,
        neutralizer: Neutralizer,
        pairwise_judge: PairwiseJudge,
        ranking_judge: RankingJudge,
        cfg: PipelineConfig,
    ):
        self._neutralizer    = neutralizer
        self._pairwise_judge = pairwise_judge
        self._ranking_judge  = ranking_judge
        self._cfg            = cfg

    # ── Data loading ──────────────────────────────────────────────────────────

    @staticmethod
    def load_and_group(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        ok = [r for r in data if r.get("status") == "ok" and r.get("response")]
        print(f"  Loaded : {len(ok)}/{len(data)} valid responses")
        groups = defaultdict(lambda: defaultdict(list))
        for r in ok:
            groups[r["scenario_id"]][f"{r['gender']}/{r['ethnicity']}"].append(r)
        return groups

    @staticmethod
    def filter_by_model(neutralized: dict, model_name: str) -> dict:
        """Return a copy of neutralized containing only entries from model_name."""
        filtered = defaultdict(lambda: defaultdict(list))
        for sid, identities in neutralized.items():
            for ikey, records in identities.items():
                for rec in records:
                    if rec["record"].get("model") == model_name:
                        filtered[sid][ikey].append(rec)
        return filtered

    # ── Neutralization ────────────────────────────────────────────────────────

    def neutralize_all(self, groups: dict) -> dict:
        total = sum(len(recs) for ids in groups.values() for recs in ids.values())
        print(f"\n  Neutralizing {total} responses...\n")

        neutralized = defaultdict(lambda: defaultdict(list))
        done = 0
        for sid, identities in sorted(groups.items()):
            for ikey, records in identities.items():
                for record in records:
                    done += 1
                    print(f"  [{done:>3}/{total}] {sid}  {ikey:<22} {record['name']:<12}", end=" ", flush=True)
                    neutralized[sid][ikey].append({
                        "text":   self._neutralizer.neutralize(record["response"], record["name"]),
                        "name":   record["name"],
                        "record": record,
                    })
                    print("✓")
                    time.sleep(0.3)
        return neutralized

    # ── Pairwise tournament ───────────────────────────────────────────────────

    def run_pairwise(self, groups: dict, neutralized: dict) -> list[dict]:
        tasks = []
        for sid in sorted(groups.keys()):
            available = [k for k in IDENTITY_KEYS if k in neutralized.get(sid, {})]
            for id_a, id_b in combinations(available, 2):
                recs_a = neutralized[sid][id_a]
                recs_b = neutralized[sid][id_b]
                m = min(self._cfg.M, len(recs_a), len(recs_b))
                for r in range(m):
                    tasks.append({
                        "sid": sid, "id_a": id_a, "id_b": id_b,
                        "rec_a": recs_a[r], "rec_b": recs_b[r],
                    })

        total = len(tasks) * 2  # ×2 for counterbalancing
        print(f"\n  Pairwise : {total} judge calls\n")
        counter = ThreadSafeCounter()

        def process(task):
            sid, id_a, id_b = task["sid"], task["id_a"], task["id_b"]
            rec_a, rec_b = task["rec_a"], task["rec_b"]

            r1 = self._pairwise_judge.judge(rec_a["text"], rec_b["text"], "A", "B")
            time.sleep(self._cfg.delay)
            counter.increment()

            r2 = self._pairwise_judge.judge(rec_b["text"], rec_a["text"], "A", "B")
            time.sleep(self._cfg.delay)
            n = counter.increment()

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
            print(f"  [{n:>5}/{total}] {n / total * 100:5.1f}%  {sid}  "
                  f"{rec_a['name']:<12} ({id_a[:10]}) vs {rec_b['name']:<12} ({id_b[:10]})  → {favored}")

            return {
                "scenario_id":          sid,
                "identity_a":           id_a,
                "identity_b":           id_b,
                "name_a":               rec_a["name"],
                "name_b":               rec_b["name"],
                "favored":              favored,
                "votes_a":              votes[id_a],
                "votes_b":              votes[id_b],
                "consequence_a_order1": r1.get("consequence_a", ""),
                "consequence_b_order1": r1.get("consequence_b", ""),
                "reasoning_order1":     r1.get("reasoning", ""),
                "reasoning_order2":     r2.get("reasoning", ""),
            }

        with ThreadPoolExecutor(max_workers=self._cfg.max_workers) as ex:
            results = list(ex.map(process, tasks))

        retry_failed(
            tasks, results, process,
            label_fn=lambda t: f"{t['sid']} {t['rec_a']['name']} vs {t['rec_b']['name']}",
            retry_delay=self._cfg.retry_delay,
        )
        return results

    # ── Ranking ───────────────────────────────────────────────────────────────

    def run_ranking(self, groups: dict, neutralized: dict) -> list[dict]:
        label_orders = [
            ["male/white", "male/arabic", "female/white", "female/arabic"],
            ["female/arabic", "male/white", "female/white", "male/arabic"],
            ["female/white", "female/arabic", "male/white", "male/arabic"],
        ]

        tasks = []
        for sid in sorted(groups.keys()):
            available = [k for k in IDENTITY_KEYS if k in neutralized.get(sid, {})]
            if len(available) < 4:
                print(f"  Warning: {sid} — only {len(available)}/4 identities, skipping ranking")
                continue
            # Zip sampling: round r pairs identity_k[r] across all 4 identities
            m = min(self._cfg.M, *(len(neutralized[sid][k]) for k in available))
            combos = [tuple(neutralized[sid][k][r] for k in available) for r in range(m)]
            for combo in combos:
                for order_idx, order in enumerate(label_orders):
                    tasks.append({"sid": sid, "combo": combo, "available": available,
                                  "order_idx": order_idx, "order": order})

        total = len(tasks)
        print(f"\n  Ranking : {total} judge calls\n")
        counter = ThreadSafeCounter()

        def process(task):
            sid, combo, available = task["sid"], task["combo"], task["available"]
            order_idx, order = task["order_idx"], task["order"]
            labels = ["A", "B", "C", "D"]

            label_to_record   = {lbl: combo[available.index(ikey)] for lbl, ikey in zip(labels, order)}
            label_to_identity = dict(zip(labels, order))

            result = self._ranking_judge.rank({lbl: rec["text"] for lbl, rec in label_to_record.items()})
            time.sleep(self._cfg.delay)
            n = counter.increment()

            names = " | ".join(label_to_record[lbl]["name"] for lbl in labels)
            print(f"  [{n:>5}/{total}] {n / total * 100:5.1f}%  {sid}  [{names}]")

            consequences = result.get("consequences", {})
            return {
                "scenario_id":       sid,
                "order_index":       order_idx,
                "label_order":       order,
                "names":             {lbl: label_to_record[lbl]["name"] for lbl in labels},
                "raw_ranking":       result.get("ranking", []),
                "label_to_identity": label_to_identity,
                "consequences":      {
                    label_to_identity[lbl]: val
                    for lbl, val in consequences.items()
                    if lbl in label_to_identity
                },
                "reasoning":         result.get("reasoning", ""),
            }

        with ThreadPoolExecutor(max_workers=self._cfg.max_workers) as ex:
            results = list(ex.map(process, tasks))

        retry_failed(
            tasks, results, process,
            label_fn=lambda t: (
                f"{t['sid']} [{' | '.join(t['combo'][t['available'].index(k)]['name'] for k in t['order'])}]"
            ),
            retry_delay=self._cfg.retry_delay,
        )
        return results
