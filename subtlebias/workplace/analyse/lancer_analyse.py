"""
analyze_pairwise.py
-------------------
Analyse a pairwise tournament CSV and saves 4 figures independently.
Output filenames are prefixed with the CSV stem (e.g. llama-3.1-8b-instant_fig1_win_rate.png).

Usage:
    python analyze_pairwise.py <file.csv>
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load ──────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    required = {"scenario_id", "identity_a", "identity_b", "favored", "votes_a", "votes_b"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {missing}")
    df["votes_a"] = pd.to_numeric(df["votes_a"], errors="coerce").fillna(0)
    df["votes_b"] = pd.to_numeric(df["votes_b"], errors="coerce").fillna(0)
    return df

# ── Compute ───────────────────────────────────────────────────────────────────

def compute(df):
    identities = sorted(set(df["identity_a"]) | set(df["identity_b"]))

    stats = {i: {"wins": 0, "draws": 0, "losses": 0, "votes": [], "n": 0} for i in identities}
    for _, row in df.iterrows():
        a, b, f = row["identity_a"], row["identity_b"], row["favored"]
        stats[a]["votes"].append(row["votes_a"])
        stats[b]["votes"].append(row["votes_b"])
        stats[a]["n"] += 1
        stats[b]["n"] += 1
        if f == a:
            stats[a]["wins"]   += 1   # favored = treated more severely
            stats[b]["losses"] += 1
        elif f == b:
            stats[b]["wins"]   += 1
            stats[a]["losses"] += 1
        else:
            stats[a]["draws"] += 1
            stats[b]["draws"] += 1

    rows = []
    for i in identities:
        s = stats[i]
        n = s["n"]
        rows.append({
            "identity":  i,
            "wins":      s["wins"],
            "draws":     s["draws"],
            "losses":    s["losses"],
            "win_pct":   round(s["wins"] / n * 100, 1) if n else 0,
            "avg_votes": round(np.mean(s["votes"]), 2) if s["votes"] else 0,
        })
    stats_df = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)

    h2h_wins  = pd.DataFrame(0.0, index=identities, columns=identities)
    h2h_total = pd.DataFrame(0.0, index=identities, columns=identities)
    for _, row in df.iterrows():
        a, b, f = row["identity_a"], row["identity_b"], row["favored"]
        h2h_total.loc[a, b] += 1
        h2h_total.loc[b, a] += 1
        if f == a:
            h2h_wins.loc[a, b] += 1
        elif f == b:
            h2h_wins.loc[b, a] += 1
        else:
            h2h_wins.loc[a, b] += 0.5
            h2h_wins.loc[b, a] += 0.5
    h2h_rate = (h2h_wins / h2h_total.replace(0, np.nan)).round(2)

    records = []
    for scenario in sorted(df["scenario_id"].unique()):
        sdf = df[df["scenario_id"] == scenario]
        for i in identities:
            v = list(sdf[sdf["identity_a"] == i]["votes_a"]) + \
                list(sdf[sdf["identity_b"] == i]["votes_b"])
            if v:
                records.append({"scenario": scenario, "identity": i, "avg_votes": round(np.mean(v), 2)})
    scenario_df = pd.DataFrame(records).pivot(index="scenario", columns="identity", values="avg_votes")

    return stats_df, h2h_rate, scenario_df, identities

# ── Terminal summary ──────────────────────────────────────────────────────────

def print_summary(stats_df):
    print("\n" + "=" * 55)
    print("  RANKING  (win = favored = treated more severely)")
    print("=" * 55)
    for _, r in stats_df.iterrows():
        bar = "█" * int(r["win_pct"] / 5)
        print(f"  {r['identity']:<22} {r['win_pct']:5.1f}%  {bar}")
    print()

# ── Plots ─────────────────────────────────────────────────────────────────────

PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def save(fig, prefix, name):
    filename = f"{prefix}{name}" if prefix else name
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename}")

def plot(stats_df, h2h_rate, scenario_df, identities, prefix=""):
    colors = {i: PALETTE[k % len(PALETTE)] for k, i in enumerate(identities)}
    ordered = stats_df["identity"].tolist()

    # Figure 1 — win rate
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = stats_df.set_index("identity").loc[ordered, "win_pct"]
    bars = ax.barh(ordered, vals, color=[colors[i] for i in ordered], height=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9)
    ax.axvline(50, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.set_xlabel("Win rate (%)")
    ax.set_title("Win rate by identity\n(win = favored = treated more severely)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    save(fig, prefix, "fig1_win_rate.png")

    # Figure 2 — average votes
    fig, ax = plt.subplots(figsize=(7, 4))
    vals2 = stats_df.set_index("identity").loc[ordered, "avg_votes"]
    bars2 = ax.barh(ordered, vals2, color=[colors[i] for i in ordered], height=0.5)
    for bar, v in zip(bars2, vals2):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 2.3)
    ax.invert_yaxis()
    ax.set_xlabel("Average votes received (max = 2)")
    ax.set_title("Average votes received by identity")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    save(fig, prefix, "fig2_avg_votes.png")

    # Figure 3 — head-to-head heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    data = h2h_rate.loc[identities, identities].values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(identities)))
    ax.set_yticks(range(len(identities)))
    ax.set_xticklabels(identities, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(identities, fontsize=8)
    for i in range(len(identities)):
        for j in range(len(identities)):
            v = data[i, j]
            txt = "—" if np.isnan(v) else f"{v:.0%}"
            col = "white" if (not np.isnan(v) and (v > 0.65 or v < 0.35)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Win rate (row vs column)")
    ax.set_title("Head-to-head win rate\n(row more severely treated than column)")
    fig.tight_layout()
    save(fig, prefix, "fig3_head_to_head.png")

    # Figure 4 — grouped bar chart by scenario
    fig, ax = plt.subplots(figsize=(10, 5))
    scenarios = scenario_df.index.tolist()
    x = np.arange(len(scenarios))
    n = len(identities)
    width = 0.8 / n
    for k, ident in enumerate(identities):
        offset = (k - n / 2 + 0.5) * width
        vals3 = [scenario_df.loc[s, ident] if ident in scenario_df.columns else 0
                 for s in scenarios]
        ax.bar(x + offset, vals3, width=width * 0.9, color=colors[ident], label=ident)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Average votes")
    ax.set_ylim(0, 2.3)
    ax.set_title("Average votes by scenario and identity")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, prefix, "fig4_by_scenario.png")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze_pairwise.py <file.csv>")

    csv_path = sys.argv[1]
    # Derive prefix from filename: "pairwise_llama-3.1-8b-instant.csv" → "llama-3.1-8b-instant_"
    stem = csv_path.rsplit("/", 1)[-1].removesuffix(".csv")
    prefix = stem.removeprefix("pairwise_") + "_"

    df = load(csv_path)
    print(f"Loaded: {len(df)} comparisons, {df['scenario_id'].nunique()} scenarios")

    stats_df, h2h_rate, scenario_df, identities = compute(df)
    print_summary(stats_df)
    plot(stats_df, h2h_rate, scenario_df, identities, prefix=prefix)