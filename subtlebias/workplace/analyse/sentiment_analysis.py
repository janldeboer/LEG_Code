"""
sentiment_analysis.py
---------------------
Sentiment analysis of neutralized LLM responses, visualised as a
ethnicity × gender heatmap per model.

Requires one of: vaderSentiment, textblob  (pip install vaderSentiment)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workplace.analyse.embedding_analysis import (
    load_neutralized_responses, DEFAULT_MODELS, DEFAULT_SCENARIOS, _tail,
)

# ── Sentiment ─────────────────────────────────────────────────────────────────

def _sentiment_vader(texts: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def _sentiment_textblob(texts: list[str]) -> list[float]:
    from textblob import TextBlob
    return [TextBlob(t).sentiment.polarity for t in texts]


def compute_sentiment(texts: list[str]) -> list[float]:
    """Return a compound sentiment score in [-1, 1] for each text.

    Each text is truncated to its first and last 20% before scoring.
    Tries vaderSentiment then textblob.
    """
    truncated = [_tail(t) for t in texts]
    for fn in (_sentiment_vader, _sentiment_textblob):
        try:
            scores = fn(truncated)
            print(f"  sentiment backend: {fn.__name__.replace('_sentiment_', '')}")
            return scores
        except Exception as exc:
            print(f"  [{fn.__name__}] unavailable: {exc}", file=sys.stderr)
    raise RuntimeError(
        "No sentiment backend found. Install one:\n"
        "  pip install vaderSentiment\n  pip install textblob"
    )


# ── Matrix ────────────────────────────────────────────────────────────────────

def build_sentiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (ethnicity × gender) pivot table of mean sentiment scores."""
    return df.pivot_table(
        values="sentiment",
        index="ethnicity",
        columns="gender",
        aggfunc="mean",
    )


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_matrices(
    matrices: dict[str, pd.DataFrame],
    title: str = "Mean sentiment by ethnicity × gender",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """One heatmap subplot per model.

    Parameters
    ----------
    matrices:
        {model_name: DataFrame(index=ethnicity, columns=gender)}
    """
    models = list(matrices.keys())
    n = len(models)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    if n == 1:
        axs = [axs]

    vmin = min(m.values.min() for m in matrices.values())
    vmax = max(m.values.max() for m in matrices.values())
    # symmetric around 0 so colours read intuitively
    bound = max(abs(vmin), abs(vmax))

    for ax, model in zip(axs, models):
        mat = matrices[model]
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=-bound, vmax=bound,
                       aspect="auto")

        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, fontsize=9)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_title(model, fontsize=9, pad=6)

        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                val = mat.values[i, j]
                if np.isnan(val):
                    continue
                color = "white" if abs(val) > bound * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=axs, shrink=0.7, label="mean sentiment (−1 neg → +1 pos)")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    return fig


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_sentiment_analysis(
    cache_paths: list[str | Path],
    scenarios: list[str] | None = None,
    models: list[str] | None = None,  # defaults to DEFAULT_MODELS
    output_dir: str | Path = ".",
    output_name: str = "sentiment_matrix.png",
) -> None:
    """Load responses → compute sentiment → plot ethnicity × gender matrix per model.

    Parameters
    ----------
    cache_paths:
        Neutralized cache JSON files to load.
    scenarios:
        Scenario IDs to include. Defaults to WP-001/002/003.
    models:
        Model names to include. Defaults to SMALL_MODELS.
    output_dir:
        Directory where the PNG is saved.
    output_name:
        Output filename.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    models = models or DEFAULT_MODELS

    df = load_neutralized_responses(cache_paths, scenarios, models)

    print(f"Computing sentiment on {len(df)} responses …")
    df["sentiment"] = compute_sentiment(df["text"].tolist())

    matrices = {}
    for model in models:
        sub = df[df["model"] == model]
        if sub.empty:
            print(f"[warn] no data for '{model}', skipping", file=sys.stderr)
            continue
        matrices[model] = build_sentiment_matrix(sub)

    out = Path(output_dir) / output_name
    plot_matrices(matrices, output_path=out)
