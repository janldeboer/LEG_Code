"""
embedding_analysis.py
---------------------
UMAP visualisation of neutralized LLM response embeddings.

Each neutralized cache file (neutralized_cache*.json) holds the pre-judge
responses that were fed into the pairwise comparison stage. This module
embeds those texts and projects them into 2D via UMAP so we can inspect
whether responses cluster by identity (gender × ethnicity).

Usage (via run_embedding_analysis.py):
    python run_embedding_analysis.py --caches neutralized_cache.json \
        neutralized_cache_all.json neutralized_cache_deepseek.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

SMALL_MODELS = ["llama-3.1-8b-instant", "mistral-7b", "deepseek-r1-7b"]
DEFAULT_MODELS = ["llama-3.1-8b-instant", "mistral-7b", "deepseek-r1-7b", "llama-3.3-70b-versatile"]
DEFAULT_SCENARIOS = ["WP-001", "WP-002", "WP-003"]
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

IDENTITY_COLORS: dict[str, str] = {
    "male/white":    "#4878D0",
    "male/arabic":   "#EE854A",
    "female/white":  "#6ACC65",
    "female/arabic": "#D65F5F",
}
MODEL_MARKERS: dict[str, str] = {}   # filled lazily in plot_umap


def _normalize_model(name: str) -> str:
    """Map Ollama-style names to config-style: 'mistral:7b' → 'mistral-7b'."""
    return name.replace(":", "-")


# ── Load ──────────────────────────────────────────────────────────────────────

def load_neutralized_responses(
    cache_paths: list[str | Path],
    scenarios: list[str],
    models: list[str],
) -> pd.DataFrame:
    """Load and merge neutralized responses from one or more cache files.

    Parameters
    ----------
    cache_paths:
        Paths to neutralized_cache*.json files to load and merge.
    scenarios:
        Scenario IDs to keep (e.g. ["WP-001", "WP-002"]).
    models:
        Normalised model names to keep (e.g. ["mistral-7b", "deepseek-r1-7b"]).

    Returns
    -------
    DataFrame with columns:
        run_id, scenario_id, model, gender, ethnicity, identity, text
    """
    rows: list[dict] = []
    seen_run_ids: set[str] = set()

    for path in cache_paths:
        path = Path(path)
        if not path.exists():
            print(f"[warn] cache file not found, skipping: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            cache: dict = json.load(f)

        for scenario_id, identities in cache.items():
            if scenario_id not in scenarios:
                continue
            for identity, entries in identities.items():
                for entry in entries:
                    record = entry.get("record", {})
                    run_id = record.get("run_id", "")
                    if run_id in seen_run_ids:
                        continue
                    model_raw = record.get("model", "")
                    model_norm = _normalize_model(model_raw)
                    if model_norm not in models:
                        continue
                    seen_run_ids.add(run_id)
                    rows.append({
                        "run_id":      run_id,
                        "scenario_id": scenario_id,
                        "model":       model_norm,
                        "gender":      record.get("gender", ""),
                        "ethnicity":   record.get("ethnicity", ""),
                        "identity":    identity,
                        "text":        entry.get("text", ""),
                    })

    if not rows:
        sys.exit(
            "[error] No responses found for the requested scenarios/models.\n"
            "        Check --caches, --scenarios, and --models arguments."
        )

    df = pd.DataFrame(rows)
    print(
        f"Loaded {len(df)} responses "
        f"({df['model'].nunique()} models, {df['scenario_id'].nunique()} scenarios)"
    )
    return df


# ── Embed ─────────────────────────────────────────────────────────────────────

# fastembed model name that roughly matches all-MiniLM-L6-v2 in quality
_FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"


def _tail(text: str, ratio: float = 0.20) -> str:
    """Keep only the last `ratio` of the text (by characters)."""
    cut = max(1, int(len(text) * ratio))
    return text[-cut:]


def _embed_fastembed(texts: list[str], model_name: str) -> np.ndarray:
    from fastembed import TextEmbedding   # type: ignore
    name = model_name if "/" in model_name else _FASTEMBED_DEFAULT
    print(f"  backend: fastembed  model: {name}")
    embedder = TextEmbedding(model_name=name)
    return np.array(list(embedder.embed(texts)))


def _embed_sentence_transformers(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer   # type: ignore
    print(f"  backend: sentence-transformers  model: {model_name}")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def _embed_tfidf(texts: list[str], **_) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    print("  backend: TF-IDF + SVD(100)  [install fastembed for semantic embeddings]")
    vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
    X = vec.fit_transform(texts)
    n_components = min(100, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X)


def compute_embeddings(
    texts: list[str],
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """Compute text embeddings, trying backends in order until one works.

    Each text is pre-processed to keep only its first and last 20% (by
    characters) before embedding, so the model focuses on the opening stance
    and closing recommendation rather than the full response body.

    Priority: fastembed (ONNX, no torch) → sentence-transformers → TF-IDF+SVD.
    """
    truncated = [_tail(t) for t in texts]
    print(f"Embedding {len(truncated)} texts (tail 20%) …")
    for fn in (_embed_fastembed, _embed_sentence_transformers, _embed_tfidf):
        try:
            return fn(truncated, embed_model_name)
        except Exception as exc:
            print(f"  [{fn.__name__}] unavailable: {exc}", file=sys.stderr)
    raise RuntimeError("No embedding backend available. Install fastembed or sentence-transformers.")


# ── UMAP ──────────────────────────────────────────────────────────────────────

def reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D via UMAP. Returns array of shape (n, 2)."""
    import umap as umap_lib                                 # lazy import
    print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) …")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


# ── Plot ──────────────────────────────────────────────────────────────────────

_MARKER_POOL = ["o", "s", "^", "D", "v", "P", "*"]

ColorBy = Literal["identity", "gender", "ethnicity", "model", "scenario_id"]
FacetBy = Literal["model", "scenario_id", None]


def _build_color_map(df: pd.DataFrame, color_by: str) -> dict[str, str]:
    """Return {value: hex_color} for any color_by column."""
    if color_by == "identity":
        return IDENTITY_COLORS
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    values = sorted(df[color_by].unique())
    return {v: palette[i % len(palette)] for i, v in enumerate(values)}


def plot_umap(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_by: ColorBy = "identity",
    marker_by: str = "scenario_id",
    title: str = "",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """2D scatter plot of UMAP coordinates.

    Parameters
    ----------
    coords:
        Shape (n, 2) array from reduce_umap.
    df:
        DataFrame aligned with coords (same row order).
    color_by:
        Column used to colour points.
    marker_by:
        Column used to vary the point marker shape.
    title:
        Figure suptitle.
    output_path:
        If given, save the figure here.
    """
    color_map = _build_color_map(df, color_by)
    marker_vals = sorted(df[marker_by].unique())
    marker_map = {v: _MARKER_POOL[i % len(_MARKER_POOL)] for i, v in enumerate(marker_vals)}

    df = df.copy()
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 6))

    for mv in marker_vals:
        mdf = df[df[marker_by] == mv]
        if mdf.empty:
            continue
        ax.scatter(
            mdf["_x"], mdf["_y"],
            c=[color_map.get(v, "#999999") for v in mdf[color_by]],
            marker=marker_map[mv],
            s=60, alpha=0.75, linewidths=0.4, edgecolors="white",
        )

    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    color_patches = [
        mpatches.Patch(color=c, label=v) for v, c in color_map.items()
        if v in df[color_by].values
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=marker_map[v], color="gray",
                   linestyle="none", markersize=7, label=v)
        for v in marker_vals
    ]

    all_handles = color_patches + marker_handles
    fig.legend(handles=all_handles, loc="lower center",
               ncol=min(4, len(all_handles)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    return fig


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_embedding_analysis(
    cache_paths: list[str | Path],
    scenarios: list[str] | None = None,
    models: list[str] | None = None,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    color_by: ColorBy = "identity",
    output_dir: str | Path = ".",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """End-to-end pipeline: load → embed → UMAP → one plot per model.

    UMAP is computed once on all data (all scenarios × all models) so
    coordinates are globally consistent. One PNG per model is saved in
    output_dir/umap/, with points coloured by identity and marker shape
    varying by scenario.

    Parameters
    ----------
    cache_paths:
        Neutralized cache JSON files to load.
    scenarios:
        Scenario IDs to analyse. Defaults to WP-001/002/003.
    models:
        Normalised model names to include. Defaults to SMALL_MODELS.
    embed_model_name:
        Embedding model identifier.
    color_by:
        Column used to colour UMAP points.
    output_dir:
        Parent directory; plots are written to output_dir/umap/.
    n_neighbors, min_dist, random_state:
        UMAP hyperparameters.
    """
    scenarios = scenarios or DEFAULT_SCENARIOS
    models = models or DEFAULT_MODELS

    umap_dir = Path(output_dir) / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    df = load_neutralized_responses(cache_paths, scenarios, models)

    for model in models:
        mask = df["model"] == model
        if not mask.any():
            print(f"[warn] no data for model '{model}', skipping", file=sys.stderr)
            continue

        sub_df = df[mask].reset_index(drop=True)
        embeddings = compute_embeddings(sub_df["text"].tolist(), embed_model_name)
        sub_coords = reduce_umap(embeddings, n_neighbors=n_neighbors,
                                 min_dist=min_dist, random_state=random_state)

        safe_model = model.replace("/", "-")
        out = umap_dir / f"{safe_model}_{color_by}.png"
        plot_umap(
            sub_coords, sub_df,
            color_by=color_by,
            marker_by="scenario_id",
            title=f"{model}  (colour: {color_by})",
            output_path=out,
        )
