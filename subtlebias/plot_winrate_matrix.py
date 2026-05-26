"""
plot_winrate_matrix.py
----------------------
Heatmap des win rates pairwise entre les 4 identités.
equal = 0.5 point pour chaque côté.

Usage:
    python plot_winrate_matrix.py
    python plot_winrate_matrix.py --csv pairwise_results_final.csv
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

IDENTITIES = ["male/white", "male/arabic", "female/white", "female/arabic"]
LABELS     = ["M. White", "M. Arabic", "F. White", "F. Arabic"]


def compute_matrix(rows: list[dict]) -> np.ndarray:
    wins  = defaultdict(float)
    total = defaultdict(int)

    for r in rows:
        id_a    = r["identity_a"]
        id_b    = r["identity_b"]
        favored = r["favored"]

        if id_a not in IDENTITIES or id_b not in IDENTITIES:
            continue

        total[(id_a, id_b)] += 1
        total[(id_b, id_a)] += 1

        if favored == id_a:
            wins[(id_a, id_b)] += 1
            wins[(id_b, id_a)] += 0
        elif favored == id_b:
            wins[(id_b, id_a)] += 1
            wins[(id_a, id_b)] += 0
        else:  # equal
            wins[(id_a, id_b)] += 0.5
            wins[(id_b, id_a)] += 0.5

    n = len(IDENTITIES)
    matrix = np.full((n, n), np.nan)
    for i, id_i in enumerate(IDENTITIES):
        for j, id_j in enumerate(IDENTITIES):
            if i == j:
                matrix[i, j] = 0.5
            elif total[(id_i, id_j)] > 0:
                matrix[i, j] = wins[(id_i, id_j)] / total[(id_i, id_j)]

    return matrix


def plot(matrix: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Diverging colormap centré sur 0.5
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=0.3, vcenter=0.5, vmax=0.7)

    im = ax.imshow(matrix, cmap=cmap, norm=norm)

    # Annotations
    for i in range(len(IDENTITIES)):
        for j in range(len(IDENTITIES)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            color = "black" if 0.38 < val < 0.62 else "white"
            diag  = i == j
            text  = "—" if diag else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    ax.set_xticks(range(len(IDENTITIES)))
    ax.set_yticks(range(len(IDENTITIES)))
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Identity judged LESS seriously →", fontsize=10, labelpad=10)
    ax.set_ylabel("← Identity judged MORE seriously", fontsize=10, labelpad=10)

    ax.set_title("Pairwise win rate matrix\n"
                 "cell [i,j] = P(row i judged more seriously than col j)",
                 fontsize=12, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Win rate (equal = 0.5)", fontsize=9)
    cbar.ax.axhline(0.5, color="black", linewidth=1.5, linestyle="--")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved : {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/pairwise_results_corrected.csv")
    parser.add_argument("--out", default="results/winrate_matrix.png")
    args = parser.parse_args()

    base = Path(__file__).parent

    with open(base / args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Chargé : {len(rows)} comparaisons")

    matrix = compute_matrix(rows)

    print("\nWin rate matrix :")
    print(f"{'':>12}", end="")
    for lbl in LABELS:
        print(f"  {lbl:>10}", end="")
    print()
    for i, lbl in enumerate(LABELS):
        print(f"{lbl:>12}", end="")
        for j in range(len(IDENTITIES)):
            val = matrix[i, j]
            cell = "  —" if i == j else f"  {val:.3f}"
            print(f"{cell:>12}", end="")
        print()

    plot(matrix, base / args.out)


if __name__ == "__main__":
    main()
