"""Rebuild the introductory Abalone figure and check its displayed numbers.

Uses the frozen pooled Abalone records, with the local E64 comparison family.
The sample tree is independently reconstructed from SciPy Kendall tau-b values
converted to tau-a using exact marginal tie counts. No simulations are rerun.
"""
from pathlib import Path
from itertools import combinations
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

ROOT = Path(__file__).resolve().parent
M = ROOT / "manuscript"
R = ROOT / "results" / "cross_section"


def main():
    design = json.loads((R / "design.json").read_text())
    names = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
             "Viscera weight", "Shell weight", "Rings"]
    data = ROOT / "data" / "cross_section" / "abalone.data"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == design["data_sha256"]["abalone.data"]
    x = pd.read_csv(data, header=None).iloc[:, 1:].to_numpy(float)
    assert x.shape == (4177, 8)
    pairs = list(combinations(range(8), 2))
    total_pairs = len(x) * (len(x) - 1) // 2
    untied = []
    for column in x.T:
        counts = np.unique(column, return_counts=True)[1]
        untied.append(total_pairs - int(np.sum(counts * (counts - 1) // 2)))
    tau = np.array([kendalltau(x[:, i], x[:, j]).statistic *
                    np.sqrt(float(untied[i]) * untied[j]) / total_pairs
                    for i, j in pairs])
    edges = pd.read_csv(R / "edges.csv")
    edges = edges[(edges.dataset == "Abalone") & (edges.stratum == "pooled")].sort_values("edge")
    assert len(edges) == len(pairs)
    assert list(zip(edges.a, edges.b)) == [(names[i], names[j]) for i, j in pairs]
    assert np.allclose(tau, edges.kendall_tau_a.to_numpy(), rtol=0, atol=1e-12)
    weights = np.zeros((8, 8))
    for (i, j), w in zip(pairs, tau):
        weights[i, j] = weights[j, i] = -w
    tree = minimum_spanning_tree(weights).tocoo()
    tree_edges = list(zip(tree.row.tolist(), tree.col.tolist()))
    selected = {frozenset(e) for e in tree_edges}
    recorded = {frozenset(pairs[k]) for k in edges.loc[edges.kendall_tree == 1, "edge"]}
    assert selected == recorded and len(selected) == 7
    assert connected_components(tree, directed=False, return_labels=False) == 1
    nodes = pd.read_csv(R / "nodes_all.csv")
    nodes = nodes[(nodes.dataset == "Abalone") & (nodes.stratum == "pooled") &
                  (nodes.method == "evalue_64") & (nodes.family == "local")].sort_values("vertex")
    assert len(nodes) == 8 and nodes.name.tolist() == names
    degree = np.bincount(np.array(tree_edges).ravel(), minlength=8)
    assert np.array_equal(degree, nodes.sample_degree.to_numpy())
    assert degree[3] == 4 and np.flatnonzero(degree == degree.max()).tolist() == [3]
    whole = nodes.loc[nodes.vertex == 3].iloc[0]
    assert (int(whole.degree_lower), int(whole.degree_upper)) == (3, 5)
    assert (int(whole.rank_lower), int(whole.rank_upper)) == (1, 3)
    repetitions = int(design["row_bootstrap_repetitions"])
    assert repetitions == 499 and float(whole.bootstrap_unique_hub_frequency) == 1.0
    evidence = json.loads((ROOT / "results/evalue/cross_section/dataset00.json").read_text())
    independent_records = [n for n in evidence["nodes"] if n["method"] == "evalue_64" and
                           n["family"] == "local" and n["vertex"] == 3]
    assert len(independent_records) == 1
    saved = independent_records[0]
    assert [saved[k] for k in ["degree_lower", "degree_upper", "rank_lower", "rank_upper"]] == [3, 5, 1, 3]
    assert float(design["alpha"]) == .05

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    orange, blue, ink, gray = "#bd5b15", "#53778d", "#263844", "#b8c1c7"
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2),
                             gridspec_kw={"width_ratios": [1.55, 1.05, 1.25]})
    pos = {0: (-1.32, .53), 1: (-1.92, 1.04), 2: (1.52, .95),
           3: (-.35, .02), 4: (-1.32, -.70), 5: (-.37, 1.07),
           6: (.72, .02), 7: (1.50, -.89)}
    ax = axes[0]
    for i, j in tree_edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color=gray, lw=1.8, zorder=1)
    others = [pos[i] for i in range(8) if i != 3]
    ax.scatter([p[0] for p in others], [p[1] for p in others], s=115,
               color=blue, edgecolors="white", linewidths=1, zorder=2)
    ax.scatter(*pos[3], s=420, color=orange, edgecolors="white", linewidths=1, zorder=3)
    labels = {0: "Length", 1: "Diameter", 2: "Height", 3: "Whole\nweight",
              4: "Shucked\nweight", 5: "Viscera\nweight", 6: "Shell\nweight", 7: "Rings"}
    offsets = {0: (-.03, .22), 1: (-.02, .22), 2: (-.04, .22), 3: (-.03, -.50),
               4: (0, -.29), 5: (0, .30), 6: (-.10, -.45), 7: (0, -.23)}
    for i in range(8):
        px, py = pos[i]; dx, dy = offsets[i]
        ax.text(px + dx, py + dy, labels[i], ha="center", va="center", fontsize=10.5,
                color=orange if i == 3 else ink,
                fontweight="bold" if i == 3 else "normal")
    ax.set_xlim(-2.42, 1.93); ax.set_ylim(-1.48, 1.64); ax.axis("off")
    ax.set_title("(a) Estimated\ncorrelation tree", color=ink, fontsize=12.5, pad=10)
    ax.text(.5, -.055, "Sample degree: 4", transform=ax.transAxes,
            ha="center", color=orange, fontsize=11)

    ax = axes[1]; ax.axis("off")
    ax.set_title("(b) Resampling\nstability", loc="center", color=ink, fontsize=12.5, pad=10)
    ax.text(.5, .77, "Whole weight is\nthe unique hub in", transform=ax.transAxes,
            ha="center", va="center", color=ink, fontsize=11.5)
    ax.text(.5, .48, "499 / 499", transform=ax.transAxes, ha="center", va="center",
            fontsize=24, fontweight="bold", color=orange)
    ax.text(.5, .25, "bootstrap trees", transform=ax.transAxes, ha="center", color=ink)
    ax.text(.5, .06, "Resampling frequency", transform=ax.transAxes, ha="center",
            color=ink, fontsize=10)

    ax = axes[2]
    ax.set_title("(c) Population\nrank", loc="center", color=ink, fontsize=12.5, pad=10)
    ax.set_xlim(.65, 8.35); ax.set_ylim(0, 1)
    ax.axhline(.51, color=gray, lw=1.5, zorder=1)
    ax.plot([1, 3], [.51, .51], color=orange, lw=7, solid_capstyle="round", zorder=2)
    ax.scatter(range(1, 9), [.51] * 8, s=32,
               c=[orange] * 3 + [gray] * 5, zorder=3)
    for rank in range(1, 9):
        ax.text(rank, .38, str(rank), ha="center", color=ink, fontsize=10)
    ax.text(.5, .76, "Whole weight: [1, 3]", transform=ax.transAxes,
            ha="center", color=orange, fontweight="bold", fontsize=12)
    ax.text(.5, .23, "Rank 1 = largest degree", transform=ax.transAxes,
            ha="center", color=ink, fontsize=10)
    ax.text(.5, .045, "E64: 95% simultaneous\ncoverage under IID", transform=ax.transAxes,
            ha="center", color=ink, fontsize=10)
    ax.axis("off")
    fig.subplots_adjust(left=.025, right=.98, top=.83, bottom=.17, wspace=.23)
    fig.savefig(M / "Fig_intro_abalone.pdf", bbox_inches="tight", pad_inches=.10)
    fig.savefig(M / "Fig_intro_abalone.png", dpi=200, bbox_inches="tight", pad_inches=.10)
    plt.close(fig)
    record = {"status": "PASS", "dataset": "Abalone", "stratum": "pooled",
              "n": 4177, "p": 8, "target": "Kendall tau-a", "method": "evalue_64",
              "family": "local", "projection": "joint cycle-path LP",
              "alpha": .05, "IID_working_model_required": True,
              "whole_weight_sample_degree": 4, "bootstrap_unique_hub_count": 499,
              "bootstrap_repetitions": 499, "degree_bounds": [3, 5], "rank_bounds": [1, 3],
              "independent_sample_tree_reconstruction": "SciPy tau-b converted to tau-a + sparse MST",
              "raw_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
              "record_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [R / "nodes_all.csv", R / "edges.csv", R / "design.json",
                                          ROOT / "results/evalue/cross_section/dataset00.json"]}}
    (ROOT / "results/intro_figure_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
