"""Monte Carlo diagnostics. All generated observations are explicitly synthetic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from network_methods import (hub_stability, mst, covariance_to_correlation,
                             demean_population_covariance)

BASE = Path(__file__).parent
CASES = ["independent_equal", "independent_unequal", "market_only",
         "market_residual", "market_only_sv_t5"]
KINDS = ["raw", "demeaned", "factor_residual"]


def dgp(case: str, n: int, t: int, rng: np.random.Generator):
    beta = np.zeros(n)
    sigma = np.ones(n)
    residual_loading = np.zeros(n)
    if case == "independent_unequal":
        sigma = np.geomspace(.12, 2.5, n)
    if case.startswith("market"):
        beta = np.linspace(.25, 1.1, n)
        sigma = np.linspace(1.4, .8, n)
        beta[0], sigma[0] = 2.2, .35
    if case == "market_residual":
        # A second common factor is dependence remaining after the market factor.
        # This DGP does not identify or model causal price transmission.
        residual_loading[1] = 1.8
        residual_loading[2:22] = np.linspace(.80, .30, 20)
    residual_cov = np.diag(sigma**2) + np.outer(residual_loading, residual_loading)
    cov = np.outer(beta, beta) + residual_cov
    burn = 500 if case.endswith("sv_t5") else 0
    total = t + burn
    if case.endswith("sv_t5"):
        innovation = rng.standard_t(5, size=(total, n + 2)) * np.sqrt(3/5)
        # Independent log-volatility AR(1), stationary E[exp(log_variance)]=1.
        log_variance = np.empty(total)
        phi, eta_sd = .95, .20
        stationary_variance = eta_sd**2 / (1-phi**2)
        log_variance[0] = rng.normal(0, np.sqrt(stationary_variance))
        for u in range(1, total):
            log_variance[u] = phi*log_variance[u-1] + rng.normal(0, eta_sd)
        scale = np.exp(.5*(log_variance - .5*stationary_variance))
        innovation *= scale[:, None]
    else:
        innovation = rng.normal(size=(total, n + 2))
    f = innovation[:, 0]
    g = innovation[:, 1]
    eps = innovation[:, 2:]
    x = f[:, None]*beta + g[:, None]*residual_loading + eps*sigma
    return x[burn:], f[burn:], cov, residual_cov


def oracle(cov: np.ndarray, residual_cov: np.ndarray, kind: str):
    if kind == "demeaned":
        chosen = demean_population_covariance(cov)
    elif kind == "factor_residual":
        chosen = residual_cov
    else:
        chosen = cov
    r = covariance_to_correlation(chosen)
    off = r[np.triu_indices(len(r), 1)]
    # No unique population tree is declared if pair correlations contain ties.
    # This deliberately conservative check also suppresses partly tied graphs.
    unique = np.unique(np.round(off, 12)).size == off.size
    edges, degree = mst(r)
    return edges, degree, unique


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, kind), group in df.groupby(["case", "transform"], sort=False):
        row = dict(case=case, transform=kind, replications=len(group))
        for metric in ["max_selection", "anchor_selection", "max_degree",
                       "degree_concentration", "reliability_share",
                       "population_edge_recovery", "mean_selection"]:
            values = group[metric].dropna().to_numpy()
            row[metric + "_mean"] = float(values.mean()) if len(values) else np.nan
            row[metric + "_mc_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values)>1 else np.nan
        events = (group["max_selection"] >= .8).astype(float).to_numpy()
        proportion = events.mean()
        m = len(events)
        z = 1.95996398454
        center = (proportion + z*z/(2*m)) / (1+z*z/m)
        half = z*np.sqrt(proportion*(1-proportion)/m + z*z/(4*m*m)) / (1+z*z/m)
        row.update(frequency_max_selection_ge_080=float(proportion),
                   frequency_wilson_low=float(center-half),
                   frequency_wilson_high=float(center+half))
        rows.append(row)
    return pd.DataFrame(rows)


def figures(df: pd.DataFrame, output: Path):
    labels = ["Independent, equal variances", "Independent, unequal variances",
              "Market factor only", "Market + residual dependence", "Market only, SV + t(5)"]
    colors = {"raw":"#264b75", "demeaned":"#b45c20", "factor_residual":"#2b7663"}
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                        "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")
    positions = np.arange(len(CASES))
    for j, kind in enumerate(KINDS):
        sub = summarize(df[df["transform"] == kind]).set_index("case").reindex(CASES)
        offset = (j-1)*.23
        axes[0].errorbar(sub["max_selection_mean"], positions+offset,
                         xerr=1.96*sub["max_selection_mc_se"], fmt="o", markersize=4,
                         color=colors[kind], label=kind.replace("_", " "))
        prop = sub["frequency_max_selection_ge_080"]
        err = np.maximum(0, np.vstack([prop-sub["frequency_wilson_low"], sub["frequency_wilson_high"]-prop]))
        axes[1].errorbar(prop, positions+offset, xerr=err, fmt="o", markersize=4, color=colors[kind])
    for ax in axes:
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", alpha=.2)
    axes[0].set_xlabel("Mean maximum hub selection frequency")
    axes[1].set_xlabel("Fraction of samples with maximum frequency >= 0.80")
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Synthetic diagnostic: stability is conditional on the return representation", fontsize=12)
    fig.savefig(output / "simulation_stability.png", dpi=300)
    fig.savefig(output / "simulation_stability.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.65, 4.0), layout="constrained")
    for j, kind in enumerate(KINDS):
        sub = summarize(df[df["transform"] == kind]).set_index("case").reindex(CASES)
        ax.errorbar(sub["max_selection_mean"], positions+(j-1)*.23,
                    xerr=1.96*sub["max_selection_mc_se"], fmt="o", markersize=4,
                    color=colors[kind], label=kind.replace("_", " "))
    ax.set_yticks(positions, ["Independent: equal variance", "Independent: unequal variance",
                             "Market factor only", "Market + residual dependence", "Market only: SV + t(5)"])
    ax.invert_yaxis(); ax.set_xlim(0,1.02)
    ax.set_xlabel("Mean maximum hub selection frequency")
    ax.grid(axis="x",alpha=.2)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.13),ncol=3,frameon=False,fontsize=8.5)
    fig.savefig(output / "manuscript_stability.png",dpi=300)
    fig.savefig(output / "manuscript_stability.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--t", type=int, default=252)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    output = BASE / args.output
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = []
    sequence = np.random.SeedSequence(args.seed).spawn(len(CASES)*args.mc)
    for ci, case in enumerate(CASES):
        for rep in range(args.mc):
            rng = np.random.default_rng(sequence[ci*args.mc+rep])
            x, f, cov, rcov = dgp(case, args.n, args.t, rng)
            for kind in KINDS:
                seed = int(rng.integers(0, 2**32-1))
                result = hub_stability(x, f, kind, args.bootstrap, args.block, seed)
                observed_anchor = int(np.argmax(result.degree))
                true_edges, _, unique = oracle(cov, rcov, kind)
                recovery = np.nan
                if unique:
                    truth = {tuple(edge) for edge in true_edges}
                    recovery = sum(tuple(edge) in truth for edge in result.edges)/(args.n-1)
                records.append(dict(case=case, replication=rep, transform=kind,
                    max_selection=float(result.selection.max()),
                    anchor_selection=float(result.selection[observed_anchor]),
                    max_degree=int(result.degree.max()),
                    degree_concentration=result.degree_concentration,
                    reliability_share=result.reliability_share,
                    population_edge_recovery=recovery,
                    mean_selection=float(result.selection.mean()), seed=seed))
                if rep == 0:
                    pd.DataFrame(dict(asset=np.arange(args.n), degree=result.degree,
                                      selection=result.selection)).to_csv(output / f"example_{case}_{kind}.csv", index=False)
            if (rep+1) % 20 == 0 or rep+1 == args.mc:
                print(f"{case}: {rep+1}/{args.mc}; elapsed {time.monotonic()-start:.1f}s", flush=True)
        # Checkpoint: previous scenarios remain recoverable if execution stops.
        pd.DataFrame(records).to_csv(output / "simulation_replications.csv", index=False)
    df = pd.DataFrame(records)
    summary = summarize(df)
    summary.to_csv(output / "simulation_summary.csv", index=False)
    config = vars(args).copy()
    config.update(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  pandas=pd.__version__, matplotlib=matplotlib.__version__,
                  elapsed_seconds=time.monotonic()-start,
                  data_type="synthetic; no observed stock returns", circular_block_bootstrap=True,
                  tail_note="Independent unit-variance t(5) innovations with independent stationary log-volatility AR(1)")
    config["source_hashes"] = {name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                               for name in ["network_methods.py", "run_simulations.py"]}
    (output / "simulation_config.json").write_text(json.dumps(config, indent=2))
    figures(df, output)
    print(summary[["case", "transform", "max_selection_mean", "frequency_max_selection_ge_080"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
