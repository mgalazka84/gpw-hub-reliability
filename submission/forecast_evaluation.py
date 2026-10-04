"""Chronological, label-maturity-aware forecast evaluation on an audited feature table.

This module does not generate empirical results without observed input data.
The network feature builder is separate from forecasting to preserve an audit trail.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from network_methods import circular_block_indices

BASE_FEATURES = ["current_y20", "current_y252", "mean_correlation", "market_vol20",
                 "market_vol252", "pc1_share", "zero_volume_share", "n_assets"]
MODEL_FEATURES = {
    "base": BASE_FEATURES,
    "degree": BASE_FEATURES + ["degree_residual"],
    "reliability": BASE_FEATURES + ["degree_residual", "reliability_residual"],
}
ALPHAS = [.01, .1, 1., 10., 100.]


def ridge_prediction(x: np.ndarray, y: np.ndarray, current: np.ndarray, alpha: float) -> float:
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1
    z, now = (x-mean)/sd, (current-mean)/sd
    ybar = y.mean()
    coef = np.linalg.solve(z.T @ z + alpha*np.eye(z.shape[1]), z.T @ (y-ybar))
    # The diversification target is mathematically bounded under long-only weights.
    return float(np.clip(ybar + now @ coef, 0, 1))


def chronological_predictions(df: pd.DataFrame, features: list[str], alpha: float,
                              minimum_training: int = 60) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for index, row in df.iterrows():
        # No final future outcome, even for an earlier origin, enters before it matures.
        mature = (df.origin_date < row.origin_date) & (df.label_end_date <= row.origin_date)
        train = df.loc[mature].dropna(subset=features + ["future_y"])
        if len(train) < minimum_training or row[features].isna().any():
            continue
        out.loc[index] = ridge_prediction(train[features].to_numpy(float),
            train.future_y.to_numpy(float), row[features].to_numpy(float), alpha)
    return out


def paired_loss_interval(differences: np.ndarray, block: int = 6,
                         repetitions: int = 5000, seed: int = 7029) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for b in range(repetitions):
        index = circular_block_indices(len(differences), min(block, len(differences)), rng)
        means[b] = differences[index].mean()
    low, high = np.quantile(means, [.025, .975])
    return dict(mean_difference=float(differences.mean()), ci95_low=float(low),
                ci95_high=float(high), block_origins=block, bootstrap=repetitions)


def validate_features(df: pd.DataFrame):
    mandatory = ["origin_date", "label_end_date", "future_y"] + MODEL_FEATURES["reliability"]
    missing = sorted(set(mandatory)-set(df))
    if missing:
        raise ValueError(f"Missing audited feature columns: {missing}")
    if df.origin_date.duplicated().any() or not df.origin_date.is_monotonic_increasing:
        raise ValueError("Origins must be unique and ordered")
    if not (df.label_end_date > df.origin_date).all():
        raise ValueError("Every forecast label must end after its origin")
    # The next forecast can share the preceding endpoint, not any return observation.
    if (df.origin_date.iloc[1:].to_numpy() < df.label_end_date.iloc[:-1].to_numpy()).any():
        raise ValueError("Primary forecast outcome windows must not overlap")
    known_y = df.future_y.dropna()
    if not known_y.between(0, 1).all():
        raise ValueError("Future diversification target must lie in [0, 1]")
    complete = df.dropna(subset=["degree_residual", "reliability_residual"])
    if (complete.reliability_residual > complete.degree_residual+1e-10).any():
        raise ValueError("Reliability share cannot exceed observed degree concentration")


def evaluate(df: pd.DataFrame, minimum_training: int = 60):
    validate_features(df)
    validation = df.origin_date.dt.year.between(2015, 2018)
    test = df.origin_date.dt.year.between(2019, 2025)
    if not validation.any() or not test.any():
        raise ValueError("Need separate 2015-2018 validation and 2019-2025 retrospective test samples")
    predictions = df[["origin_date", "label_end_date", "future_y"]].copy()
    configuration = dict(alphas=ALPHAS, minimum_training=minimum_training,
                         validation_years=[2015,2018], test_years=[2019,2025], models={})
    for model, features in MODEL_FEATURES.items():
        choices = []
        for alpha in ALPHAS:
            pred = chronological_predictions(df, features, alpha, minimum_training)
            available = validation & pred.notna() & df.future_y.notna()
            if available.sum() < 24:
                raise ValueError(f"Insufficient validation origins for {model}")
            score = float(((pred[available]-df.future_y[available])**2).mean())
            choices.append((score, alpha))
        _, chosen = min(choices)
        predictions[model] = chronological_predictions(df, features, chosen, minimum_training)
        configuration["models"][model] = dict(alpha=chosen, features=features,
                                               validation_mse_by_alpha=choices)
    predictions["persistence"] = df.current_y20
    final = predictions.loc[test].dropna(subset=["future_y"] + list(MODEL_FEATURES) + ["persistence"])
    if len(final) < 36:
        raise ValueError("Too few common test origins; no primary forecasting result produced")
    mse = {model:float(((final[model]-final.future_y)**2).mean())
           for model in list(MODEL_FEATURES)+["persistence"]}
    differences = ((final.degree-final.future_y)**2 - (final.reliability-final.future_y)**2).to_numpy()
    result = dict(test_origins=len(final), first_origin=str(final.origin_date.min().date()),
                  last_origin=str(final.origin_date.max().date()), mse=mse,
                  primary_comparison="degree MSE minus reliability MSE; positive favors reliability",
                  primary_interval=paired_loss_interval(differences, 6),
                  sensitivity_intervals=[paired_loss_interval(differences, block) for block in [3,12]],
                  evaluation="retrospective; historical outcomes already exist; no prospective preregistration")
    return predictions, configuration, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features", type=Path)
    parser.add_argument("--output", type=Path, default=Path("empirical_results"))
    args = parser.parse_args()
    frame = pd.read_csv(args.features, parse_dates=["origin_date", "label_end_date"])
    predictions, config, result = evaluate(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "frozen_forecast_models.json").write_text(json.dumps(config, indent=2))
    (args.output / "forecast_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
