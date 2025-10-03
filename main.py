
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .data_load import load_data
from .features import engineer
from .baselines import naive_predict_last, naive_predict_last_week, ridge_baseline_pipeline
from .model_mlp import time_series_grid_search
from .evaluate import regression_metrics, plot_pred_vs_actual, plot_residuals

def temporal_split(df_feat: pd.DataFrame, splits=(0.7, 0.15, 0.15), target="sales"):
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    n = len(df_feat)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    train = df_feat.iloc[:n_train]
    val = df_feat.iloc[n_train:n_train+n_val]
    test = df_feat.iloc[n_train+n_val:]
    def clean(part):
        return part.dropna(axis=0, how="any")
    train, val, test = map(clean, (train, val, test))
    X_train, y_train = train.drop(columns=[target]), train[target].to_numpy()
    X_val, y_val = val.drop(columns=[target]), val[target].to_numpy()
    X_test, y_test = test.drop(columns=[target]), test[target].to_numpy()
    return (X_train, y_train, train), (X_val, y_val, val), (X_test, y_test, test)

def main(args):
    df = load_data(args.csv)
    df_feat = engineer(df)
    (X_train, y_train, df_tr), (X_val, y_val, df_va), (X_test, y_test, df_te) = temporal_split(
        df_feat, splits=tuple(args.split), target=args.target
    )
    naive1_pred = naive_predict_last(df_te)
    naive7_pred = naive_predict_last_week(df_te)
    ridge_pipe = ridge_baseline_pipeline(pd.concat([X_train, X_val]))
    ridge_pipe.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
    ridge_pred = ridge_pipe.predict(X_test)
    from .evaluate import regression_metrics
    m_naive1 = regression_metrics(y_test, naive1_pred)
    m_naive7 = regression_metrics(y_test, naive7_pred)
    m_ridge = regression_metrics(y_test, ridge_pred)
    print("\n=== Baseline metrics (Test) ===")
    print(f"Naive last (t-1): {m_naive1}")
    print(f"Naive last week (t-7): {m_naive7}")
    print(f"Ridge baseline: {m_ridge}")
    X_tr_full = X_train
    y_tr_full = y_train
    gs = time_series_grid_search(X_tr_full, y_tr_full, n_splits=args.cv_splits)
    print("\nBest params:", gs.best_params_)
    print("Best CV score (negative MAE):", gs.best_score_)
    best_model = gs.best_estimator_
    best_model.fit(pd.concat([X_train, X_val]), np.concatenate([y_train, y_val]))
    mlp_pred = best_model.predict(X_test)
    m_mlp = regression_metrics(y_test, mlp_pred)
    print("\n=== MLP metrics (Test) ===")
    print(m_mlp)
    figures_dir = Path("figures")
    plot_pred_vs_actual(df_te, y_test, naive1_pred, figures_dir / "pred_vs_actual_naive1.png")
    plot_pred_vs_actual(df_te, y_test, ridge_pred, figures_dir / "pred_vs_actual_ridge.png")
    plot_pred_vs_actual(df_te, y_test, mlp_pred, figures_dir / "pred_vs_actual_mlp.png")
    plot_residuals(y_test, mlp_pred, figures_dir / "residuals_mlp.png")
    print(f"\nSaved plots -> {figures_dir.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="Path to raw CSV file.")
    p.add_argument("--target", type=str, default="sales")
    p.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], help="Train/Val/Test fractions")
    p.add_argument("--cv_splits", type=int, default=5)
    args = p.parse_args()
    main(args)
