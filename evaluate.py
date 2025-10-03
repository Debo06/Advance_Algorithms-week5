
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": mape}

def plot_pred_vs_actual(df_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(df_test["date"].to_numpy(), y_true, label="Actual")
    plt.plot(df_test["date"].to_numpy(), y_pred, label="Predicted")
    plt.xlabel("Date"); plt.ylabel("Sales"); plt.title("Predicted vs Actual Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, outpath: str | Path) -> None:
    res = y_true - y_pred
    plt.figure()
    plt.hist(res, bins=40)
    plt.xlabel("Residual"); plt.ylabel("Frequency"); plt.title("Residual Distribution")
    plt.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
