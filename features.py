
from __future__ import annotations
import pandas as pd

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out

def _group_keys(df: pd.DataFrame) -> list[str]:
    return [c for c in ["store_id", "item_id"] if c in df.columns]

def add_lag_features(df: pd.DataFrame, target: str = "sales", lags: list[int] = [1, 7]) -> pd.DataFrame:
    out = df.copy()
    keys = _group_keys(out)
    if keys:
        gb = out.groupby(keys, group_keys=False)
        for L in lags:
            out[f"{target}_lag{L}"] = gb[target].shift(L)
    else:
        for L in lags:
            out[f"{target}_lag{L}"] = out[target].shift(L)
    return out

def add_rolling_features(df: pd.DataFrame, target: str = "sales", windows: list[int] = [7, 14, 28]) -> pd.DataFrame:
    out = df.copy()
    keys = _group_keys(out)
    if keys:
        gb = out.groupby(keys, group_keys=False)
        for w in windows:
            roll = gb[target].shift(1).rolling(w, min_periods=max(2, w//2))
            out[f"{target}_rollmean_{w}"] = roll.mean()
            out[f"{target}_rollstd_{w}"] = roll.std()
    else:
        for w in windows:
            roll = out[target].shift(1).rolling(w, min_periods=max(2, w//2))
            out[f"{target}_rollmean_{w}"] = roll.mean()
            out[f"{target}_rollstd_{w}"] = roll.std()
    return out

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_lag_features(out, "sales", lags=[1, 7])
    out = add_rolling_features(out, "sales", windows=[7, 14, 28])
    return out
