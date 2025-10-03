
from __future__ import annotations
import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["date", "sales"]

def load_data(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError("Input must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some dates could not be parsed; check source format.")
    group_cols = [c for c in ["store_id", "item_id"] if c in df.columns]
    sort_cols = group_cols + ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    if df["sales"].isna().any():
        raise ValueError("Non-numeric values in 'sales' after coercion.")
    return df
