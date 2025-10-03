
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def naive_predict_last(df: pd.DataFrame) -> np.ndarray:
    if "sales_lag1" not in df.columns:
        raise ValueError("Require 'sales_lag1' for naive t-1 baseline.")
    return df["sales_lag1"].to_numpy()

def naive_predict_last_week(df: pd.DataFrame) -> np.ndarray:
    if "sales_lag7" not in df.columns:
        raise ValueError("Require 'sales_lag7' for naive t-7 baseline.")
    return df["sales_lag7"].to_numpy()

def ridge_baseline_pipeline(X_train: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    for idc in ["store_id", "item_id"]:
        if idc in X_train.columns and X_train[idc].dtype != "object":
            cat_cols.append(idc)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )
    ridge = Ridge(alpha=1.0, random_state=0)
    pipe = Pipeline([("prep", pre), ("model", ridge)])
    return pipe
