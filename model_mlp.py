
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def build_pipeline(X_sample: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in X_sample.columns if X_sample[c].dtype == "object"]
    for idc in ["store_id", "item_id"]:
        if idc in X_sample.columns and X_sample[idc].dtype != "object":
            cat_cols.append(idc)
    cat_cols = sorted(set(cat_cols))
    num_cols = [c for c in X_sample.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop"
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, ),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        random_state=42,
        early_stopping=True,
        max_iter=500,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=False,
    )
    return Pipeline([("prep", pre), ("model", mlp)])

def time_series_grid_search(X_train: pd.DataFrame, y_train: np.ndarray, n_splits: int = 5) -> GridSearchCV:
    pipe = build_pipeline(X_train)
    param_grid = {
        "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "model__learning_rate_init": [1e-3, 5e-4],
        "model__max_iter": [400, 800],
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )
    gs.fit(X_train, y_train)
    return gs
