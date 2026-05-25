"""
XGBoost model for option price prediction.
Trained on features from data/processed/; compared against BS baseline.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import joblib
import logging

log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_model.json"


def load_data():
    X = pd.read_csv(PROC_DIR / "features.csv")
    y = pd.read_csv(PROC_DIR / "targets.csv").squeeze()
    return X, y


def train(X: pd.DataFrame, y: pd.Series, params=None) -> xgb.XGBRegressor:
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "early_stopping_rounds": 30,
        "eval_metric": "rmse",
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    model.save_model(MODEL_PATH)
    log.info("XGBoost model saved -> %s", MODEL_PATH)
    return model


def evaluate(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model


def predict(features: pd.DataFrame) -> np.ndarray:
    model = load_model()
    return model.predict(features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    model = train(X, y)
    metrics = evaluate(model, X_test, y_test)
    log.info("XGBoost test metrics: %s", metrics)
