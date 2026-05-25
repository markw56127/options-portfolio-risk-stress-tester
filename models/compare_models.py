"""
Runs all models on the test split and prints a comparison table.
Outputs: models/comparison_results.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

log = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent
PROC_DIR = MODEL_DIR.parent / "data" / "processed"


def run_comparison():
    from models.black_scholes import bs_price
    from models.xgboost_model import load_model as load_xgb, evaluate as eval_xgb
    from models.lstm_model import load_model as load_lstm, evaluate as eval_lstm

    X = pd.read_csv(PROC_DIR / "features.csv")
    y = pd.read_csv(PROC_DIR / "targets.csv").squeeze()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    results = []

    # XGBoost
    try:
        xgb_model = load_xgb()
        xgb_metrics = eval_xgb(xgb_model, X_test, y_test)
        results.append({"model": "XGBoost", **xgb_metrics})
    except Exception as e:
        log.warning("XGBoost eval failed: %s", e)

    # LSTM
    try:
        lstm_model = load_lstm(input_size=X.shape[1])
        lstm_metrics = eval_lstm(lstm_model, X_test.values, y_test.values)
        results.append({"model": "LSTM", **lstm_metrics})
    except Exception as e:
        log.warning("LSTM eval failed: %s", e)

    df = pd.DataFrame(results)
    out = MODEL_DIR / "comparison_results.csv"
    df.to_csv(out, index=False)
    log.info("\n%s", df.to_string(index=False))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_comparison()
