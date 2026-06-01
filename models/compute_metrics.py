"""
Compute MAPE, RMSE, and R² for all three pricing models on the test split.

Run after scripts/run_pipeline.py has trained the models:
    python -m models.compute_metrics

Two comparisons are reported:
  1. BS vs actual market price (lastPrice from yfinance raw data)
  2. XGBoost and LSTM vs their training target (bs_price, 15% test split)

Outputs: models/comparison_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent


def mape(y_true, y_pred):
    mask = np.abs(y_true) > 0.01
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "mape":  round(mape(y_true, y_pred), 3),
        "rmse":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":    round(float(r2_score(y_true, y_pred)), 4),
        "n":     len(y_true),
    }


def eval_black_scholes():
    """Compare bs_price (analytical) vs lastPrice (actual market) from raw data."""
    raw_files = list(RAW_DIR.glob("options_*.csv"))
    if not raw_files:
        log.warning("No raw options CSVs found — skipping BS vs market comparison.")
        return None

    frames = [pd.read_csv(f) for f in raw_files]
    df = pd.concat(frames, ignore_index=True)

    required = {"bs_price", "lastPrice"}
    if not required.issubset(df.columns):
        log.warning("Raw data missing bs_price or lastPrice columns.")
        return None

    df = df.dropna(subset=["bs_price", "lastPrice"])
    df = df[(df["lastPrice"] > 0.01) & (df["bs_price"] > 0.01)]

    m = metrics(df["lastPrice"].values, df["bs_price"].values)
    log.info("Black-Scholes  MAPE=%.2f%%  RMSE=%.4f  R²=%.4f  (n=%d)",
             m["mape"], m["rmse"], m["r2"], m["n"])
    return {"model": "Black-Scholes", **m}


def eval_xgboost():
    import xgboost as xgb
    model_path = MODEL_DIR / "xgb_model.json"
    if not model_path.exists():
        log.warning("xgb_model.json not found — run the pipeline first.")
        return None

    X = pd.read_csv(PROC_DIR / "features.csv")
    y = pd.read_csv(PROC_DIR / "targets.csv").squeeze()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    preds = model.predict(X_test)

    m = metrics(y_test.values, preds)
    log.info("XGBoost        MAPE=%.2f%%  RMSE=%.4f  R²=%.4f  (n=%d)",
             m["mape"], m["rmse"], m["r2"], m["n"])
    return {"model": "XGBoost", **m}


def eval_lstm():
    import torch
    from models.lstm_model import LSTMPricer, OptionsDataset, SEQUENCE_LEN, BATCH_SIZE
    from torch.utils.data import DataLoader

    model_path = MODEL_DIR / "lstm_model.pt"
    if not model_path.exists():
        log.warning("lstm_model.pt not found — run the pipeline first.")
        return None

    X = pd.read_csv(PROC_DIR / "features.csv").values
    y = pd.read_csv(PROC_DIR / "targets.csv").squeeze().values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = LSTMPricer(input_size=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    ds = OptionsDataset(X_test, y_test, seq_len=SEQUENCE_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.extend(model(xb).cpu().numpy())
            targets.extend(yb.numpy())

    m = metrics(np.array(targets), np.array(preds))
    log.info("LSTM           MAPE=%.2f%%  RMSE=%.4f  R²=%.4f  (n=%d)",
             m["mape"], m["rmse"], m["r2"], m["n"])
    return {"model": "LSTM", **m}


def run():
    results = []

    log.info("=== Black-Scholes vs actual market price (lastPrice) ===")
    bs = eval_black_scholes()
    if bs:
        results.append(bs)

    log.info("=== XGBoost vs bs_price target (test split) ===")
    xgb = eval_xgboost()
    if xgb:
        results.append(xgb)

    log.info("=== LSTM vs bs_price target (test split) ===")
    lstm = eval_lstm()
    if lstm:
        results.append(lstm)

    if not results:
        log.error("No results — make sure the pipeline has been run.")
        return

    df = pd.DataFrame(results).set_index("model")
    out = MODEL_DIR / "comparison_results.csv"
    df.to_csv(out)

    print("\n" + "=" * 55)
    print("MODEL COMPARISON RESULTS")
    print("=" * 55)
    print(df.to_string())
    print("=" * 55)
    print(f"\nNote: BS is compared against actual market price (lastPrice).")
    print("      XGBoost/LSTM are compared against the bs_price training target.")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    run()
