"""
Preprocesses raw options + price data into model-ready feature matrices.
Outputs: data/processed/features.csv and data/processed/targets.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import logging

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    "moneyness", "T", "impliedVolatility", "delta", "gamma", "vega", "theta",
    "volume", "openInterest", "bid", "ask",
]
TARGET_COL = "bs_price"  # predict option price; swap for realized P&L if desired


def load_all_options() -> pd.DataFrame:
    frames = []
    for p in RAW_DIR.glob("options_*.csv"):
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No options CSVs found in data/raw/. Run fetch_market_data.py first.")
    return pd.concat(frames, ignore_index=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread"] = df["ask"] - df["bid"]
    df["spread_pct"] = df["spread"] / df["S"].clip(lower=1)
    df["log_moneyness"] = np.log(df["moneyness"].clip(lower=1e-4))
    df["is_call"] = (df["flag"] == "call").astype(int)
    # IV smile skew proxy: deviation from ATM
    atm_iv = df.groupby(["ticker", "expiration"])["impliedVolatility"].transform(
        lambda x: x.iloc[(x.index - x.index[len(x) // 2]).argmin()] if len(x) > 0 else np.nan
    )
    df["iv_skew"] = df["impliedVolatility"] - atm_iv
    return df


def build_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = load_all_options()
    df = engineer_features(df)

    extra = ["spread", "spread_pct", "log_moneyness", "is_call", "iv_skew"]
    feat_cols = [c for c in FEATURE_COLS + extra if c in df.columns]
    df_clean = df[feat_cols + [TARGET_COL]].dropna()

    X = df_clean[feat_cols]
    y = df_clean[TARGET_COL]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols, index=X.index)
    joblib.dump(scaler, PROC_DIR / "scaler.pkl")

    X_scaled.to_csv(PROC_DIR / "features.csv", index=False)
    y.to_csv(PROC_DIR / "targets.csv", index=False, header=True)
    log.info("Processed dataset: %d rows, %d features", len(X_scaled), len(feat_cols))
    return X_scaled, y


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
