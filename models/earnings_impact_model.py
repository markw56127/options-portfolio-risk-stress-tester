"""
Earnings Impact Model: predicts post-earnings option price change.

Training pipeline:
  1. Load historical earnings CSVs (eps_surprise_pct, stock_move_pct)
  2. Build features per earnings event (options context from BS model)
  3. Train XGBoost to predict stock_move_pct from earnings + options features
  4. At inference: predict stock_move → apply delta-gamma approximation → option ΔP

This separates the earnings signal from the generic options pricing model.
"""

import logging
from pathlib import Path

from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from scipy.stats import norm

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).parent / "earnings_model.json"
SCALER_PATH = Path(__file__).parent / "earnings_scaler.pkl"

FEATURES = [
    "eps_surprise_pct",
    "historical_move_std",
    "moneyness",
    "dte",
    "iv",
    "is_call",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs_greeks(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if flag == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return {"delta": delta, "gamma": gamma}


def _typical_iv(ticker: str, default: float = 0.25) -> float:
    """Estimate a typical IV for the ticker using recent historical vol."""
    try:
        prices = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)["Close"]
        if prices.empty:
            return default
        returns = np.log(prices / prices.shift(1)).dropna()
        return float(returns.std() * np.sqrt(252))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Training data construction
# ---------------------------------------------------------------------------

def build_training_data(tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    For each historical earnings event, build a row of training features.
    We synthesise moneyness / DTE / IV variation to enrich the dataset.
    """
    if tickers is None:
        tickers = [p.stem.replace("earnings_", "") for p in RAW_DIR.glob("earnings_*.csv")]
    if not tickers:
        raise FileNotFoundError("No earnings CSVs found. Run data/fetch_earnings.py first.")

    rows = []
    for ticker in tickers:
        path = RAW_DIR / f"earnings_{ticker}.csv"
        if not path.exists():
            log.warning("Missing %s, skipping", path)
            continue

        df = pd.read_csv(path)
        if df.empty or "eps_surprise_pct" not in df.columns:
            continue

        hist_std = float(df["stock_move_pct"].std()) if len(df) > 1 else 0.03
        iv = _typical_iv(ticker)

        for _, ev in df.iterrows():
            surprise = float(ev["eps_surprise_pct"])
            target = float(ev["stock_move_pct"])
            if np.isnan(target):
                continue
            # Augment each real event with several moneyness / DTE scenarios
            for moneyness in [0.90, 0.95, 1.00, 1.05, 1.10]:
                for dte in [14, 30, 60]:
                    for is_call in [0, 1]:
                        rows.append({
                            "eps_surprise_pct": surprise,
                            "historical_move_std": hist_std,
                            "moneyness": moneyness,
                            "dte": dte,
                            "iv": iv,
                            "is_call": is_call,
                            "stock_move_pct": target,
                        })

    if not rows:
        raise ValueError("No training rows constructed. Check earnings CSVs have stock_move_pct.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Train / save / load
# ---------------------------------------------------------------------------

def train(tickers: Optional[List[str]] = None) -> xgb.XGBRegressor:
    df = build_training_data(tickers)
    X = df[FEATURES].values
    y = df["stock_move_pct"].values

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    model.save_model(str(MODEL_PATH))
    log.info("Earnings model saved → %s  (trained on %d rows)", MODEL_PATH, len(df))
    return model


def load_model() -> xgb.XGBRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained earnings model at {MODEL_PATH}. Run train() first.")
    m = xgb.XGBRegressor()
    m.load_model(str(MODEL_PATH))
    return m


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_option_change(
    ticker: str,
    strike: float,
    dte: int,
    eps_surprise_pct: float,
    flag: str = "call",
    r: float = 0.05,
) -> dict:
    """
    Predict how an option's price changes after an earnings event.

    Returns a dict with:
      predicted_stock_move_pct, current_price, predicted_price,
      predicted_option_change_pct, delta, gamma
    """
    model = load_model()

    # Current spot + IV
    try:
        spot_data = yf.download(ticker, period="5d", auto_adjust=True, progress=False)["Close"]
        S = float(spot_data.iloc[-1])
    except Exception:
        S = strike  # fallback: assume ATM

    iv = _typical_iv(ticker)
    hist_df_path = RAW_DIR / f"earnings_{ticker}.csv"
    if hist_df_path.exists():
        hist_df = pd.read_csv(hist_df_path)
        hist_std = float(hist_df["stock_move_pct"].std()) if len(hist_df) > 1 else 0.03
    else:
        hist_std = 0.03

    moneyness = S / strike
    T = dte / 365.0
    is_call = 1 if flag == "call" else 0

    feat = np.array([[eps_surprise_pct, hist_std, moneyness, dte, iv, is_call]])
    predicted_move = float(model.predict(feat)[0])

    # BS option price before earnings
    greeks = _bs_greeks(S, strike, T, r, iv, flag)
    d1 = (np.log(S / strike) + (r + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T)) if T > 0 else 0
    d2 = d1 - iv * np.sqrt(T) if T > 0 else 0
    if flag == "call":
        current_price = S * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    else:
        current_price = strike * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    current_price = max(current_price, 0.01)

    # Delta-gamma approximation: dP ≈ delta*S*move + 0.5*gamma*(S*move)²
    dS = S * predicted_move
    dP = greeks["delta"] * dS + 0.5 * greeks["gamma"] * dS ** 2
    predicted_price = max(current_price + dP, 0.0)
    change_pct = dP / current_price if current_price > 0 else 0.0

    return {
        "ticker": ticker,
        "strike": strike,
        "dte": dte,
        "flag": flag,
        "spot": round(S, 2),
        "iv": round(iv, 4),
        "predicted_stock_move_pct": round(predicted_move, 4),
        "current_price": round(current_price, 4),
        "predicted_price": round(predicted_price, 4),
        "predicted_option_change_pct": round(change_pct, 4),
        "delta": round(greeks["delta"], 4),
        "gamma": round(greeks["gamma"], 6),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
    result = predict_option_change("AAPL", strike=200, dte=30, eps_surprise_pct=0.10)
    print(result)
