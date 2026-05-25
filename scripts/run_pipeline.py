"""End-to-end pipeline: fetch data → preprocess → train all models → compare."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from data.fetch_market_data import fetch_price_history, fetch_options_chain, build_vol_surface
from data.fetch_earnings import fetch_all_earnings
from data.preprocess import build_dataset
from models.xgboost_model import train as train_xgb, load_data
from models.lstm_model import train as train_lstm
from models.earnings_impact_model import train as train_earnings
from models.compare_models import run_comparison

TICKERS = ["SPY", "AAPL", "TSLA", "NVDA", "QQQ"]

if __name__ == "__main__":
    print("=== Step 1: Fetch market data (prices + options) ===")
    fetch_price_history(TICKERS)
    for t in TICKERS:
        opts = fetch_options_chain(t)
        if not opts.empty:
            build_vol_surface(opts)

    print("=== Step 2: Fetch earnings history ===")
    fetch_all_earnings(TICKERS)

    print("=== Step 3: Preprocess options data ===")
    X, y = build_dataset()

    print("=== Step 4: Train XGBoost options pricing model ===")
    X_df, y_s = load_data()
    train_xgb(X_df, y_s)

    print("=== Step 5: Train LSTM options pricing model ===")
    train_lstm(X.values, y.values)

    print("=== Step 6: Train earnings impact model ===")
    train_earnings(TICKERS)

    print("=== Step 7: Compare models ===")
    run_comparison()

    print("=== Pipeline complete. Start the app with: docker-compose up ===")
