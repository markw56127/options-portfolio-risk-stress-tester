"""
Earnings data collection via yfinance.
Fetches historical EPS (actual vs estimate) and computes post-earnings stock moves.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TICKERS = ["SPY", "AAPL", "TSLA", "NVDA", "QQQ"]


def fetch_earnings_history(ticker: str) -> pd.DataFrame:
    """
    Return historical quarterly earnings with EPS surprise and next-day stock move.
    Columns: date, eps_estimate, eps_actual, eps_surprise_pct, stock_move_pct
    """
    t = yf.Ticker(ticker)

    # --- EPS history ---
    try:
        ed = t.earnings_dates
        if ed is None or ed.empty:
            log.warning("%s: no earnings_dates available", ticker)
            return pd.DataFrame()
        # earnings_dates index is the earnings datetime; columns include EPS Estimate / Reported EPS
        ed = ed.copy()
        ed.index = pd.to_datetime(ed.index).tz_localize(None)
        ed = ed.rename(columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
        })
        ed = ed[["eps_estimate", "eps_actual"]].dropna(subset=["eps_actual"])
    except Exception as exc:
        log.warning("%s: could not fetch earnings_dates: %s", ticker, exc)
        return pd.DataFrame()

    # --- Stock prices to compute next-day move ---
    try:
        earliest = ed.index.min() - timedelta(days=3)
        latest = ed.index.max() + timedelta(days=3)
        prices = yf.download(ticker, start=earliest, end=latest,
                             auto_adjust=True, progress=False)["Close"]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
    except Exception as exc:
        log.warning("%s: could not fetch prices for earnings move: %s", ticker, exc)
        prices = pd.Series(dtype=float)

    rows = []
    for date, row in ed.iterrows():
        eps_est = row["eps_estimate"]
        eps_act = row["eps_actual"]
        # EPS surprise as fraction of |estimate|
        if eps_est != 0:
            surprise_pct = (eps_act - eps_est) / abs(eps_est)
        else:
            surprise_pct = 0.0

        # Next-day stock return after earnings
        stock_move = np.nan
        if not prices.empty:
            future = prices[prices.index > date]
            past = prices[prices.index <= date]
            if not future.empty and not past.empty:
                next_close = future.iloc[0]
                prev_close = past.iloc[-1]
                stock_move = float((next_close - prev_close) / prev_close)

        rows.append({
            "date": date.date(),
            "eps_estimate": float(eps_est),
            "eps_actual": float(eps_act),
            "eps_surprise_pct": float(surprise_pct),
            "stock_move_pct": stock_move,
        })

    df = pd.DataFrame(rows).dropna(subset=["stock_move_pct"])
    df = df.sort_values("date").reset_index(drop=True)

    out = RAW_DIR / f"earnings_{ticker}.csv"
    df.to_csv(out, index=False)
    log.info("%s: saved %d earnings events → %s", ticker, len(df), out)
    return df


def get_upcoming_earnings(ticker: str) -> dict:
    """
    Return the next scheduled earnings date and current EPS estimate.
    Falls back gracefully if unavailable.
    """
    t = yf.Ticker(ticker)
    result = {
        "ticker": ticker,
        "upcoming_date": None,
        "eps_estimate": None,
    }
    try:
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed.index = pd.to_datetime(ed.index).tz_localize(None)
            future = ed[ed.index > pd.Timestamp.now()]
            if not future.empty:
                next_row = future.sort_index().iloc[0]
                result["upcoming_date"] = str(next_row.name.date())
                est = next_row.get("EPS Estimate")
                result["eps_estimate"] = float(est) if pd.notna(est) else None
    except Exception as exc:
        log.warning("%s: get_upcoming_earnings failed: %s", ticker, exc)
    return result


def load_earnings_history(ticker: str) -> pd.DataFrame:
    """Load saved earnings CSV; fetch fresh if missing."""
    path = RAW_DIR / f"earnings_{ticker}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return fetch_earnings_history(ticker)


def fetch_all_earnings(tickers: List[str] = DEFAULT_TICKERS) -> None:
    for ticker in tickers:
        try:
            fetch_earnings_history(ticker)
        except Exception as exc:
            log.error("%s: unhandled error: %s", ticker, exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_all_earnings()
