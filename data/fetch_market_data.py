"""
Fetches historical prices and options chain data (Greeks + IV) via yfinance.
Outputs structured CSVs to data/raw/ for downstream model training.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Historical price + realized volatility
# ---------------------------------------------------------------------------

def fetch_price_history(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    """Download adjusted close prices and compute log returns + rolling vol."""
    log.info("Downloading price history for %s", tickers)
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    returns = np.log(raw / raw.shift(1)).dropna()

    # 21-day (monthly) realized volatility, annualized
    rv = returns.rolling(21).std() * np.sqrt(252)

    out = pd.concat(
        {"price": raw, "log_return": returns, "realized_vol_21d": rv}, axis=1
    )
    out.columns = ["_".join(c) for c in out.columns]
    path = RAW_DIR / "price_history.csv"
    out.to_csv(path)
    log.info("Saved price history -> %s", path)
    return out


# ---------------------------------------------------------------------------
# Options chain: implied vol surface + Greeks
# ---------------------------------------------------------------------------

def _bs_greeks(S, K, T, r, sigma, flag="call"):
    """Black-Scholes d1/d2 based Greeks."""
    if T <= 0 or sigma <= 0:
        return {}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        delta = norm.cdf(d1)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% move in vol
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if flag == "call" else norm.cdf(-d2))
    ) / 365
    return {"bs_price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def fetch_options_chain(ticker: str, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Pull all available expiry chains from yfinance, append BS Greeks,
    and return a flat DataFrame suitable for model training.
    """
    log.info("Fetching options chain for %s", ticker)
    t = yf.Ticker(ticker)
    S = t.fast_info["lastPrice"]
    expirations = t.options

    frames = []
    for exp in expirations:
        exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        T = max((exp_dt - datetime.now()).days / 365, 1e-6)
        try:
            chain = t.option_chain(exp)
        except Exception as e:
            log.warning("Could not fetch chain for %s %s: %s", ticker, exp, e)
            continue

        for df, flag in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df.copy()
            df["expiration"] = exp
            df["dte"] = (exp_dt - datetime.now()).days
            df["T"] = T
            df["S"] = S
            df["flag"] = flag
            df["ticker"] = ticker
            df["moneyness"] = df["strike"] / S

            greeks = df.apply(
                lambda row: pd.Series(
                    _bs_greeks(S, row["strike"], T, risk_free_rate, row.get("impliedVolatility", 0.2), flag)
                ),
                axis=1,
            )
            df = pd.concat([df, greeks], axis=1)
            frames.append(df)

    if not frames:
        log.error("No options data returned for %s", ticker)
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    cols = [
        "ticker", "flag", "expiration", "dte", "T", "S", "strike", "moneyness",
        "lastPrice", "bid", "ask", "volume", "openInterest",
        "impliedVolatility", "bs_price", "delta", "gamma", "vega", "theta",
    ]
    result = result[[c for c in cols if c in result.columns]]
    path = RAW_DIR / f"options_{ticker}.csv"
    result.to_csv(path, index=False)
    log.info("Saved options chain -> %s (%d rows)", path, len(result))
    return result


# ---------------------------------------------------------------------------
# Volatility surface (IV by strike x expiry)
# ---------------------------------------------------------------------------

def build_vol_surface(options_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot implied vol into a strike-moneyness x DTE surface."""
    calls = options_df[options_df["flag"] == "call"].copy()
    calls["moneyness_bucket"] = pd.cut(
        calls["moneyness"], bins=[0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3], labels=False
    )
    surface = calls.pivot_table(
        index="moneyness_bucket", columns="dte", values="impliedVolatility", aggfunc="median"
    )
    path = RAW_DIR / "vol_surface.csv"
    surface.to_csv(path)
    log.info("Saved vol surface -> %s", path)
    return surface


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = ["SPY", "AAPL", "TSLA", "NVDA", "QQQ"]

if __name__ == "__main__":
    prices = fetch_price_history(DEFAULT_TICKERS)
    for ticker in DEFAULT_TICKERS:
        opts = fetch_options_chain(ticker)
        if not opts.empty:
            build_vol_surface(opts)
    log.info("Data collection complete.")
