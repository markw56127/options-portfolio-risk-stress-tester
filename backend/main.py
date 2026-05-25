"""FastAPI backend: VaR calculation, options pricing, and earnings impact endpoints."""

import io
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent.parent))
from models.black_scholes import bs_greeks, bs_price, historical_var, portfolio_var
from data.fetch_earnings import get_upcoming_earnings, load_earnings_history
from models.earnings_impact_model import predict_option_change

log = logging.getLogger(__name__)

app = FastAPI(
    title="Portfolio Risk Stress-Tester API",
    description="Options pricing, VaR calculation, and earnings impact prediction.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class Position(BaseModel):
    ticker: str
    weight: float
    sigma_daily: float


class VaRRequest(BaseModel):
    positions: list[Position]
    portfolio_value: float = 100_000.0
    confidence: float = 0.95
    horizon_days: int = 1


class BSRequest(BaseModel):
    S: float
    K: float
    T: float
    r: float = 0.05
    sigma: float
    flag: str = "call"


class EarningsImpactRequest(BaseModel):
    ticker: str
    strike: float
    dte: int
    eps_surprise_pct: float  # e.g. 0.10 for a 10% beat
    flag: str = "call"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# VaR endpoints (existing)
# ---------------------------------------------------------------------------

@app.post("/var/parametric", tags=["VaR"])
def compute_parametric_var(req: VaRRequest):
    positions = [{"weight": p.weight, "sigma_daily": p.sigma_daily} for p in req.positions]
    return portfolio_var(positions, req.portfolio_value, req.confidence, req.horizon_days)


@app.post("/var/historical", tags=["VaR"])
async def compute_historical_var(file: UploadFile = File(...), confidence: float = 0.95):
    """Accept a CSV with a 'returns' column and compute historical VaR."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not parse CSV")
    if "returns" not in df.columns:
        raise HTTPException(400, "CSV must have a 'returns' column")
    var = historical_var(df["returns"].dropna().values, confidence)
    return {"var": round(var, 6), "confidence": confidence, "n_observations": len(df)}


@app.post("/var/portfolio_csv", tags=["VaR"])
async def var_from_portfolio_csv(file: UploadFile = File(...), confidence: float = 0.95):
    """Accept a portfolio CSV (ticker, weight, sigma_daily) and return parametric VaR."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "Could not parse CSV")
    required = {"ticker", "weight", "sigma_daily"}
    if not required.issubset(df.columns):
        raise HTTPException(400, f"CSV must contain columns: {required}")
    positions = df[["weight", "sigma_daily"]].to_dict("records")
    result = portfolio_var(positions, portfolio_value=1.0, confidence=confidence)
    result["tickers"] = df["ticker"].tolist()
    return result


# ---------------------------------------------------------------------------
# Options pricing (existing + new chain endpoint)
# ---------------------------------------------------------------------------

@app.post("/options/price", tags=["Options"])
def option_price(req: BSRequest):
    price = bs_price(req.S, req.K, req.T, req.r, req.sigma, req.flag)
    greeks = bs_greeks(req.S, req.K, req.T, req.r, req.sigma, req.flag)
    return {"price": round(price, 4), **{k: round(v, 6) for k, v in greeks.items()}}


@app.get("/options/chain/{ticker}", tags=["Options"])
def get_options_chain(ticker: str, max_strikes: int = 20):
    """
    Fetch the live options chain for a ticker via yfinance.
    Returns the nearest expiry with calls and puts (columns: strike, lastPrice, impliedVolatility, delta, gamma, bid, ask, inTheMoney).
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            raise HTTPException(404, f"No options found for {ticker}")
        # Use nearest expiry
        exp = exps[0]
        chain = t.option_chain(exp)

        spot_data = yf.download(ticker, period="2d", auto_adjust=True, progress=False)["Close"]
        S = float(spot_data.iloc[-1]) if not spot_data.empty else None

        def enrich(df, flag):
            df = df.copy()
            df["flag"] = flag
            # Keep only the most liquid strikes (nearest to ATM)
            if S and len(df) > max_strikes:
                df["dist"] = abs(df["strike"] - S)
                df = df.nsmallest(max_strikes, "dist").drop(columns="dist")
            keep = ["strike", "lastPrice", "bid", "ask", "impliedVolatility",
                    "volume", "openInterest", "inTheMoney", "flag"]
            return df[[c for c in keep if c in df.columns]].fillna(0)

        calls = enrich(chain.calls, "call").to_dict("records")
        puts = enrich(chain.puts, "put").to_dict("records")

        return {"ticker": ticker, "expiry": exp, "spot": S, "calls": calls, "puts": puts}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to fetch options chain: {exc}")


# ---------------------------------------------------------------------------
# Stress test (existing)
# ---------------------------------------------------------------------------

@app.post("/stress/scenario", tags=["VaR"])
def stress_test(req: VaRRequest, shock_pct: float = -0.10):
    """Apply a market shock and compare base vs stressed VaR."""
    base_pos = [{"weight": p.weight, "sigma_daily": p.sigma_daily} for p in req.positions]
    shocked_pos = [
        {"weight": p.weight, "sigma_daily": p.sigma_daily * (1 + abs(shock_pct))}
        for p in req.positions
    ]
    base = portfolio_var(base_pos, req.portfolio_value, req.confidence, req.horizon_days)
    stressed = portfolio_var(shocked_pos, req.portfolio_value, req.confidence, req.horizon_days)
    return {
        "base_var": base["var"],
        "stressed_var": stressed["var"],
        "shock_pct": shock_pct,
        "var_increase_pct": round((stressed["var"] - base["var"]) / base["var"] * 100, 2),
    }


# ---------------------------------------------------------------------------
# Earnings endpoints (new)
# ---------------------------------------------------------------------------

@app.get("/earnings/{ticker}", tags=["Earnings"])
def get_earnings_info(ticker: str):
    """
    Return upcoming earnings date + EPS estimate, plus historical move stats.
    """
    upcoming = get_upcoming_earnings(ticker.upper())
    hist = load_earnings_history(ticker.upper())
    stats = {}
    if not hist.empty and "stock_move_pct" in hist.columns:
        moves = hist["stock_move_pct"].dropna()
        stats = {
            "historical_avg_move_pct": round(float(moves.mean()), 4),
            "historical_move_std": round(float(moves.std()), 4),
            "n_events": int(len(moves)),
        }
    return {**upcoming, **stats}


@app.get("/earnings/history/{ticker}", tags=["Earnings"])
def get_earnings_history(ticker: str):
    """Return historical earnings events with EPS surprise and stock move."""
    hist = load_earnings_history(ticker.upper())
    if hist.empty:
        raise HTTPException(404, f"No earnings history for {ticker}. Run the data pipeline first.")
    # Return most recent 8 quarters
    hist = hist.tail(8).copy()
    hist["date"] = hist["date"].astype(str)
    return hist.to_dict("records")


@app.post("/earnings/impact", tags=["Earnings"])
def earnings_impact(req: EarningsImpactRequest):
    """
    Predict how a specific option's price changes after an earnings event.
    Uses an XGBoost model trained on historical earnings × options data.
    """
    try:
        result = predict_option_change(
            ticker=req.ticker.upper(),
            strike=req.strike,
            dte=req.dte,
            eps_surprise_pct=req.eps_surprise_pct,
            flag=req.flag,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model not trained yet: {exc}. Run scripts/run_pipeline.py first.")
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")
