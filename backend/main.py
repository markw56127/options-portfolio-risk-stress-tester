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
from scipy.stats import norm as sp_norm

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


class GreeksSurfaceRequest(BaseModel):
    K: float
    T: float
    r: float = 0.05
    sigma: float
    flag: str = "call"
    S_min_pct: int = 70
    S_max_pct: int = 130


class ThetaDecayRequest(BaseModel):
    S: float
    K: float
    T_max_days: int
    r: float = 0.05
    sigma: float
    flag: str = "call"


class EarningsImpactRequest(BaseModel):
    ticker: str
    strike: float
    dte: int
    eps_surprise_pct: float
    flag: str = "call"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# VaR endpoints
# ---------------------------------------------------------------------------

@app.post("/var/parametric", tags=["VaR"])
def compute_parametric_var(req: VaRRequest):
    positions = [{"weight": p.weight, "sigma_daily": p.sigma_daily} for p in req.positions]
    return portfolio_var(positions, req.portfolio_value, req.confidence, req.horizon_days)


@app.post("/var/historical", tags=["VaR"])
async def compute_historical_var(file: UploadFile = File(...), confidence: float = 0.95):
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
# Options pricing
# ---------------------------------------------------------------------------

@app.post("/options/price", tags=["Options"])
def option_price(req: BSRequest):
    price = bs_price(req.S, req.K, req.T, req.r, req.sigma, req.flag)
    greeks = bs_greeks(req.S, req.K, req.T, req.r, req.sigma, req.flag)
    # Probability of being ITM at expiry: N(d2) for call, N(-d2) for put
    if req.T > 0 and req.sigma > 0:
        d1 = (np.log(req.S / req.K) + (req.r + 0.5 * req.sigma**2) * req.T) / (req.sigma * np.sqrt(req.T))
        d2 = d1 - req.sigma * np.sqrt(req.T)
        pop = float(sp_norm.cdf(d2) if req.flag == "call" else sp_norm.cdf(-d2))
    else:
        pop = 1.0 if ((req.flag == "call" and req.S > req.K) or
                      (req.flag == "put" and req.S < req.K)) else 0.0
    return {
        "price": round(price, 4),
        **{k: round(v, 6) for k, v in greeks.items()},
        "pop": round(pop, 4),
    }


@app.post("/options/greeks_surface", tags=["Options"])
def greeks_surface(req: GreeksSurfaceRequest):
    """Bulk-compute price + Greeks across a spot range. Used for client-side chart rendering."""
    spots = [req.K * x / 100 for x in range(req.S_min_pct, req.S_max_pct + 1)]
    return [
        {
            "spot": round(s, 4),
            "price": round(bs_price(s, req.K, req.T, req.r, req.sigma, req.flag), 4),
            **{k: round(v, 6) for k, v in bs_greeks(s, req.K, req.T, req.r, req.sigma, req.flag).items()},
        }
        for s in spots
    ]


@app.post("/options/theta_decay", tags=["Options"])
def theta_decay_curve(req: ThetaDecayRequest):
    """Compute option price for each day from T_max_days down to 1 (spot held fixed)."""
    return [
        {"days": d, "price": round(bs_price(req.S, req.K, d / 365.0, req.r, req.sigma, req.flag), 4)}
        for d in range(req.T_max_days, 0, -1)
    ]


@app.get("/options/chain/{ticker}", tags=["Options"])
def get_options_chain(ticker: str, max_strikes: int = 20):
    """Fetch live options chain for a ticker. Returns nearest expiry calls + puts."""
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            raise HTTPException(404, f"No options found for {ticker}")
        exp = exps[0]
        chain = t.option_chain(exp)

        spot_data = yf.download(ticker, period="2d", auto_adjust=True, progress=False)["Close"]
        spot_data = spot_data.squeeze()
        S = float(spot_data.iloc[-1]) if not spot_data.empty else None

        def enrich(df, flag):
            df = df.copy()
            df["flag"] = flag
            if S and len(df) > max_strikes:
                df["dist"] = abs(df["strike"] - S)
                df = df.nsmallest(max_strikes, "dist").drop(columns="dist")
            keep = ["strike", "lastPrice", "bid", "ask", "impliedVolatility",
                    "volume", "openInterest", "inTheMoney", "flag"]
            return df[[c for c in keep if c in df.columns]].fillna(0)

        return {
            "ticker": ticker,
            "expiry": exp,
            "spot": S,
            "calls": enrich(chain.calls, "call").to_dict("records"),
            "puts": enrich(chain.puts, "put").to_dict("records"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to fetch options chain: {exc}")


@app.get("/options/price_history/{ticker}", tags=["Options"])
def get_option_price_history(
    ticker: str,
    strike: float = 100.0,
    dte: int = 30,
    flag: str = "call",
    period: str = "1M",
    r: float = 0.05,
):
    """
    Compute theoretical option price over historical spot prices using BS + rolling IV.
    At each past date the option is treated as a fresh contract with the selected DTE.
    """
    period_yf = {"1W": "30d", "1M": "60d", "3M": "120d", "6M": "270d"}.get(period, "60d")
    period_n = {"1W": 7, "1M": 21, "3M": 63, "6M": 126}.get(period, 21)
    T_fixed = dte / 365.0

    try:
        raw = yf.download(ticker, period=period_yf, auto_adjust=True, progress=False)["Close"]
        prices = raw.squeeze()
        if prices.empty:
            raise HTTPException(404, f"No price data for {ticker}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to fetch prices: {exc}")

    log_ret = np.log(prices / prices.shift(1)).dropna()
    roll_vol = log_ret.rolling(window=min(20, max(len(log_ret) - 1, 1))).std() * np.sqrt(252)
    fallback_iv = float(roll_vol.dropna().iloc[-1]) if not roll_vol.dropna().empty else 0.25

    rows = []
    for date, spot in prices.iloc[-period_n:].items():
        ts = pd.Timestamp(date)
        try:
            iv = float(roll_vol.loc[ts])
            if not np.isfinite(iv) or iv <= 0:
                iv = fallback_iv
        except (KeyError, TypeError):
            iv = fallback_iv
        opt = bs_price(float(spot), strike, T_fixed, r, iv, flag)
        rows.append({
            "date": str(ts.date()),
            "spot": round(float(spot), 2),
            "theoretical_price": round(max(opt, 0.0), 4),
            "iv": round(iv, 4),
        })

    return {"ticker": ticker, "strike": strike, "dte": dte, "flag": flag, "period": period, "history": rows}


# ---------------------------------------------------------------------------
# Stress test
# ---------------------------------------------------------------------------

@app.post("/stress/scenario", tags=["VaR"])
def stress_test(req: VaRRequest, shock_pct: float = -0.10):
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
# Earnings endpoints
# ---------------------------------------------------------------------------

@app.get("/earnings/{ticker}", tags=["Earnings"])
def get_earnings_info(ticker: str):
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
    hist = load_earnings_history(ticker.upper())
    if hist.empty:
        raise HTTPException(404, f"No earnings history for {ticker}. Run the data pipeline first.")
    hist = hist.tail(8).copy()
    hist["date"] = hist["date"].astype(str)
    return hist.to_dict("records")


@app.post("/earnings/impact", tags=["Earnings"])
def earnings_impact(req: EarningsImpactRequest):
    try:
        return predict_option_change(
            ticker=req.ticker.upper(),
            strike=req.strike,
            dte=req.dte,
            eps_surprise_pct=req.eps_surprise_pct,
            flag=req.flag,
        )
    except FileNotFoundError as exc:
        raise HTTPException(503, f"Model not trained: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")


# ---------------------------------------------------------------------------
# Multi-ticker comparison
# ---------------------------------------------------------------------------

@app.get("/compare/prices", tags=["Comparison"])
def compare_prices(tickers: str, period: str = "3M"):
    """Return historical closing prices for up to 5 comma-separated tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:5]
    if not ticker_list:
        raise HTTPException(400, "No tickers provided.")
    yf_period = {"1W": "7d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y"}.get(period, "3mo")
    result = {}
    for tk in ticker_list:
        try:
            raw = yf.download(tk, period=yf_period, auto_adjust=True, progress=False)["Close"]
            prices = raw.squeeze()
            if not prices.empty:
                result[tk] = [
                    {"date": str(pd.Timestamp(d).date()), "price": round(float(p), 2)}
                    for d, p in prices.items()
                ]
        except Exception:
            continue
    return result
