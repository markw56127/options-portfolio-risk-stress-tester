# Portfolio Risk Stress-Tester

**STATS 418 Final Project — Mark Wang, UCLA Department of Statistics & Data Science**

A full-stack system for options pricing, earnings impact prediction, and portfolio risk quantification. Collects live market data via Yahoo Finance, trains three complementary pricing models (Black-Scholes, XGBoost, LSTM) plus an earnings impact model, and exposes everything through a FastAPI backend and Streamlit dashboard.

> **Live App:** [Streamlit — TBD after deployment](#)
> **API Docs:** [Cloud Run — TBD after deployment](#)

---

## Project Structure

```
stats418_final_project/
├── data/
│   ├── fetch_market_data.py   # yfinance: prices, options chains, vol surface
│   ├── fetch_earnings.py      # yfinance: EPS history, upcoming earnings dates
│   └── preprocess.py          # feature engineering, StandardScaler, train/val/test split
├── models/
│   ├── black_scholes.py            # analytical pricing, Greeks, VaR
│   ├── xgboost_model.py            # gradient boosting on options features
│   ├── lstm_model.py               # PyTorch LSTM for temporal dynamics
│   ├── earnings_impact_model.py    # XGBoost + delta-gamma: EPS surprise → option ΔP
│   └── compare_models.py           # MAPE / R² comparison across models
├── backend/
│   └── main.py                # FastAPI: pricing, VaR, earnings impact, options chain
├── frontend/
│   └── app.py                 # Streamlit: earnings predictor, VaR dashboard, BS pricer
├── scripts/
│   └── run_pipeline.py        # end-to-end: fetch → preprocess → train all models
├── presentation/              # proposal and final slides
├── eda_proposal.py            # generates proposal presentation figures
├── docker-compose.yml
└── requirements.txt
```

---

## Quickstart

### Option A — Docker (recommended)

```bash
# 1. Train models first (generates data and model weights)
python scripts/run_pipeline.py

# 2. Start both services
docker-compose up --build
```

- **Frontend:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

### Option B — Local (no Docker)

```bash
pip install -r requirements.txt

# Train models
python scripts/run_pipeline.py

# Terminal 1 — backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run frontend/app.py
```

---

## Features

### Earnings Impact Predictor *(main feature)*
Enter any ticker to see its upcoming earnings date and EPS estimate, then slide an EPS surprise % slider to predict how a specific option's price would change. Powered by an XGBoost model trained on historical earnings events, with a delta-gamma approximation for option price impact.

### Portfolio VaR & Stress Test
Upload a portfolio CSV (`ticker, weight, sigma_daily`) to compute parametric Value-at-Risk at any confidence level and horizon. Run a stress scenario with a configurable market shock to see how VaR changes.

### Black-Scholes Option Pricer
Interactive pricer for calls and puts with live Greeks output (Delta, Gamma, Vega, Theta). Includes a Greeks-vs-spot visualization across a user-defined range.

---

## Data

Five tickers: **SPY, AAPL, NVDA, TSLA, QQQ** — all collected live via `yfinance`.

| Data Type | Records | Description |
|-----------|---------|-------------|
| Options Chain | ~5,100 | All available expirations with strikes |
| Greeks | ~5,100 | Delta, Gamma, Vega, Theta via Black-Scholes |
| Price History | ~1,260 | 252 trading days × 5 tickers |
| Volatility Surface | ~300 | IV by moneyness bucket and DTE |
| Earnings History | ~20 | EPS actual vs estimate, post-earnings stock moves |

---

## Models

| Model | Approach | Target |
|-------|----------|--------|
| Black-Scholes | Analytical, closed-form | Option price (baseline) |
| XGBoost | Gradient boosting on moneyness, IV, DTE, Greeks | Option price |
| LSTM | 21-day rolling window, PyTorch | Option price (temporal) |
| Earnings Impact | XGBoost → delta-gamma approximation | Post-earnings option ΔP |

---

## API Endpoints

Full interactive docs at `/docs` (Swagger UI).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/var/parametric` | Parametric VaR from portfolio JSON |
| `POST` | `/var/historical` | Historical VaR from returns CSV |
| `POST` | `/var/portfolio_csv` | Parametric VaR from portfolio CSV |
| `POST` | `/options/price` | Black-Scholes price + Greeks |
| `GET` | `/options/chain/{ticker}` | Live options chain from yfinance |
| `GET` | `/earnings/{ticker}` | Upcoming date, EPS estimate, historical move stats |
| `GET` | `/earnings/history/{ticker}` | Last 8 quarters of EPS surprises + stock moves |
| `POST` | `/earnings/impact` | Predict option price change given EPS surprise |
| `POST` | `/stress/scenario` | VaR comparison under a market shock |

### Portfolio CSV format

```csv
ticker,weight,sigma_daily
SPY,0.4,0.008
AAPL,0.3,0.015
TSLA,0.3,0.025
```

---

## Deployment

| Service | Platform | URL |
|---------|----------|-----|
| Frontend (Streamlit) | Streamlit Community Cloud | TBD |
| Backend API (FastAPI) | Google Cloud Run | TBD |

*URLs will be updated after deployment. Both services must be live through June 9, 2026.*

---

## Solution Architecture

*Architecture diagram to be added.*

```
Yahoo Finance API
      │
      ▼
data/fetch_market_data.py   data/fetch_earnings.py
      │                              │
      └──────────┬───────────────────┘
                 ▼
         data/preprocess.py
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
models/xgboost_model.py   models/lstm_model.py
models/earnings_impact_model.py
       │
       ▼
backend/main.py  (FastAPI — Cloud Run)
       │
       ▼
frontend/app.py  (Streamlit — Streamlit Cloud)
       │
       ▼
    Browser
```

---

## AI Assistant Usage

*Full documentation to be added.*

This project was developed with significant assistance from **Claude Code** (Anthropic). Key areas where AI assistance was used:

- **Code creation** — Editing and writing of individual lines of code
- **Language** - README language and writing, timeline and chart creation
- **Frontend** — Streamlit layout, Plotly chart configuration, session state management
- **Deployment** — Dockerfile optimization, Cloud Run configuration, Streamlit secrets

---

## Requirements

Python 3.10+. See `requirements.txt` for pinned versions.

```
yfinance, pandas, numpy, scipy
scikit-learn, xgboost, torch
fastapi, uvicorn, pydantic
streamlit, plotly
```

---

## Timeline

**Apr 27 – June 1, 2026** (5 weeks)

| Phase | Dates | Status |
|-------|-------|--------|
| Data collection & EDA | Apr 27 – May 10 | In Progress |
| Feature engineering | May 4 – May 11 | In Progress |
| Black-Scholes baseline | May 4 – May 11 | Complete |
| XGBoost + Earnings models | May 11 – May 25 | Planned |
| LSTM model | May 18 – May 25 | Planned |
| Model comparison | May 25 – June 1 | Planned |
| Deployment & docs | May 25 – June 1 | Planned |
