# Portfolio Risk Stress-Tester

**STATS 418 Final Project — Mark Wang, UCLA Department of Statistics & Data Science**

A full-stack system for options pricing and portfolio risk quantification. Collects live market data via Yahoo Finance, trains three complementary pricing models (Black-Scholes, XGBoost, LSTM), and exposes results through a FastAPI backend and Streamlit dashboard.

---

## Project Structure

```
stats418_final_project/
├── data/
│   ├── fetch_market_data.py   # yfinance data collection (prices, options chains, vol surface)
│   └── preprocess.py          # feature engineering, train/val/test split
├── models/
│   ├── black_scholes.py       # analytical pricing, Greeks, VaR
│   ├── xgboost_model.py       # gradient boosting on options features
│   ├── lstm_model.py          # recurrent network for temporal dynamics
│   └── compare_models.py      # MAPE / VaR accuracy comparison
├── backend/
│   └── main.py                # FastAPI: /var, /options/price, /stress/scenario
├── frontend/
│   └── app.py                 # Streamlit dashboard
├── scripts/
│   └── run_pipeline.py        # end-to-end: fetch → preprocess → train
├── eda_proposal.py            # generates proposal presentation figures
└── docker-compose.yml
```

---

## Quickstart

### Option A — Docker (recommended)

```bash
docker-compose up --build
```

- Frontend: http://localhost:8501
- API docs: http://localhost:8000/docs

### Option B — Local

```bash
pip install -r requirements.txt

# Run the full data + training pipeline
python scripts/run_pipeline.py

# Start backend (in one terminal)
uvicorn backend.main:app --reload --port 8000

# Start frontend (in another terminal)
streamlit run frontend/app.py
```

---

## Data

Five tickers collected via `yfinance`: **SPY, AAPL, NVDA, TSLA, QQQ**

| Data Type | Records | Description |
|-----------|---------|-------------|
| Options Chain | ~5,100 | All available expirations with strikes |
| Greeks | ~5,100 | Delta, Gamma, Vega, Theta (computed via Black-Scholes) |
| Price History | ~1,260 | 252 trading days × 5 tickers |
| Volatility Surface | ~300 | IV organized by moneyness bucket and DTE |

---

## Models

| Model | Approach | Expected MAPE |
|-------|----------|---------------|
| Black-Scholes | Analytical, closed-form | ~20–25% (baseline) |
| XGBoost | Gradient boosting on moneyness, IV, DTE, Greeks | ~8–12% |
| LSTM | 21-day rolling window, temporal dynamics | ~10–15% |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/var/parametric` | Parametric VaR from portfolio positions |
| `POST` | `/var/historical` | Historical VaR from a returns CSV |
| `POST` | `/var/portfolio_csv` | Parametric VaR from a portfolio CSV upload |
| `POST` | `/options/price` | Black-Scholes price + Greeks |
| `POST` | `/stress/scenario` | VaR under a configurable price shock |
| `GET` | `/health` | Health check |

### Portfolio CSV format

```csv
ticker,weight,sigma_daily
SPY,0.4,0.008
AAPL,0.3,0.015
TSLA,0.3,0.025
```

---

## Dashboard Features

- **Portfolio upload** — drag-and-drop CSV, renders holdings table and weight pie chart
- **Parametric VaR** — variance-covariance VaR at configurable confidence level (90–99%) and horizon (1–21 days)
- **Stress testing** — apply a market shock (-1% to -30%) and compare base vs. stressed VaR
- **Options pricer** — interactive Black-Scholes pricer with live Greeks output

---

## Timeline

Project runs **Apr 27 – June 1, 2026** (5 weeks).

| Phase | Dates | Status |
|-------|-------|--------|
| Data collection & EDA | Apr 27 – May 10 | In Progress |
| Feature engineering | May 4 – May 11 | In Progress |
| Black-Scholes baseline | May 4 – May 11 | Planned |
| XGBoost model | May 11 – May 25 | Planned |
| LSTM model | May 18 – May 25 | Planned |
| Model comparison | May 25 – June 1 | Planned |
| Deployment & docs | May 25 – June 1 | Planned |

---

## Requirements

Python 3.10+. Key dependencies:

```
yfinance, pandas, numpy, scipy
scikit-learn, xgboost, torch
fastapi, uvicorn, pydantic
streamlit, plotly
```

See `requirements.txt` for pinned versions.
