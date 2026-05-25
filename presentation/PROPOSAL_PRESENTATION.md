# STATS 418 Final Project: Proposal Presentation
## Portfolio Risk Analysis & Option Pricing Models

**Duration:** 5-7 minutes  
**Total Slides:** 6 (1 title + 5 content)

---

## SLIDE 1: Title Slide
**Visual:** Project title, team, date, course number

### Speaker Notes:
Good morning/afternoon. I'm presenting a project on **Portfolio Risk Analysis and Option Pricing Models**. 

The goal is to build a comprehensive system that:
- Collects real market data (historical prices and options chains)
- Performs exploratory analysis to understand market structure
- Develops and compares multiple modeling approaches for option pricing
- Enables portfolio risk quantification through Value-at-Risk calculations

This is particularly relevant for quantitative finance and risk management applications.

---

## SLIDE 2: Data Collection Approach & Progress

**Visual Elements:**
- Data pipeline completion status (4 components)
- Ticker coverage bar chart (SPY, AAPL, NVDA, TSLA, QQQ)
- Current data volume by type (~11,760 total records collected)
- Data type distribution pie chart

### Speaker Notes (1.5 minutes):

**Data Sources:**
We're using Yahoo Finance (yfinance) to automatically collect:
1. **Historical Price Data** - 252 trading days (~1 year) for 5 major indices/stocks
2. **Options Chains** - All available expiration dates with complete Greeks
3. **Computed Greeks** - Delta, Gamma, Vega, Theta using Black-Scholes formulas
4. **Volatility Surface** - IV organized by strike and expiration

**Progress:**
- Data collection: 100% complete - we have ~5,100 options records across tickers
- Feature engineering: 100% complete - all Greeks and derived features computed
- Black-Scholes baseline: 100% complete - analytical benchmark ready
- Volatility surface: 85% complete - currently analyzing IV smile/skew patterns

**Coverage:**
- **5 tickers:** SPY, AAPL, NVDA, TSLA, QQQ (mix of indices and mega-cap stocks)
- **~1000 options records per ticker** providing diverse moneyness, expiration, and volatility regimes
- **252 days of price history** with computed daily returns and realized volatility

---

## SLIDE 3: Exploratory Data Analysis

**Visual Elements:**
- Implied volatility distribution by ticker (SPY vs AAPL vs NVDA vol levels)
- Call price vs moneyness scatter (colored by DTE — hockey stick + time value)
- Realized volatility over time (time-varying vol motivates ML over constant-IV models)

### Speaker Notes (1.5 minutes):

**Key EDA Findings:**

1. **Implied Volatility by Ticker:**
   - SPY: moderate IV (~15%), stable index
   - AAPL: higher IV (~25%), reflecting individual stock risk
   - NVDA: highest IV (~35%), indicating tech sector volatility premium
   - This spread motivates ticker-specific modeling rather than one-size-fits-all

2. **Call Price vs Moneyness:** Prices rise sharply once moneyness exceeds 1.0 (ITM). Longer DTE options (yellow) are worth more than short-dated ones (purple) at the same moneyness — this is time value, and it's exactly what our models need to capture.

3. **Realized Volatility:** 21-day rolling vol shows clear time-variation — it is not constant. This directly contradicts Black-Scholes' key assumption and motivates ML models that can learn from changing market conditions.

---

## SLIDE 4: Proposed Modeling Approaches

**Visual Elements:**
- Model 1: Black-Scholes fit (actual vs. predicted scatter)
- Model 2: Feature importance for ML models
- Model comparison: interpretability vs. performance trade-off
- Greeks risk profile (delta vs. gamma scatter)

### Speaker Notes (1.5 minutes):

**Three Complementary Models:**

1. **Black-Scholes Baseline** (Analytical)
   - Closed-form option pricing with constant IV
   - Pros: fast, interpretable, provides Greeks analytically
   - Cons: ignores smile/skew, assumes lognormal returns and constant vol
   - Use case: benchmark and baseline risk calculations
   - Expected accuracy: ~75-80% vs. market prices

2. **XGBoost Regressor** (Machine Learning)
   - Gradient boosting on options features
   - Key features: moneyness, implied vol, DTE, delta, gamma, vega
   - Pros: captures non-linear patterns, learns the smile implicitly
   - Cons: black-box, slower inference than BS
   - Use case: predict market prices when IV surface is complex
   - Expected accuracy: ~88-92%

3. **LSTM Neural Network** (Deep Learning - Time Series)
   - Recurrent network trained on temporal sequences
   - Input: rolling window of price/vol/Greeks history
   - Pros: captures temporal dynamics, can forecast next-day prices
   - Cons: requires more data, longer training time, overfitting risk
   - Use case: dynamic pricing and risk forecasting
   - Expected accuracy: ~85-90%

**Evaluation Metrics:**
- Mean Absolute Percentage Error (MAPE) on held-out test set
- Greeks accuracy (delta, gamma matching)
- Value-at-Risk prediction accuracy

---

## SLIDE 5: Timeline & Milestones

**Visual Elements:**
- Gantt chart showing all project phases
- 5-week timeline (Apr 27 – June 1) with overlapping tasks
- NOW marker at week 1 (May 4), deadline marker at week 5 (June 1)

### Speaker Notes (1 minute):

**Project Timeline (5 weeks: Apr 27 – June 1):**

| Week (Project) | Dates | Phase | Status |
|----------------|-------|-------|--------|
| 0–1.5 | Apr 27 – May 10 | Data Collection & Cleaning | In Progress |
| 0.5–1.5 | May 4 – May 10 | Feature Engineering & EDA | In Progress |
| 1–2 | May 4 – May 11 | Black-Scholes Baseline | Planned |
| 1.5–3 | May 11 – May 25 | XGBoost Model Development | Planned |
| 2–3.5 | May 18 – May 25 | LSTM Model Development | Planned |
| 3–4 | May 25 – June 1 | Model Comparison & Analysis | Planned |
| 4–5 | May 25 – June 1 | Documentation, API, Deployment | Planned |

**Key Milestones:**
- **Week 0 (Apr 27):** Project start — data collection underway
- **Week 1 (May 4):** Proposal presentation — we are here
- **Week 3 (May 18):** XGBoost & LSTM models trained, comparison begins
- **Week 5 (June 1):** Full system complete — final slides due

**Risk Mitigation:**
- Started early on data collection to handle API limits
- Black-Scholes baseline provides fallback if ML models underperform
- Overlapping tasks allow for parallel progress

---

## SUMMARY: What Problem Are We Solving?

**Problem:** Options are complex financial instruments. Standard models (Black-Scholes) make unrealistic assumptions. ML models can be more accurate but are hard to interpret.

**Solution:** We develop a **comparative framework** that:
1. Understands market structure through EDA (moneyness, expiration, smile)
2. Builds multiple models with different trade-offs (interpretability vs. accuracy)
3. Enables risk managers to choose the right model for their use case
4. Provides a complete API and UI for real-world deployment

**Impact:** Better option pricing → better portfolio hedging → reduced market risk for institutions

---

## Backup Slides (if time permits)

### A. Data Preprocessing Details
- Handling missing IV values (use ATM estimates)
- Outlier detection on bid-ask spreads
- Feature scaling for ML (StandardScaler)
- Train/val/test split (70/15/15)

### B. Expected Model Performance
- BS: Fast baseline, limited accuracy
- XGBoost: Good balance of speed & accuracy
- LSTM: Best for time-series, slower inference

### C. Risk Metrics: Value-at-Risk (VaR)
- Portfolio VaR at 95% confidence (1-day horizon)
- Stress testing under market shocks
- Greeks-based hedging calculations

---

## Key Questions Likely to Be Asked

**Q: Why not just use Black-Scholes if it's faster?**
A: BS doesn't capture the volatility smile, which costs money in OTM hedging. ML models learn this implicitly from data.

**Q: How much data do you need for XGBoost/LSTM?**
A: XGBoost is happy with 1000+ records. LSTM benefits from temporal sequences, so we'll window the data (21-day lookback).

**Q: What happens if market regime changes?**
A: Good point. We're planning retraining pipelines. In crisis periods, realized vol spikes and the smile intensifies—our models should adapt.

**Q: Can you trade on these predictions?**
A: Not directly—we're building risk models, not trading systems. But accurate pricing is the foundation for profitable trading strategies.

**Q: How do Greeks help?**
A: Delta tells us directional exposure, gamma tells us convexity risk, vega tells us vol sensitivity. Portfolio managers use Greeks to hedge and manage risk.

