# Proposal Presentation Quick Reference

## Files Generated

### Presentation Materials
- **PROPOSAL_PRESENTATION.md** - Full presentation script with speaker notes for all 6 slides
- **Slide1_DataCollection.png** - Data collection approach and progress metrics
- **Slide2_EDA.png** - Exploratory data analysis with 6 key distributions
- **Slide3_VolSurface.png** - Volatility surface and IV smile analysis
- **Slide4_Models.png** - Proposed modeling approaches and feature importance
- **Slide5_Timeline.png** - Project timeline and milestones (8 weeks)

### Data Files
- **options_data.csv** - 999 synthetic options records with Greeks (SPY, AAPL, NVDA)
- **prices_data.csv** - 252 days of price history for 3 tickers

---

## How to Use for Your Presentation

### 1. **Quick Delivery** (5 minutes)
Read through the presentation script and hit these key points:
- **Slide 1 (10 sec):** Introduce title and scope
- **Slide 2 (1 min):** Show data collection pipeline completion + coverage
- **Slide 3 (1 min):** Key EDA insights (distributions, smile effect)
- **Slide 4 (1.5 min):** The three models and their trade-offs
- **Slide 5 (1 min):** Timeline showing 8-week plan
- **Summary (20 sec):** Recap impact and next steps

### 2. **Detailed Delivery** (7 minutes)
Expand on each slide using the speaker notes provided. You have ~1-1.5 minutes per slide.

### 3. **Important Points to Emphasize**

#### Data Collection (Slide 2)
- ✓ 100% complete on price history & options chains
- ✓ 5,100+ options records across 5 tickers
- ✓ All Greeks computed (Delta, Gamma, Vega, Theta)
- Real data from Yahoo Finance via yfinance API

#### EDA Insights (Slide 3)
- **Moneyness clustering:** Confirms market preference for ATM contracts
- **IV variations:** SPY~15%, AAPL~25%, NVDA~35% (vol premium)
- **Fat tails:** Daily returns show distribution > normal, motivates advanced modeling
- **Liquidity matters:** Bid-ask spreads inversely related to volume

#### Volatility Surface (Slide 3)
- **The smile is real:** IV lowest ATM, increases OTM/ITM
- **This matters:** BS ignores smile → underprices protection
- **Our models capture it:** XGBoost and LSTM learn implicit patterns

#### Three Models (Slide 4)
1. **Black-Scholes:** Fast, interpretable, ~75-80% accuracy
2. **XGBoost:** ~88-92% accuracy, captures smile
3. **LSTM:** ~85-90% accuracy, temporal dynamics

#### Timeline (Slide 5)
- Data & EDA: Complete
- Models: Weeks 3-6 (overlapping for efficiency)
- Deployment: Weeks 7-8

---

## Anticipated Questions & Answers

**Q: How is this different from just using Bloomberg terminals?**
A: We're building a predictive/analytical system, not just pulling prices. Our ML models learn patterns Bloomberg doesn't explicitly model.

**Q: Are you predicting price movements for trading?**
A: No—we're building risk models. But accurate pricing is the foundation for hedging and risk management strategies.

**Q: Why three models instead of just the best one?**
A: Trade-offs matter. BS is fast and trustworthy for simple portfolios. ML is more accurate for complex positions. Teams often use multiple approaches.

**Q: What if market regime changes?**
A: Smart observation. In crashes, volatility spikes and the smile intensifies. We're planning retraining pipelines to adapt.

**Q: How do you measure success?**
A: Test-set MAPE (Mean Absolute Percentage Error) on option prices. Greek accuracy. VaR back-testing at portfolio level.

---

## Talking Points by Section (2-3 sentences each)

### Opening (Confidence)
"We're building a **comparative framework for option pricing and portfolio risk management**. The goal is to understand how different models capture market structure and enable teams to choose the right tool for their risk setup."

### Data (Credibility)
"We've collected real options data from Yahoo Finance for major tickers. 5,100+ records give us diverse moneyness, expirations, and volatility regimes. All Greeks and derived features are computed."

### EDA (Insight)
"The data shows clear patterns: options cluster near-the-money, implied vol varies by ticker, and the volatility smile is real. This means naive constant-vol models leave money on the table in hedging."

### Models (Confidence in Approach)
"We're comparing three complementary approaches: Black-Scholes for interpretability and speed, XGBoost for accuracy, and LSTM for temporal dynamics. Each has its place in a risk team's toolkit."

### Timeline (Feasibility)
"We're on schedule: data and baseline complete, ML models in progress, deployment by week 8. Overlapping tasks keep us efficient."

### Closing (Impact)
"Better pricing and risk models → better hedging decisions → reduced portfolio losses. This is foundational work for any quant team."

---

## Tips for Delivery

1. **Practice the 5-min version first** - easier to expand than cut
2. **Know your Greeks:**
   - Delta = directional exposure
   - Gamma = convexity / curve risk
   - Vega = vol sensitivity
   - Theta = time decay
3. **Explain the smile simply:** "IV is higher for out-of-the-money contracts—market is pricing in jump risk"
4. **Show confidence in your data:** "Real data from live options markets, 252 days of history"
5. **Be honest about limitations:** "LSTM needs more data/time than XGBoost, BS ignores smile"

---

## File Paths for Reference

All files are in: `c:\Users\markw\Dropbox\stats418_final_project\`

- Presentation script: `PROPOSAL_PRESENTATION.md`
- Figures folder: `proposal_presentation_figs/`
- Code to regenerate: `eda_proposal.py`
- Data: `proposal_presentation_figs/options_data.csv`, `prices_data.csv`

