"""Streamlit frontend: Earnings Impact Predictor, Portfolio VaR, Options Pricer."""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# API base URL — overridden via Streamlit secrets when deployed
try:
    API = st.secrets["API_URL"]
except Exception:
    API = "http://localhost:8000"

st.set_page_config(page_title="Portfolio Risk Stress-Tester", layout="wide", page_icon="📈")
st.title("📈 Portfolio Risk Stress-Tester")

tab1, tab2, tab3 = st.tabs([
    "🔔 Earnings Impact Predictor",
    "📊 Portfolio VaR & Stress Test",
    "🧮 Black-Scholes Pricer",
])


# ===========================================================================
# TAB 1 — EARNINGS IMPACT PREDICTOR
# ===========================================================================
with tab1:
    st.header("Earnings Impact Predictor")
    st.markdown(
        "Enter a ticker to see its upcoming earnings, then use the slider to simulate "
        "how a beat or miss might move the option price."
    )

    col_input, col_info = st.columns([1, 2])

    with col_input:
        ticker = st.text_input("Ticker symbol", value="AAPL").upper().strip()
        load_btn = st.button("Load Earnings Data", type="primary")

    # Session state to persist data across slider interactions
    if "earnings_info" not in st.session_state:
        st.session_state.earnings_info = None
    if "earnings_history" not in st.session_state:
        st.session_state.earnings_history = []
    if "options_chain" not in st.session_state:
        st.session_state.options_chain = None
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = None

    if load_btn or (ticker != st.session_state.current_ticker and ticker):
        st.session_state.current_ticker = ticker
        with st.spinner(f"Fetching data for {ticker}…"):
            try:
                r = requests.get(f"{API}/earnings/{ticker}", timeout=30)
                st.session_state.earnings_info = r.json() if r.ok else None
            except Exception as e:
                st.error(f"Could not reach API: {e}")
                st.session_state.earnings_info = None

            try:
                r = requests.get(f"{API}/earnings/history/{ticker}", timeout=30)
                st.session_state.earnings_history = r.json() if r.ok else []
            except Exception:
                st.session_state.earnings_history = []

            try:
                r = requests.get(f"{API}/options/chain/{ticker}", timeout=30)
                st.session_state.options_chain = r.json() if r.ok else None
            except Exception:
                st.session_state.options_chain = None

    info = st.session_state.earnings_info
    history = st.session_state.earnings_history
    chain = st.session_state.options_chain

    if info:
        with col_info:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Next Earnings", info.get("upcoming_date") or "Unknown")
            eps_est = info.get("eps_estimate")
            m2.metric("EPS Estimate", f"${eps_est:.2f}" if eps_est else "N/A")
            avg_move = info.get("historical_avg_move_pct")
            m3.metric("Avg Post-Earnings Move", f"{avg_move*100:.1f}%" if avg_move else "N/A")
            n = info.get("n_events", 0)
            m4.metric("Historical Events", str(n))

        st.divider()

        # --- Historical earnings chart ---
        if history:
            hist_df = pd.DataFrame(history)
            hist_df["eps_surprise_pct"] = hist_df["eps_surprise_pct"] * 100
            hist_df["stock_move_pct"] = hist_df["stock_move_pct"] * 100
            hist_df["color"] = hist_df["eps_surprise_pct"].apply(
                lambda x: "Beat" if x >= 0 else "Miss"
            )
            fig_hist = px.bar(
                hist_df,
                x="date",
                y="stock_move_pct",
                color="color",
                color_discrete_map={"Beat": "#2ecc71", "Miss": "#e74c3c"},
                labels={"stock_move_pct": "Next-Day Stock Move (%)", "date": "Earnings Date", "color": ""},
                title=f"{ticker} — Historical Post-Earnings Stock Moves",
                hover_data={"eps_surprise_pct": ":.1f"},
            )
            fig_hist.update_layout(height=280, margin=dict(t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # --- Options chain + predictor ---
        st.subheader("Predict Option Price Change")

        col_opt, col_pred = st.columns([1, 1])

        with col_opt:
            # Build strike list from chain, or let user type manually
            if chain and (chain.get("calls") or chain.get("puts")):
                spot = chain.get("spot")
                all_strikes = sorted(set(
                    [r["strike"] for r in chain.get("calls", [])] +
                    [r["strike"] for r in chain.get("puts", [])]
                ))
                expiry = chain.get("expiry", "")
                st.caption(f"Live chain — expiry: **{expiry}** | Spot: **${spot:.2f}**" if spot else "")
                strike = st.selectbox("Strike price", all_strikes,
                                      index=len(all_strikes)//2 if all_strikes else 0)
                dte = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)
            else:
                spot = None
                st.info("Live chain unavailable. Enter parameters manually.")
                strike = st.number_input("Strike price", min_value=1.0, value=200.0, step=1.0)
                dte = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)

            flag = st.radio("Option type", ["call", "put"], horizontal=True)

        with col_pred:
            eps_surprise = st.slider(
                "EPS Surprise (%)",
                min_value=-30, max_value=30, value=0, step=1,
                help="Negative = miss vs estimate, Positive = beat vs estimate",
                format="%d%%",
            )

            label = "🟢 Beat" if eps_surprise > 0 else ("🔴 Miss" if eps_surprise < 0 else "Neutral")
            st.caption(f"Scenario: **{label}** by {abs(eps_surprise)}%")

            if st.button("Predict Impact", type="primary"):
                with st.spinner("Running model…"):
                    try:
                        resp = requests.post(
                            f"{API}/earnings/impact",
                            json={
                                "ticker": ticker,
                                "strike": float(strike),
                                "dte": int(dte),
                                "eps_surprise_pct": eps_surprise / 100,
                                "flag": flag,
                            },
                            timeout=30,
                        )
                        if resp.ok:
                            r = resp.json()
                            st.success("Prediction complete")

                            delta_sign = "▲" if r["predicted_option_change_pct"] >= 0 else "▼"
                            color = "normal" if r["predicted_option_change_pct"] >= 0 else "inverse"

                            pm1, pm2, pm3 = st.columns(3)
                            pm1.metric(
                                "Predicted Stock Move",
                                f"{r['predicted_stock_move_pct']*100:+.1f}%",
                            )
                            pm2.metric(
                                f"{flag.capitalize()} Price Before",
                                f"${r['current_price']:.2f}",
                            )
                            pm3.metric(
                                f"{flag.capitalize()} Price After",
                                f"${r['predicted_price']:.2f}",
                                delta=f"{r['predicted_option_change_pct']*100:+.1f}%",
                            )

                            with st.expander("Model details"):
                                st.json({
                                    "spot": r.get("spot"),
                                    "implied_vol": r.get("iv"),
                                    "delta": r.get("delta"),
                                    "gamma": r.get("gamma"),
                                    "method": "XGBoost → delta-gamma approximation",
                                })
                        else:
                            st.error(f"API error {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        # --- Live options chain table ---
        if chain and chain.get("calls"):
            with st.expander("Live options chain"):
                calls_df = pd.DataFrame(chain["calls"])
                puts_df = pd.DataFrame(chain["puts"])
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Calls")
                    st.dataframe(calls_df, use_container_width=True, hide_index=True)
                with c2:
                    st.caption("Puts")
                    st.dataframe(puts_df, use_container_width=True, hide_index=True)

    else:
        st.info("Enter a ticker and click **Load Earnings Data** to begin.")


# ===========================================================================
# TAB 2 — PORTFOLIO VAR & STRESS TEST
# ===========================================================================
with tab2:
    st.header("Portfolio VaR & Stress Test")

    st.sidebar.header("VaR Settings")
    confidence = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    horizon = st.sidebar.selectbox("Horizon (days)", [1, 5, 10, 21], index=0)
    shock_pct = st.sidebar.slider("Stress Shock (%)", -30, -1, -10) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Portfolio CSV format:**\n```\nticker,weight,sigma_daily\nSPY,0.4,0.008\nAAPL,0.3,0.015\nTSLA,0.3,0.025\n```"
    )

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Portfolio")
        st.dataframe(df, use_container_width=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Parametric VaR")
            portfolio_value = st.number_input("Portfolio Value ($)", value=100_000, step=1_000)
            if st.button("Calculate VaR"):
                payload = {
                    "positions": df[["ticker", "weight", "sigma_daily"]].to_dict("records"),
                    "portfolio_value": portfolio_value,
                    "confidence": confidence,
                    "horizon_days": horizon,
                }
                try:
                    resp = requests.post(f"{API}/var/parametric", json=payload, timeout=10)
                    result = resp.json()
                    st.metric("VaR ($)", f"${result['var']:,.2f}")
                    st.metric("Portfolio Daily σ", f"{result['portfolio_sigma']*100:.3f}%")
                except Exception as e:
                    st.error(f"API error: {e}")

        with col2:
            st.subheader("Stress Test")
            if st.button("Run Stress Scenario"):
                payload = {
                    "positions": df[["ticker", "weight", "sigma_daily"]].to_dict("records"),
                    "portfolio_value": portfolio_value,
                    "confidence": confidence,
                    "horizon_days": horizon,
                }
                try:
                    resp = requests.post(
                        f"{API}/stress/scenario", json=payload,
                        params={"shock_pct": shock_pct}, timeout=10,
                    )
                    r = resp.json()
                    st.metric("Base VaR ($)", f"${r['base_var']:,.2f}")
                    st.metric("Stressed VaR ($)", f"${r['stressed_var']:,.2f}",
                              delta=f"+{r['var_increase_pct']}%")
                except Exception as e:
                    st.error(f"API error: {e}")

        with col3:
            st.subheader("Portfolio Weights")
            fig = px.pie(df, names="ticker", values="weight", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily Volatility by Position")
        fig2 = px.bar(df, x="ticker", y="sigma_daily", color="ticker",
                      labels={"sigma_daily": "Daily σ"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload a portfolio CSV to get started.")


# ===========================================================================
# TAB 3 — BLACK-SCHOLES PRICER
# ===========================================================================
with tab3:
    st.header("Black-Scholes Option Pricer")
    st.markdown("Compute analytical price and Greeks for a single option.")

    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot (S)", value=100.0, min_value=0.01)
    K = c2.number_input("Strike (K)", value=100.0, min_value=0.01)
    T = c3.number_input("Time to Expiry (years)", value=0.25, min_value=0.001, step=0.05)
    sigma = c4.number_input("Implied Vol (σ)", value=0.20, min_value=0.001, step=0.01)
    flag = c5.selectbox("Type", ["call", "put"])

    if st.button("Price Option", type="primary"):
        try:
            resp = requests.post(
                f"{API}/options/price",
                json={"S": S, "K": K, "T": T, "r": 0.05, "sigma": sigma, "flag": flag},
                timeout=10,
            )
            if resp.ok:
                r = resp.json()
                labels = {"price": "Price", "delta": "Delta", "gamma": "Gamma",
                          "vega": "Vega", "theta": "Theta (daily)"}
                cols = st.columns(len(r))
                for col, (key, val) in zip(cols, r.items()):
                    col.metric(labels.get(key, key.capitalize()), f"{val:.4f}")
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

    st.divider()
    st.subheader("Greeks vs. Spot Price")
    st.markdown("Visualise how delta and gamma change across a range of spot prices.")

    spot_range = st.slider("Spot range (% of strike)", 70, 130, (80, 120))
    spots = [K * x / 100 for x in range(spot_range[0], spot_range[1] + 1)]

    try:
        delta_vals, gamma_vals = [], []
        for s in spots:
            resp = requests.post(
                f"{API}/options/price",
                json={"S": s, "K": K, "T": T, "r": 0.05, "sigma": sigma, "flag": flag},
                timeout=5,
            )
            if resp.ok:
                r = resp.json()
                delta_vals.append(r.get("delta", 0))
                gamma_vals.append(r.get("gamma", 0))

        if delta_vals:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spots, y=delta_vals, name="Delta", line=dict(color="#3498db", width=2)))
            fig.add_trace(go.Scatter(x=spots, y=gamma_vals, name="Gamma", line=dict(color="#e74c3c", width=2),
                                     yaxis="y2"))
            fig.add_vline(x=K, line_dash="dash", line_color="gray", annotation_text="Strike")
            fig.update_layout(
                xaxis_title="Spot Price",
                yaxis_title="Delta",
                yaxis2=dict(title="Gamma", overlaying="y", side="right"),
                height=350,
                margin=dict(t=20, b=20),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Could not render Greeks chart — check API connection.")
