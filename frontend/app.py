"""Streamlit frontend: Earnings Impact Predictor, Portfolio VaR, Options Pricer, Comparison."""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    API = st.secrets["API_URL"]
except Exception:
    API = "http://localhost:8000"

st.set_page_config(page_title="Portfolio Risk Stress-Tester", layout="wide")
st.title("Portfolio Risk Stress-Tester")

tab1, tab2, tab3, tab4 = st.tabs([
    "Earnings Impact Predictor",
    "Portfolio VaR & Stress Test",
    "Black-Scholes Pricer",
    "Multi-Ticker Comparison",
])


# ===========================================================================
# TAB 1 — EARNINGS IMPACT PREDICTOR
# ===========================================================================
with tab1:
    st.header("Earnings Impact Predictor")

    # Session state init
    for k, v in [("options_chain", None), ("current_ticker", None),
                 ("ph_key", ""), ("ph_data", None),
                 ("earnings_info", None), ("earnings_history", []),
                 ("earnings_loaded_for", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # --- Ticker input with search button ---
    _col_input, _col_btn = st.columns([8, 1])
    with _col_input:
        ticker = st.text_input(
            "Ticker symbol", value="AAPL",
            help="Press Enter or click the search button to load."
        ).upper().strip()
    with _col_btn:
        st.markdown('<div style="margin-top:1.75rem"></div>', unsafe_allow_html=True)
        _search_btn = st.button("→", type="primary", use_container_width=True, help="Search")

    _should_load = _search_btn or (ticker and ticker != st.session_state.current_ticker)
    if ticker and _should_load:
        st.session_state.current_ticker = ticker
        st.session_state.ph_key = ""
        if ticker != st.session_state.earnings_loaded_for:
            st.session_state.earnings_info = None
            st.session_state.earnings_history = []
        with st.spinner(f"Loading {ticker} options chain…"):
            try:
                r = requests.get(f"{API}/options/chain/{ticker}", timeout=30)
                chain_data = r.json() if r.ok else None
                st.session_state.options_chain = chain_data
                if chain_data:
                    spot_val = chain_data.get("spot")
                    if spot_val:
                        st.session_state["bs_S"] = float(spot_val)
                        calls = chain_data.get("calls", [])
                        if calls:
                            atm = min(calls, key=lambda c: abs(c["strike"] - spot_val))
                            st.session_state["bs_K"] = float(atm["strike"])
            except Exception as e:
                st.error(f"API unreachable: {e}")
                st.session_state.options_chain = None

    chain = st.session_state.options_chain

    # -----------------------------------------------------------------------
    # SECTION 1 — OPTION SELECTOR + PRICE HISTORY (auto-loaded)
    # -----------------------------------------------------------------------
    if chain:
        spot = chain.get("spot")
        expiry = chain.get("expiry", "")
        all_strikes = sorted(set(
            [c["strike"] for c in chain.get("calls", [])] +
            [p["strike"] for p in chain.get("puts", [])]
        ))

        sel1, sel2, sel3, info_col = st.columns([2, 1, 1, 2])
        with sel1:
            strike = st.selectbox(
                "Strike", all_strikes,
                index=len(all_strikes) // 2 if all_strikes else 0,
            )
        with sel2:
            dte = st.number_input("DTE", min_value=1, max_value=365, value=30)
        with sel3:
            flag = st.radio("Type", ["call", "put"], horizontal=True)
        with info_col:
            if spot:
                st.metric("Spot Price", f"${spot:.2f}",
                          help=f"Nearest expiry: {expiry}")
    else:
        if ticker:
            st.info("Live chain unavailable — enter parameters manually.")
        spot = None
        man1, man2, man3 = st.columns(3)
        strike = man1.number_input("Strike", min_value=1.0, value=200.0, step=1.0)
        dte = man2.number_input("DTE", min_value=1, max_value=365, value=30)
        flag = man3.radio("Type", ["call", "put"], horizontal=True)

    # Theoretical price history chart — cached, re-fetches when params change
    period_choice = st.radio("Period", ["1W", "1M", "3M", "6M"], index=1,
                             horizontal=True, key="t1_period")
    ph_key = f"{ticker}_{strike}_{dte}_{flag}_{period_choice}"
    if ticker and st.session_state.ph_key != ph_key:
        with st.spinner("Loading price history…"):
            try:
                r2 = requests.get(
                    f"{API}/options/price_history/{ticker}",
                    params={"strike": strike, "dte": dte, "flag": flag, "period": period_choice},
                    timeout=30,
                )
                st.session_state.ph_data = r2.json() if r2.ok else None
                st.session_state.ph_key = ph_key
            except Exception:
                st.session_state.ph_data = None

    ph = st.session_state.ph_data
    if ph and ph.get("history"):
        ph_df = pd.DataFrame(ph["history"])
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=ph_df["date"], y=ph_df["theoretical_price"],
            name=f"{flag.capitalize()} K={int(strike)}",
            line=dict(color="#3498db", width=2),
        ))
        fig_ph.add_trace(go.Scatter(
            x=ph_df["date"], y=ph_df["spot"],
            name="Spot Price",
            line=dict(color="#95a5a6", width=1, dash="dot"),
            yaxis="y2",
        ))
        fig_ph.update_layout(
            xaxis_title="Date",
            yaxis_title="Option Price ($)",
            yaxis2=dict(title="Spot Price ($)", overlaying="y", side="right"),
            height=280, margin=dict(t=20, b=20),
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig_ph, use_container_width=True)
    elif ticker:
        st.info("Option price history unavailable for this ticker.")

    if chain and chain.get("calls"):
        with st.expander("Live options chain"):
            lc1, lc2 = st.columns(2)
            lc1.caption("Calls")
            lc1.dataframe(pd.DataFrame(chain["calls"]), use_container_width=True, hide_index=True)
            lc2.caption("Puts")
            lc2.dataframe(pd.DataFrame(chain["puts"]), use_container_width=True, hide_index=True)

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 2 — EARNINGS HISTORY (separate button)
    # -----------------------------------------------------------------------
    st.subheader("Earnings History")
    earnings_btn = st.button("Load Earnings Data", type="secondary")
    if earnings_btn and ticker:
        st.session_state.earnings_loaded_for = ticker
        with st.spinner(f"Fetching earnings for {ticker}…"):
            try:
                r = requests.get(f"{API}/earnings/{ticker}", timeout=30)
                st.session_state.earnings_info = r.json() if r.ok else None
            except Exception as e:
                st.error(f"API unreachable: {e}")
                st.session_state.earnings_info = None
            try:
                r = requests.get(f"{API}/earnings/history/{ticker}", timeout=30)
                st.session_state.earnings_history = r.json() if r.ok else []
            except Exception:
                st.session_state.earnings_history = []

    info = st.session_state.earnings_info
    history = st.session_state.earnings_history

    if info:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Next Earnings", info.get("upcoming_date") or "Not scheduled")
        eps_est = info.get("eps_estimate")
        m2.metric("EPS Estimate", f"${eps_est:.2f}" if eps_est else "N/A")
        avg_move = info.get("historical_avg_move_pct")
        m3.metric("Avg Post-Earnings Move", f"{avg_move*100:.1f}%" if avg_move else "N/A")
        m4.metric("Historical Events", str(info.get("n_events", 0)))

        if history:
            hdf = pd.DataFrame(history)
            hdf["eps_pct"] = hdf["eps_surprise_pct"] * 100
            hdf["move_pct"] = hdf["stock_move_pct"] * 100
            hdf["color"] = hdf["eps_pct"].apply(lambda x: "Beat" if x >= 0 else "Miss")
            fig_hist = px.bar(
                hdf, x="date", y="move_pct", color="color",
                color_discrete_map={"Beat": "#2ecc71", "Miss": "#e74c3c"},
                labels={"move_pct": "Next-Day Stock Move (%)", "date": "Date", "color": ""},
                title=f"{st.session_state.earnings_loaded_for} — Historical Post-Earnings Stock Moves",
                hover_data={"eps_pct": ":.1f"},
            )
            fig_hist.update_layout(height=260, margin=dict(t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No earnings history available for this ticker (ETFs and some stocks have no earnings data).")
    else:
        st.caption("Click **Load Earnings Data** to see upcoming earnings date, EPS estimate, and historical post-earnings moves.")

    st.divider()

    # -----------------------------------------------------------------------
    # SECTION 3 — PREDICT IMPACT
    # -----------------------------------------------------------------------
    st.subheader("Predict Option Price Change")
    pred_col, res_col = st.columns([1, 1])
    with pred_col:
        eps_surprise = st.slider(
            "EPS Surprise (%)", min_value=-30, max_value=30, value=0, step=1,
            format="%d%%",
            help="Negative = miss vs estimate, Positive = beat vs estimate",
        )
        label = "Beat" if eps_surprise > 0 else ("Miss" if eps_surprise < 0 else "Neutral")
        st.caption(f"Scenario: **{label}** by {abs(eps_surprise)}%")
        predict_btn = st.button("Predict Impact", type="primary")

    with res_col:
        if predict_btn and ticker:
            with st.spinner("Running model…"):
                try:
                    resp = requests.post(
                        f"{API}/earnings/impact",
                        json={"ticker": ticker, "strike": float(strike), "dte": int(dte),
                              "eps_surprise_pct": eps_surprise / 100, "flag": flag},
                        timeout=30,
                    )
                    if resp.ok:
                        rv = resp.json()
                        st.success("Prediction complete")
                        pm1, pm2, pm3 = st.columns(3)
                        pm1.metric("Predicted Stock Move",
                                   f"{rv['predicted_stock_move_pct']*100:+.1f}%")
                        pm2.metric(f"{flag.capitalize()} Price Before",
                                   f"${rv['current_price']:.2f}")
                        pm3.metric(f"{flag.capitalize()} Price After",
                                   f"${rv['predicted_price']:.2f}",
                                   delta=f"{rv['predicted_option_change_pct']*100:+.1f}%")
                        with st.expander("Model details"):
                            st.json({
                                "spot": rv.get("spot"),
                                "implied_vol": rv.get("iv"),
                                "delta": rv.get("delta"),
                                "gamma": rv.get("gamma"),
                                "method": "XGBoost → delta-gamma approximation",
                            })
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ===========================================================================
# TAB 2 — PORTFOLIO VAR & STRESS TEST
# ===========================================================================
with tab2:
    st.header("Portfolio VaR & Stress Test")


    # Settings inline (no sidebar clutter)
    s1, s2, s3 = st.columns(3)
    confidence = s1.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01, key="var_conf")
    horizon = s2.selectbox("Horizon (days)", [1, 5, 10, 21], index=0, key="var_horiz")
    shock_pct = s3.slider("Stress Shock (%)", -30, -1, -10, key="var_shock") / 100

    st.caption(
        "**Portfolio CSV format:** `ticker,weight,sigma_daily`  —  "
        "example: `SPY,0.4,0.008 | AAPL,0.3,0.015 | TSLA,0.3,0.025`"
    )

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])

    if uploaded:
        df_port = pd.read_csv(uploaded)
        st.subheader("Portfolio")
        st.dataframe(df_port, use_container_width=True)

        vc1, vc2, vc3 = st.columns(3)

        with vc1:
            st.subheader("Parametric VaR")
            portfolio_value = st.number_input("Portfolio Value ($)", value=100_000, step=1_000)
            if st.button("Calculate VaR"):
                payload = {
                    "positions": df_port[["ticker", "weight", "sigma_daily"]].to_dict("records"),
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

        with vc2:
            st.subheader("Stress Test")
            if st.button("Run Stress Scenario"):
                payload = {
                    "positions": df_port[["ticker", "weight", "sigma_daily"]].to_dict("records"),
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

        with vc3:
            st.subheader("Portfolio Weights")
            fig_pie = px.pie(df_port, names="ticker", values="weight", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Daily Volatility by Position")
        fig_bar = px.bar(df_port, x="ticker", y="sigma_daily", color="ticker",
                         labels={"sigma_daily": "Daily σ"})
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info(
            "Upload a portfolio CSV to get started.\n\n"
            "**Format:** `ticker,weight,sigma_daily`\n```\n"
            "SPY,0.4,0.008\nAAPL,0.3,0.015\nTSLA,0.3,0.025\n```"
        )


# ===========================================================================
# TAB 3 — BLACK-SCHOLES PRICER
# ===========================================================================
with tab3:
    st.header("Black-Scholes Option Pricer")

    st.markdown(
        "Compute price, Greeks, probability of profit, theta decay, and payoff diagram "
        "for any option. Click **Price Option** to update all charts. "
        "Spot and strike auto-populate when you load a ticker in the Earnings tab."
    )

    bc1, bc2, bc3, bc4, bc5 = st.columns(5)
    S_bs = bc1.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, key="bs_S")
    K_bs = bc2.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, key="bs_K")
    T_bs = bc3.number_input("Time to Expiry (years)", value=0.25, min_value=0.001,
                             step=0.05, key="bs_T")
    sigma_bs = bc4.number_input("Implied Vol (σ)", value=0.20, min_value=0.001,
                                 step=0.01, key="bs_sigma")
    flag_bs = bc5.selectbox("Type", ["call", "put"], key="bs_flag")

    if st.button("Price Option", type="primary"):
        with st.spinner("Computing…"):
            try:
                # Single price + greeks call
                r_price = requests.post(
                    f"{API}/options/price",
                    json={"S": S_bs, "K": K_bs, "T": T_bs, "r": 0.05,
                          "sigma": sigma_bs, "flag": flag_bs},
                    timeout=10,
                )
                # Bulk Greeks surface (70–130% of K)
                r_surf = requests.post(
                    f"{API}/options/greeks_surface",
                    json={"K": K_bs, "T": T_bs, "r": 0.05, "sigma": sigma_bs,
                          "flag": flag_bs, "S_min_pct": 70, "S_max_pct": 130},
                    timeout=10,
                )
                # Theta decay (days to 0)
                T_max_days = max(int(T_bs * 365), 2)
                r_decay = requests.post(
                    f"{API}/options/theta_decay",
                    json={"S": S_bs, "K": K_bs, "T_max_days": T_max_days,
                          "r": 0.05, "sigma": sigma_bs, "flag": flag_bs},
                    timeout=10,
                )
                if r_price.ok:
                    st.session_state["bs_result"] = r_price.json()
                    st.session_state["bs_params"] = {
                        "S": S_bs, "K": K_bs, "T": T_bs, "sigma": sigma_bs, "flag": flag_bs
                    }
                if r_surf.ok:
                    st.session_state["bs_surface"] = r_surf.json()
                if r_decay.ok:
                    st.session_state["bs_decay"] = r_decay.json()
            except Exception as e:
                st.error(f"Request failed: {e}")

    for key in ["bs_result", "bs_params", "bs_surface", "bs_decay"]:
        if key not in st.session_state:
            st.session_state[key] = None

    result = st.session_state["bs_result"]
    params = st.session_state["bs_params"]
    surface = st.session_state["bs_surface"]
    decay = st.session_state["bs_decay"]

    if result and params:
        price = result["price"]
        flag_p = params["flag"]
        K_p = params["K"]
        S_p = params["S"]
        T_p = params["T"]

        # Pricing row
        lbl = {"price": "Option Price", "delta": "Delta", "gamma": "Gamma",
               "vega": "Vega (per 1%)", "theta": "Theta (daily)", "pop": "P(Profit / ITM)"}
        disp_keys = ["price", "delta", "gamma", "vega", "theta", "pop"]
        pcols = st.columns(len(disp_keys))
        for col, key in zip(pcols, disp_keys):
            val = result.get(key, 0)
            col.metric(lbl[key], f"{val:.4f}")

        # Break-even
        breakeven = K_p + price if flag_p == "call" else K_p - price
        bc_col, pop_col = st.columns(2)
        bc_col.metric("Break-even at Expiry", f"${breakeven:.2f}",
                      help="Spot price at which the option exactly covers its premium at expiry")
        pop_col.metric("Probability of Profit", f"{result.get('pop', 0)*100:.1f}%",
                       help="N(d₂) for call / N(−d₂) for put — probability of expiring ITM")

        st.divider()

        # Greeks vs. Spot + Theta (single bulk API call, already done)
        if surface:
            st.subheader("Greeks vs. Spot Price")
            surf_df = pd.DataFrame(surface)
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(
                x=surf_df["spot"], y=surf_df["delta"], name="Delta",
                line=dict(color="#3498db", width=2),
            ))
            fig_g.add_trace(go.Scatter(
                x=surf_df["spot"], y=surf_df["gamma"], name="Gamma",
                line=dict(color="#e74c3c", width=2), yaxis="y2",
            ))
            fig_g.add_trace(go.Scatter(
                x=surf_df["spot"], y=surf_df["theta"], name="Theta (daily $)",
                line=dict(color="#f39c12", width=2, dash="dot"), yaxis="y2",
            ))
            fig_g.add_vline(x=K_p, line_dash="dash", line_color="gray")
            fig_g.add_vline(x=S_p, line_dash="dot", line_color="#2ecc71")
            fig_g.update_layout(
                xaxis_title="Spot Price",
                yaxis=dict(title="Delta"),
                yaxis2=dict(title="Gamma / Theta", overlaying="y", side="right"),
                height=350, margin=dict(t=20, b=20),
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Strike (gray dashed): ${K_p:.0f}  |  Current spot (green dotted): ${S_p:.0f}")

        st.divider()

        # Theta Decay chart
        if decay:
            st.subheader("Theta Decay — Option Price vs. Days to Expiry")
            dec_df = pd.DataFrame(decay).sort_values("days")
            intrinsic = max(S_p - K_p, 0) if flag_p == "call" else max(K_p - S_p, 0)
            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(
                x=dec_df["days"], y=dec_df["price"],
                name="Option Price", fill="tozeroy",
                line=dict(color="#3498db", width=2),
                fillcolor="rgba(52,152,219,0.12)",
            ))
            fig_d.add_hline(y=intrinsic, line_dash="dash", line_color="#e74c3c",
                            annotation_text=f"Intrinsic ${intrinsic:.2f}")
            fig_d.update_layout(
                xaxis_title="Days to Expiry",
                yaxis_title="Option Price ($)",
                xaxis=dict(autorange="reversed"),
                height=280, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_d, use_container_width=True)

        st.divider()

        # Payoff Diagram at Expiration (pure math, no API call)
        st.subheader("Payoff at Expiration")
        spot_range = [K_p * x / 100 for x in range(60, 141)]
        if flag_p == "call":
            pnl = [max(s - K_p, 0) - price for s in spot_range]
        else:
            pnl = [max(K_p - s, 0) - price for s in spot_range]

        fig_po = go.Figure()
        fig_po.add_trace(go.Scatter(
            x=spot_range, y=pnl, mode="lines",
            line=dict(color="#3498db", width=2.5), name="P&L",
        ))
        # Green fill above zero, red below
        fig_po.add_trace(go.Scatter(
            x=spot_range, y=[max(v, 0) for v in pnl],
            fill="tozeroy", fillcolor="rgba(46,204,113,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_po.add_trace(go.Scatter(
            x=spot_range, y=[min(v, 0) for v in pnl],
            fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_po.add_hline(y=0, line_color="gray", line_width=1)
        fig_po.add_vline(x=K_p, line_dash="dash", line_color="gray")
        fig_po.add_vline(x=breakeven, line_dash="dot", line_color="#2ecc71")
        fig_po.add_vline(x=S_p, line_dash="dot", line_color="#f39c12")
        fig_po.update_layout(
            xaxis_title="Spot Price at Expiry",
            yaxis_title="P&L ($)",
            height=300, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_po, use_container_width=True)
        st.caption(
            f"Strike (gray dashed): ${K_p:.0f}  |  "
            f"Break-even (green dotted): ${breakeven:.2f}  |  "
            f"Spot (orange dotted): ${S_p:.0f}"
        )

    else:
        st.info("Enter parameters above and click **Price Option** to see price, Greeks, theta decay, and payoff diagram.")


# ===========================================================================
# TAB 4 — MULTI-TICKER COMPARISON
# ===========================================================================
with tab4:
    st.header("Multi-Ticker Comparison")

    st.markdown(
        "Compare historical stock performance or theoretical option prices across up to 5 tickers. "
        "Enable **Normalize** to compare % change from the start of the period."
    )

    ct1, ct2, ct3 = st.columns([2, 1, 1])
    with ct1:
        comp_tickers_input = st.text_input("Tickers (comma-separated, max 5)",
                                            value="AAPL, NVDA, TSLA")
    with ct2:
        comp_period = st.radio("Period", ["1W", "1M", "3M", "6M", "1Y"],
                               index=2, horizontal=True, key="comp_period")
    with ct3:
        normalize = st.toggle("Normalize (% from start)", value=True)

    comp_mode = st.radio(
        "Compare",
        ["Stock prices", "Theoretical option prices"],
        horizontal=True,
    )

    comp_strike, comp_dte, comp_flag = 0.0, 30, "call"
    if comp_mode == "Theoretical option prices":
        op1, op2, op3 = st.columns(3)
        comp_strike = op1.number_input("Strike (0 = use live spot as ATM)",
                                        value=0.0, step=1.0)
        comp_dte = op2.number_input("DTE", min_value=1, max_value=365, value=30,
                                     key="comp_dte")
        comp_flag = op3.radio("Type", ["call", "put"], horizontal=True, key="comp_flag")

    COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    if st.button("Compare", type="primary"):
        tickers = [t.strip().upper() for t in comp_tickers_input.split(",") if t.strip()][:5]
        if not tickers:
            st.warning("Enter at least one ticker.")
        else:
            with st.spinner("Fetching comparison data…"):
                fig_comp = go.Figure()

                if comp_mode == "Stock prices":
                    try:
                        r_comp = requests.get(
                            f"{API}/compare/prices",
                            params={"tickers": ",".join(tickers), "period": comp_period},
                            timeout=30,
                        )
                        if r_comp.ok:
                            data = r_comp.json()
                            for i, (tk, records) in enumerate(data.items()):
                                df_t = pd.DataFrame(records)
                                y = df_t["price"]
                                if normalize and not df_t.empty:
                                    y = (y / y.iloc[0] - 1) * 100
                                fig_comp.add_trace(go.Scatter(
                                    x=df_t["date"], y=y, name=tk,
                                    line=dict(color=COLORS[i % len(COLORS)], width=2),
                                ))
                        else:
                            st.error(f"API error: {r_comp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

                else:  # Theoretical option prices
                    for i, tk in enumerate(tickers):
                        try:
                            # Resolve ATM strike if needed
                            if comp_strike <= 0:
                                chain_r = requests.get(f"{API}/options/chain/{tk}", timeout=15)
                                if chain_r.ok:
                                    spot_val = chain_r.json().get("spot")
                                    calls = chain_r.json().get("calls", [])
                                    if spot_val and calls:
                                        k_val = min(calls, key=lambda c: abs(c["strike"] - spot_val))["strike"]
                                    else:
                                        k_val = spot_val or 100.0
                                else:
                                    k_val = 100.0
                            else:
                                k_val = comp_strike

                            r_ph = requests.get(
                                f"{API}/options/price_history/{tk}",
                                params={"strike": k_val, "dte": comp_dte,
                                        "flag": comp_flag, "period": comp_period},
                                timeout=30,
                            )
                            if r_ph.ok:
                                ph_data = r_ph.json().get("history", [])
                                if ph_data:
                                    df_t = pd.DataFrame(ph_data)
                                    y = df_t["theoretical_price"]
                                    if normalize and not df_t.empty:
                                        y = (y / y.iloc[0] - 1) * 100
                                    fig_comp.add_trace(go.Scatter(
                                        x=df_t["date"], y=y,
                                        name=f"{tk} (K={int(k_val)})",
                                        line=dict(color=COLORS[i % len(COLORS)], width=2),
                                    ))
                        except Exception:
                            st.warning(f"Could not fetch data for {tk}.")

                fig_comp.update_layout(
                    xaxis_title="Date",
                    yaxis_title="% Change from Start" if normalize else (
                        "Price ($)" if comp_mode == "Stock prices" else "Theoretical Option Price ($)"
                    ),
                    height=450, margin=dict(t=20, b=20),
                    legend=dict(x=0.01, y=0.99),
                )
                if normalize:
                    fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)

                st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.info("Select tickers and click **Compare** to generate the chart.")
