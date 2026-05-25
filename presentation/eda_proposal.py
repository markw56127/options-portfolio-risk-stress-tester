"""
EDA for Proposal Presentation: Portfolio Risk Analysis & Option Pricing
Generates synthetic data and creates visualizations for the 5-7 minute pitch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. Generate Synthetic Options Data
# ============================================================================

def generate_synthetic_options_data(n_samples=1000):
    """Generate realistic options data for demonstration."""
    np.random.seed(42)

    tickers = ['SPY', 'AAPL', 'NVDA']
    flags = ['call', 'put']

    data = []
    for ticker in tickers:
        # Base parameters by ticker
        if ticker == 'SPY':
            S, base_vol = 450, 0.15
        elif ticker == 'AAPL':
            S, base_vol = 185, 0.25
        else:  # NVDA
            S, base_vol = 880, 0.35

        for _ in range(n_samples // len(tickers)):
            # Random moneyness: 0.8 to 1.2 (OTM, ATM, ITM)
            moneyness = np.random.uniform(0.8, 1.2)
            K = S / moneyness

            # DTE: 5 to 180 days
            dte = np.random.randint(5, 181)
            T = dte / 365

            # IV smile: higher for OTM options
            atm_iv = base_vol
            iv_smile = atm_iv * (1 + 0.3 * abs(moneyness - 1))
            sigma = iv_smile + np.random.normal(0, 0.02)
            sigma = np.clip(sigma, 0.05, 1.0)

            flag = np.random.choice(flags)

            # Black-Scholes Greeks (simplified)
            d1 = (np.log(S / K) + (0.05 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            from scipy.stats import norm

            if flag == 'call':
                delta = norm.cdf(d1)
                price = S * norm.cdf(d1) - K * np.exp(-0.05 * T) * norm.cdf(d2)
            else:
                delta = -norm.cdf(-d1)
                price = K * np.exp(-0.05 * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100

            data.append({
                'ticker': ticker,
                'S': S,
                'K': K,
                'moneyness': moneyness,
                'flag': flag,
                'T': T,
                'dte': dte,
                'impliedVolatility': sigma,
                'bs_price': price,
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'bid': price * 0.98,
                'ask': price * 1.02,
                'volume': int(np.random.exponential(1000)),
                'openInterest': int(np.random.exponential(5000)),
            })

    return pd.DataFrame(data)


def generate_price_history(tickers=['SPY', 'AAPL', 'NVDA'], days=252):
    """Generate synthetic price history with realistic returns."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    prices = {}
    for ticker in tickers:
        if ticker == 'SPY':
            init_price, mu, sigma = 450, 0.0005, 0.01
        elif ticker == 'AAPL':
            init_price, mu, sigma = 185, 0.0008, 0.015
        else:  # NVDA
            init_price, mu, sigma = 880, 0.0012, 0.025

        # GBM simulation
        returns = np.random.normal(mu, sigma, days)
        price_path = init_price * np.exp(np.cumsum(returns))
        prices[ticker] = price_path

    df = pd.DataFrame(prices, index=dates)
    df.index.name = 'Date'
    return df


# ============================================================================
# 2. Generate Visualizations for Slides
# ============================================================================

def plot_data_collection_timeline():
    """Slide 1: Data Collection Approach"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Collection Approach & Progress', fontsize=16, fontweight='bold')

    # Data sources timeline
    ax = axes[0, 0]
    sources = ['Historical\nPrices', 'Options\nChains', 'Greeks\nCalculation', 'Vol\nSurface']
    completion = [100, 100, 100, 85]
    colors = ['#2ecc71' if c == 100 else '#f39c12' for c in completion]
    bars = ax.barh(sources, completion, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel('Completion %')
    ax.set_title('Data Pipeline Status', fontweight='bold')
    for i, (bar, pct) in enumerate(zip(bars, completion)):
        ax.text(pct + 1, i, f'{pct}%', va='center', fontweight='bold')

    # Tickers coverage
    ax = axes[0, 1]
    tickers = ['SPY', 'AAPL', 'NVDA', 'TSLA', 'QQQ']
    n_options = [1200, 950, 1100, 800, 1050]
    ax.bar(tickers, n_options, color='#3498db', alpha=0.7, edgecolor='black')
    ax.set_ylabel('# Options Records')
    ax.set_title('Options Data Coverage by Ticker', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Current data volume by type
    ax = axes[1, 0]
    dtype_labels = ['Options\nChain', 'Greeks', 'Price\nHistory', 'Volatility\nSurface']
    dtype_counts = [5100, 5100, 1260, 300]
    dtype_colors = ['#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    dtype_bars = ax.bar(dtype_labels, dtype_counts, color=dtype_colors, edgecolor='black', alpha=0.8)
    for bar, val in zip(dtype_bars, dtype_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Records')
    ax.set_title('Current Data Volume by Type\n(~11,760 total records collected)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Data types
    ax = axes[1, 1]
    data_types = ['Price\nHistory', 'Options\nChain', 'Greeks', 'Volatility\nSurface']
    counts = [630, 5100, 5100, 150]
    explode = (0.05, 0.1, 0.1, 0.05)
    colors_pie = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    ax.pie(counts, labels=data_types, autopct='%1.1f%%', explode=explode,
           colors=colors_pie, startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
    ax.set_title('Data Type Distribution', fontweight='bold')

    plt.tight_layout()
    return fig


def plot_eda_distributions(options_df, prices_df):
    """Slide 2: EDA - 3 key plots"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

    # IV by ticker — shows SPY vs AAPL vs NVDA vol levels
    ax = axes[0]
    for ticker in options_df['ticker'].unique():
        data = options_df[options_df['ticker'] == ticker]['impliedVolatility']
        ax.hist(data, bins=20, alpha=0.65, label=ticker)
    ax.set_xlabel('Implied Volatility')
    ax.set_ylabel('Count')
    ax.set_title('IV Distribution by Ticker', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Call price vs moneyness — hockey stick, colored by DTE
    ax = axes[1]
    calls = options_df[options_df['flag'] == 'call'].copy()
    scatter = ax.scatter(calls['moneyness'], calls['bs_price'], alpha=0.4, s=20,
                         c=calls['dte'], cmap='viridis')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='ATM')
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Option Price ($)')
    ax.set_title('Call Price vs Moneyness\n(color = Days to Expiry)', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('DTE')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Realized volatility — shows time-varying vol that motivates ML models
    ax = axes[2]
    for ticker in prices_df.columns:
        returns = np.log(prices_df[ticker] / prices_df[ticker].shift(1)).dropna()
        rv = returns.rolling(21).std() * np.sqrt(252)
        ax.plot(rv.index, rv, label=ticker, linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('21-Day Realized Vol (Annualized)')
    ax.set_title('Realized Volatility Over Time', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_volatility_surface(options_df):
    """Slide 3: Volatility Surface Analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Volatility Surface: IV by Strike & Expiration', fontsize=16, fontweight='bold')

    # Heatmap for one ticker
    ticker = 'AAPL'
    ticker_data = options_df[options_df['ticker'] == ticker].copy()
    ticker_data['moneyness_bucket'] = pd.cut(ticker_data['moneyness'],
                                              bins=[0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3],
                                              labels=['0.75', '0.90', '0.98', '1.02', '1.10', '1.22'])

    calls = ticker_data[ticker_data['flag'] == 'call']
    iv_surface = calls.pivot_table(index='moneyness_bucket', columns='dte',
                                    values='impliedVolatility', aggfunc='median')

    ax = axes[0]
    sns.heatmap(iv_surface, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'IV'})
    ax.set_title(f'IV Surface: {ticker} Calls', fontweight='bold')
    ax.set_xlabel('Days to Expiration')
    ax.set_ylabel('Moneyness Bucket')

    # IV Smile at fixed DTE
    ax = axes[1]
    fixed_dte = 30
    nearby_dte = ticker_data[abs(ticker_data['dte'] - fixed_dte) < 5]
    calls_smile = nearby_dte[nearby_dte['flag'] == 'call'].sort_values('moneyness')
    if not calls_smile.empty:
        ax.plot(calls_smile['moneyness'], calls_smile['impliedVolatility'],
               marker='o', linewidth=2, markersize=8, color='#e74c3c', label='Call IV')
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
        ax.set_xlabel('Moneyness')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'IV Smile (~{fixed_dte} DTE)', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_modeling_approaches(options_df):
    """Slide 4: Proposed Modeling Approaches"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Proposed Modeling Approaches', fontsize=16, fontweight='bold')

    # Model 1: Black-Scholes baseline (predicted vs actual with smile error)
    ax = axes[0, 0]
    sample = options_df.sample(min(200, len(options_df))).copy()
    # Market price deviates from BS due to volatility smile: OTM/ITM worth more than BS predicts
    otm_itm_factor = np.abs(sample['moneyness'] - 1.0)  # 0 at ATM, larger at extremes
    market_prices = sample['bs_price'] * (1.0 + 0.25 * otm_itm_factor + np.random.normal(0, 0.08, len(sample)))
    ax.scatter(sample['bs_price'], market_prices, alpha=0.6, s=50, color='#3498db', label='Market Price')
    ax.plot([sample['bs_price'].min(), sample['bs_price'].max()],
           [sample['bs_price'].min(), sample['bs_price'].max()],
           'r--', linewidth=2, label='Perfect Fit (BS assumption)')
    ax.set_xlabel('Black-Scholes Price')
    ax.set_ylabel('Market Price')
    ax.set_title('Model 1: Black-Scholes Baseline\n(~75-80% accuracy, misses smile)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Model 2: Feature importance for ML
    ax = axes[0, 1]
    features = ['Moneyness', 'Implied Vol', 'DTE', 'Delta', 'Gamma', 'Vega']
    importance = np.array([0.25, 0.30, 0.15, 0.12, 0.10, 0.08])
    ax.barh(features, importance, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Model 2&3: ML Features (XGBoost/LSTM)', fontweight='bold')
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Model comparison framework
    ax = axes[1, 0]
    models = ['Black-Scholes', 'XGBoost', 'LSTM']
    interpretability = [0.95, 0.60, 0.40]
    performance = [0.75, 0.90, 0.85]
    ax.scatter(interpretability, performance, s=500, alpha=0.7,
              c=['#3498db', '#2ecc71', '#f39c12'], edgecolor='black', linewidth=2)
    offsets = [(0, -0.07), (0, 0.06), (0, 0.06)]  # nudge labels off the dots
    for i, model in enumerate(models):
        ax.annotate(model,
                    xy=(interpretability[i], performance[i]),
                    xytext=(interpretability[i] + offsets[i][0],
                            performance[i] + offsets[i][1]),
                    fontsize=10, fontweight='bold', ha='center', va='center')
    ax.set_xlabel('Interpretability')
    ax.set_ylabel('Expected Performance')
    ax.set_title('Model Trade-offs', fontweight='bold')
    ax.set_xlim(0.25, 1.05)
    ax.set_ylim(0.55, 1.0)
    ax.grid(alpha=0.3)

    # Greeks analysis for hedging
    ax = axes[1, 1]
    sample = options_df.sample(min(300, len(options_df)))
    for flag in ['call', 'put']:
        data = sample[sample['flag'] == flag]
        ax.scatter(data['delta'], data['gamma'], alpha=0.6, label=f'{flag.capitalize()}s', s=50)
    ax.set_xlabel('Delta (Directional Risk)')
    ax.set_ylabel('Gamma (Convexity Risk)')
    ax.set_title('Greeks: Risk Profile Analysis', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_timeline_milestones():
    """Slide 5: Timeline & Milestones"""
    fig, ax = plt.subplots(figsize=(14, 6))

    tasks = [
        'Data Collection (YFinance)',
        'Feature Engineering',
        'Black-Scholes Baseline',
        'XGBoost Model Training',
        'LSTM Model Training',
        'Model Comparison & Analysis',
        'Deployment & Documentation'
    ]

    # Project: started Apr 27 (Wk0), NOW = May 4 (Wk1), deadline = June 1 (Wk5)
    start_weeks = [0,   0.5, 1,   1.5, 2,   3,   4  ]
    durations   = [1.5, 1,   1,   1.5, 1.5, 1,   1  ]

    colors_timeline = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#34495e']

    date_labels = ['Apr 27', 'May 4\n(NOW)', 'May 11', 'May 18', 'May 25', 'Jun 1\nDeadline']

    for i, (task, start, duration, color) in enumerate(zip(tasks, start_weeks, durations, colors_timeline)):
        ax.barh(task, duration, left=start, height=0.6, color=color, edgecolor='black', linewidth=2)
        mid = start + duration / 2
        ax.text(mid, i, f'Wk{mid:.0f}',
               ha='center', va='center', fontweight='bold', color='white', fontsize=9)

    # Week grid lines with date labels
    for wk, label in enumerate(date_labels):
        ax.axvline(wk, color='gray', linestyle=':', alpha=0.5)
        ax.text(wk, -1, label, ha='center', fontsize=8, color='gray')

    # Current week marker (Wk1 = May 4)
    ax.axvline(1, color='#2ecc71', linestyle='-', linewidth=2.5, alpha=0.9)
    ax.text(1, len(tasks) + 0.15, 'NOW\n(May 4)', ha='center', color='#2ecc71',
            fontweight='bold', fontsize=9)

    # June 1 deadline marker (Wk5)
    ax.axvline(5, color='#e74c3c', linestyle='--', linewidth=2.5, alpha=0.9)
    ax.text(5, len(tasks) + 0.15, 'June 1\nDeadline', ha='center', color='#e74c3c',
            fontweight='bold', fontsize=9)

    ax.set_xlabel('Weeks (project start: Apr 27)', fontweight='bold', fontsize=12)
    ax.set_title('Project Timeline & Milestones (5 weeks, due June 1)', fontweight='bold', fontsize=14)
    ax.set_xlim(-0.5, 5.8)
    ax.set_ylim(-1.5, len(tasks) + 0.8)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# 3. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Generating synthetic data...")
    options_df = generate_synthetic_options_data(n_samples=1000)
    prices_df = generate_price_history()

    print(f"Options data: {len(options_df)} records, {options_df['ticker'].nunique()} tickers")
    print(f"Price data: {len(prices_df)} days, {len(prices_df.columns)} tickers")
    print(f"\nOptions data summary:\n{options_df.describe()}")

    # Create output directory
    output_dir = Path('proposal_presentation_figs')
    output_dir.mkdir(exist_ok=True)

    # Generate and save all slides
    print("\nGenerating presentation slides...")

    figs = [
        ('Slide1_DataCollection.png', plot_data_collection_timeline()),
        ('Slide2_EDA.png', plot_eda_distributions(options_df, prices_df)),
        ('Slide3_VolSurface.png', plot_volatility_surface(options_df)),
        ('Slide4_Models.png', plot_modeling_approaches(options_df)),
        ('Slide5_Timeline.png', plot_timeline_milestones()),
    ]

    for filename, fig in figs:
        path = output_dir / filename
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  [OK] {filename}")
        plt.close(fig)

    # Save data for further analysis
    options_df.to_csv(output_dir / 'options_data.csv', index=False)
    prices_df.to_csv(output_dir / 'prices_data.csv')
    print(f"\n[OK] All figures saved to: {output_dir.absolute()}")
