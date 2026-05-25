"""
Black-Scholes baseline: analytical option pricing + VaR.
Used as the benchmark all other models are compared against.
"""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if flag == "call" else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * (norm.cdf(d2) if flag == "call" else norm.cdf(-d2))
    ) / 365
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric-free VaR from a returns array."""
    return float(np.percentile(returns, (1 - confidence) * 100))


def parametric_var(portfolio_value: float, sigma_daily: float, confidence: float = 0.95) -> float:
    """Variance-covariance VaR assuming normality."""
    z = norm.ppf(confidence)
    return portfolio_value * sigma_daily * z


def portfolio_var(
    positions: list[dict],  # [{"weight": 0.4, "sigma_daily": 0.012}, ...]
    portfolio_value: float,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """
    Compute portfolio VaR using a weighted variance-covariance approach.
    Assumes zero correlation as a conservative baseline (positions are each stocks or options).
    """
    weights = np.array([p["weight"] for p in positions])
    sigmas = np.array([p["sigma_daily"] for p in positions])
    # Independent assets: portfolio sigma = sqrt(sum of w_i^2 * sigma_i^2)
    portfolio_sigma = np.sqrt(np.sum((weights * sigmas) ** 2)) * np.sqrt(horizon_days)
    z = norm.ppf(confidence)
    var = portfolio_value * portfolio_sigma * z
    return {
        "var": round(var, 4),
        "portfolio_sigma": round(portfolio_sigma, 6),
        "confidence": confidence,
        "horizon_days": horizon_days,
    }
