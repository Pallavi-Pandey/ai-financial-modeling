"""
Synthetic financial data generator.

Produces 3 years (36 months) of realistic monthly business data with
trend, seasonality, and correlated features, and saves the result to
data/sample_data.csv.
"""

import os
import numpy as np
import pandas as pd


def generate_financial_data(seed: int = 42, save: bool = True) -> pd.DataFrame:
    """Generate synthetic monthly financial data.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.
    save:
        If True, write the DataFrame to ``data/sample_data.csv`` relative to
        this file's parent directory.

    Returns
    -------
    pd.DataFrame
        36-row DataFrame with all generated columns.
    """
    rng = np.random.default_rng(seed)

    # --- Date range: 2022-01 through 2024-12 (36 months) ---
    dates = pd.date_range(start="2022-01-01", periods=36, freq="MS")
    n = len(dates)
    t = np.arange(n)

    # --- Units sold: upward trend + noise ---
    trend = 800 + t * 8  # grows ~288 units over the period
    noise = rng.integers(-100, 101, size=n)
    units_sold = np.clip(trend + noise, 800, 1500).astype(int)

    # --- Price per unit: slow drift + small noise ---
    price_base = 55.0
    price_drift = rng.normal(0, 1.5, size=n).cumsum() * 0.3
    price_per_unit = np.clip(price_base + price_drift + rng.uniform(-5, 5, size=n), 45, 75).round(2)

    # --- Marketing spend: seasonal surges (Q4 heavier) ---
    marketing_base = 10_000 + 3_000 * np.sin(2 * np.pi * t / 12 + np.pi)
    marketing_spend = np.clip(
        marketing_base + rng.uniform(-2_000, 2_000, size=n), 5_000, 20_000
    ).round(0)

    # --- Operating costs: slow inflation + noise ---
    operating_base = 20_000 + t * 120
    operating_costs = np.clip(
        operating_base + rng.uniform(-3_000, 3_000, size=n), 15_000, 35_000
    ).round(0)

    # --- Seasonality factor: sine wave (peak in summer/holiday) ---
    seasonality_factor = (1 + 0.25 * np.sin(2 * np.pi * t / 12 - np.pi / 6)).round(4)

    # --- Economic index: random walk around 100 ---
    shocks = rng.normal(0, 0.8, size=n)
    economic_index = np.clip(100 + shocks.cumsum(), 85, 120).round(2)

    # --- Derived columns ---
    revenue = (units_sold * price_per_unit).round(2)
    profit = (revenue - operating_costs).round(2)

    df = pd.DataFrame(
        {
            "date": dates,
            "units_sold": units_sold,
            "price_per_unit": price_per_unit,
            "marketing_spend": marketing_spend,
            "operating_costs": operating_costs,
            "seasonality_factor": seasonality_factor,
            "economic_index": economic_index,
            "revenue": revenue,
            "profit": profit,
        }
    )

    if save:
        out_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")

    return df


if __name__ == "__main__":
    df = generate_financial_data(save=True)
    print(df.head(10).to_string(index=False))
