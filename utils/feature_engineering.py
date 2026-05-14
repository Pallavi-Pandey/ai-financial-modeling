"""
Feature engineering and preprocessing for financial time-series data.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import TEST_SIZE


class FeatureEngineer:
    """Loads raw CSV data and engineers features for ML modeling."""

    # Columns that must exist in the raw CSV
    REQUIRED_COLUMNS = [
        "date",
        "units_sold",
        "price_per_unit",
        "marketing_spend",
        "operating_costs",
        "seasonality_factor",
        "economic_index",
        "revenue",
        "profit",
    ]

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data, parse dates, and sort chronologically.

        Parameters
        ----------
        filepath:
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Clean DataFrame sorted by date.
        """
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the DataFrame.

        Features added
        --------------
        - Lag features : revenue_lag_1, revenue_lag_2, revenue_lag_3
        - Rolling means : revenue_rolling_3, revenue_rolling_6
        - Calendar      : month, quarter (one-hot encoded)
        - growth_rate   : month-over-month revenue change (%)
        - marketing_roi : revenue / marketing_spend
        - profit_margin : profit / revenue

        Parameters
        ----------
        df:
            DataFrame returned by :meth:`load_data`.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with extra feature columns appended.
        """
        df = df.copy()

        # --- Lag features ---
        for lag in (1, 2, 3):
            df[f"revenue_lag_{lag}"] = df["revenue"].shift(lag)

        # --- Rolling means ---
        df["revenue_rolling_3"] = df["revenue"].shift(1).rolling(window=3).mean()
        df["revenue_rolling_6"] = df["revenue"].shift(1).rolling(window=6).mean()

        # --- Calendar features (one-hot) ---
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

        month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int)
        quarter_dummies = pd.get_dummies(df["quarter"], prefix="quarter", drop_first=True, dtype=int)
        df = pd.concat([df, month_dummies, quarter_dummies], axis=1)

        # --- Derived ratios ---
        df["growth_rate"] = df["revenue"].pct_change() * 100
        df["marketing_roi"] = df["revenue"] / df["marketing_spend"].replace(0, np.nan)
        df["profit_margin"] = df["profit"] / df["revenue"].replace(0, np.nan)

        return df

    def prepare_for_modeling(
        self,
        df: pd.DataFrame,
        target: str = "revenue",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """Split the engineered DataFrame into train/test sets.

        Drops rows with NaN values introduced by lagging/rolling, then
        performs an 80/20 chronological split (no shuffling).

        Parameters
        ----------
        df:
            DataFrame with engineered features.
        target:
            Name of the target column.

        Returns
        -------
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
        """
        # Columns that must not be used as features
        drop_cols = {
            "date",
            target,
            # Drop the raw revenue-derived columns when target is revenue to
            # avoid data leakage (profit is revenue minus costs, for example)
            "profit",
            "month",
            "quarter",
        }

        feature_cols = [c for c in df.columns if c not in drop_cols]

        # Drop rows with NaN (from lag / rolling windows)
        modeling_df = df[feature_cols + [target]].dropna()

        X = modeling_df[feature_cols]
        y = modeling_df[target]

        # Chronological split — no shuffle
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False
        )

        return X_train, X_test, y_train, y_test, feature_cols
