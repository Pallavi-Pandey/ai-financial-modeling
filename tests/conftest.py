import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def sample_df():
    """36-month financial DataFrame matching the required schema."""
    rng = np.random.default_rng(42)
    n = 36
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    units = rng.integers(800, 1500, n).astype(float)
    price = rng.uniform(45, 75, n)
    revenue = (units * price).round(2)
    marketing = rng.uniform(5000, 20000, n).round(0)
    costs = rng.uniform(15000, 35000, n).round(0)
    return pd.DataFrame({
        "date": dates,
        "units_sold": units,
        "price_per_unit": price,
        "marketing_spend": marketing,
        "operating_costs": costs,
        "seasonality_factor": rng.uniform(0.75, 1.25, n).round(4),
        "economic_index": rng.uniform(85, 120, n).round(2),
        "revenue": revenue,
        "profit": (revenue - costs).round(2),
    })


@pytest.fixture(scope="session")
def engineered_split(sample_df):
    from utils.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_features = fe.create_features(sample_df)
    return fe.prepare_for_modeling(df_features)


@pytest.fixture(scope="session")
def trained_forecaster(engineered_split):
    from models.forecasting import FinancialForecaster
    X_train, X_test, y_train, y_test, feature_names = engineered_split
    f = FinancialForecaster()
    f.train(X_train, y_train)
    return f, X_train, X_test, y_train, y_test, feature_names
