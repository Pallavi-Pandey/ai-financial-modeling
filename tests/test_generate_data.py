import numpy as np
import pandas as pd
import pytest

from data.generate_data import generate_financial_data
from utils.feature_engineering import FeatureEngineer


def test_generates_36_rows():
    df = generate_financial_data(save=False)
    assert len(df) == 36


def test_has_all_required_columns():
    df = generate_financial_data(save=False)
    for col in FeatureEngineer.REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing: {col}"


def test_revenue_equals_units_times_price():
    df = generate_financial_data(save=False)
    expected = (df["units_sold"] * df["price_per_unit"]).round(2)
    pd.testing.assert_series_equal(df["revenue"], expected, check_names=False)


def test_profit_equals_revenue_minus_costs():
    df = generate_financial_data(save=False)
    expected = (df["revenue"] - df["operating_costs"]).round(2)
    pd.testing.assert_series_equal(df["profit"], expected, check_names=False)


def test_reproducible_with_same_seed():
    df1 = generate_financial_data(seed=42, save=False)
    df2 = generate_financial_data(seed=42, save=False)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_data():
    df1 = generate_financial_data(seed=42, save=False)
    df2 = generate_financial_data(seed=99, save=False)
    assert not df1["revenue"].equals(df2["revenue"])


def test_units_sold_within_clip_range():
    df = generate_financial_data(save=False)
    assert df["units_sold"].min() >= 800
    assert df["units_sold"].max() <= 1500


def test_price_per_unit_within_clip_range():
    df = generate_financial_data(save=False)
    assert df["price_per_unit"].min() >= 45
    assert df["price_per_unit"].max() <= 75


def test_marketing_spend_within_clip_range():
    df = generate_financial_data(save=False)
    assert df["marketing_spend"].min() >= 5000
    assert df["marketing_spend"].max() <= 20000


def test_operating_costs_within_clip_range():
    df = generate_financial_data(save=False)
    assert df["operating_costs"].min() >= 15000
    assert df["operating_costs"].max() <= 35000


def test_dates_are_monthly():
    df = generate_financial_data(save=False)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    diffs = df["date"].diff().dropna().dt.days
    assert diffs.between(28, 31).all()


def test_dates_span_three_years():
    df = generate_financial_data(save=False)
    assert df["date"].iloc[0].year == 2022
    assert df["date"].iloc[-1].year == 2024


def test_save_writes_file(tmp_path, monkeypatch):
    import data.generate_data as gd
    monkeypatch.setattr(gd.os.path, "dirname", lambda _: str(tmp_path))
    df = gd.generate_financial_data(seed=42, save=True)
    assert (tmp_path / "sample_data.csv").exists()
    loaded = pd.read_csv(tmp_path / "sample_data.csv")
    assert len(loaded) == 36
