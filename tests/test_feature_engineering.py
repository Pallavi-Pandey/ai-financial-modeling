import numpy as np
import pandas as pd
import pytest

from utils.feature_engineering import FeatureEngineer


@pytest.fixture
def fe():
    return FeatureEngineer()


@pytest.fixture
def base_df():
    rng = np.random.default_rng(0)
    n = 36
    dates = pd.date_range("2022-01-01", periods=n, freq="MS")
    units = rng.uniform(800, 1500, n)
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
        "seasonality_factor": rng.uniform(0.75, 1.25, n),
        "economic_index": rng.uniform(85, 120, n),
        "revenue": revenue,
        "profit": (revenue - costs).round(2),
    })


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_loads_valid_csv(self, fe, base_df, tmp_path):
        path = str(tmp_path / "data.csv")
        base_df.to_csv(path, index=False)
        df = fe.load_data(path)
        assert len(df) == 36

    def test_parses_date_column(self, fe, base_df, tmp_path):
        path = str(tmp_path / "data.csv")
        base_df.to_csv(path, index=False)
        df = fe.load_data(path)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_sorts_by_date(self, fe, base_df, tmp_path):
        shuffled = base_df.sample(frac=1, random_state=7)
        path = str(tmp_path / "shuffled.csv")
        shuffled.to_csv(path, index=False)
        df = fe.load_data(path)
        assert (df["date"].values == base_df.sort_values("date")["date"].values).all()

    def test_raises_for_missing_columns(self, fe, base_df, tmp_path):
        incomplete = base_df.drop(columns=["revenue", "units_sold"])
        path = str(tmp_path / "bad.csv")
        incomplete.to_csv(path, index=False)
        with pytest.raises(ValueError):
            fe.load_data(path)

    def test_accepts_extra_columns(self, fe, base_df, tmp_path):
        extra = base_df.copy()
        extra["bonus_col"] = 99
        path = str(tmp_path / "extra.csv")
        extra.to_csv(path, index=False)
        df = fe.load_data(path)
        assert "bonus_col" in df.columns


# ---------------------------------------------------------------------------
# create_features
# ---------------------------------------------------------------------------

class TestCreateFeatures:
    def test_lag_columns_present(self, fe, base_df):
        df = fe.create_features(base_df)
        assert "revenue_lag_1" in df.columns
        assert "revenue_lag_2" in df.columns
        assert "revenue_lag_3" in df.columns

    def test_lag_1_matches_previous_row(self, fe, base_df):
        df = fe.create_features(base_df)
        assert df["revenue_lag_1"].iloc[1] == base_df["revenue"].iloc[0]

    def test_lag_2_matches_two_rows_back(self, fe, base_df):
        df = fe.create_features(base_df)
        assert df["revenue_lag_2"].iloc[2] == base_df["revenue"].iloc[0]

    def test_rolling_mean_columns_present(self, fe, base_df):
        df = fe.create_features(base_df)
        assert "revenue_rolling_3" in df.columns
        assert "revenue_rolling_6" in df.columns

    def test_rolling_mean_shifted_no_lookahead(self, fe, base_df):
        # At row index 1, rolling_3 should be NaN (not enough past data after shift)
        df = fe.create_features(base_df)
        assert pd.isna(df["revenue_rolling_3"].iloc[0])

    def test_month_dummies_present(self, fe, base_df):
        df = fe.create_features(base_df)
        month_cols = [c for c in df.columns if c.startswith("month_")]
        assert len(month_cols) > 0

    def test_quarter_dummies_present(self, fe, base_df):
        df = fe.create_features(base_df)
        quarter_cols = [c for c in df.columns if c.startswith("quarter_")]
        assert len(quarter_cols) > 0

    def test_growth_rate_present(self, fe, base_df):
        df = fe.create_features(base_df)
        assert "growth_rate" in df.columns

    def test_marketing_roi_present(self, fe, base_df):
        df = fe.create_features(base_df)
        assert "marketing_roi" in df.columns

    def test_profit_margin_present(self, fe, base_df):
        df = fe.create_features(base_df)
        assert "profit_margin" in df.columns

    def test_zero_marketing_spend_gives_nan_roi(self, fe, base_df):
        df_mod = base_df.copy()
        df_mod.loc[5, "marketing_spend"] = 0
        df = fe.create_features(df_mod)
        assert pd.isna(df.loc[5, "marketing_roi"])

    def test_zero_revenue_gives_nan_profit_margin(self, fe, base_df):
        df_mod = base_df.copy()
        df_mod.loc[5, "revenue"] = 0
        df = fe.create_features(df_mod)
        assert pd.isna(df.loc[5, "profit_margin"])

    def test_does_not_modify_original_dataframe(self, fe, base_df):
        original_cols = list(base_df.columns)
        fe.create_features(base_df)
        assert list(base_df.columns) == original_cols


# ---------------------------------------------------------------------------
# prepare_for_modeling
# ---------------------------------------------------------------------------

class TestPrepareForModeling:
    def test_returns_five_values(self, fe, base_df):
        df = fe.create_features(base_df)
        result = fe.prepare_for_modeling(df)
        assert len(result) == 5

    def test_no_leakage_columns_in_X(self, fe, base_df):
        df = fe.create_features(base_df)
        X_train, X_test, *_ = fe.prepare_for_modeling(df)
        for col in ["date", "revenue", "profit", "month", "quarter"]:
            assert col not in X_train.columns
            assert col not in X_test.columns

    def test_no_nans_in_training_set(self, fe, base_df):
        df = fe.create_features(base_df)
        X_train, X_test, y_train, y_test, _ = fe.prepare_for_modeling(df)
        assert not X_train.isnull().any().any()
        assert not y_train.isnull().any()

    def test_chronological_split_order(self, fe, base_df):
        df = fe.create_features(base_df)
        X_train, X_test, *_ = fe.prepare_for_modeling(df)
        assert X_train.index.max() < X_test.index.min()

    def test_split_is_approximately_80_20(self, fe, base_df):
        df = fe.create_features(base_df)
        X_train, X_test, *_ = fe.prepare_for_modeling(df)
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_feature_names_excludes_target(self, fe, base_df):
        df = fe.create_features(base_df)
        _, _, _, _, feature_names = fe.prepare_for_modeling(df)
        assert "revenue" not in feature_names

    def test_feature_names_is_list(self, fe, base_df):
        df = fe.create_features(base_df)
        _, _, _, _, feature_names = fe.prepare_for_modeling(df)
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_y_train_matches_revenue(self, fe, base_df):
        df = fe.create_features(base_df)
        X_train, _, y_train, _, _ = fe.prepare_for_modeling(df)
        assert y_train.name == "revenue"
