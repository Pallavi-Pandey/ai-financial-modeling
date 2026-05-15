import numpy as np
import pandas as pd
import pytest

from models.forecasting import FinancialForecaster


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------

class TestTrainAndPredict:
    def test_predict_before_train_raises(self, engineered_split):
        X_train, *_ = engineered_split
        f = FinancialForecaster()
        with pytest.raises(RuntimeError, match="trained"):
            f.predict(X_train)

    def test_train_sets_is_trained_flag(self, engineered_split):
        X_train, _, y_train, *_ = engineered_split
        f = FinancialForecaster()
        assert not f._is_trained
        f.train(X_train, y_train)
        assert f._is_trained

    def test_predict_returns_ndarray(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        preds = f.predict(X_test)
        assert isinstance(preds, np.ndarray)

    def test_predict_length_matches_input(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        assert len(f.predict(X_test)) == len(X_test)

    def test_predict_is_mean_of_both_models(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        expected = (f.xgb_model.predict(X_test) + f.rf_model.predict(X_test)) / 2.0
        np.testing.assert_allclose(f.predict(X_test), expected)

    def test_stores_feature_names_after_train(self, trained_forecaster, engineered_split):
        f, *_ = trained_forecaster
        _, _, _, _, feature_names = engineered_split
        assert f._feature_names == feature_names


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_all_metric_keys(self, trained_forecaster):
        f, _, X_test, _, y_test, _ = trained_forecaster
        metrics = f.evaluate(X_test, y_test)
        assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "MAPE"}

    def test_mae_is_non_negative(self, trained_forecaster):
        f, _, X_test, _, y_test, _ = trained_forecaster
        assert f.evaluate(X_test, y_test)["MAE"] >= 0

    def test_rmse_gte_mae(self, trained_forecaster):
        f, _, X_test, _, y_test, _ = trained_forecaster
        m = f.evaluate(X_test, y_test)
        assert m["RMSE"] >= m["MAE"]

    def test_r2_at_most_one(self, trained_forecaster):
        f, _, X_test, _, y_test, _ = trained_forecaster
        assert f.evaluate(X_test, y_test)["R2"] <= 1.0

    def test_mape_nan_when_all_y_true_zero(self, trained_forecaster):
        f, X_train, *_ = trained_forecaster
        y_zeros = pd.Series(np.zeros(len(X_train)), index=X_train.index)
        metrics = f.evaluate(X_train, y_zeros)
        assert np.isnan(metrics["MAPE"])

    def test_mape_finite_for_nonzero_targets(self, trained_forecaster):
        f, _, X_test, _, y_test, _ = trained_forecaster
        assert np.isfinite(f.evaluate(X_test, y_test)["MAPE"])


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestGetFeatureImportance:
    def test_returns_dataframe(self, trained_forecaster):
        f, _, _, _, _, feature_names = trained_forecaster
        fi = f.get_feature_importance(feature_names)
        assert isinstance(fi, pd.DataFrame)

    def test_has_feature_and_importance_columns(self, trained_forecaster):
        f, _, _, _, _, feature_names = trained_forecaster
        fi = f.get_feature_importance(feature_names)
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_sorted_descending(self, trained_forecaster):
        f, _, _, _, _, feature_names = trained_forecaster
        fi = f.get_feature_importance(feature_names)
        assert (fi["importance"].diff().dropna() <= 0).all()

    def test_importances_sum_to_approximately_one(self, trained_forecaster):
        f, _, _, _, _, feature_names = trained_forecaster
        fi = f.get_feature_importance(feature_names)
        assert abs(fi["importance"].sum() - 1.0) < 1e-5

    def test_uses_stored_feature_names_when_none_given(self, trained_forecaster):
        f, *_ = trained_forecaster
        fi = f.get_feature_importance()
        assert len(fi) > 0


# ---------------------------------------------------------------------------
# Forecast future
# ---------------------------------------------------------------------------

class TestForecastFuture:
    def test_forecast_before_train_raises(self, engineered_split):
        X_train, *_ = engineered_split
        f = FinancialForecaster()
        with pytest.raises(RuntimeError, match="trained"):
            f.forecast_future(X_train.iloc[[-1]])

    def test_returns_correct_number_of_periods(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]], periods=6)
        assert len(df) == 6

    def test_custom_periods(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]], periods=3)
        assert len(df) == 3

    def test_output_columns(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]])
        assert {"period", "forecast", "lower_bound", "upper_bound"}.issubset(df.columns)

    def test_period_labels(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]], periods=3)
        assert list(df["period"]) == ["M+1", "M+2", "M+3"]

    def test_uncertainty_bands_grow_with_horizon(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]], periods=6)
        band_width = df["upper_bound"] - df["lower_bound"]
        assert (band_width.diff().dropna() >= 0).all()

    def test_lower_bound_below_forecast(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]])
        assert (df["lower_bound"] <= df["forecast"]).all()

    def test_upper_bound_above_forecast(self, trained_forecaster):
        f, _, X_test, *_ = trained_forecaster
        df = f.forecast_future(X_test.iloc[[-1]])
        assert (df["upper_bound"] >= df["forecast"]).all()


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_preserves_predictions(self, trained_forecaster, tmp_path):
        f, _, X_test, *_ = trained_forecaster
        path = str(tmp_path / "model.joblib")
        f.save_model(path)

        f2 = FinancialForecaster()
        f2.load_model(path)
        assert f2._is_trained
        np.testing.assert_allclose(f2.predict(X_test), f.predict(X_test))

    def test_load_restores_feature_names(self, trained_forecaster, tmp_path):
        f, *_ = trained_forecaster
        path = str(tmp_path / "model.joblib")
        f.save_model(path)

        f2 = FinancialForecaster()
        f2.load_model(path)
        assert f2._feature_names == f._feature_names

    def test_predict_after_load_raises_no_error(self, trained_forecaster, tmp_path):
        f, _, X_test, *_ = trained_forecaster
        path = str(tmp_path / "model.joblib")
        f.save_model(path)

        f2 = FinancialForecaster()
        f2.load_model(path)
        result = f2.predict(X_test)
        assert result is not None
