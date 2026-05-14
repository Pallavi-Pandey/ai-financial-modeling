"""
Financial forecasting using an XGBoost + RandomForest ensemble.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PARAMS, FORECAST_PERIODS


class FinancialForecaster:
    """Ensemble forecaster combining XGBoost and RandomForest."""

    def __init__(self) -> None:
        xgb_params = MODEL_PARAMS["xgboost"]
        rf_params = MODEL_PARAMS["random_forest"]

        self.xgb_model = XGBRegressor(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            random_state=xgb_params["random_state"],
            verbosity=0,
        )
        self.rf_model = RandomForestRegressor(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            random_state=rf_params["random_state"],
        )
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit both models on the training set.

        Parameters
        ----------
        X_train:
            Feature matrix.
        y_train:
            Target values.
        """
        self.xgb_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self._is_trained = True
        self._feature_names: List[str] = list(X_train.columns)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the ensemble prediction (simple average of both models).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self._check_trained()
        xgb_preds = self.xgb_model.predict(X)
        rf_preds = self.rf_model.predict(X)
        return (xgb_preds + rf_preds) / 2.0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Compute standard regression metrics on the test set.

        Returns
        -------
        dict
            Keys: MAE, RMSE, R2, MAPE.
        """
        self._check_trained()
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE: avoid division by zero
        nonzero_mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask]))
            * 100
            if nonzero_mask.any()
            else float("nan")
        )

        return {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape), 2),
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self, feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Return a DataFrame with sorted ensemble feature importances.

        Parameters
        ----------
        feature_names:
            List of feature names. Falls back to names stored during
            training if not provided.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance`` (sorted descending).
        """
        self._check_trained()
        names = feature_names or self._feature_names

        xgb_importance = self.xgb_model.feature_importances_
        rf_importance = self.rf_model.feature_importances_
        ensemble_importance = (xgb_importance + rf_importance) / 2.0

        df = pd.DataFrame({"feature": names, "importance": ensemble_importance})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Future forecast
    # ------------------------------------------------------------------

    def forecast_future(
        self,
        last_row_features: pd.DataFrame,
        periods: int = FORECAST_PERIODS,
    ) -> pd.DataFrame:
        """Iteratively forecast *periods* months ahead.

        Each iteration uses the prediction from the previous step to update
        the lag features before predicting the next period.

        Parameters
        ----------
        last_row_features:
            A single-row DataFrame containing the most recent feature values.
        periods:
            Number of future months to forecast.

        Returns
        -------
        pd.DataFrame
            Columns: ``period``, ``forecast``, ``lower_bound``, ``upper_bound``.
        """
        self._check_trained()

        current_features = last_row_features.copy()
        forecasts: List[float] = []

        # Estimate uncertainty from individual model spread during training
        # (simple heuristic: ±5 % per period, growing with horizon)
        uncertainty_pct = 0.05

        for step in range(1, periods + 1):
            pred = float(self.predict(current_features)[0])
            forecasts.append(pred)

            # Roll lag features forward
            for lag in (3, 2):
                lag_col = f"revenue_lag_{lag}"
                prev_lag_col = f"revenue_lag_{lag - 1}"
                if lag_col in current_features.columns and prev_lag_col in current_features.columns:
                    current_features[lag_col] = current_features[prev_lag_col].values

            if "revenue_lag_1" in current_features.columns:
                current_features["revenue_lag_1"] = pred

            # Update rolling means naively
            if "revenue_rolling_3" in current_features.columns:
                current_features["revenue_rolling_3"] = pred
            if "revenue_rolling_6" in current_features.columns:
                current_features["revenue_rolling_6"] = pred

        results = []
        for step, forecast_val in enumerate(forecasts, start=1):
            margin = forecast_val * uncertainty_pct * step
            results.append(
                {
                    "period": f"M+{step}",
                    "forecast": round(forecast_val, 2),
                    "lower_bound": round(forecast_val - margin, 2),
                    "upper_bound": round(forecast_val + margin, 2),
                }
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Serialize both models to *path* using joblib."""
        self._check_trained()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "xgb_model": self.xgb_model,
                "rf_model": self.rf_model,
                "feature_names": self._feature_names,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Restore models from a joblib file saved with :meth:`save_model`."""
        data = joblib.load(path)
        self.xgb_model = data["xgb_model"]
        self.rf_model = data["rf_model"]
        self._feature_names = data["feature_names"]
        self._is_trained = True
        print(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
