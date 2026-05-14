"""
Scenario simulation engine for what-if analysis.

Applies percentage changes to key business levers and uses the trained
forecasting model to estimate the revenue impact.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd

from models.forecasting import FinancialForecaster


class ScenarioEngine:
    """Run what-if scenario simulations against a trained forecaster."""

    def __init__(self, model: FinancialForecaster, base_data: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        model:
            A trained :class:`~models.forecasting.FinancialForecaster`.
        base_data:
            The full feature-engineered DataFrame (rows with NaN already
            dropped).  The last row is used as the baseline for simulations.
        """
        self.model = model
        self.base_data = base_data.copy()
        # Baseline: last available row of features
        self._base_row = base_data.iloc[[-1]].copy()

    # ------------------------------------------------------------------
    # Single-variable simulations
    # ------------------------------------------------------------------

    def simulate_price_change(self, price_delta_pct: float) -> Dict[str, float]:
        """Estimate revenue impact of changing ``price_per_unit``.

        Parameters
        ----------
        price_delta_pct:
            Percentage change (e.g. 10 means +10 %).

        Returns
        -------
        dict
            ``baseline_revenue``, ``new_revenue``, ``revenue_change``,
            ``revenue_change_pct``.
        """
        return self._simulate_column_change("price_per_unit", price_delta_pct)

    def simulate_marketing_change(self, marketing_delta_pct: float) -> Dict[str, float]:
        """Estimate revenue impact of changing ``marketing_spend``."""
        return self._simulate_column_change("marketing_spend", marketing_delta_pct)

    def simulate_demand_change(self, demand_delta_pct: float) -> Dict[str, float]:
        """Estimate revenue impact of changing ``units_sold``."""
        return self._simulate_column_change("units_sold", demand_delta_pct)

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(
        self,
        variable: str,
        range_pct: List[float] = None,
    ) -> pd.DataFrame:
        """Sweep a variable across a range of percentage changes.

        Parameters
        ----------
        variable:
            Column name to vary.  Must be one of ``price_per_unit``,
            ``marketing_spend``, or ``units_sold``.
        range_pct:
            List of percentage deltas to test.  Defaults to
            ``[-30, -20, -10, 0, 10, 20, 30]``.

        Returns
        -------
        pd.DataFrame
            Columns: ``delta_pct``, ``baseline_revenue``, ``new_revenue``,
            ``revenue_change``, ``revenue_change_pct``.
        """
        if range_pct is None:
            range_pct = [-30, -20, -10, 0, 10, 20, 30]

        sim_fn_map = {
            "price_per_unit": self.simulate_price_change,
            "marketing_spend": self.simulate_marketing_change,
            "units_sold": self.simulate_demand_change,
        }
        if variable not in sim_fn_map:
            raise ValueError(
                f"Unknown variable '{variable}'. "
                f"Choose from: {list(sim_fn_map.keys())}"
            )

        rows = []
        for pct in range_pct:
            result = sim_fn_map[variable](pct)
            rows.append({"delta_pct": pct, **result})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Multi-scenario comparison
    # ------------------------------------------------------------------

    def compare_scenarios(self, scenarios_list: List[Dict[str, Union[str, float]]]) -> pd.DataFrame:
        """Compare multiple named scenarios.

        Parameters
        ----------
        scenarios_list:
            List of dicts, each with keys:

            - ``name`` (str) – human-readable label
            - ``variable`` (str) – column to change
            - ``delta_pct`` (float) – percentage change

        Returns
        -------
        pd.DataFrame
            Summary with columns: ``scenario``, ``variable``, ``delta_pct``,
            ``baseline_revenue``, ``new_revenue``, ``revenue_change``,
            ``revenue_change_pct``.

        Example
        -------
        >>> engine.compare_scenarios([
        ...     {"name": "Price +10%", "variable": "price_per_unit", "delta_pct": 10},
        ...     {"name": "Marketing -20%", "variable": "marketing_spend", "delta_pct": -20},
        ... ])
        """
        sim_fn_map = {
            "price_per_unit": self.simulate_price_change,
            "marketing_spend": self.simulate_marketing_change,
            "units_sold": self.simulate_demand_change,
        }

        rows = []
        for scenario in scenarios_list:
            name = scenario.get("name", "Unnamed")
            variable = scenario["variable"]
            delta_pct = float(scenario["delta_pct"])

            if variable not in sim_fn_map:
                raise ValueError(f"Unknown variable '{variable}'.")

            result = sim_fn_map[variable](delta_pct)
            rows.append(
                {
                    "scenario": name,
                    "variable": variable,
                    "delta_pct": delta_pct,
                    **result,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _simulate_column_change(self, column: str, delta_pct: float) -> Dict[str, float]:
        """Apply *delta_pct* to *column* in the base row and get a prediction."""
        baseline_pred = float(self.model.predict(self._base_row)[0])

        if column not in self._base_row.columns:
            # Column not present in feature set — return no change
            return {
                "baseline_revenue": round(baseline_pred, 2),
                "new_revenue": round(baseline_pred, 2),
                "revenue_change": 0.0,
                "revenue_change_pct": 0.0,
            }

        modified_row = self._base_row.copy()
        original_value = float(modified_row[column].iloc[0])
        modified_row[column] = original_value * (1 + delta_pct / 100.0)

        new_pred = float(self.model.predict(modified_row)[0])
        revenue_change = new_pred - baseline_pred
        revenue_change_pct = (revenue_change / baseline_pred * 100) if baseline_pred != 0 else 0.0

        return {
            "baseline_revenue": round(baseline_pred, 2),
            "new_revenue": round(new_pred, 2),
            "revenue_change": round(revenue_change, 2),
            "revenue_change_pct": round(revenue_change_pct, 4),
        }
