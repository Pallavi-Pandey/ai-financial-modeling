import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from models.scenario_engine import ScenarioEngine


@pytest.fixture
def scenario_engine(trained_forecaster):
    f, X_train, *_ = trained_forecaster
    return ScenarioEngine(f, X_train), f, X_train


# ---------------------------------------------------------------------------
# Single-variable simulations
# ---------------------------------------------------------------------------

class TestSimulateColumnChange:
    def test_price_increase_returns_expected_keys(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_price_change(10.0)
        assert set(result.keys()) == {
            "baseline_revenue", "new_revenue", "revenue_change", "revenue_change_pct"
        }

    def test_price_increase_raises_revenue(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_price_change(10.0)
        assert result["new_revenue"] > result["baseline_revenue"]

    def test_price_decrease_lowers_revenue(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_price_change(-10.0)
        assert result["new_revenue"] < result["baseline_revenue"]

    def test_zero_delta_produces_no_change(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_price_change(0.0)
        assert result["revenue_change"] == pytest.approx(0.0, abs=1e-2)
        assert result["revenue_change_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_marketing_change_returns_dict(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_marketing_change(20.0)
        assert "new_revenue" in result

    def test_demand_change_returns_dict(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_demand_change(15.0)
        assert "new_revenue" in result

    def test_missing_column_returns_no_change(self, scenario_engine):
        se, *_ = scenario_engine
        result = se._simulate_column_change("nonexistent_column_xyz", 10.0)
        assert result["revenue_change"] == 0.0
        assert result["revenue_change_pct"] == 0.0
        assert result["baseline_revenue"] == result["new_revenue"]

    def test_zero_baseline_revenue_no_division_error(self, scenario_engine):
        se, f, _ = scenario_engine
        with patch.object(f, "predict", return_value=np.array([0.0])):
            result = se._simulate_column_change("price_per_unit", 10.0)
        assert result["revenue_change_pct"] == 0.0

    def test_revenue_change_is_new_minus_baseline(self, scenario_engine):
        se, *_ = scenario_engine
        result = se.simulate_price_change(5.0)
        expected_change = round(result["new_revenue"] - result["baseline_revenue"], 2)
        assert result["revenue_change"] == pytest.approx(expected_change, abs=0.01)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_returns_dataframe(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.run_sensitivity_analysis("price_per_unit")
        assert isinstance(df, pd.DataFrame)

    def test_default_range_has_seven_rows(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.run_sensitivity_analysis("price_per_unit")
        assert len(df) == 7

    def test_delta_pct_column_present(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.run_sensitivity_analysis("marketing_spend")
        assert "delta_pct" in df.columns

    def test_custom_range(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.run_sensitivity_analysis("units_sold", range_pct=[-10, 0, 10])
        assert len(df) == 3

    def test_zero_delta_row_has_no_change(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.run_sensitivity_analysis("price_per_unit")
        zero_row = df[df["delta_pct"] == 0]
        assert len(zero_row) == 1
        assert zero_row.iloc[0]["revenue_change_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_unknown_variable_raises_value_error(self, scenario_engine):
        se, *_ = scenario_engine
        with pytest.raises(ValueError, match="Unknown variable"):
            se.run_sensitivity_analysis("bad_variable_name")

    def test_all_three_valid_variables(self, scenario_engine):
        se, *_ = scenario_engine
        for var in ["price_per_unit", "marketing_spend", "units_sold"]:
            df = se.run_sensitivity_analysis(var, range_pct=[0])
            assert len(df) == 1


# ---------------------------------------------------------------------------
# Compare scenarios
# ---------------------------------------------------------------------------

class TestCompareScenarios:
    def test_returns_dataframe(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.compare_scenarios([
            {"name": "Price Up", "variable": "price_per_unit", "delta_pct": 10},
            {"name": "Mkt Up", "variable": "marketing_spend", "delta_pct": 20},
        ])
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self, scenario_engine):
        se, *_ = scenario_engine
        scenarios = [
            {"name": "A", "variable": "price_per_unit", "delta_pct": 5},
            {"name": "B", "variable": "units_sold", "delta_pct": -10},
            {"name": "C", "variable": "marketing_spend", "delta_pct": 15},
        ]
        df = se.compare_scenarios(scenarios)
        assert len(df) == 3

    def test_scenario_names_in_output(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.compare_scenarios([
            {"name": "MyScenario", "variable": "price_per_unit", "delta_pct": 0},
        ])
        assert "MyScenario" in df["scenario"].values

    def test_output_has_required_columns(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.compare_scenarios([
            {"name": "X", "variable": "price_per_unit", "delta_pct": 5},
        ])
        for col in ["scenario", "variable", "delta_pct", "baseline_revenue",
                    "new_revenue", "revenue_change", "revenue_change_pct"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_unknown_variable_raises_value_error(self, scenario_engine):
        se, *_ = scenario_engine
        with pytest.raises(ValueError):
            se.compare_scenarios([
                {"name": "Bad", "variable": "not_a_real_column", "delta_pct": 5}
            ])

    def test_single_scenario_returns_one_row(self, scenario_engine):
        se, *_ = scenario_engine
        df = se.compare_scenarios([
            {"name": "Baseline", "variable": "price_per_unit", "delta_pct": 0}
        ])
        assert len(df) == 1
