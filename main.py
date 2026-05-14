"""
CLI entry point for the AI-Powered Financial Modeling system.

Usage:
    python main.py                  # Run full pipeline
    python main.py --generate-data  # Regenerate synthetic data, then run pipeline
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# Ensure project root is importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_PATH, FORECAST_PERIODS
from data.generate_data import generate_financial_data
from insights.ai_insights import AIInsightsEngine
from models.forecasting import FinancialForecaster
from models.scenario_engine import ScenarioEngine
from utils.feature_engineering import FeatureEngineer


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def _indent(text: str, prefix: str = "  ") -> str:
    return textwrap.indent(text, prefix)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_generate_data(force: bool = False) -> str:
    """Generate (or locate) the sample CSV."""
    _section("Step 1: Data Generation")
    csv_path = os.path.join(os.path.dirname(__file__), DATA_PATH)

    if force or not os.path.exists(csv_path):
        print("  Generating synthetic financial data…")
        generate_financial_data(save=True)
    else:
        print(f"  Using existing data at {csv_path}")

    return csv_path


def step_feature_engineering(csv_path: str):
    """Load data and create features."""
    _section("Step 2: Feature Engineering")
    fe = FeatureEngineer()
    df_raw = fe.load_data(csv_path)
    df_features = fe.create_features(df_raw)
    X_train, X_test, y_train, y_test, feature_names = fe.prepare_for_modeling(df_features)

    print(f"  Raw data shape:       {df_raw.shape}")
    print(f"  Feature matrix shape: {df_features.dropna().shape}")
    print(f"  Training samples:     {len(X_train)}")
    print(f"  Test samples:         {len(X_test)}")
    print(f"  Number of features:   {len(feature_names)}")

    return df_raw, df_features, X_train, X_test, y_train, y_test, feature_names


def step_train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train ensemble and evaluate."""
    _section("Step 3: Model Training & Evaluation")
    forecaster = FinancialForecaster()
    forecaster.train(X_train, y_train)
    metrics = forecaster.evaluate(X_test, y_test)

    print("  Evaluation Metrics:")
    print(f"    MAE  : ${metrics['MAE']:>12,.2f}")
    print(f"    RMSE : ${metrics['RMSE']:>12,.2f}")
    print(f"    R²   : {metrics['R2']:>14.4f}")
    print(f"    MAPE : {metrics['MAPE']:>13.2f}%")

    print("\n  Top 5 Feature Importances:")
    fi_df = forecaster.get_feature_importance(feature_names)
    for _, row in fi_df.head(5).iterrows():
        print(f"    {row['feature']:<35s} {row['importance']:.4f}")

    return forecaster, metrics, fi_df


def step_forecast(forecaster: FinancialForecaster, X_test):
    """Generate future forecasts."""
    _section(f"Step 4: {FORECAST_PERIODS}-Month Ahead Forecast")
    forecast_df = forecaster.forecast_future(X_test.iloc[[-1]].copy())
    print(f"\n{_indent(forecast_df.to_string(index=False))}")
    return forecast_df


def step_scenarios(forecaster: FinancialForecaster, df_features, feature_names):
    """Run a standard set of scenario simulations."""
    _section("Step 5: Scenario Simulation")
    base_data = df_features[feature_names].dropna()
    engine = ScenarioEngine(forecaster, base_data)

    scenarios = [
        {"name": "Price +10%",         "variable": "price_per_unit",  "delta_pct": 10},
        {"name": "Price -10%",         "variable": "price_per_unit",  "delta_pct": -10},
        {"name": "Marketing +20%",     "variable": "marketing_spend", "delta_pct": 20},
        {"name": "Marketing -20%",     "variable": "marketing_spend", "delta_pct": -20},
        {"name": "Demand +15%",        "variable": "units_sold",      "delta_pct": 15},
        {"name": "Demand -15%",        "variable": "units_sold",      "delta_pct": -15},
        {"name": "Baseline (0%)",      "variable": "price_per_unit",  "delta_pct":  0},
    ]
    scenario_df = engine.compare_scenarios(scenarios)

    display_cols = ["scenario", "baseline_revenue", "new_revenue", "revenue_change", "revenue_change_pct"]
    print(f"\n{_indent(scenario_df[display_cols].round(2).to_string(index=False))}")
    return scenario_df


def step_ai_insights(metrics, fi_df, scenario_df, df_raw, forecast_df):
    """Request LLM insights from Claude."""
    _section("Step 6: AI-Powered Insights (Claude)")
    ai_engine = AIInsightsEngine()

    print("\n  [Performance Insights]")
    perf_text = ai_engine.generate_performance_insights(metrics, fi_df)
    print(_indent(textwrap.fill(perf_text, width=80)))

    print("\n  [Scenario Recommendations]")
    scen_text = ai_engine.generate_scenario_recommendations(scenario_df.to_dict())
    print(_indent(textwrap.fill(scen_text, width=80)))

    print("\n  [Executive Summary]")
    exec_text = ai_engine.generate_executive_summary(
        {
            "metrics": metrics,
            "data_stats": df_raw[["revenue", "profit"]].describe().round(2).to_string(),
            "forecast": forecast_df.to_string(index=False),
            "scenarios": scenario_df[["scenario", "revenue_change"]].to_string(index=False),
            "feature_importance": fi_df.head(5).to_string(index=False),
        }
    )
    print(_indent(textwrap.fill(exec_text, width=80)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Powered Financial Modeling CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python main.py                  # Run full pipeline with existing data
              python main.py --generate-data  # Regenerate data first, then run pipeline
            """
        ),
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Regenerate synthetic data even if sample_data.csv already exists.",
    )
    args = parser.parse_args()

    _banner("AI-Powered Financial Modeling & Decision Intelligence")

    # --- Run pipeline ---
    csv_path = step_generate_data(force=args.generate_data)
    df_raw, df_features, X_train, X_test, y_train, y_test, feature_names = (
        step_feature_engineering(csv_path)
    )
    forecaster, metrics, fi_df = step_train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )
    forecast_df = step_forecast(forecaster, X_test)
    scenario_df = step_scenarios(forecaster, df_features, feature_names)
    step_ai_insights(metrics, fi_df, scenario_df, df_raw, forecast_df)

    _banner("Pipeline Complete")
    print("\nTo launch the interactive dashboard:")
    print("  streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
