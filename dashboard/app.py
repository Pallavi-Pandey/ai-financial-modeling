"""
Streamlit interactive dashboard for the AI-Powered Financial Modeling system.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

# Allow imports from the project root regardless of where streamlit is invoked
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DATA_PATH
from data.generate_data import generate_financial_data
from insights.ai_insights import AIInsightsEngine
from models.forecasting import FinancialForecaster
from models.scenario_engine import ScenarioEngine
from utils.feature_engineering import FeatureEngineer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Intelligence",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI-Powered Financial Modeling & Decision Intelligence")
st.markdown(
    "Combine machine-learning forecasting, what-if scenario simulation, "
    "and Claude-powered narrative insights — all in one place."
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
defaults: dict = {
    "df_raw": None,
    "df_features": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "feature_names": None,
    "forecaster": None,
    "metrics": None,
    "forecast_df": None,
    "scenario_df": None,
    "insights_engine": AIInsightsEngine(),
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Helper: cached data loading / feature engineering
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_engineer(filepath: str) -> tuple:
    """Load CSV and produce feature-engineered splits (cached)."""
    fe = FeatureEngineer()
    df_raw = fe.load_data(filepath)
    df_feat = fe.create_features(df_raw)
    X_train, X_test, y_train, y_test, feature_names = fe.prepare_for_modeling(df_feat)
    return df_raw, df_feat, X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.button("Use Sample Data", type="primary")

    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "..", DATA_PATH)
        if not os.path.exists(sample_path):
            with st.spinner("Generating sample data…"):
                generate_financial_data(save=True)
        with st.spinner("Loading data…"):
            result = load_and_engineer(sample_path)
        (
            st.session_state.df_raw,
            st.session_state.df_features,
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test,
            st.session_state.feature_names,
        ) = result
        # Reset downstream state when new data is loaded
        st.session_state.forecaster = None
        st.session_state.metrics = None
        st.session_state.forecast_df = None
        st.session_state.scenario_df = None
        st.success("Sample data loaded!")

    if uploaded_file is not None:
        tmp_path = "/tmp/uploaded_financial.csv"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Loading uploaded data…"):
            result = load_and_engineer(tmp_path)
        (
            st.session_state.df_raw,
            st.session_state.df_features,
            st.session_state.X_train,
            st.session_state.X_test,
            st.session_state.y_train,
            st.session_state.y_test,
            st.session_state.feature_names,
        ) = result
        st.session_state.forecaster = None
        st.session_state.metrics = None
        st.session_state.forecast_df = None
        st.session_state.scenario_df = None
        st.success("File uploaded and processed!")

    st.divider()
    st.header("Model Parameters")
    n_estimators = st.slider("Estimators (XGB & RF)", 50, 300, 100, 50)
    max_depth_xgb = st.slider("XGB Max Depth", 2, 8, 4)
    max_depth_rf = st.slider("RF Max Depth", 2, 12, 6)

    st.divider()
    st.caption("Set ANTHROPIC_API_KEY to enable AI insights.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Overview", "Forecasting", "Scenario Simulation", "AI Insights"]
)

# ==========================================================================
# TAB 1 — Data Overview
# ==========================================================================
with tab1:
    if st.session_state.df_raw is None:
        st.info("Load data from the sidebar to get started.")
    else:
        df = st.session_state.df_raw

        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
        col3.metric("Avg Profit", f"${df['profit'].mean():,.0f}")
        col4.metric(
            "Profit Margin",
            f"{(df['profit'] / df['revenue']).mean() * 100:.1f}%",
        )

        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        st.subheader("Revenue & Profit Over Time")
        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=df["date"], y=df["revenue"],
                name="Revenue", line=dict(color="#2196F3", width=2),
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=df["date"], y=df["profit"],
                name="Profit", line=dict(color="#4CAF50", width=2),
            )
        )
        fig_ts.update_layout(
            xaxis_title="Date", yaxis_title="USD ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_mkt = px.bar(
                df, x="date", y="marketing_spend",
                title="Monthly Marketing Spend",
                color="marketing_spend",
                color_continuous_scale="Blues",
            )
            fig_mkt.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_mkt, use_container_width=True)

        with col_b:
            fig_corr = px.scatter(
                df, x="marketing_spend", y="revenue",
                title="Marketing Spend vs Revenue",
                trendline="ols",
                color="profit",
                color_continuous_scale="RdYlGn",
            )
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================================================
# TAB 2 — Forecasting
# ==========================================================================
with tab2:
    if st.session_state.X_train is None:
        st.info("Load data first (sidebar).")
    else:
        if st.button("Train Model", type="primary"):
            with st.spinner("Training XGBoost + RandomForest ensemble…"):
                from xgboost import XGBRegressor
                from sklearn.ensemble import RandomForestRegressor

                forecaster = FinancialForecaster()
                # Apply sidebar parameters to models
                forecaster.xgb_model = XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth_xgb,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                )
                forecaster.rf_model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth_rf,
                    random_state=42,
                )
                forecaster.train(
                    st.session_state.X_train, st.session_state.y_train
                )
                metrics = forecaster.evaluate(
                    st.session_state.X_test, st.session_state.y_test
                )
                forecast_df = forecaster.forecast_future(
                    st.session_state.X_test.iloc[[-1]].copy()
                )
                st.session_state.forecaster = forecaster
                st.session_state.metrics = metrics
                st.session_state.forecast_df = forecast_df
            st.success("Model trained successfully!")

        if st.session_state.metrics:
            st.subheader("Model Evaluation Metrics")
            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"${m['MAE']:,.2f}")
            c2.metric("RMSE", f"${m['RMSE']:,.2f}")
            c3.metric("R²", f"{m['R2']:.4f}")
            c4.metric("MAPE", f"{m['MAPE']:.2f}%")

            st.subheader("Feature Importance")
            fi_df = st.session_state.forecaster.get_feature_importance(
                st.session_state.feature_names
            )
            fig_fi = px.bar(
                fi_df.head(15), x="importance", y="feature",
                orientation="h", title="Top 15 Feature Importances",
                color="importance", color_continuous_scale="Viridis",
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            st.plotly_chart(fig_fi, use_container_width=True)

            st.subheader("6-Month Revenue Forecast")
            fc = st.session_state.forecast_df
            fig_fc = go.Figure()
            fig_fc.add_trace(
                go.Scatter(
                    x=fc["period"], y=fc["forecast"],
                    name="Forecast", mode="lines+markers",
                    line=dict(color="#2196F3", width=3),
                    marker=dict(size=8),
                )
            )
            fig_fc.add_trace(
                go.Scatter(
                    x=pd.concat([fc["period"], fc["period"].iloc[::-1]]),
                    y=pd.concat([fc["upper_bound"], fc["lower_bound"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(33,150,243,0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Confidence Band",
                )
            )
            fig_fc.update_layout(
                xaxis_title="Period", yaxis_title="Revenue ($)", height=400
            )
            st.plotly_chart(fig_fc, use_container_width=True)
            st.dataframe(fc, use_container_width=True)

# ==========================================================================
# TAB 3 — Scenario Simulation
# ==========================================================================
with tab3:
    if st.session_state.forecaster is None:
        st.info("Train the model first (Forecasting tab).")
    else:
        st.subheader("What-If Scenario Controls")

        col_sliders, col_results = st.columns([1, 2])

        with col_sliders:
            price_delta = st.slider(
                "Price Change (%)", -50, 50, 0, 5,
                help="% change in price_per_unit",
            )
            marketing_delta = st.slider(
                "Marketing Spend Change (%)", -50, 50, 0, 5,
                help="% change in marketing_spend",
            )
            demand_delta = st.slider(
                "Demand / Units Sold Change (%)", -50, 50, 0, 5,
                help="% change in units_sold",
            )

            run_btn = st.button("Run Simulation", type="primary")

        if run_btn:
            df_feat = st.session_state.df_features
            feature_cols = st.session_state.feature_names
            base_data = df_feat[feature_cols].dropna()

            engine = ScenarioEngine(st.session_state.forecaster, base_data)
            scenarios = [
                {
                    "name": f"Price {price_delta:+d}%",
                    "variable": "price_per_unit",
                    "delta_pct": price_delta,
                },
                {
                    "name": f"Marketing {marketing_delta:+d}%",
                    "variable": "marketing_spend",
                    "delta_pct": marketing_delta,
                },
                {
                    "name": f"Demand {demand_delta:+d}%",
                    "variable": "units_sold",
                    "delta_pct": demand_delta,
                },
                {
                    "name": "Baseline (No Change)",
                    "variable": "price_per_unit",
                    "delta_pct": 0,
                },
            ]
            scenario_df = engine.compare_scenarios(scenarios)
            st.session_state.scenario_df = scenario_df

        with col_results:
            if st.session_state.scenario_df is not None:
                sc_df = st.session_state.scenario_df

                st.subheader("Scenario Comparison Results")
                display_df = sc_df[
                    ["scenario", "delta_pct", "baseline_revenue", "new_revenue",
                     "revenue_change", "revenue_change_pct"]
                ].copy()
                display_df.columns = [
                    "Scenario", "Delta %", "Baseline Revenue", "New Revenue",
                    "Revenue Change", "Change %"
                ]
                st.dataframe(display_df.round(2), use_container_width=True)

                fig_sc = px.bar(
                    sc_df,
                    x="scenario",
                    y="revenue_change",
                    color="revenue_change",
                    color_continuous_scale="RdYlGn",
                    title="Revenue Impact by Scenario ($)",
                    labels={"revenue_change": "Revenue Change ($)", "scenario": "Scenario"},
                )
                fig_sc.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_sc, use_container_width=True)

# ==========================================================================
# TAB 4 — AI Insights
# ==========================================================================
with tab4:
    st.subheader("Claude-Powered Business Intelligence")

    if st.session_state.metrics is None:
        st.info("Train the model first (Forecasting tab) to unlock AI insights.")
    else:
        engine_ai: AIInsightsEngine = st.session_state.insights_engine

        col_btn1, col_btn2, col_btn3 = st.columns(3)

        # --- Performance Insights ---
        with col_btn1:
            if st.button("Model Performance Insights", use_container_width=True):
                with st.spinner("Asking Claude for performance insights…"):
                    fi_df = st.session_state.forecaster.get_feature_importance(
                        st.session_state.feature_names
                    )
                    text = engine_ai.generate_performance_insights(
                        st.session_state.metrics, fi_df
                    )
                st.markdown("### Model Performance Analysis")
                st.markdown(text)

        # --- Scenario Recommendations ---
        with col_btn2:
            if st.button("Scenario Recommendations", use_container_width=True):
                if st.session_state.scenario_df is None:
                    st.warning("Run a scenario simulation first.")
                else:
                    with st.spinner("Asking Claude for strategic recommendations…"):
                        text = engine_ai.generate_scenario_recommendations(
                            st.session_state.scenario_df.to_dict()
                        )
                    st.markdown("### Strategic Scenario Recommendations")
                    st.markdown(text)

        # --- Executive Summary ---
        with col_btn3:
            if st.button("Executive Summary", use_container_width=True):
                with st.spinner("Generating executive summary…"):
                    df_raw = st.session_state.df_raw
                    fi_df = st.session_state.forecaster.get_feature_importance(
                        st.session_state.feature_names
                    )
                    all_data: dict = {
                        "metrics": st.session_state.metrics,
                        "data_stats": df_raw[["revenue", "profit", "marketing_spend"]]
                        .describe()
                        .round(2)
                        .to_string(),
                        "feature_importance": fi_df.head(5).to_string(index=False),
                    }
                    if st.session_state.forecast_df is not None:
                        all_data["forecast"] = st.session_state.forecast_df.to_string(
                            index=False
                        )
                    if st.session_state.scenario_df is not None:
                        all_data["scenarios"] = st.session_state.scenario_df[
                            ["scenario", "revenue_change", "revenue_change_pct"]
                        ].to_string(index=False)

                    text = engine_ai.generate_executive_summary(all_data)

                st.markdown("### Executive Intelligence Summary")
                st.markdown(text)

        st.divider()
        st.caption(
            "AI insights are generated by Anthropic's Claude (claude-sonnet-4-6). "
            "Ensure ANTHROPIC_API_KEY is set in your environment."
        )
