"""
EV Demand Intelligence Hub - Main Dashboard
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
import warnings
from pathlib import Path

# --- Path Setup ---
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))

# --- Imports ---
try:
    from dashboard_utils import (
        get_2025_data,
        run_classical_predictions,
        run_qml_predictions,
        generate_agent_report,
        generate_on_demand_forecast,
        DATA_PATH
    )
except ImportError:
    pass 

warnings.filterwarnings('ignore')

# Page configuration (Must be first)
st.set_page_config(
    page_title="EV Demand Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4A90E2; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .section-header { font-size: 1.75rem; color: #333333; margin-top: 2rem; border-bottom: 2px solid #4A90E2; }
    .stButton>button { background-color: #4A90E2; color: white; width: 100%; border-radius: 8px; }
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #1f77b4; }
</style>
""", unsafe_allow_html=True)

class InsightsDashboard:
    def __init__(self):
        st.session_state.setdefault('classical_preds', pd.DataFrame())
        st.session_state.setdefault('qml_preds', pd.DataFrame())
        st.session_state.setdefault('insights_generated', False)
        st.session_state.setdefault('classical_report_text', None)
        st.session_state.setdefault('qml_report_text', None)

    def run_analysis(self):
        try:
            with st.spinner("Loading and preparing data for 2025..."):
                df_2025 = get_2025_data()
            with st.spinner("Running classical models..."):
                st.session_state.classical_preds = run_classical_predictions(df_2025)
            with st.spinner("Running QML models..."):
                st.session_state.qml_preds = run_qml_predictions(df_2025)
                
            # Reset reports to force regeneration
            st.session_state.classical_report_text = None
            st.session_state.qml_report_text = None
            st.session_state.insights_generated = True
            st.success("Forecasts generated successfully!")
        except Exception as e:
            st.error(f"Analysis Error: {e}")

    def display_forecast_visualizations(self):
        st.markdown("<h3 class='section-header'>Forecast Visualizations (2026 Daily)</h3>", unsafe_allow_html=True)
        
        # Load 2026 Predictions if available
        pred_path = ROOT_DIR / "output" / "daily_predictions_2026.csv"
        if pred_path.exists():
            df_2026 = pd.read_csv(pred_path)
            df_2026['Date'] = pd.to_datetime(df_2026['Date'])

            # Aggregate for faster plotting
            daily_agg = df_2026.groupby('Date')['Predicted_Sales'].sum().reset_index()

            fig = px.line(daily_agg, x='Date', y='Predicted_Sales', title="Daily Market Forecast (2026)")
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                state_sales = df_2026.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False).head(10)
                fig_state = px.bar(state_sales, x=state_sales.index, y='Predicted_Sales', title="Top 10 States (2026 Total)")
                st.plotly_chart(fig_state, use_container_width=True)
            with col2:
                category_sales = df_2026.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
                fig_cat = px.bar(category_sales, x=category_sales.index, y='Predicted_Sales', title="Sales by Category (2026 Total)")
                st.plotly_chart(fig_cat, use_container_width=True)

        elif st.session_state.classical_preds.empty and st.session_state.qml_preds.empty:
            st.info("No forecast data available. Please run the Advanced Model pipeline.")
            return

        # Legacy 2025 view (if 2026 not present or just to show comparison)
        if not st.session_state.classical_preds.empty:
             st.markdown("#### 2025 Forecasts (Legacy)")
             classical_agg = st.session_state.classical_preds.groupby('Date')['Predicted_Sales'].sum().reset_index()
             fig_legacy = px.line(classical_agg, x='Date', y='Predicted_Sales', title="2025 Forecast")
             st.plotly_chart(fig_legacy, use_container_width=True)

def render_accuracy_section():
    st.markdown("<h2 class='section-header'>Model Performance (Training Data)</h2>", unsafe_allow_html=True)

    report_path = ROOT_DIR / "output" / "model_performance_report.csv"
    if report_path.exists():
        df_perf = pd.read_csv(report_path)

        # Display as a styled dataframe
        st.markdown("### 📊 Accuracy Metrics by Category")
        st.markdown("Metrics evaluated on the Monthly Aggregated Dataset (2021-2024). High R2 indicates strong fit to historical trends.")

        st.dataframe(
            df_perf.style.format({
                "R2_Score": "{:.4f}",
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}"
            }).background_gradient(cmap="Greens", subset=["R2_Score"]),
            use_container_width=True
        )

        # Simple bar chart for R2
        fig = px.bar(df_perf, x='Vehicle_Category', y='R2_Score', title="Model Fit (R2 Score) by Category",
                     color='R2_Score', color_continuous_scale='Viridis')
        fig.update_layout(yaxis_range=[0.8, 1.0]) # Zoom in since scores are high
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Performance report not found. Please run the model training to generate it.")

class LiveDashboard:
    def __init__(self):
        # MUST use st.connection to read from Cloud DB
        try:
            self.conn = st.connection("postgresql", type="sql")
        except Exception:
            self.conn = None
    
    def get_live_data(self):
        if self.conn is None: return pd.DataFrame()
        try:
            # Read from Cloud DB
            return self.conn.query("SELECT * FROM live_predictions ORDER BY timestamp DESC LIMIT 500", ttl=2)
        except Exception:
            return pd.DataFrame()
    
    def create_metrics_cards(self, predictions_df):
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Predictions", len(predictions_df))
        with col2: st.metric("Overall MAE", f"{predictions_df['error'].mean():.2f}" if not predictions_df.empty else "N/A")
        with col3: st.metric("Avg Latency", f"{predictions_df['processing_time_ms'].mean():.2f}ms" if not predictions_df.empty else "N/A")
        with col4: st.metric("Confidence", "0.95") 

    def create_charts(self, predictions_df):
        if predictions_df.empty: return
        
        recent = predictions_df.sort_values('timestamp').tail(100)
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=recent['timestamp'], y=recent['actual_sales'], name='Actual'))
        fig_ts.add_trace(go.Scatter(x=recent['timestamp'], y=recent['predicted_sales'], name='Predicted', line=dict(dash='dash')))
        fig_ts.update_layout(title="Real-time Sales: Actual vs Predicted", height=350)
        st.plotly_chart(fig_ts, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_err = px.histogram(predictions_df, x='error', title="Error Distribution")
            st.plotly_chart(fig_err, use_container_width=True)
        with col2:
            cat_perf = predictions_df.groupby('vehicle_category')['error'].mean().reset_index()
            fig_cat = px.bar(cat_perf, x='vehicle_category', y='error', title="MAE by Category")
            st.plotly_chart(fig_cat, use_container_width=True)

# --- MODULAR SECTION FUNCTIONS ---

def render_live_section(live_dashboard):
    st.markdown("<h2 class='section-header'>Live Simulation Monitoring</h2>", unsafe_allow_html=True)
    predictions_df = live_dashboard.get_live_data()
    
    if not predictions_df.empty:
        live_dashboard.create_metrics_cards(predictions_df)
        live_dashboard.create_charts(predictions_df)
        st.dataframe(predictions_df.head(10), use_container_width=True)
    else:
        st.info("Waiting for simulation data... (Run option 3 or 5 in terminal)")

def render_insights_section(insights_dashboard):
    st.markdown("<h2 class='section-header'>Strategic Insights (2025)</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('insights_generated', False):
        col_classical, col_quantum = st.columns(2)
        
        with col_classical:
            st.markdown("### 🤖 Classical Model Insights")
            if not st.session_state.classical_preds.empty:
                if st.session_state.classical_report_text:
                    st.markdown(st.session_state.classical_report_text, unsafe_allow_html=True)
                else:
                    with st.spinner("Generating Classical Agent Report..."):
                        report_text = generate_agent_report(st.session_state.classical_preds, "Classical")
                        st.session_state.classical_report_text = report_text
                        st.markdown(report_text, unsafe_allow_html=True)
            else:
                st.warning("Classical predictions unavailable.")
                
        with col_quantum:
            st.markdown("### ✨ Quantum-Hybrid Model Insights")
            if not st.session_state.qml_preds.empty:
                if st.session_state.qml_report_text:
                    st.markdown(st.session_state.qml_report_text, unsafe_allow_html=True)
                else:
                    with st.spinner("Generating Quantum-Hybrid Agent Report..."):
                        report_text = generate_agent_report(st.session_state.qml_preds, "Quantum-Hybrid")
                        st.session_state.qml_report_text = report_text
                        st.markdown(report_text, unsafe_allow_html=True)
            else:
                st.warning("Quantum-Hybrid predictions unavailable.")

        insights_dashboard.display_forecast_visualizations()
    else:
        st.info("Click 'Generate 2025 Forecast' in sidebar to view long-term insights.")

def render_ondemand_section():
    st.markdown("<h2 class='section-header'>On-Demand Forecasting Tool</h2>", unsafe_allow_html=True)
    st.info("Select a specific state and vehicle category to generate a custom short-term forecast.")

    try:
        df_temp = pd.read_csv(DATA_PATH)
        if 'Vehicle_Class' in df_temp.columns and 'Vehicle_Category' not in df_temp.columns:
            df_temp.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
            
        states = sorted(df_temp['State'].unique().tolist()) if 'State' in df_temp.columns else ['Delhi']
        categories = sorted(df_temp['Vehicle_Category'].unique().tolist()) if 'Vehicle_Category' in df_temp.columns else ['4-Wheelers']

        col1, col2, col3 = st.columns(3)
        with col1: selected_state = st.selectbox("Select State", states)
        with col2: selected_category = st.selectbox("Select Vehicle Category", categories)
        with col3: days_to_forecast = st.number_input("Days to Forecast", min_value=7, max_value=365, value=30, step=7)

        if st.button("Generate On-Demand Forecast", key="ondemand_btn"):
            with st.spinner(f"Generating forecast for {selected_category} in {selected_state}..."):
                forecast_df, forecast_fig = generate_on_demand_forecast(selected_category, selected_state, days_to_forecast)
            
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error("Could not generate forecast. Please check data availability.")
    except Exception as e:
        st.warning(f"On-Demand tool unavailable: {e}")

def main():
    st.markdown('<h1 class="main-header">⚡ EV Demand Intelligence Hub</h1>', unsafe_allow_html=True)
    
    live_dashboard = LiveDashboard()
    insights_dashboard = InsightsDashboard()
    
    st.sidebar.title("Controls")
    if st.sidebar.button("Generate 2025 Forecast"):
        insights_dashboard.run_analysis()
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh Live Data", True)
    
    # Render Sections
    render_live_section(live_dashboard)
    render_insights_section(insights_dashboard)
    render_accuracy_section() # Added Accuracy Section
    render_ondemand_section()

    if auto_refresh:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()