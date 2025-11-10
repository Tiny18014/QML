"""
EV Demand Intelligence Hub - Main Dashboard
CORRECTED VERSION - Fixed all identified errors
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict
import json
from pathlib import Path
import sys
import warnings

# --- Agent and Model Imports ---
sys.path.append(str(Path(__file__).parent.resolve()))

try:
    from dashboard_utils import (
        get_2025_data,
        run_classical_predictions,
        run_qml_predictions,
        generate_agent_report,
        generate_on_demand_forecast,
        DATA_PATH
    )
except ImportError as e:
    st.error(f"Failed to import from dashboard_utils.py: {e}")
    st.error("Please make sure 'dashboard_utils.py' is in the same 'scripts' folder.")
    st.stop()

warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="EV Demand Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.75rem;
        color: #333333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4A90E2;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        font-size: 1.1rem;
        width: 100%;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        color: white;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class InsightsDashboard:
    """Handles the logic for the Strategic Insights tab."""
    def __init__(self):
        # FIX #2: Use setdefault to avoid race conditions
        st.session_state.setdefault('classical_preds', pd.DataFrame())
        st.session_state.setdefault('qml_preds', pd.DataFrame())
        st.session_state.setdefault('insights_generated', False)
        st.session_state.setdefault('classical_report_text', None)
        st.session_state.setdefault('qml_report_text', None)

    def run_analysis(self):
        """
        Performs the full forecasting pipeline for both classical and QML models.
        Resets the report cache.
        """
        try:
            with st.spinner("Loading and preparing data for 2025..."):
                df_2025 = get_2025_data()

            with st.spinner("Running classical models to generate 2025 forecast..."):
                st.session_state.classical_preds = run_classical_predictions(df_2025)

            with st.spinner("Running QML model to generate 2025 forecast..."):
                st.session_state.qml_preds = run_qml_predictions(df_2025)
                
            # Reset report text to force new LLM call
            st.session_state.classical_report_text = None
            st.session_state.qml_report_text = None
                
            st.session_state.insights_generated = True
            st.success("Forecasts and insights generated successfully! Generating initial reports...")
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            import traceback
            st.exception(traceback.format_exc())

    def display_forecast_visualizations(self):
        st.markdown("<h3 class='section-header'>Forecast Visualizations (2025)</h3>", unsafe_allow_html=True)
        
        # FIX #10: Safe aggregation with empty checks
        if st.session_state.classical_preds.empty and st.session_state.qml_preds.empty:
            st.info("No forecast data available. Generate a forecast first.")
            return

        classical_agg = pd.DataFrame()
        qml_agg = pd.DataFrame()
        
        if not st.session_state.classical_preds.empty:
            classical_agg = st.session_state.classical_preds.groupby('Date')['Predicted_Sales'].sum().reset_index()
        
        if not st.session_state.qml_preds.empty:
            qml_agg = st.session_state.qml_preds.groupby('Date')['Predicted_Sales'].sum().reset_index()

        if classical_agg.empty and qml_agg.empty:
            st.warning("Aggregation produced no data.")
            return

        fig = px.line(title="Overall Market Forecast (2025)")
        if not classical_agg.empty:
            fig.add_scatter(x=classical_agg['Date'], y=classical_agg['Predicted_Sales'], name="Classical Forecast", mode='lines')
        if not qml_agg.empty:
            fig.add_scatter(x=qml_agg['Date'], y=qml_agg['Predicted_Sales'], name="QML Forecast", mode='lines', line=dict(dash='dash'))
        
        fig.update_layout(xaxis_title="Date", yaxis_title="Total Predicted Sales")
        st.plotly_chart(fig, width='stretch') 

        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.classical_preds.empty:
                state_sales = st.session_state.classical_preds.groupby('State')['Predicted_Sales'].sum().sort_values(ascending=False).head(10)
                fig_state = px.bar(state_sales, x=state_sales.index, y='Predicted_Sales', title="Top 10 States (Classical)", labels={'y': 'Total Sales', 'x': 'State'})
                st.plotly_chart(fig_state, width='stretch')
        with col2:
            if not st.session_state.classical_preds.empty:
                category_sales = st.session_state.classical_preds.groupby('Vehicle_Category')['Predicted_Sales'].sum().sort_values(ascending=False)
                fig_cat = px.bar(category_sales, x=category_sales.index, y='Predicted_Sales', title="Sales by Category (Classical)", labels={'y': 'Total Sales', 'x': 'Category'})
                st.plotly_chart(fig_cat, width='stretch')

    def display_data_explorer(self):
        st.markdown("<h3 class='section-header'>Data Explorer</h3>", unsafe_allow_html=True)
        if st.session_state.classical_preds.empty:
            st.info("No data to explore. Generate a forecast first.")
            return
            
        # Merge classical and QML predictions for comparison
        merged_df = st.session_state.classical_preds[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']].rename(columns={'Predicted_Sales': 'Classical_Predicted_Sales'})
        if not st.session_state.qml_preds.empty:
            qml_df = st.session_state.qml_preds[['Date', 'State', 'Vehicle_Category', 'Predicted_Sales']].rename(columns={'Predicted_Sales': 'QML_Predicted_Sales'})
            merged_df = pd.merge(merged_df, qml_df, on=['Date', 'State', 'Vehicle_Category'], how='outer')
        
        st.dataframe(merged_df, width='stretch')
        if st.button("Download Predictions as CSV"):
            csv = merged_df.to_csv(index=False)
            st.download_button("Download CSV", csv, f"predictions_comparison_{datetime.now().strftime('%Y%m%d')}.csv", 'text/csv')

class LiveDashboard:
    def __init__(self, db_path: str = 'output/live_predictions.db'):
        self.db_path = Path(__file__).parent.parent / db_path
        self.auto_refresh = True
        self.refresh_interval = 2  # seconds
    
    def get_live_data(self):
        """Fetch live data from database with proper connection handling"""
        # FIX #5: Proper connection cleanup with try-finally
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get predictions
            predictions_df = pd.read_sql_query("""
                SELECT * FROM live_predictions 
                ORDER BY timestamp DESC 
                LIMIT 500
            """, conn)
            
            # Get performance metrics
            metrics_df = pd.DataFrame()
            try:
                metrics_df = pd.read_sql_query("""
                    SELECT * FROM performance_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, conn)
            except sqlite3.OperationalError:
                pass  # Table might not exist yet
            
            return predictions_df, metrics_df
            
        except Exception as e:
            st.error(f"Error loading data from database ({self.db_path}): {e}")
            return pd.DataFrame(), pd.DataFrame()
        finally:
            if conn:
                conn.close()
    
    def create_metrics_cards(self, predictions_df, metrics_df):
        """Create metrics cards for the dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_predictions = len(predictions_df)
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            if len(predictions_df) > 0:
                overall_mae = predictions_df['error'].mean()
                st.metric("Overall MAE", f"{overall_mae:.2f}")
            else:
                st.metric("Overall MAE", "N/A")
        
        with col3:
            if len(predictions_df) > 0:
                avg_processing_time = predictions_df['processing_time_ms'].mean()
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}ms")
            else:
                st.metric("Avg Processing Time", "N/A")
        
        with col4:
            if len(predictions_df) > 0:
                avg_confidence = predictions_df['model_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            else:
                st.metric("Avg Confidence", "N/A")
    
    def create_time_series_chart(self, predictions_df):
        """Create time series chart of predictions vs actual"""
        if len(predictions_df) == 0:
            st.warning("No data available for time series chart")
            return
        
        predictions_df = predictions_df.sort_values('timestamp')
        recent_data = predictions_df.tail(100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['actual_sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['predicted_sales'],
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Real-time EV Sales: Actual vs Predicted",
            xaxis_title="Time",
            yaxis_title="Sales Quantity",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def create_error_distribution(self, predictions_df):
        """Create error distribution chart"""
        if len(predictions_df) == 0:
            st.warning("No data available for error distribution")
            return
        
        fig = px.histogram(
            predictions_df, 
            x='error', 
            nbins=30,
            title="Prediction Error Distribution",
            labels={'error': 'Absolute Error', 'count': 'Frequency'}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
    
    def create_category_performance(self, predictions_df):
        """Create category performance chart"""
        if len(predictions_df) == 0:
            st.warning("No data available for category performance")
            return
        
        category_mae = predictions_df.groupby('vehicle_category')['error'].mean().reset_index()
        
        fig = px.bar(
            category_mae,
            x='vehicle_category',
            y='error',
            title="MAE by Vehicle Category",
            labels={'error': 'Mean Absolute Error', 'vehicle_category': 'Vehicle Category'}
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')
    
    def create_state_heatmap(self, predictions_df):
        """Create state performance heatmap"""
        if len(predictions_df) == 0:
            st.warning("No data available for state heatmap")
            return
        
        state_category_error = predictions_df.groupby(['state', 'vehicle_category'])['error'].mean().reset_index()
        pivot_table = state_category_error.pivot(index='state', columns='vehicle_category', values='error')
        
        fig = px.imshow(
            pivot_table,
            title="Prediction Error by State and Vehicle Category",
            color_continuous_scale='RdYlBu_r',
            aspect='auto'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    def show_recent_predictions(self, predictions_df):
        """Show recent predictions table with safe formatting"""
        # FIX #7: Safe timestamp formatting
        if len(predictions_df) == 0:
            st.warning("No recent predictions available")
            return
        
        recent = predictions_df.head(10)[['timestamp', 'state', 'vehicle_category', 
                                         'actual_sales', 'predicted_sales', 'error', 
                                         'model_confidence']].copy()
        
        # Safe timestamp formatting
        try:
            recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')
        except Exception:
            recent['timestamp'] = recent['timestamp'].astype(str)
        
        recent['actual_sales'] = recent['actual_sales'].fillna(0).round(0).astype(int)
        recent['predicted_sales'] = recent['predicted_sales'].fillna(0).round(0).astype(int)
        recent['error'] = recent['error'].fillna(0).round(2)
        recent['model_confidence'] = recent['model_confidence'].fillna(0).round(3)
        
        st.dataframe(recent, width='stretch')
    
    def create_performance_timeline(self, metrics_df):
        """Create performance metrics timeline"""
        if len(metrics_df) == 0:
            st.warning("No metrics data available")
            return
        
        mae_data = metrics_df[metrics_df['metric_name'] == 'overall_mae'].copy()
        
        if len(mae_data) == 0:
            st.warning("No MAE timeline data available")
            return
        
        mae_data = mae_data.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mae_data['timestamp'],
            y=mae_data['metric_value'],
            mode='lines+markers',
            name='Overall MAE',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Time",
            yaxis_title="MAE",
            height=300
        )
        
        st.plotly_chart(fig, width='stretch')

def main():
    st.markdown('<h1 class="main-header">⚡ EV Demand Intelligence Hub</h1>', unsafe_allow_html=True)

    # Initialize Dashboards
    live_dashboard = LiveDashboard()
    insights_dashboard = InsightsDashboard()
    
    # --- Sidebar Controls ---
    st.sidebar.title("Dashboard Controls")
    
    if st.sidebar.button("Generate 2025 Forecast & Insights", key="initial_run_button", 
                         help="Run all models (Classical and Quantum) for the 2025 forecast."):
        insights_dashboard.run_analysis()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live Simulation Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh Live Data", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 2)
    status_placeholder = st.sidebar.empty()
    
    # =================================================================
    # SECTION 1: STRATEGIC INSIGHTS
    # =================================================================
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Strategic Insights: 2025 Forecast & Executive Summary</h2>", unsafe_allow_html=True)
    
    if st.session_state.get('insights_generated', False):
        col_classical, col_quantum = st.columns(2)
        
        with col_classical:
            st.markdown("### 🤖 Classical Model Insights")
            if not st.session_state.classical_preds.empty:
                if st.session_state.classical_report_text:
                    st.markdown(st.session_state.classical_report_text, unsafe_allow_html=True)
                else:
                    with st.spinner("Generating Classical Agent Report (One-time LLM Call)..."):
                        report_text = generate_agent_report(st.session_state.classical_preds, "Classical")
                        st.session_state.classical_report_text = report_text
                        st.markdown(report_text, unsafe_allow_html=True)
            else:
                st.warning("Classical predictions unavailable. Run the forecast.")
                
        with col_quantum:
            st.markdown("### ✨ Quantum-Hybrid Model Insights")
            if not st.session_state.qml_preds.empty:
                if st.session_state.qml_report_text:
                    st.markdown(st.session_state.qml_report_text, unsafe_allow_html=True)
                else:
                    with st.spinner("Generating Quantum-Hybrid Agent Report (One-time LLM Call)..."):
                        report_text = generate_agent_report(st.session_state.qml_preds, "Quantum-Hybrid")
                        st.session_state.qml_report_text = report_text
                        st.markdown(report_text, unsafe_allow_html=True)
            else:
                st.warning("Quantum-Hybrid predictions unavailable. Run the forecast.")
    else:
        st.info("Click the 'Generate 2025 Forecast & Insights' button in the sidebar to run the models and populate this section.")

    if st.session_state.get('insights_generated', False):
        insights_dashboard.display_forecast_visualizations()
        insights_dashboard.display_data_explorer()

    # =================================================================
    # SECTION 2: LIVE SIMULATION MONITORING
    # =================================================================
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Live Simulation Monitoring (Real-time Performance)</h2>", unsafe_allow_html=True)
    
    predictions_df, metrics_df = live_dashboard.get_live_data()

    if not predictions_df.empty:
        st.markdown("### 📊 Key Performance Metrics")
        live_dashboard.create_metrics_cards(predictions_df, metrics_df)
        
        st.markdown("### 📈 Real-time Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            live_dashboard.create_time_series_chart(predictions_df)
        with col2:
            live_dashboard.create_error_distribution(predictions_df)
        
        st.markdown("### ⏱️ Performance Breakdown")
        col3, col4 = st.columns(2)
        with col3:
            live_dashboard.create_category_performance(predictions_df)
        with col4:
            live_dashboard.create_performance_timeline(metrics_df)
        
        st.markdown("### 🗺️ Geographic Performance")
        live_dashboard.create_state_heatmap(predictions_df)
        
        st.markdown("### 🔄 Recent Predictions Table")
        live_dashboard.show_recent_predictions(predictions_df)
    else:
        st.warning("No live data available. Make sure the simulation is running!")
        st.markdown("""
        ### 🚀 To start the simulation:
        1. Go back to your terminal.
        2. Choose option '3', '5', or '6' from the `run_pipeline.py` menu.
        3. Come back to this dashboard to see live predictions.
        """)

    # =================================================================
    # SECTION 3: ON-DEMAND FORECASTING
    # =================================================================
    st.markdown("---")
    st.markdown("<h2 class='section-header'>On-Demand Forecasting Tool</h2>", unsafe_allow_html=True)
    st.info("Select a specific state and vehicle category to generate a custom short-term forecast using the classical models.")

    try:
        # FIX #3: Improved dropdown data loading with better error handling
        @st.cache_data
        def get_dropdown_data():
            df = pd.read_csv(DATA_PATH)
            
            # Standardize column names first
            if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
                df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
            
            # Safe column access with dropna and better defaults
            states = sorted(df['State'].dropna().unique().tolist()) if 'State' in df.columns else ['Delhi']
            categories = sorted(df['Vehicle_Category'].dropna().unique().tolist()) if 'Vehicle_Category' in df.columns else ['4-Wheelers']
            
            return states, categories

        states, categories = get_dropdown_data()

        # FIX #4: Safe default index with empty list check
        if not states or not categories:
            st.error("Unable to load state/category data. Please check your dataset.")
            st.stop()

        default_state_index = states.index("Delhi") if "Delhi" in states else 0
        default_category_index = categories.index("4-Wheelers") if "4-Wheelers" in categories else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_state = st.selectbox("Select State", states, index=default_state_index)
        with col2:
            selected_category = st.selectbox("Select Vehicle Category", categories, index=default_category_index)
        with col3:
            days_to_forecast = st.number_input("Days to Forecast", min_value=7, max_value=365, value=30, step=7)

        if st.button("Generate On-Demand Forecast", key="ondemand_btn"):
            with st.spinner(f"Generating {days_to_forecast}-day forecast for {selected_category} in {selected_state}..."):
                forecast_df, forecast_fig = generate_on_demand_forecast(selected_category, selected_state, days_to_forecast)
            
            if forecast_fig:
                st.plotly_chart(forecast_fig, width='stretch')
                st.dataframe(forecast_df, width='stretch')
            else:
                st.error(f"Could not generate forecast: {forecast_df}")

    except Exception as e:
        st.error(f"An error occurred while setting up the on-demand forecast: {e}")
        import traceback
        st.exception(traceback.format_exc())
        
    # Auto-refresh logic for live data
    if not predictions_df.empty and auto_refresh:
        status_placeholder.markdown("**Status:** <span class='status-running'>🟢 Live</span>", unsafe_allow_html=True)
        time.sleep(refresh_interval)
        st.rerun()
    elif auto_refresh:
        status_placeholder.markdown("**Status:** <span class='status-stopped'>🔴 Waiting for Data</span>", unsafe_allow_html=True)
    else:
        status_placeholder.markdown("**Status:** <span class='status-stopped'>⚫ Disabled</span>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()