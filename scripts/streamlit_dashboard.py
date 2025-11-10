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
# Ensure the script can find sibling modules
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
        if 'classical_preds' not in st.session_state:
            st.session_state.classical_preds = pd.DataFrame()
        if 'qml_preds' not in st.session_state:
            st.session_state.qml_preds = pd.DataFrame()
        if 'insights_generated' not in st.session_state:
            st.session_state.insights_generated = False

    def run_analysis(self):
        """
        Performs the full forecasting pipeline for both classical and QML models.
        """
        try:
            with st.spinner("Loading and preparing data for 2025..."):
                df_2025 = get_2025_data()

            with st.spinner("Running classical models to generate 2025 forecast..."):
                st.session_state.classical_preds = run_classical_predictions(df_2025)

            with st.spinner("Running QML model to generate 2025 forecast..."):
                st.session_state.qml_preds = run_qml_predictions(df_2025)
                
            st.session_state.insights_generated = True
            st.success("Forecasts and insights generated successfully!")
        
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            import traceback
            st.exception(traceback.format_exc())

    def display_executive_summary(self):
        st.markdown("<h3 class='section-header'>Executive Summary</h3>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Classical Model Insights", "Quantum-Hybrid Model Insights"])

        with tab1:
            if not st.session_state.classical_preds.empty:
                generate_agent_report(st.session_state.classical_preds, "Classical")
            else:
                st.warning("Classical model predictions are not available. Click 'Generate Forecast' to run.")
        
        with tab2:
            if not st.session_state.qml_preds.empty:
                generate_agent_report(st.session_state.qml_preds, "Quantum-Hybrid")
            else:
                st.warning("Quantum-Hybrid model predictions are not available. Click 'Generate Forecast' to run.")

    def display_forecast_visualizations(self):
        st.markdown("<h3 class='section-header'>Forecast Visualizations (2025)</h3>", unsafe_allow_html=True)
        if st.session_state.classical_preds.empty and st.session_state.qml_preds.empty:
            return

        classical_agg = st.session_state.classical_preds.groupby('Date')['Predicted_Sales'].sum().reset_index()
        qml_agg = st.session_state.qml_preds.groupby('Date')['Predicted_Sales'].sum().reset_index()

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
        """Fetch live data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get predictions
            predictions_df = pd.read_sql_query("""
                SELECT * FROM live_predictions 
                ORDER BY timestamp DESC 
                LIMIT 500
            """, conn)
            
            # Get performance metrics (if table exists)
            metrics_df = pd.DataFrame()
            try:
                metrics_df = pd.read_sql_query("""
                    SELECT * FROM performance_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """, conn)
            except sqlite3.OperationalError:
                pass # Table might not exist, which is fine
            
            conn.close()
            
            return predictions_df, metrics_df
            
        except Exception as e:
            st.error(f"Error loading data from database ({self.db_path}): {e}")
            return pd.DataFrame(), pd.DataFrame()
    
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
        
        # Sort by timestamp
        predictions_df = predictions_df.sort_values('timestamp')
        
        # Limit to last 100 predictions for better visualization
        recent_data = predictions_df.tail(100)
        
        fig = go.Figure()
        
        # Add actual sales
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['actual_sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add predicted sales
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
        
        # Calculate MAE by category
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
        
        # Create state-category performance matrix
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
        """Show recent predictions table"""
        if len(predictions_df) == 0:
            st.warning("No recent predictions available")
            return
        
        # Show last 10 predictions
        recent = predictions_df.head(10)[['timestamp', 'state', 'vehicle_category', 
                                         'actual_sales', 'predicted_sales', 'error', 
                                         'model_confidence']].copy()
        
        # Format columns
        recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%H:%M:%S')
        recent['actual_sales'] = recent['actual_sales'].round(0).astype(int)
        recent['predicted_sales'] = recent['predicted_sales'].round(0).astype(int)
        recent['error'] = recent['error'].round(2)
        recent['model_confidence'] = recent['model_confidence'].round(3)
        
        st.dataframe(recent, width='stretch')
    
    def create_performance_timeline(self, metrics_df):
        """Create performance metrics timeline"""
        if len(metrics_df) == 0:
            st.warning("No metrics data available")
            return
        
        # Filter for overall MAE
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

    # --- Initialize Dashboards ---
    live_dashboard = LiveDashboard()
    insights_dashboard = InsightsDashboard()
    
    # --- Sidebar Controls ---
    st.sidebar.title("Dashboard Controls")
    
    # 1. Automatic Analysis Run / Rerun Button
    # Run the forecast once initially or on Rerun button click
    if st.sidebar.button("Generate 2025 Forecast & Insights", key="initial_run_button", 
                         help="Run all models (Classical and Quantum) for the 2025 forecast."):
        insights_dashboard.run_analysis()
    
    # Live data refresh controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Live Simulation Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh Live Data", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 2)
    st.sidebar.markdown("**Status:** <span class='status-running'>🟢 Live</span>", unsafe_allow_html=True)

    # --- MAIN CONTENT: ALL SECTIONS STACKED ---
    
    # =================================================================
    # SECTION 1: STRATEGIC INSIGHTS (Agent Reports & Forecast Visuals)
    # =================================================================
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Strategic Insights: 2025 Forecast & Executive Summary</h2>", unsafe_allow_html=True)
    
    # 1a. Executive Summaries (Classical vs. Quantum-Hybrid Agents)
    if st.session_state.get('insights_generated', False):
        col_classical, col_quantum = st.columns(2)
        
        with col_classical:
            st.markdown("### 🤖 Classical Model Insights")
            if not st.session_state.classical_preds.empty:
                generate_agent_report(st.session_state.classical_preds, "Classical")
            else:
                st.warning("Classical predictions unavailable. Run the forecast.")
                
        with col_quantum:
            st.markdown("### ✨ Quantum-Hybrid Model Insights")
            if not st.session_state.qml_preds.empty:
                generate_agent_report(st.session_state.qml_preds, "Quantum-Hybrid")
            else:
                st.warning("Quantum-Hybrid predictions unavailable. Run the forecast.")
    else:
        st.info("Click the 'Generate 2025 Forecast & Insights' button in the sidebar to run the models and populate this section.")

    # 1b. Forecast Visualizations
    if st.session_state.get('insights_generated', False):
        insights_dashboard.display_forecast_visualizations()
        insights_dashboard.display_data_explorer() # Includes Download button

    # =================================================================
    # SECTION 2: LIVE SIMULATION MONITORING (Real-time Data)
    # =================================================================
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Live Simulation Monitoring (Real-time Performance)</h2>", unsafe_allow_html=True)
    
    predictions_df, metrics_df = live_dashboard.get_live_data()

    if not predictions_df.empty:
        # Row 1: Metrics
        st.markdown("### 📊 Key Performance Metrics")
        live_dashboard.create_metrics_cards(predictions_df, metrics_df)
        
        # Row 2: Time Series & Error Distribution (Side-by-side)
        st.markdown("### 📈 Real-time Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            live_dashboard.create_time_series_chart(predictions_df)
        with col2:
            live_dashboard.create_error_distribution(predictions_df)
        
        # Row 3: Category Performance & Performance Timeline (Side-by-side)
        st.markdown("### ⏱️ Performance Breakdown")
        col3, col4 = st.columns(2)
        with col3:
            live_dashboard.create_category_performance(predictions_df)
        with col4:
            live_dashboard.create_performance_timeline(metrics_df)
        
        # Row 4: Geographic Heatmap (Full Width)
        st.markdown("### 🗺️ Geographic Performance")
        live_dashboard.create_state_heatmap(predictions_df)
        
        # Row 5: Recent Predictions Table (Full Width)
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
        # Load data once and cache it
        @st.cache_data
        def get_dropdown_data():
            df = pd.read_csv(DATA_PATH)
            # Ensure columns exist before calling unique()
            states = sorted(df['State'].unique()) if 'State' in df.columns else []
            categories = sorted(df['Vehicle_Category'].unique()) if 'Vehicle_Category' in df.columns else []
            return states, categories

        states, categories = get_dropdown_data()

        # Use safe index access or default values
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
                st.error(f"Could not generate forecast. Please check the logs.")

    except Exception as e:
        st.error(f"An error occurred while setting up the on-demand forecast: {e}")
        
    # --- Auto-refresh logic for live data ---
    if not predictions_df.empty and auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()