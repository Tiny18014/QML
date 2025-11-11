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

# =================================================================
# REPLAY FUNCTION FOR DEPLOYED APP
# =================================================================
def display_live_simulation_replay():
    """
    Replays a pre-run simulation from the database to simulate a live feed.
    This is ideal for deployed environments like Streamlit Community Cloud.
    """
    st.markdown("<h2 class='section-header'>Live Simulation Replay</h2>", unsafe_allow_html=True)
    st.info("This is a dynamic replay of the latest simulation run. To update the data, run the simulation locally and push the `live_predictions.db` file to GitHub.")

    db_path = Path(__file__).parent.parent / 'output/live_predictions.db'

    # --- Initialization and Data Loading ---
    if 'replay_data' not in st.session_state:
        if not db_path.exists():
            st.error("Simulation database (`live_predictions.db`) not found. Please run a simulation locally and commit the database file.")
            return
        try:
            conn = sqlite3.connect(db_path)
            # Load data sorted from oldest to newest for correct replay
            st.session_state.replay_data = pd.read_sql_query("SELECT * FROM live_predictions ORDER BY timestamp ASC", conn)
            conn.close()
            st.session_state.replay_index = 0
            st.session_state.replay_df = pd.DataFrame(columns=st.session_state.replay_data.columns)
        except Exception as e:
            st.error(f"Failed to load simulation data: {e}")
            return

    if st.session_state.replay_data.empty:
        st.warning("The simulation database is empty. No data to replay.")
        return

    # --- Replay Control Buttons ---
    col1, col2, col3 = st.columns([1,1,8])
    with col1:
        if st.button("Play/Pause", key="play_pause"):
            st.session_state.running_replay = not st.session_state.get('running_replay', False)
    with col2:
        if st.button("Reset", key="reset_replay"):
            st.session_state.replay_index = 0
            st.session_state.replay_df = pd.DataFrame(columns=st.session_state.replay_data.columns)
            st.session_state.running_replay = False


    # --- Main Replay Logic ---
    if st.session_state.get('running_replay', False) and st.session_state.replay_index < len(st.session_state.replay_data):
        # Add the next row to the display dataframe
        next_row_index = st.session_state.replay_index
        # Use pd.concat instead of append
        st.session_state.replay_df = pd.concat([st.session_state.replay_df, st.session_state.replay_data.iloc[[next_row_index]]], ignore_index=True)
        st.session_state.replay_index += 1
    
    # --- Display Metrics and Charts ---
    replay_df = st.session_state.replay_df
    
    if replay_df.empty:
        st.info("Press 'Play/Pause' to start the simulation replay.")
    else:
        # --- Metrics ---
        st.markdown("### 📊 Key Performance Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Total Predictions Replayed", f"{len(replay_df)}")
        metric_col2.metric("Overall Mean Absolute Error (MAE)", f"{replay_df['error'].mean():.2f}")
        metric_col3.metric("Avg. Processing Time", f"{replay_df['processing_time_ms'].mean():.2f} ms")

        # --- Chart ---
        st.markdown("### 📈 Replay Trend")
        chart_df = replay_df.sort_values('timestamp')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['actual_sales'], mode='lines', name='Actual Sales', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=chart_df['timestamp'], y=chart_df['predicted_sales'], mode='lines', name='Predicted Sales', line=dict(color='crimson', dash='dot')))
        fig.update_layout(title="Actual vs. Predicted Sales Replay", xaxis_title="Timestamp", yaxis_title="EV Sales Quantity")
        st.plotly_chart(fig, use_container_width=True)

        # --- Data Table ---
        st.markdown("### 🔄 Replayed Predictions Data")
        st.dataframe(replay_df.sort_values('timestamp', ascending=False))

    # --- Auto-refresh loop ---
    if st.session_state.get('running_replay', False):
        if st.session_state.replay_index >= len(st.session_state.replay_data):
            st.success("Replay finished!")
            st.session_state.running_replay = False
        
        time.sleep(0.5) # Adjust speed of replay here
        st.rerun()


def main():
    st.markdown('<h1 class="main-header">⚡ EV Demand Intelligence Hub</h1>', unsafe_allow_html=True)

    PAGES = {
        "On-Demand Forecast": display_on_demand_forecast,
        "Live Simulation Replay": display_live_simulation_replay,
        # Add other pages if you have them, e.g., "Model Performance"
    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page_function = PAGES[selection]
    page_function()

# The display_on_demand_forecast function remains largely the same
def display_on_demand_forecast():
    st.markdown("<h2 class='section-header'>On-Demand Forecasting Tool</h2>", unsafe_allow_html=True)
    st.info("Select a specific state and vehicle category to generate a custom short-term forecast.")

    try:
        @st.cache_data
        def get_dropdown_data():
            df = pd.read_csv(DATA_PATH)
            if 'Vehicle_Class' in df.columns and 'Vehicle_Category' not in df.columns:
                df.rename(columns={'Vehicle_Class': 'Vehicle_Category'}, inplace=True)
            states = sorted(df['State'].dropna().unique().tolist())
            categories = sorted(df['Vehicle_Category'].dropna().unique().tolist())
            return states, categories

        states, categories = get_dropdown_data()

        default_state_index = states.index("Delhi") if "Delhi" in states else 0
        default_category_index = categories.index("4-Wheelers") if "4-Wheelers" in categories else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_state = st.selectbox("Select State", states, index=default_state_index)
        with col2:
            selected_category = st.selectbox("Select Vehicle Category", categories, index=default_category_index)
        with col3:
            days_to_forecast = st.number_input("Days to Forecast", min_value=7, max_value=90, value=30, step=7)

        if st.button("Generate On-Demand Forecast", key="ondemand_btn"):
            with st.spinner(f"Generating {days_to_forecast}-day forecast..."):
                forecast_df, forecast_fig = generate_on_demand_forecast(selected_category, selected_state, days_to_forecast)
            
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error(f"Could not generate forecast: {forecast_df}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
