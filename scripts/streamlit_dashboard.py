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

# Page configuration
st.set_page_config(
    page_title="EV Sales Live Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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

class LiveDashboard:
    def __init__(self, db_path: str = 'output/live_predictions.db'):
        self.db_path = db_path
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
            
            # Get performance metrics
            metrics_df = pd.read_sql_query("""
                SELECT * FROM performance_metrics 
                ORDER BY timestamp DESC 
                LIMIT 100
            """, conn)
            
            conn.close()
            
            return predictions_df, metrics_df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
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
        
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        st.dataframe(recent, use_container_width=True)
    
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
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🚗 EV Sales Live Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = LiveDashboard()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)
    
    # Manual refresh button
    if st.sidebar.button("Refresh Now"):
        st.rerun()
    
    # Status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:** <span class='status-running'>🟢 Live</span>", unsafe_allow_html=True)
    
    # Get live data
    predictions_df, metrics_df = dashboard.get_live_data()
    
    # Main dashboard
    if len(predictions_df) > 0:
        # Metrics cards
        st.markdown("## 📊 Live Metrics")
        dashboard.create_metrics_cards(predictions_df, metrics_df)
        
        # Charts section
        st.markdown("## 📈 Real-time Visualizations")
        
        # Time series chart
        col1, col2 = st.columns(2)
        with col1:
            dashboard.create_time_series_chart(predictions_df)
        with col2:
            dashboard.create_error_distribution(predictions_df)
        
        # Category and state performance
        col3, col4 = st.columns(2)
        with col3:
            dashboard.create_category_performance(predictions_df)
        with col4:
            dashboard.create_performance_timeline(metrics_df)
        
        # State heatmap
        st.markdown("## 🗺️ Geographic Performance")
        dashboard.create_state_heatmap(predictions_df)
        
        # Recent predictions
        st.markdown("## 🔄 Recent Predictions")
        dashboard.show_recent_predictions(predictions_df)
        
        # Data export
        st.markdown("## 📥 Export Data")
        if st.button("Download Predictions CSV"):
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"live_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
    
    else:
        st.warning("No live data available. Make sure the simulation is running!")
        st.markdown("""
        ### 🚀 To start the simulation:
        1. Run the live simulation script: `python live_simulation.py`
        2. Choose your preferred delay settings
        3. Come back to this dashboard to see live predictions
        """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()