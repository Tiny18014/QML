"""
Enhanced EV Demand Intelligence Dashboard
Integrates all visualization components with model inference
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))
sys.path.append(str(ROOT_DIR / "dashboard" / "backend"))
sys.path.append(str(ROOT_DIR / "dashboard" / "frontend" / "components"))
sys.path.append(str(ROOT_DIR / "dashboard" / "frontend" / "utils"))

# Import backend services
try:
    from model_loader import ModelLoader
    from inference_engine import InferenceEngine
    from metrics_service import MetricsService
except ImportError as e:
    st.error(f"Backend import error: {e}")
    ModelLoader = None
    InferenceEngine = None
    MetricsService = None

# Import frontend components
try:
    from sentiment_gauge import render_sentiment_dashboard
    from volatility_plot import render_volatility_dashboard
    from trend_lines import render_trend_lines_dashboard
    from market_health import render_market_health_dashboard
    from data_refresh import DataRefreshManager, render_refresh_controls
except ImportError as e:
    st.error(f"Frontend import error: {e}")

# Import existing utilities
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
    st.warning("Could not import dashboard_utils - some features may be limited")
    DATA_PATH = ROOT_DIR / "data" / "EV_Dataset.csv"

# Page configuration
st.set_page_config(
    page_title="EV Demand Intelligence Hub - Enhanced",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        border-bottom: 2px solid #4A90E2; 
        padding-bottom: 0.5rem;
    }
    .stButton>button { 
        background-color: #4A90E2; 
        color: white; 
        width: 100%; 
        border-radius: 8px; 
    }
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid #1f77b4; 
    }
</style>
""", unsafe_allow_html=True)


class EnhancedDashboard:
    """Enhanced dashboard with all visualization components."""
    
    def __init__(self):
        """Initialize dashboard components."""
        # Initialize session state
        if 'predictions_df' not in st.session_state:
            st.session_state.predictions_df = pd.DataFrame()
        
        if 'metrics' not in st.session_state:
            st.session_state.metrics = {}
        
        # Initialize backend services
        models_dir = ROOT_DIR / "models"
        output_dir = ROOT_DIR / "output"
        
        if ModelLoader and InferenceEngine and MetricsService:
            self.model_loader = ModelLoader(str(models_dir))
            self.inference_engine = InferenceEngine(str(models_dir))
            self.metrics_service = MetricsService(str(output_dir), str(models_dir))
        else:
            self.model_loader = None
            self.inference_engine = None
            self.metrics_service = None
        
        # Initialize refresh manager
        self.refresh_manager = DataRefreshManager(
            str(models_dir),
            str(output_dir),
            str(ROOT_DIR / "data"),
            str(ROOT_DIR / "predictions")
        )
    
    def load_predictions(self):
        """Load prediction data from files."""
        # Try to load from refresh manager
        df = self.refresh_manager.get_latest_predictions()
        
        if not df.empty:
            st.session_state.predictions_df = df
            return df
        
        # Fallback: try legacy method
        if 'run_classical_predictions' in globals():
            try:
                with st.spinner("Loading and preparing data..."):
                    df_2025 = get_2025_data()
                with st.spinner("Running predictions..."):
                    df = run_classical_predictions(df_2025)
                st.session_state.predictions_df = df
                return df
            except Exception as e:
                st.error(f"Error loading predictions: {e}")
        
        return pd.DataFrame()
    
    def load_metrics(self):
        """Load model metrics."""
        if self.metrics_service:
            metrics = self.metrics_service.get_metrics_summary()
            st.session_state.metrics = metrics
            return metrics
        return {}
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">⚡ EV Demand Intelligence Hub - Enhanced</h1>', 
                   unsafe_allow_html=True)
        
        # Quick stats
        if not st.session_state.predictions_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            df = st.session_state.predictions_df
            
            with col1:
                total = df['Predicted_Sales'].sum()
                st.metric("Total Predicted Sales", f"{total:,}")
            
            with col2:
                if 'Date' in df.columns:
                    num_days = df['Date'].nunique()
                    st.metric("Forecast Days", f"{num_days}")
            
            with col3:
                if 'State' in df.columns:
                    num_states = df['State'].nunique()
                    st.metric("States Covered", f"{num_states}")
            
            with col4:
                if 'Vehicle_Category' in df.columns:
                    num_categories = df['Vehicle_Category'].nunique()
                    st.metric("Vehicle Categories", f"{num_categories}")
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Data loading
        st.sidebar.markdown("### Data Management")
        
        if st.sidebar.button("🔄 Load Latest Predictions", key="load_pred"):
            with st.spinner("Loading predictions..."):
                self.load_predictions()
            st.sidebar.success("✅ Predictions loaded!")
        
        if st.sidebar.button("📊 Load Model Metrics", key="load_metrics"):
            with st.spinner("Loading metrics..."):
                self.load_metrics()
            st.sidebar.success("✅ Metrics loaded!")
        
        # Refresh controls
        auto_refresh = render_refresh_controls(self.refresh_manager)
        
        return auto_refresh
    
    def render_main_content(self):
        """Render main dashboard content."""
        df = st.session_state.predictions_df
        metrics = st.session_state.metrics
        
        if df.empty:
            st.info("👈 Click 'Load Latest Predictions' in the sidebar to get started")
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview",
            "💹 Sentiment Analysis", 
            "📈 Volatility & Risk",
            "🎯 Trends & Forecasts",
            "🏥 Market Health"
        ])
        
        with tab1:
            self.render_overview_tab(df, metrics)
        
        with tab2:
            render_sentiment_dashboard(df, title="Real-time Sentiment Analysis")
        
        with tab3:
            render_volatility_dashboard(df, title="Market Volatility & Risk Analysis")
        
        with tab4:
            render_trend_lines_dashboard(df, title="Brand-Specific Trends & Forecasts")
        
        with tab5:
            render_market_health_dashboard(df, metrics, title="Market Health Monitoring")
    
    def render_overview_tab(self, df: pd.DataFrame, metrics: Dict):
        """Render overview tab with comprehensive analysis."""
        st.markdown("### 📊 Comprehensive Market Overview")
        
        # Agno Agent Integration
        st.markdown("#### 🤖 AI-Powered Strategic Insights")
        
        col_agent1, col_agent2 = st.columns(2)
        
        with col_agent1:
            st.markdown("##### Classical Model Analysis")
            if 'generate_agent_report' in globals():
                try:
                    report = generate_agent_report(df, "Classical")
                    st.markdown(report, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Agent analysis unavailable: {e}")
            else:
                st.info("Agent integration not available")
        
        with col_agent2:
            st.markdown("##### Market Correlation Analysis")
            st.markdown("""
            **Key Observations:**
            - Sentiment trends correlate with demand fluctuations
            - Seasonal patterns drive category-specific variations
            - Regional disparities indicate market maturity levels
            """)
        
        # Quick visualizations
        st.markdown("#### 📈 Quick Insights")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Daily trend
            if 'Date' in df.columns:
                daily_sales = df.groupby('Date')['Predicted_Sales'].sum().reset_index()
                st.line_chart(daily_sales.set_index('Date'))
                st.caption("Daily Sales Forecast Trend")
        
        with viz_col2:
            # Category distribution
            if 'Vehicle_Category' in df.columns:
                cat_sales = df.groupby('Vehicle_Category')['Predicted_Sales'].sum()
                st.bar_chart(cat_sales)
                st.caption("Sales by Vehicle Category")
        
        # On-Demand Forecasting Tool
        st.markdown("---")
        st.markdown("#### 🎯 On-Demand Forecasting Tool")
        
        if 'generate_on_demand_forecast' in globals():
            self.render_ondemand_section(df)
        else:
            st.info("On-demand forecasting not available")
    
    def render_ondemand_section(self, df: pd.DataFrame):
        """Render on-demand forecasting section."""
        col1, col2, col3 = st.columns(3)
        
        states = sorted(df['State'].unique().tolist()) if 'State' in df.columns else ['Delhi']
        categories = sorted(df['Vehicle_Category'].unique().tolist()) if 'Vehicle_Category' in df.columns else ['4-Wheelers']
        
        with col1:
            selected_state = st.selectbox("Select State", states, key="ondemand_state")
        
        with col2:
            selected_category = st.selectbox("Select Category", categories, key="ondemand_cat")
        
        with col3:
            days = st.number_input("Days to Forecast", min_value=7, max_value=365, value=30, key="ondemand_days")
        
        if st.button("🚀 Generate Custom Forecast", key="ondemand_btn"):
            with st.spinner(f"Generating forecast for {selected_category} in {selected_state}..."):
                forecast_df, forecast_fig = generate_on_demand_forecast(
                    selected_category, selected_state, days
                )
            
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Show summary
                total_forecast = forecast_df['Forecasted_Sales'].sum()
                avg_daily = forecast_df['Forecasted_Sales'].mean()
                
                sum_col1, sum_col2 = st.columns(2)
                with sum_col1:
                    st.metric("Total Forecasted Sales", f"{total_forecast:,}")
                with sum_col2:
                    st.metric("Average Daily Sales", f"{avg_daily:,.0f}")
                
                # Download option
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data",
                    data=csv,
                    file_name=f"forecast_{selected_category}_{selected_state}_{days}d.csv",
                    mime="text/csv"
                )
            else:
                st.error("Could not generate forecast")


def main():
    """Main application entry point."""
    dashboard = EnhancedDashboard()
    
    # Render sidebar
    auto_refresh = dashboard.render_sidebar()
    
    # Render header
    dashboard.render_header()
    
    # Render main content
    dashboard.render_main_content()
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
