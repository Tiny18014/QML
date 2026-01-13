"""
Volatility Plot Component
Time-series volatility visualization with interactive features
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List


def compute_volatility(df: pd.DataFrame,
                      value_col: str = 'Predicted_Sales',
                      window: int = 7) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation).
    
    Args:
        df: Input DataFrame
        value_col: Column to compute volatility for
        window: Rolling window size
        
    Returns:
        DataFrame with volatility column
    """
    df = df.copy()
    df = df.sort_values('Date')
    
    # Compute rolling std
    df['volatility'] = df[value_col].rolling(window=window, min_periods=1).std()
    
    # Compute percentage volatility
    df['volatility_pct'] = (df['volatility'] / df[value_col].rolling(window=window, min_periods=1).mean()) * 100
    
    return df


def create_volatility_plot(df: pd.DataFrame,
                           date_col: str = 'Date',
                           volatility_col: str = 'volatility',
                           title: str = "Sales Volatility Over Time") -> go.Figure:
    """
    Create volatility visualization.
    
    Args:
        df: DataFrame with volatility data
        date_col: Name of date column
        volatility_col: Name of volatility column
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add volatility line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[volatility_col],
        mode='lines',
        name='Volatility',
        fill='tozeroy',
        line=dict(color='purple', width=2),
        fillcolor='rgba(128, 0, 128, 0.2)'
    ))
    
    # Add threshold lines if applicable
    if volatility_col in df.columns:
        mean_vol = df[volatility_col].mean()
        std_vol = df[volatility_col].std()
        
        fig.add_hline(y=mean_vol, line_dash="dash", line_color="blue",
                     annotation_text=f"Mean: {mean_vol:.1f}")
        fig.add_hline(y=mean_vol + std_vol, line_dash="dot", line_color="red",
                     annotation_text=f"High: {mean_vol + std_vol:.1f}")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Volatility (Std Dev)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


def create_volatility_heatmap(df: pd.DataFrame,
                              group_by: str = 'State',
                              value_col: str = 'Predicted_Sales') -> go.Figure:
    """
    Create heatmap showing volatility by category.
    
    Args:
        df: DataFrame with sales data
        group_by: Column to group by (State, Vehicle_Category, etc.)
        value_col: Column to analyze
        
    Returns:
        Plotly figure object
    """
    if 'Date' not in df.columns or group_by not in df.columns:
        # Return empty figure
        return go.Figure()
    
    # Compute volatility by group and date
    df_grouped = df.groupby([pd.Grouper(key='Date', freq='W'), group_by])[value_col].std().reset_index()
    df_grouped.columns = ['Date', group_by, 'volatility']
    
    # Pivot for heatmap
    df_pivot = df_grouped.pivot(index=group_by, columns='Date', values='volatility')
    
    # Get top groups by average volatility
    top_groups = df_pivot.mean(axis=1).nlargest(10).index
    df_pivot = df_pivot.loc[top_groups]
    
    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale='RdYlGn_r',
        colorbar=dict(title="Volatility")
    ))
    
    fig.update_layout(
        title=f"Volatility Heatmap by {group_by}",
        xaxis_title="Date",
        yaxis_title=group_by,
        height=400
    )
    
    return fig


def analyze_volatility_events(df: pd.DataFrame,
                              volatility_col: str = 'volatility',
                              threshold: Optional[float] = None) -> pd.DataFrame:
    """
    Identify high volatility events.
    
    Args:
        df: DataFrame with volatility data
        volatility_col: Name of volatility column
        threshold: Volatility threshold (default: mean + 2*std)
        
    Returns:
        DataFrame with high volatility events
    """
    if volatility_col not in df.columns:
        return pd.DataFrame()
    
    if threshold is None:
        mean_vol = df[volatility_col].mean()
        std_vol = df[volatility_col].std()
        threshold = mean_vol + 2 * std_vol
    
    # Find events above threshold
    high_vol = df[df[volatility_col] > threshold].copy()
    
    if 'Date' in high_vol.columns:
        high_vol = high_vol.sort_values('Date', ascending=False)
    
    return high_vol


def render_volatility_dashboard(predictions_df: pd.DataFrame,
                                title: str = "Market Volatility Analysis"):
    """
    Render complete volatility dashboard section.
    
    Args:
        predictions_df: DataFrame with prediction data
        title: Dashboard section title
    """
    st.markdown(f"### {title}")
    
    if predictions_df.empty or 'Date' not in predictions_df.columns:
        st.warning("No data available for volatility analysis")
        return
    
    # Prepare data
    df_daily = predictions_df.groupby('Date').agg({
        'Predicted_Sales': 'sum'
    }).reset_index()
    
    # Compute volatility
    df_daily = compute_volatility(df_daily, window=7)
    
    # Volatility settings
    col_settings1, col_settings2 = st.columns(2)
    
    with col_settings1:
        window_size = st.slider("Rolling Window (days)", 3, 30, 7, key="vol_window")
    
    with col_settings2:
        view_type = st.selectbox("View Type", 
                                ["Time Series", "Heatmap", "Both"],
                                key="vol_view")
    
    # Recompute with selected window
    if window_size != 7:
        df_daily = compute_volatility(df_daily, window=window_size)
    
    # Visualization
    if view_type in ["Time Series", "Both"]:
        vol_fig = create_volatility_plot(df_daily, volatility_col='volatility')
        st.plotly_chart(vol_fig, use_container_width=True)
    
    if view_type in ["Heatmap", "Both"]:
        if 'State' in predictions_df.columns:
            heatmap_fig = create_volatility_heatmap(predictions_df, group_by='State')
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Volatility metrics
    st.markdown("#### Volatility Indicators")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        avg_vol = df_daily['volatility'].mean()
        st.metric("Avg Volatility", f"{avg_vol:,.1f}")
    
    with metric_cols[1]:
        current_vol = df_daily['volatility'].iloc[-1]
        st.metric("Current Volatility", f"{current_vol:,.1f}")
    
    with metric_cols[2]:
        max_vol = df_daily['volatility'].max()
        max_vol_date = df_daily.loc[df_daily['volatility'].idxmax(), 'Date']
        st.metric("Peak Volatility", f"{max_vol:,.1f}",
                 delta=f"{max_vol_date.strftime('%Y-%m-%d')}")
    
    with metric_cols[3]:
        # Volatility trend
        recent_vol = df_daily.tail(7)['volatility'].mean()
        older_vol = df_daily.head(7)['volatility'].mean()
        vol_trend = ((recent_vol - older_vol) / older_vol * 100) if older_vol > 0 else 0
        st.metric("Volatility Trend", f"{vol_trend:+.1f}%")
    
    # High volatility events
    st.markdown("#### High Volatility Events")
    
    high_vol_events = analyze_volatility_events(df_daily)
    
    if not high_vol_events.empty:
        display_cols = ['Date', 'Predicted_Sales', 'volatility']
        display_cols = [c for c in display_cols if c in high_vol_events.columns]
        
        st.dataframe(
            high_vol_events[display_cols].head(10),
            use_container_width=True
        )
    else:
        st.info("No significant volatility events detected")
