"""
Sentiment Gauge Visualization Component
Displays real-time sentiment metrics with visual indicators
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any


def create_sentiment_gauge(value: float, 
                           title: str = "Market Sentiment",
                           min_val: float = 0,
                           max_val: float = 100,
                           thresholds: Optional[Dict[str, float]] = None) -> go.Figure:
    """
    Create a gauge chart for sentiment visualization.
    
    Args:
        value: Current sentiment value
        title: Chart title
        min_val: Minimum value for gauge
        max_val: Maximum value for gauge
        thresholds: Dictionary with 'low' and 'high' threshold values
        
    Returns:
        Plotly figure object
    """
    if thresholds is None:
        thresholds = {'low': 40, 'high': 70}
    
    # Determine color based on value
    if value < thresholds['low']:
        color = "red"
    elif value < thresholds['high']:
        color = "yellow"
    else:
        color = "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': (thresholds['low'] + thresholds['high']) / 2},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, thresholds['low']], 'color': 'rgba(255, 0, 0, 0.2)'},
                {'range': [thresholds['low'], thresholds['high']], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [thresholds['high'], max_val], 'color': 'rgba(0, 255, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "black", 'family': "Arial"}
    )
    
    return fig


def create_sentiment_trend(df: pd.DataFrame,
                           date_col: str = 'Date',
                           sentiment_col: str = 'sentiment',
                           title: str = "Sentiment Trend") -> go.Figure:
    """
    Create a line chart showing sentiment over time.
    
    Args:
        df: DataFrame with sentiment data
        date_col: Name of date column
        sentiment_col: Name of sentiment column
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[sentiment_col],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add threshold lines
    fig.add_hline(y=40, line_dash="dash", line_color="red", 
                  annotation_text="Low Threshold")
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="High Threshold")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def compute_sentiment_from_sales(predictions_df: pd.DataFrame,
                                 actual_col: str = 'EV_Sales_Quantity',
                                 predicted_col: str = 'Predicted_Sales') -> float:
    """
    Compute sentiment score from sales data.
    Higher predicted vs actual = higher sentiment.
    
    Args:
        predictions_df: DataFrame with predictions
        actual_col: Name of actual sales column
        predicted_col: Name of predicted sales column
        
    Returns:
        Sentiment score (0-100)
    """
    if predictions_df.empty:
        return 50.0  # Neutral
    
    # Check if both columns exist
    has_actual = actual_col in predictions_df.columns
    has_predicted = predicted_col in predictions_df.columns
    
    if not has_predicted:
        return 50.0
    
    predicted_total = predictions_df[predicted_col].sum()
    
    if has_actual:
        actual_total = predictions_df[actual_col].sum()
        
        if actual_total > 0:
            # Sentiment based on growth
            growth_rate = ((predicted_total - actual_total) / actual_total) * 100
            
            # Map growth rate to 0-100 scale
            # -20% = 0, 0% = 50, +20% = 100
            sentiment = 50 + (growth_rate * 2.5)
            sentiment = max(0, min(100, sentiment))  # Clamp to 0-100
        else:
            sentiment = 50.0
    else:
        # Use trend if available
        if len(predictions_df) > 1:
            recent = predictions_df.tail(30)[predicted_col].mean()
            older = predictions_df.head(30)[predicted_col].mean()
            
            if older > 0:
                trend = ((recent - older) / older) * 100
                sentiment = 50 + (trend * 2.5)
                sentiment = max(0, min(100, sentiment))
            else:
                sentiment = 50.0
        else:
            sentiment = 50.0
    
    return float(sentiment)


def render_sentiment_dashboard(predictions_df: pd.DataFrame,
                               title: str = "Market Sentiment Analysis"):
    """
    Render complete sentiment dashboard section.
    
    Args:
        predictions_df: DataFrame with prediction data
        title: Dashboard section title
    """
    st.markdown(f"### {title}")
    
    # Compute overall sentiment
    sentiment_score = compute_sentiment_from_sales(predictions_df)
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Sentiment gauge
        gauge_fig = create_sentiment_gauge(
            sentiment_score,
            title="Current Sentiment"
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        # Sentiment interpretation
        if sentiment_score >= 70:
            st.success("🟢 **Strong Positive Sentiment**")
            st.write("Market shows strong growth indicators")
        elif sentiment_score >= 40:
            st.warning("🟡 **Neutral Sentiment**")
            st.write("Market is stable with moderate outlook")
        else:
            st.error("🔴 **Weak Sentiment**")
            st.write("Market shows concerning indicators")
    
    with col2:
        # Sentiment trend over time
        if 'Date' in predictions_df.columns and len(predictions_df) > 5:
            # Aggregate by date for trend
            daily_sentiment = predictions_df.groupby('Date').agg({
                'Predicted_Sales': 'sum'
            }).reset_index()
            
            # Compute rolling sentiment
            daily_sentiment['sentiment'] = daily_sentiment['Predicted_Sales'].pct_change(7).fillna(0) * 500 + 50
            daily_sentiment['sentiment'] = daily_sentiment['sentiment'].clip(0, 100)
            
            trend_fig = create_sentiment_trend(daily_sentiment)
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.info("Insufficient data for sentiment trend visualization")
    
    # Additional metrics
    st.markdown("#### Sentiment Indicators")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total_predicted = predictions_df['Predicted_Sales'].sum()
        st.metric("Total Predicted Sales", f"{total_predicted:,}")
    
    with metric_cols[1]:
        avg_daily = predictions_df.groupby('Date')['Predicted_Sales'].sum().mean() if 'Date' in predictions_df.columns else 0
        st.metric("Avg Daily Sales", f"{avg_daily:,.0f}")
    
    with metric_cols[2]:
        if 'State' in predictions_df.columns:
            top_state = predictions_df.groupby('State')['Predicted_Sales'].sum().idxmax()
            st.metric("Top State", top_state)
        else:
            st.metric("Top State", "N/A")
    
    with metric_cols[3]:
        if 'Vehicle_Category' in predictions_df.columns:
            top_category = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum().idxmax()
            st.metric("Top Category", top_category)
        else:
            st.metric("Top Category", "N/A")
