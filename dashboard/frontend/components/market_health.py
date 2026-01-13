"""
Market Health Monitoring Component
At-a-glance KPI cards and health indicators
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def compute_market_health_score(predictions_df: pd.DataFrame,
                                metrics: Optional[Dict[str, float]] = None) -> float:
    """
    Compute overall market health score (0-100).
    
    Args:
        predictions_df: DataFrame with prediction data
        metrics: Optional dictionary with additional metrics
        
    Returns:
        Health score (0-100)
    """
    if predictions_df.empty:
        return 50.0
    
    score_components = []
    
    # Component 1: Growth trend (30%)
    if 'Date' in predictions_df.columns and len(predictions_df) > 14:
        recent = predictions_df.tail(7)['Predicted_Sales'].mean()
        older = predictions_df.head(7)['Predicted_Sales'].mean()
        
        if older > 0:
            growth = ((recent - older) / older) * 100
            # Map -10% to 0, 0% to 50, +10% to 100
            growth_score = 50 + (growth * 5)
            growth_score = max(0, min(100, growth_score))
            score_components.append(('growth', growth_score, 0.30))
    
    # Component 2: Volatility (20%) - lower is better
    if 'Predicted_Sales' in predictions_df.columns:
        if 'Date' in predictions_df.columns:
            daily_sales = predictions_df.groupby('Date')['Predicted_Sales'].sum()
            cv = (daily_sales.std() / daily_sales.mean()) * 100 if daily_sales.mean() > 0 else 100
        else:
            cv = (predictions_df['Predicted_Sales'].std() / predictions_df['Predicted_Sales'].mean()) * 100
        
        # Map CV: 0% = 100, 50% = 50, 100% = 0
        volatility_score = max(0, 100 - cv)
        score_components.append(('volatility', volatility_score, 0.20))
    
    # Component 3: Market coverage (20%)
    coverage_score = 70  # Default
    if 'State' in predictions_df.columns:
        num_states = predictions_df['State'].nunique()
        # Assume 36 states in India
        coverage_score = min(100, (num_states / 36) * 100)
    score_components.append(('coverage', coverage_score, 0.20))
    
    # Component 4: Diversity (15%)
    diversity_score = 70  # Default
    if 'Vehicle_Category' in predictions_df.columns:
        sales_by_cat = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum()
        # Shannon diversity index
        total = sales_by_cat.sum()
        proportions = sales_by_cat / total
        entropy = -np.sum(proportions * np.log(proportions + 1e-10))
        max_entropy = np.log(len(sales_by_cat))
        diversity_score = (entropy / max_entropy * 100) if max_entropy > 0 else 50
    score_components.append(('diversity', diversity_score, 0.15))
    
    # Component 5: Additional metrics (15%)
    if metrics:
        r2 = metrics.get('r2', 0.7)
        mae = metrics.get('mae', 100)
        
        # R2 score: 0.9+ = 100, 0.7 = 70, 0.5 = 50
        r2_score = max(0, min(100, r2 * 100))
        score_components.append(('accuracy', r2_score, 0.15))
    else:
        score_components.append(('accuracy', 70, 0.15))
    
    # Calculate weighted average
    total_score = sum(score * weight for _, score, weight in score_components)
    
    return round(total_score, 1)


def create_health_gauge(health_score: float) -> go.Figure:
    """
    Create health score gauge visualization.
    
    Args:
        health_score: Health score (0-100)
        
    Returns:
        Plotly figure object
    """
    # Determine color
    if health_score >= 75:
        color = "green"
    elif health_score >= 50:
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Health Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [50, 75], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [75, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
            ]
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_kpi_card(title: str, value: str, delta: Optional[str] = None,
                   delta_color: str = "normal") -> str:
    """
    Create HTML for a KPI card.
    
    Args:
        title: KPI title
        value: KPI value
        delta: Optional change indicator
        delta_color: Color for delta (normal, inverse, off)
        
    Returns:
        HTML string
    """
    delta_html = ""
    if delta:
        color = "green" if delta_color == "normal" and delta.startswith("+") else "red"
        delta_html = f'<div style="font-size: 14px; color: {color};">{delta}</div>'
    
    html = f"""
    <div style="
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4A90E2;
        margin: 5px 0;
    ">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 24px; font-weight: bold; color: #333;">{value}</div>
        {delta_html}
    </div>
    """
    return html


def detect_alerts(predictions_df: pd.DataFrame,
                 metrics: Optional[Dict[str, float]] = None) -> List[Dict[str, str]]:
    """
    Detect significant changes or alerts.
    
    Args:
        predictions_df: DataFrame with prediction data
        metrics: Optional metrics dictionary
        
    Returns:
        List of alert dictionaries
    """
    alerts = []
    
    if predictions_df.empty:
        return alerts
    
    # Alert 1: Significant growth
    if 'Date' in predictions_df.columns and len(predictions_df) > 14:
        recent = predictions_df.tail(7)['Predicted_Sales'].mean()
        older = predictions_df.head(7)['Predicted_Sales'].mean()
        
        if older > 0:
            growth = ((recent - older) / older) * 100
            
            if growth > 20:
                alerts.append({
                    'type': 'success',
                    'title': 'Strong Growth Detected',
                    'message': f'Sales increased by {growth:.1f}% in recent period'
                })
            elif growth < -20:
                alerts.append({
                    'type': 'warning',
                    'title': 'Sales Decline Alert',
                    'message': f'Sales decreased by {abs(growth):.1f}% in recent period'
                })
    
    # Alert 2: High volatility
    if 'Date' in predictions_df.columns:
        daily_sales = predictions_df.groupby('Date')['Predicted_Sales'].sum()
        cv = (daily_sales.std() / daily_sales.mean()) * 100 if daily_sales.mean() > 0 else 0
        
        if cv > 50:
            alerts.append({
                'type': 'warning',
                'title': 'High Market Volatility',
                'message': f'Coefficient of variation: {cv:.1f}%'
            })
    
    # Alert 3: Category concentration
    if 'Vehicle_Category' in predictions_df.columns:
        sales_by_cat = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum()
        top_cat_share = (sales_by_cat.max() / sales_by_cat.sum() * 100)
        
        if top_cat_share > 60:
            top_cat = sales_by_cat.idxmax()
            alerts.append({
                'type': 'info',
                'title': 'Market Concentration',
                'message': f'{top_cat} dominates with {top_cat_share:.1f}% market share'
            })
    
    return alerts


def render_market_health_dashboard(predictions_df: pd.DataFrame,
                                   metrics: Optional[Dict[str, float]] = None,
                                   title: str = "Market Health Monitoring"):
    """
    Render complete market health dashboard section.
    
    Args:
        predictions_df: DataFrame with prediction data
        metrics: Optional metrics dictionary
        title: Dashboard section title
    """
    st.markdown(f"### {title}")
    
    if predictions_df.empty:
        st.warning("No data available for market health analysis")
        return
    
    # Compute health score
    health_score = compute_market_health_score(predictions_df, metrics)
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Health gauge
        health_fig = create_health_gauge(health_score)
        st.plotly_chart(health_fig, use_container_width=True)
        
        # Health interpretation
        if health_score >= 75:
            st.success("🟢 **Excellent Market Health**")
        elif health_score >= 50:
            st.warning("🟡 **Moderate Market Health**")
        else:
            st.error("🔴 **Weak Market Health**")
    
    with col2:
        # KPI Cards
        st.markdown("#### Key Performance Indicators")
        
        kpi_col1, kpi_col2 = st.columns(2)
        
        with kpi_col1:
            # Total sales
            total_sales = predictions_df['Predicted_Sales'].sum()
            st.markdown(create_kpi_card(
                "Total Predicted Sales",
                f"{total_sales:,}",
                None
            ), unsafe_allow_html=True)
            
            # Average daily
            if 'Date' in predictions_df.columns:
                avg_daily = predictions_df.groupby('Date')['Predicted_Sales'].sum().mean()
                st.markdown(create_kpi_card(
                    "Avg Daily Sales",
                    f"{avg_daily:,.0f}",
                    None
                ), unsafe_allow_html=True)
        
        with kpi_col2:
            # Market coverage
            if 'State' in predictions_df.columns:
                num_states = predictions_df['State'].nunique()
                st.markdown(create_kpi_card(
                    "States Covered",
                    f"{num_states}",
                    None
                ), unsafe_allow_html=True)
            
            # Categories
            if 'Vehicle_Category' in predictions_df.columns:
                num_cats = predictions_df['Vehicle_Category'].nunique()
                st.markdown(create_kpi_card(
                    "Vehicle Categories",
                    f"{num_cats}",
                    None
                ), unsafe_allow_html=True)
    
    # Alerts section
    st.markdown("#### Market Alerts & Notifications")
    
    alerts = detect_alerts(predictions_df, metrics)
    
    if alerts:
        for alert in alerts:
            if alert['type'] == 'success':
                st.success(f"**{alert['title']}**: {alert['message']}")
            elif alert['type'] == 'warning':
                st.warning(f"**{alert['title']}**: {alert['message']}")
            else:
                st.info(f"**{alert['title']}**: {alert['message']}")
    else:
        st.info("No significant alerts at this time")
    
    # Drill-down section
    with st.expander("📊 Drill-Down Analysis"):
        drill_col1, drill_col2 = st.columns(2)
        
        with drill_col1:
            # Top states
            if 'State' in predictions_df.columns:
                st.markdown("**Top 10 States by Sales**")
                top_states = predictions_df.groupby('State')['Predicted_Sales'].sum().nlargest(10)
                st.bar_chart(top_states)
        
        with drill_col2:
            # Category distribution
            if 'Vehicle_Category' in predictions_df.columns:
                st.markdown("**Sales by Category**")
                cat_dist = predictions_df.groupby('Vehicle_Category')['Predicted_Sales'].sum()
                st.bar_chart(cat_dist)
