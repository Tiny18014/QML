"""
Brand-Specific Trend Lines Component
Multi-brand comparison and forecast visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List


def create_trend_comparison(df: pd.DataFrame,
                            group_col: str = 'Vehicle_Category',
                            date_col: str = 'Date',
                            value_col: str = 'Predicted_Sales',
                            title: str = "Brand Trend Comparison") -> go.Figure:
    """
    Create multi-line chart comparing trends across brands/categories.
    
    Args:
        df: DataFrame with sales data
        group_col: Column to group by (brand, category, etc.)
        date_col: Date column name
        value_col: Value column to plot
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Aggregate by date and group
    df_agg = df.groupby([date_col, group_col])[value_col].sum().reset_index()
    
    fig = px.line(
        df_agg,
        x=date_col,
        y=value_col,
        color=group_col,
        title=title,
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def create_forecast_vs_actual(df: pd.DataFrame,
                              date_col: str = 'Date',
                              actual_col: Optional[str] = 'EV_Sales_Quantity',
                              predicted_col: str = 'Predicted_Sales',
                              category: Optional[str] = None) -> go.Figure:
    """
    Create chart comparing forecast vs actual values.
    
    Args:
        df: DataFrame with data
        date_col: Date column name
        actual_col: Actual values column (if available)
        predicted_col: Predicted values column
        category: Specific category to filter by
        
    Returns:
        Plotly figure object
    """
    if category:
        df = df[df['Vehicle_Category'] == category].copy()
    
    # Aggregate by date
    agg_dict = {predicted_col: 'sum'}
    if actual_col and actual_col in df.columns:
        agg_dict[actual_col] = 'sum'
    
    df_agg = df.groupby(date_col).agg(agg_dict).reset_index()
    
    fig = go.Figure()
    
    # Add actual if available
    if actual_col and actual_col in df_agg.columns:
        fig.add_trace(go.Scatter(
            x=df_agg[date_col],
            y=df_agg[actual_col],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
    
    # Add predicted
    fig.add_trace(go.Scatter(
        x=df_agg[date_col],
        y=df_agg[predicted_col],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    title = f"Forecast vs Actual"
    if category:
        title += f" - {category}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_growth_rate_chart(df: pd.DataFrame,
                             group_col: str = 'Vehicle_Category',
                             value_col: str = 'Predicted_Sales',
                             periods: int = 7) -> go.Figure:
    """
    Create chart showing growth rates by category.
    
    Args:
        df: DataFrame with sales data
        group_col: Column to group by
        value_col: Value column
        periods: Number of periods for growth calculation
        
    Returns:
        Plotly figure object
    """
    # Compute growth rates
    df_sorted = df.sort_values('Date')
    df_sorted['growth_rate'] = df_sorted.groupby(group_col)[value_col].pct_change(periods) * 100
    
    # Get latest growth rate for each category
    latest_growth = df_sorted.groupby(group_col)['growth_rate'].last().reset_index()
    latest_growth = latest_growth.sort_values('growth_rate', ascending=True)
    
    # Create bar chart
    fig = go.Figure(go.Bar(
        x=latest_growth['growth_rate'],
        y=latest_growth[group_col],
        orientation='h',
        marker=dict(
            color=latest_growth['growth_rate'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Growth %")
        )
    ))
    
    fig.update_layout(
        title=f"{periods}-Day Growth Rate by Category",
        xaxis_title="Growth Rate (%)",
        yaxis_title="Category",
        template='plotly_white',
        height=400
    )
    
    return fig


def create_market_share_chart(df: pd.DataFrame,
                              group_col: str = 'Vehicle_Category',
                              value_col: str = 'Predicted_Sales') -> go.Figure:
    """
    Create pie chart showing market share.
    
    Args:
        df: DataFrame with sales data
        group_col: Column to group by
        value_col: Value column
        
    Returns:
        Plotly figure object
    """
    # Aggregate by category
    df_agg = df.groupby(group_col)[value_col].sum().reset_index()
    df_agg = df_agg.sort_values(value_col, ascending=False)
    
    fig = go.Figure(go.Pie(
        labels=df_agg[group_col],
        values=df_agg[value_col],
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Market Share by Category",
        template='plotly_white',
        height=400
    )
    
    return fig


def render_trend_lines_dashboard(predictions_df: pd.DataFrame,
                                 title: str = "Brand-Specific Trends & Forecasts"):
    """
    Render complete trend lines dashboard section.
    
    Args:
        predictions_df: DataFrame with prediction data
        title: Dashboard section title
    """
    st.markdown(f"### {title}")
    
    if predictions_df.empty:
        st.warning("No data available for trend analysis")
        return
    
    # Settings
    col_set1, col_set2, col_set3 = st.columns(3)
    
    with col_set1:
        group_options = ['Vehicle_Category', 'State']
        group_options = [opt for opt in group_options if opt in predictions_df.columns]
        
        if group_options:
            group_by = st.selectbox("Group By", group_options, key="trend_group")
        else:
            group_by = 'Vehicle_Category'
    
    with col_set2:
        if group_by in predictions_df.columns:
            categories = sorted(predictions_df[group_by].unique())
            selected_categories = st.multiselect(
                f"Filter {group_by}",
                options=categories,
                default=categories[:5] if len(categories) > 5 else categories,
                key="trend_filter"
            )
        else:
            selected_categories = []
    
    with col_set3:
        chart_type = st.selectbox(
            "Chart Type",
            ["Trend Lines", "Growth Rate", "Market Share", "Forecast vs Actual"],
            key="trend_chart_type"
        )
    
    # Filter data
    if selected_categories and group_by in predictions_df.columns:
        df_filtered = predictions_df[predictions_df[group_by].isin(selected_categories)].copy()
    else:
        df_filtered = predictions_df.copy()
    
    # Render selected chart
    if chart_type == "Trend Lines":
        if 'Date' in df_filtered.columns and group_by in df_filtered.columns:
            trend_fig = create_trend_comparison(df_filtered, group_col=group_by)
            st.plotly_chart(trend_fig, use_container_width=True)
        else:
            st.error("Required columns not available")
    
    elif chart_type == "Growth Rate":
        if group_by in df_filtered.columns:
            growth_fig = create_growth_rate_chart(df_filtered, group_col=group_by)
            st.plotly_chart(growth_fig, use_container_width=True)
        else:
            st.error("Required columns not available")
    
    elif chart_type == "Market Share":
        if group_by in df_filtered.columns:
            share_fig = create_market_share_chart(df_filtered, group_col=group_by)
            st.plotly_chart(share_fig, use_container_width=True)
        else:
            st.error("Required columns not available")
    
    elif chart_type == "Forecast vs Actual":
        if selected_categories and len(selected_categories) > 0:
            category = selected_categories[0]
            forecast_fig = create_forecast_vs_actual(df_filtered, category=category)
            st.plotly_chart(forecast_fig, use_container_width=True)
        else:
            forecast_fig = create_forecast_vs_actual(df_filtered)
            st.plotly_chart(forecast_fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### Trend Summary")
    
    summary_cols = st.columns(4)
    
    with summary_cols[0]:
        if group_by in df_filtered.columns:
            num_categories = df_filtered[group_by].nunique()
            st.metric(f"Total {group_by}", num_categories)
        else:
            st.metric("Categories", "N/A")
    
    with summary_cols[1]:
        total_sales = df_filtered['Predicted_Sales'].sum()
        st.metric("Total Predicted Sales", f"{total_sales:,}")
    
    with summary_cols[2]:
        if 'Date' in df_filtered.columns:
            date_range = (df_filtered['Date'].max() - df_filtered['Date'].min()).days
            st.metric("Date Range (days)", date_range)
        else:
            st.metric("Date Range", "N/A")
    
    with summary_cols[3]:
        avg_daily = df_filtered.groupby('Date')['Predicted_Sales'].sum().mean() if 'Date' in df_filtered.columns else 0
        st.metric("Avg Daily Sales", f"{avg_daily:,.0f}")
    
    # Top performers table
    if group_by in df_filtered.columns:
        st.markdown(f"#### Top Performers by {group_by}")
        
        top_performers = df_filtered.groupby(group_by).agg({
            'Predicted_Sales': ['sum', 'mean', 'std']
        }).reset_index()
        
        top_performers.columns = [group_by, 'Total Sales', 'Avg Sales', 'Std Dev']
        top_performers = top_performers.sort_values('Total Sales', ascending=False)
        
        st.dataframe(top_performers.head(10), use_container_width=True)
