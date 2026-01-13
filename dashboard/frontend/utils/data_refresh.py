"""
Data Refresh Utilities
Auto-refresh and data update mechanisms for the dashboard
"""

# Make streamlit optional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataRefreshManager:
    """
    Manages automatic data refresh for the dashboard.
    Checks for new model artifacts and prediction data.
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 output_dir: str = "output",
                 data_dir: str = "data",
                 predictions_dir: str = "predictions"):
        """
        Initialize the data refresh manager.
        
        Args:
            models_dir: Directory containing model files
            output_dir: Directory containing output files
            data_dir: Directory containing data files
            predictions_dir: Directory containing predictions
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.predictions_dir = Path(predictions_dir)
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.predictions_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check if there are new updates available.
        
        Returns:
            Dictionary with update status
        """
        updates = {
            'models_updated': False,
            'data_updated': False,
            'predictions_updated': False,
            'last_checked': datetime.now().isoformat()
        }
        
        # Check if we have stored timestamps
        if STREAMLIT_AVAILABLE:
            if 'last_model_check' not in st.session_state:
                st.session_state.last_model_check = None
            
            if 'last_data_check' not in st.session_state:
                st.session_state.last_data_check = None
            
            last_model_check = st.session_state.last_model_check
            last_data_check = st.session_state.last_data_check
        else:
            last_model_check = None
            last_data_check = None
        
        # Check models directory
        if self.models_dir.exists():
            latest_model = self._get_latest_file(self.models_dir, '*.pkl')
            if latest_model:
                model_time = datetime.fromtimestamp(latest_model.stat().st_mtime)
                
                if last_model_check is None or model_time > last_model_check:
                    updates['models_updated'] = True
                    updates['latest_model'] = latest_model.name
                    updates['model_time'] = model_time.isoformat()
        
        # Check data directory
        if self.data_dir.exists():
            latest_data = self._get_latest_file(self.data_dir, '*.csv')
            if latest_data:
                data_time = datetime.fromtimestamp(latest_data.stat().st_mtime)
                
                if last_data_check is None or data_time > last_data_check:
                    updates['data_updated'] = True
                    updates['latest_data'] = latest_data.name
                    updates['data_time'] = data_time.isoformat()
        
        # Check predictions
        if self.predictions_dir.exists():
            latest_pred = self._get_latest_file(self.predictions_dir, '*.csv')
            if latest_pred:
                pred_time = datetime.fromtimestamp(latest_pred.stat().st_mtime)
                updates['predictions_updated'] = True
                updates['latest_prediction'] = latest_pred.name
                updates['prediction_time'] = pred_time.isoformat()
        
        return updates
    
    def _get_latest_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """Get the most recently modified file matching pattern."""
        files = list(directory.glob(pattern))
        
        if not files:
            return None
        
        # Sort by modification time
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0]
    
    def mark_as_checked(self):
        """Mark current time as last check time."""
        now = datetime.now()
        if STREAMLIT_AVAILABLE:
            st.session_state.last_model_check = now
            st.session_state.last_data_check = now
    
    def get_latest_predictions(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load the most recent prediction file.
        
        Args:
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with predictions
        """
        # Check multiple possible locations
        possible_files = [
            self.output_dir / "daily_predictions_2026.csv",
            self.predictions_dir / "latest_predictions.csv",
            self.output_dir / "predictions.csv"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'Date' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'])
                    
                    if limit and len(df) > limit:
                        df = df.tail(limit)
                    
                    logger.info(f"Loaded predictions from {file_path}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.warning("No prediction files found")
        return pd.DataFrame()
    
    def should_refresh(self, 
                      interval_seconds: int = 60,
                      force: bool = False) -> bool:
        """
        Determine if data should be refreshed.
        
        Args:
            interval_seconds: Minimum seconds between refreshes
            force: Force refresh regardless of interval
            
        Returns:
            True if should refresh
        """
        if force:
            return True
        
        if 'last_refresh_time' not in st.session_state:
            st.session_state.last_refresh_time = None
        
        if st.session_state.last_refresh_time is None:
            return True
        
        time_since_refresh = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        
        return time_since_refresh >= interval_seconds
    
    def perform_refresh(self) -> Dict[str, Any]:
        """
        Perform data refresh operation.
        
        Returns:
            Dictionary with refresh results
        """
        st.session_state.last_refresh_time = datetime.now()
        
        results = {
            'success': True,
            'timestamp': st.session_state.last_refresh_time.isoformat(),
            'updates': self.check_for_updates()
        }
        
        # Mark as checked
        self.mark_as_checked()
        
        return results


def setup_auto_refresh(interval_seconds: int = 300, 
                       enabled: bool = True) -> bool:
    """
    Setup automatic refresh for the dashboard.
    
    Args:
        interval_seconds: Refresh interval in seconds
        enabled: Whether auto-refresh is enabled
        
    Returns:
        True if refresh was triggered
    """
    if not enabled:
        return False
    
    # Initialize session state
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    if 'last_auto_refresh' not in st.session_state:
        st.session_state.last_auto_refresh = datetime.now()
    
    # Check if it's time to refresh
    time_since_refresh = (datetime.now() - st.session_state.last_auto_refresh).total_seconds()
    
    if time_since_refresh >= interval_seconds:
        st.session_state.last_auto_refresh = datetime.now()
        st.session_state.refresh_count += 1
        return True
    
    return False


def render_refresh_controls(refresh_manager: DataRefreshManager):
    """
    Render refresh control UI.
    
    Args:
        refresh_manager: DataRefreshManager instance
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Data Refresh")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh",
        value=True,
        help="Automatically refresh data every 5 minutes"
    )
    
    # Refresh interval
    if auto_refresh:
        interval = st.sidebar.select_slider(
            "Refresh Interval",
            options=[60, 120, 300, 600, 1800],
            value=300,
            format_func=lambda x: f"{x//60} min" if x >= 60 else f"{x} sec"
        )
        st.session_state.refresh_interval = interval
    
    # Manual refresh button
    if st.sidebar.button("Refresh Now", key="manual_refresh"):
        with st.spinner("Refreshing data..."):
            results = refresh_manager.perform_refresh()
        
        if results['success']:
            st.sidebar.success("✅ Data refreshed!")
            
            # Show what was updated
            updates = results['updates']
            if updates.get('models_updated'):
                st.sidebar.info(f"📦 New model: {updates.get('latest_model', 'Unknown')}")
            if updates.get('data_updated'):
                st.sidebar.info(f"📊 New data: {updates.get('latest_data', 'Unknown')}")
        else:
            st.sidebar.error("❌ Refresh failed")
    
    # Show last refresh time
    if 'last_refresh_time' in st.session_state and st.session_state.last_refresh_time:
        time_ago = (datetime.now() - st.session_state.last_refresh_time).total_seconds()
        
        if time_ago < 60:
            time_str = f"{int(time_ago)} seconds ago"
        elif time_ago < 3600:
            time_str = f"{int(time_ago/60)} minutes ago"
        else:
            time_str = f"{int(time_ago/3600)} hours ago"
        
        st.sidebar.caption(f"Last refresh: {time_str}")
    
    return auto_refresh
