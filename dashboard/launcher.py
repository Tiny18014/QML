#!/usr/bin/env python3
"""
Dashboard Launcher
Quick start script for the EV Demand Forecasting Dashboard
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the dashboard."""
    print("=" * 60)
    print("EV Demand Forecasting Dashboard Launcher")
    print("=" * 60)
    
    # Get root directory
    root_dir = Path(__file__).parent.parent
    
    print("\nChoose a dashboard option:")
    print("1. Enhanced Dashboard (Recommended)")
    print("2. Legacy Dashboard")
    print("3. API Server")
    print("4. Run Tests")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Launching Enhanced Dashboard...")
        app_path = root_dir / "dashboard" / "frontend" / "app.py"
        subprocess.run(["streamlit", "run", str(app_path)])
    
    elif choice == "2":
        print("\n🚀 Launching Legacy Dashboard...")
        app_path = root_dir / "scripts" / "streamlit_dashboard.py"
        subprocess.run(["streamlit", "run", str(app_path)])
    
    elif choice == "3":
        print("\n🚀 Launching API Server...")
        api_path = root_dir / "dashboard" / "backend" / "api.py"
        subprocess.run(["python", str(api_path)])
    
    elif choice == "4":
        print("\n🧪 Running Tests...")
        test_path = root_dir / "dashboard" / "tests" / "integration_test.py"
        subprocess.run(["python", str(test_path)])
    
    elif choice == "5":
        print("\n👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("\n❌ Invalid choice. Please run again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
