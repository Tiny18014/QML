#!/usr/bin/env python3
"""
EV Sales Live Pipeline Launcher
Provides an easy way to start the simulation and dashboard, now with improved
path management and structure.
"""

import subprocess
import sys
import os
import time
import threading
import argparse
from pathlib import Path
import sqlite3

# Define project structure paths at the top for clarity
# This makes it easy to change if the structure evolves.
# The ROOT_DIR is the parent of the directory containing this script (scripts/).
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# --- Key File Paths ---
DATASET_PATH = DATA_DIR / "EV_Dataset.csv"
MODEL_PATH = MODELS_DIR / "ev_model.pkl"
DB_PATH = OUTPUT_DIR / "live_predictions.db"
SIMULATION_SCRIPT_PATH = SCRIPTS_DIR / "live_simulation.py"
DASHBOARD_SCRIPT_PATH = SCRIPTS_DIR / "streamlit_dashboard.py"
TRAINING_SCRIPT_PATH = SCRIPTS_DIR / "demand_forecast.py"


def check_requirements():
    """Check if required files and directories exist."""
    print("🔎 Checking requirements...")
    
    # Ensure directories exist
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    required_files = {
        "Dataset": DATASET_PATH,
        "Trained Model": MODEL_PATH
    }
    
    missing_items = []
    for name, path in required_files.items():
        if not path.exists():
            missing_items.append(f"{name} (at {path})")
            
    if missing_items:
        print("\n❌ Missing required files:")
        for item in missing_items:
            print(f"   - {item}")
        print("\n💡 Please ensure you have:")
        print(f"   1. The EV dataset at: {DATASET_PATH}")
        print(f"   2. A trained model at: {MODEL_PATH}")
        print("   👉 You may need to run the 'Train Model' option first.")
        return False
    
    print("✅ Requirements met.")
    return True

def run_simulation(delay=1.0, max_records=None):
    """Run the live simulation."""
    print(f"🚀 Starting live simulation with {delay}s delay...")
    
    # The 'live_simulation' module should be importable if scripts/ is a package
    # or if the path is managed correctly.
    sys.path.insert(0, str(SCRIPTS_DIR))
    from live_simulation import LiveEVDataSimulator
    
    simulator = LiveEVDataSimulator(
        data_path=str(DATASET_PATH),
        model_path=str(MODEL_PATH),
        db_path=str(DB_PATH)
    )
    
    try:
        simulator.start_simulation(delay_seconds=delay, max_records=max_records)
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user.")
        simulator.stop_simulation()
    except Exception as e:
        print(f"❌ Simulation error: {e}")
    finally:
        # Clean up the path modification
        if str(SCRIPTS_DIR) in sys.path:
            sys.path.remove(str(SCRIPTS_DIR))


def run_dashboard():
    """Run the Streamlit dashboard."""
    print("📈 Starting Streamlit dashboard...")
    
    if not DASHBOARD_SCRIPT_PATH.exists():
        print(f"❌ Dashboard script not found: {DASHBOARD_SCRIPT_PATH}")
        return
    
    try:
        # Use subprocess.run with streamlit's command-line interface
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(DASHBOARD_SCRIPT_PATH)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it: pip install streamlit")

def run_both():
    """Run both simulation and dashboard in separate threads."""
    print("🎯 Starting both simulation and dashboard...")
    
    # Start simulation in a background thread
    sim_thread = threading.Thread(target=run_simulation, args=(1.0, None), daemon=True)
    sim_thread.start()
    
    print("⏳ Giving simulation time to initialize...")
    time.sleep(5)  # Increased sleep time to ensure DB is created
    
    # Start dashboard in the main thread
    run_dashboard()

def train_model():
    """Train the model by running the demand_forecast.py script."""
    print("🔄 Training the model...")
    
    if not TRAINING_SCRIPT_PATH.exists():
        print(f"❌ Training script not found: {TRAINING_SCRIPT_PATH}")
        return False
    
    try:
        # Execute the training script
        result = subprocess.run(
            [sys.executable, str(TRAINING_SCRIPT_PATH)],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        print("✅ Model training completed successfully.")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        return False

def setup_environment():
    """Create all necessary project directories."""
    print("🛠️  Setting up environment directories...")
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    print("✅ Environment setup complete.")

def show_status():
    """Show the current status of the pipeline files and database."""
    print("\n📊 Pipeline Status")
    print("=" * 50)
    
    files_to_check = {
        "📁 Dataset": DATASET_PATH,
        "🤖 Trained Model": MODEL_PATH,
        "🎯 Simulation Script": SIMULATION_SCRIPT_PATH,
        "📈 Dashboard Script": DASHBOARD_SCRIPT_PATH,
        "🗄️  Database": DB_PATH
    }
    
    for name, path in files_to_check.items():
        if path.exists():
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ {name}: Present ({size_mb:.2f} MB)")
            except Exception:
                 print(f"✅ {name}: Present")
        else:
            print(f"❌ {name}: Missing")
            
    # Check database record count
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM live_predictions")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"📈 Database Records: {count} predictions logged.")
        except sqlite3.Error as e:
            print(f"⚠️ Could not check database records: {e}")
            
    print("=" * 50)

def interactive_menu():
    """Display the interactive command-line menu."""
    # Set the working directory to the project root
    os.chdir(ROOT_DIR)
    print(f"📍 Working directory set to: {ROOT_DIR}")

    while True:
        print("\n🚗 EV Sales Live Pipeline Launcher")
        print("=" * 40)
        print("1. 📊 Show Status")
        print("2. 🔄 Train Model")
        print("3. 🚀 Run Simulation Only")
        print("4. 📈 Run Dashboard Only")
        print("5. 🎯 Run Both (Simulation + Dashboard)")
        print("6. ⚡ Quick Test (50 records)")
        print("7. 🛠️  Setup Environment")
        print("8. ❌ Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            show_status()
        elif choice == "2":
            train_model()
        elif choice == "3":
            if not check_requirements(): continue
            try:
                delay = float(input("Enter delay between predictions (seconds, default 1.0): ") or "1.0")
                run_simulation(delay)
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
        elif choice == "4":
            run_dashboard()
        elif choice == "5":
            if not check_requirements(): continue
            run_both()
        elif choice == "6":
            if not check_requirements(): continue
            print("🧪 Running quick test with 50 records...")
            run_simulation(delay=0.5, max_records=50)
        elif choice == "7":
            setup_environment()
        elif choice == "8":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def main():
    """Main entry point for command-line arguments."""
    os.chdir(ROOT_DIR)
    
    parser = argparse.ArgumentParser(description="EV Sales Live Pipeline Launcher")
    parser.add_argument("--mode", choices=["simulation", "dashboard", "both", "train", "status", "test"], 
                        help="Directly run a specific mode without the interactive menu.")
    parser.add_argument("--delay", type=float, default=1.0, 
                        help="Delay in seconds for simulation mode.")
    parser.add_argument("--max-records", type=int, 
                        help="Max records for simulation mode.")
    
    args = parser.parse_args()
    
    if not args.mode:
        interactive_menu()
    else:
        print(f"📍 Working directory: {os.getcwd()}")
        if args.mode == "status":
            show_status()
        elif args.mode == "train":
            train_model()
        elif args.mode in ["simulation", "test"]:
            if check_requirements():
                max_r = 50 if args.mode == "test" else args.max_records
                run_simulation(args.delay, max_r)
        elif args.mode == "dashboard":
            run_dashboard()
        elif args.mode == "both":
            if check_requirements():
                run_both()

if __name__ == "__main__":
    main()
