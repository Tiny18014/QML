#!/usr/bin/env python3
"""
Git Hook Trigger for EV Data Push Pipeline
This script can be used as a Git hook to automatically trigger the data push pipeline
when changes are detected in the data folder.
"""

import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
from pathlib import Path

def trigger_data_push_pipeline():
    """Trigger the data push pipeline when called from Git hook."""
    print("🔄 Git hook detected changes in data folder. Triggering data push pipeline...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.resolve()
    
    # Change to project root
    os.chdir(project_root)
    
    try:
        # Run the data push pipeline
        result = subprocess.run([
            sys.executable, 
            "scripts/run_pipeline.py", 
            "--mode", "datapush"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Data push pipeline completed successfully!")
        print("Output:", result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data push pipeline failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Error triggering pipeline: {e}")
        return False

if __name__ == "__main__":
    trigger_data_push_pipeline()
