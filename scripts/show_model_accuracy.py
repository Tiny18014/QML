#!/usr/bin/env python3
"""
Show Model Accuracy
===================
Runs the accuracy evaluation logic and displays the report for each vehicle category.
"""

import sys
import pandas as pd
from pathlib import Path

# Add scripts dir to path to import trainer
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(ROOT_DIR / "scripts"))

try:
    from advanced_model_trainer import generate_model_performance_report
except ImportError:
    print("❌ Could not import 'advanced_model_trainer'. Make sure you are in the project root.")
    sys.exit(1)

def main():
    print("📊 Evaluating Model Accuracy...")
    report_path = generate_model_performance_report()

    if report_path and report_path.exists():
        df = pd.read_csv(report_path)
        print("\n" + "="*60)
        print("🏆 MODEL ACCURACY REPORT (Training Data)")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        print(f"📄 Report saved to: {report_path}")
    else:
        print("❌ Failed to generate report.")

if __name__ == "__main__":
    main()
