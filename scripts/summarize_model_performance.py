# scripts/summarize_model_performance.py
import pickle
from pathlib import Path
import pandas as pd

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = ROOT_DIR / "models"

def summarize_performance():
    """
    Finds all category-specific models, loads them, and prints a 
    summary of their performance metrics in a clean table.
    """
    print(f"🔎 Searching for models in: {MODELS_DIR}")

    # Find all advanced model files
    model_files = list(MODELS_DIR.glob("advanced_model_*.pkl"))

    if not model_files:
        print("❌ No advanced models found. Please train the models first.")
        return

    performance_data = []

    print(f"Found {len(model_files)} models. Extracting performance data...")
    for model_path in model_files:
        try:
            # Extract the category name from the filename
            category_name = model_path.stem.replace("advanced_model_", "")

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            scores = model_data.get('test_scores', {}).get('optimized', {})
            mae = scores.get('MAE')
            r2 = scores.get('R2')

            if mae is not None and r2 is not None:
                performance_data.append({
                    "Vehicle Category": category_name,
                    "MAE": mae,
                    "R² Score": r2
                })
        except Exception as e:
            print(f"⚠️ Could not read or process {model_path.name}: {e}")

    if not performance_data:
        print("❌ No valid performance data could be extracted from the model files.")
        return

    # Create a pandas DataFrame for clean printing
    performance_df = pd.DataFrame(performance_data)
    performance_df = performance_df.sort_values(by="R² Score", ascending=False).reset_index(drop=True)

    print("\n\n📊====== Model Performance Summary ======📊")
    print(performance_df.to_string())
    print("===========================================")

if __name__ == "__main__":
    summarize_performance()