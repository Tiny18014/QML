# scripts/check_model_performance.py
import pickle
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
MODEL_PATH = ROOT_DIR / "models" / "advanced_ev_model.pkl"

def check_performance():
    """Loads the advanced model artifact and prints its stored performance metrics."""
    print(f"🔎 Loading model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        print("❌ Model file not found. Please train the advanced model first.")
        return

    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        print("✅ Model artifact loaded successfully.")
        
        # Access the test_scores dictionary saved during training
        scores = model_data.get('test_scores', {}).get('optimized', {})
        
        if not scores:
            print("⚠️ No performance scores found in the model file.")
            return
            
        mae = scores.get('MAE')
        r2 = scores.get('R2')
        
        print("\n📊 Stored Performance Metrics on Test Set 📊")
        print("==============================================")
        if mae is not None:
            print(f"  🏆 Mean Absolute Error (MAE): {mae:.4f}")
        if r2 is not None:
            print(f"  🏆 R-squared (R²):            {r2:.4f}")
        print("==============================================")

    except Exception as e:
        print(f"❌ An error occurred while reading the model file: {e}")

if __name__ == "__main__":
    check_performance()
    