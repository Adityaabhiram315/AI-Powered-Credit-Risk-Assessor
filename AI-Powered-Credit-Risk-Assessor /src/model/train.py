import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.preprocess import load_and_split_data, scale_data
from src.model.model_utils import get_model, save_model, save_scaler
from src.config import DATA_PATH, MODEL_PATH

def main():
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    model = get_model()
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), use_best_model=True)
    save_model(model, MODEL_PATH)
    save_scaler(scaler, "model/scaler.pkl")
    print("Training complete. Model and scaler saved.")

if __name__ == "__main__":
    main()