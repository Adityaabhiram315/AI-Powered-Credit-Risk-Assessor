import catboost
from catboost import CatBoostClassifier
import joblib

def get_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

def save_model(model, path):
    model.save_model(path)

def load_model(path):
    model = CatBoostClassifier()
    model.load_model(path)
    return model

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)