import os

DATA_PATH = os.getenv("DATA_PATH", "data/credit_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model/credit_model.cbm")
RANDOM_SEED = 42