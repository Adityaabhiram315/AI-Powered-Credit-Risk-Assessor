# AI-Powered Credit Risk Assessor

## Features
- Uses CatBoost, a state-of-the-art ML model for tabular data
- REST API with FastAPI
- Dockerized deployment
- Modular, testable codebase

## Usage

1. **Train the model** (put your dataset in `data/credit_data.csv` with a `default` column):

    ```bash
    python src/model/train.py
    ```

2. **Serve the API**:

    ```bash
    uvicorn src.api.main:app --reload
    ```

3. **Prediction example**:

    ```bash
    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{"age": 45, "income": 80000, "loan_amount": 15000}'
    ```

4. **Docker**:

    ```bash
    docker build -t credit-risk-assessor .
    docker run -p 8000:8000 credit-risk-assessor
    ```

## Adaptation
- Add/remove features in `CreditFeatures` and preprocessing as per your dataset.
- For advanced usage, add Streamlit or model versioning.