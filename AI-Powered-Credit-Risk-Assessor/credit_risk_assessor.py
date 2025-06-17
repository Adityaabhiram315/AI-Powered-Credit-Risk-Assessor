import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import joblib

# 1. Load Data
data = pd.read_csv('credit_data.csv')  # Replace with your dataset path

# 2. Preprocess Data
X = data.drop(['default'], axis=1)
y = data['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate Model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Save Model
joblib.dump(clf, 'credit_model.joblib')

# 6. Setup API
app = Flask(__name__)
model = joblib.load('credit_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'credit_risk': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)