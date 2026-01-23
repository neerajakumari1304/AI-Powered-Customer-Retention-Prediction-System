from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Note: You must save your trained model and scaler to these files first
# based on your best performer: Logistic Regression (Accuracy: 0.806)
# and your best scaler: Standard Scaler [cite: 1, 114]
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found. Please export them from your notebook.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Capture all 20 features from the form
    input_data = {
        'telecom_partner': request.form.get('telecom_partner'),
        'gender': request.form.get('gender'),
        'SeniorCitizen': int(request.form.get('SeniorCitizen')),
        'Partner': request.form.get('Partner'),
        'Dependents': request.form.get('Dependents'),
        'tenure': float(request.form.get('tenure')),
        'PhoneService': request.form.get('PhoneService'),
        'MultipleLines': request.form.get('MultipleLines'),
        'InternetService': request.form.get('InternetService'),
        'OnlineSecurity': request.form.get('OnlineSecurity'),
        'OnlineBackup': request.form.get('OnlineBackup'),
        'DeviceProtection': request.form.get('DeviceProtection'),
        'TechSupport': request.form.get('TechSupport'),
        'StreamingTV': request.form.get('StreamingTV'),
        'StreamingMovies': request.form.get('StreamingMovies'),
        'Contract': request.form.get('Contract'),
        'PaperlessBilling': request.form.get('PaperlessBilling'),
        'PaymentMethod': request.form.get('PaymentMethod'),
        'MonthlyCharges': float(request.form.get('MonthlyCharges')),
        'TotalCharges': float(request.form.get('TotalCharges'))
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Pre-processing: Label Encoding for Categorical Columns [cite: 111, 112]
    # In a production app, use the fitted LabelEncoder objects from your training
    categorical_cols = [
        'telecom_partner', 'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # Note: Replace this placeholder logic with your actual label mappings
    # e.g., jio=0, bsnl=1, airtel=2, vodafone=3 [cite: 98-108]
    for col in categorical_cols:
        # This is a simplified mapping logic; use your encoder.transform() in production
        df[col] = df[col].astype('category').cat.codes

    # Scaling Numerical Features using Standard Scaler [cite: 114]
    scaled_features = scaler.transform(df)

    # Prediction
    prediction = model.predict(scaled_features)
    result = "Likely to Churn" if prediction[0] == 1 else "Likely to Stay"

    return render_template('index.html', prediction_text=f'Customer Status: {result}')


if __name__ == "__main__":
    app.run(debug=True)