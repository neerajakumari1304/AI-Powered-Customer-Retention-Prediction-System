from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# -------- LOAD SAVED OBJECTS --------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
train_columns = pickle.load(open("train_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- GET USER INPUT --------
        data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": float(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "InternetService": request.form["InternetService"],
            "Contract": request.form["Contract"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        df = pd.DataFrame([data])

        # -------- SAME ENCODING AS TRAINING --------
        df = pd.get_dummies(df, drop_first=True)

        # -------- ALIGN FEATURES --------
        df = df.reindex(columns=train_columns, fill_value=0)

        # -------- SCALE --------
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=train_columns
        )
        # -------- PROBABILITY PREDICTION --------
        churn_prob = model.predict_proba(df_scaled)[0][1]

        # -------- DECISION THRESHOLD --------
        if churn_prob >= 0.4:
            result = f"Customer Will Churn ❌ (Risk: {round(churn_prob * 100, 1)}%)"
        else:
            result = f"Customer Will Stay ✅ (Risk: {round(churn_prob * 100, 1)}%)"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
