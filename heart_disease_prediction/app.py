from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)
model = pickle.load(open("model/heart_disease_rf_model.pkl", "rb"))

# Define form inputs (must match training features)
input_features = ['age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
                  'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
                  'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form[feature]) for feature in input_features]
    except KeyError as e:
        return f"Missing field in form: {e}", 400

    df = pd.DataFrame([values], columns=input_features)
    prediction = model.predict(df)[0]
    result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

    return render_template("result.html", result=result_text, prediction=prediction)

if __name__ == "__main__":
     port = int(os.environ.get("PORT", 5000))  # Render sets the PORT dynamically
     print(">>> Starting app with host binding:", port)

     app.run(host="0.0.0.0", port=port, debug=True)
