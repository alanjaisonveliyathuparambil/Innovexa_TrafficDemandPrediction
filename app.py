from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("traffic_prediction_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    junction = int(request.form["junction"])
    hour = int(request.form["hour"])
    day = int(request.form["day"])
    month = int(request.form["month"])
    year = int(request.form["year"])
    day_of_week = int(request.form["day_of_week"])

    input_data = pd.DataFrame({
        "Junction": [junction],
        "Hour": [hour],
        "Day": [day],
        "Month": [month],
        "Year": [year],
        "DayOfWeek": [day_of_week]
    })

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Vehicle Count: {round(prediction)}"
    )

if __name__ == "__main__":
    app.run(debug=True)