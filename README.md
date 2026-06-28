# 🚦 Traffic Demand Prediction using LightGBM & Flask

## 📌 Project Overview

This project focuses on predicting traffic demand (vehicle count) using Machine Learning techniques. Historical traffic data is analyzed to understand traffic patterns, perform feature engineering, and build predictive models. A Flask web application is also developed to allow users to predict traffic demand through a simple web interface.

---

## 🎯 Objectives

- Understand and analyze the traffic dataset.
- Perform Exploratory Data Analysis (EDA).
- Create useful time-based features from the DateTime column.
- Train Machine Learning models for traffic prediction.
- Compare model performance.
- Deploy the best model using Flask.

---

## 📂 Dataset Information

**Dataset:** Traffic Prediction Dataset

**Number of Records:** 48,120

**Number of Features:** 4 Original Features

### Original Columns

- DateTime
- Junction
- Vehicles
- ID

### Engineered Features

- Hour
- Day
- Month
- Year
- DayOfWeek

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Flask
- Joblib

---

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Dataset Understanding
- Missing Value Analysis
- Statistical Summary
- Traffic Distribution
- Average Traffic by Hour
- Traffic Distribution by Junction
- Correlation Heatmap

---

## 🤖 Machine Learning Models

### XGBoost Regressor

- MAE: **2.894**
- RMSE: **4.906**
- R² Score: **0.9409**

### LightGBM Regressor

- MAE: **2.852**
- RMSE: **4.693**
- R² Score: **0.9459**

### Best Model

🏆 **LightGBM Regressor**

LightGBM achieved the best performance and was selected as the final prediction model.

---

## 🌐 Flask Web Application

The trained LightGBM model was integrated with a Flask web application.

Users can enter:

- Junction
- Hour
- Day
- Month
- Year
- Day of Week

The application predicts the expected number of vehicles for the given traffic conditions.

---

## 📁 Project Structure

```
Innovexa_project3
│
├── Traffic.csv
├── project3.py
├── app.py
├── traffic_prediction_model.pkl
├── README.md
│
├── templates
│   └── index.html
│
└── output_images
    ├── traffic_distribution.png
    ├── average_traffic_by_hour.png
    ├── traffic_by_junction.png
    ├── correlation_heatmap.png
    └── flask_prediction.png
```

---

## 📈 Results

The developed prediction system successfully learns traffic patterns using historical traffic data and predicts future vehicle counts with high accuracy.

The Flask application provides a simple and user-friendly interface for real-time traffic demand prediction.

---

## 🚀 Future Improvements

- Include weather information.
- Add holiday and event data.
- Predict traffic for multiple future time intervals.
- Deploy the application on a cloud platform.
- Improve UI with charts and dashboard.

---

## 👨‍💻 Author

**Alan Jaison**

B.Tech Computer Science & Engineering (Data Science)

Christ College of Engineering (Autonomous)

Innovexa Data Science Internship Project