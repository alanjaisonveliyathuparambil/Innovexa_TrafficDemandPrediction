import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Traffic.csv")

print("TRAFFIC DEMAND PREDICTION")
print("-" * 50)

print("\n1. First 5 Rows:")
print(df.head())

print("\n2. Dataset Shape:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n3. Column Names:")
print(df.columns)

print("\n4. Data Types:")
print(df.dtypes)

print("\n5. Missing Values:")
print(df.isnull().sum())

print("\n6. Statistical Summary:")
print(df.describe())

print("\n7. Dataset Information:")
df.info()

df["DateTime"] = pd.to_datetime(df["DateTime"])

print("\n8. DateTime Converted Successfully!")
print(df["DateTime"].head())

df["Hour"] = df["DateTime"].dt.hour
df["Day"] = df["DateTime"].dt.day
df["Month"] = df["DateTime"].dt.month
df["Year"] = df["DateTime"].dt.year
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

print("\n9. New Features Created Successfully!")
print(df[["DateTime", "Hour", "Day", "Month", "Year", "DayOfWeek"]].head())

plt.figure(figsize=(8, 5))
sns.histplot(df["Vehicles"], bins=30)
plt.title("Distribution of Vehicle Count")
plt.xlabel("Number of Vehicles")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="Hour", y="Vehicles", errorbar=None)
plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Junction", y="Vehicles")
plt.title("Traffic Distribution by Junction")
plt.xlabel("Junction")
plt.ylabel("Vehicle Count")
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=["int64", "float64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

X = df[["Junction", "Hour", "Day", "Month", "Year", "DayOfWeek"]]

y = df["Vehicles"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n14. Machine Learning Data Prepared Successfully!")

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

lgb_model = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

lgb_model.fit(X_train, y_train)

lgb_pred = lgb_model.predict(X_test)

def evaluate_model(model_name, y_test, predictions):
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name} Results:")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

evaluate_model("XGBoost", y_test, xgb_pred)
evaluate_model("LightGBM", y_test, lgb_pred)

import joblib

joblib.dump(lgb_model, "traffic_prediction_model.pkl")
print("\nTraffic prediction model saved successfully!")