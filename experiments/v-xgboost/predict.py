import os
import pickle
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def raise_file_missing_exception(filepath: str):
    raise Exception(
        f"File {label_path} was not found. Did you run train.py? Also ensure the file is in the correct location, working directory is {os.getcwd()}"
    )


if len(sys.argv) != 5:
    print("Usage: python predict.py <MM-DD> <HH:MM> <LATITUDE> <LONGITUDE>")
    sys.exit(1)

date_input = sys.argv[1]
time_input = sys.argv[2]
latitude = sys.argv[3]
longitude = sys.argv[4]

# Load models and encoders
le = LabelEncoder()

label_path = "label_encoder.pkl"
if not os.path.exists(label_path):
    raise_file_missing_exception(label_path)

with open(label_path, "rb") as f:
    le = pickle.load(f)

model_path = "crime_prediction_model.json"
if not os.path.exists(model_path):
    raise_file_missing_exception(model_path)

model_category = xgb.XGBClassifier()
model_category.load_model(model_path)

df = pd.DataFrame([{"date": date_input, "time": time_input}])

df["datetime"] = pd.to_datetime(
    "2021-" + df["date"] + " " + df["time"],
    format="%Y-%m-%d %H:%M",
    errors="coerce",
)

df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["dow"] = df["datetime"].dt.weekday

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["lat"] = float(latitude)
df["lng"] = float(longitude)

features = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "lat",
    "lng",
]

pred_category = model_category.predict(df[features])

print(
    f"Predicted incident to happen at {date_input} {time_input} (lat: {latitude}, lng: {longitude}): {le.inverse_transform(pred_category)[0]}",
)
