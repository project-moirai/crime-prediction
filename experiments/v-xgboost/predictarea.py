import os
import pickle
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from interfaces import PredictionModel


class XGBoostPredictionModel(PredictionModel):
    def __init__(self):
        self.le = LabelEncoder()

        # Try to find label_encoder.pkl in current working directory, then in script directory
        label_path = "label_encoder.pkl"
        if not os.path.exists(label_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_label_path = os.path.join(script_dir, "label_encoder.pkl")
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                raise Exception(f"File {label_path} was not found. Did you run train.py? Also ensure the file is in the correct location, working directory is {os.getcwd()}")
        with open(label_path, "rb") as f:
            self.le = pickle.load(f)

        # Try to find crime_prediction_model.json in current working directory, then in script directory
        model_path = "crime_prediction_model.json"
        if not os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_model_path = os.path.join(script_dir, "crime_prediction_model.json")
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
            else:
                raise Exception(f"File {model_path} was not found. Did you run train.py? Also ensure the file is in the correct location, working directory is {os.getcwd()}")
        self.model_category = xgb.XGBClassifier()
        self.model_category.load_model(model_path)

    def predict(self, date, time, lat, lng):
        df = pd.DataFrame([{"date": date, "time": time}])

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

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["lat"] = float(lat)
        df["lng"] = float(lng)

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

        proba = self.model_category.predict_proba(df[features])[0]

        pred_category = np.argmax(proba)

        predicted_label = self.le.inverse_transform([pred_category])[0]

        confidence = proba[pred_category]
        return predicted_label, confidence