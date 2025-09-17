import os
import pickle
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from ..interfaces import PredictionModel


class XGBoostPredictionModel(PredictionModel):
    def __init__(self):
        self.le = LabelEncoder()

        label_path = "./models/xgboost/label_encoder.pkl"
        with open(label_path, "rb") as f:
            self.le = pickle.load(f)

        model_path = "./models/xgboost/crime_prediction_model.json"
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
