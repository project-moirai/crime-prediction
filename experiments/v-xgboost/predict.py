import os
import pickle
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import argparse

from llm_helper import generate_message


def predict(date_input: str, time_input: str, latitude: float, longitude: float, model_dir: str = "."):
    """
    Predicts crime category based on date, time, and location.

    Args:
        date_input (str): Date in MM-DD format.
        time_input (str): Time in HH:MM format.
        latitude (float): Latitude coordinate.
        longitude (float): Longitude coordinate.
        model_dir (str): Directory containing the model and label encoder.

    Returns:
        str: A human-readable prediction message.
    """

    def raise_file_missing_exception(filepath: str):
        raise Exception(
            f"File {filepath} was not found. Did you run train.py? Also ensure the file is in the correct location, working directory is {os.getcwd()}"
        )

    # Load models and encoders
    le = LabelEncoder()

    label_path = os.path.join(model_dir, "label_encoder.pkl")
    if not os.path.exists(label_path):
        raise_file_missing_exception(label_path)

    with open(label_path, "rb") as f:
        le = pickle.load(f)

    model_path = os.path.join(model_dir, "crime_prediction_model.json")
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

    pred_label = le.inverse_transform(pred_category)[0]
    output_msg = f"Predicted incident to happen at {date_input} {time_input} (lat: {latitude}, lng: {longitude}): {pred_label}"

    # Generate human-friendly message
    try:
        prob = None
        if hasattr(model_category, "predict_proba"):
            try:
                prob = float(model_category.predict_proba(df[features])[0].max())
            except Exception:
                prob = None

        datetime_str = f"{date_input} {time_input}"
        human_friendly_message = generate_message(
            prediction_label=pred_label,
            probability=prob,
            lat=latitude,
            lng=longitude,
            datetime_str=datetime_str,
        )
        return human_friendly_message
    except Exception as e:
        return f"{output_msg}\n(Could not generate human-friendly message: {e})"


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py <MM-DD> <HH:MM> <LATITUDE> <LONGITUDE>")
        sys.exit(1)

    date_input = sys.argv[1]
    time_input = sys.argv[2]
    latitude = float(sys.argv[3])
    longitude = float(sys.argv[4])

    result_message = predict(date_input, time_input, latitude, longitude)
    print(result_message)
