import os
import pickle
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import argparse

from llm_helper import generate_message
from predictarea import XGBoostPredictionModel


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

    # Try to find label_encoder.pkl in model_dir, then in script directory
    label_path = os.path.join(model_dir, "label_encoder.pkl")
    if not os.path.exists(label_path):
        # Try script directory (where this file is located)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_label_path = os.path.join(script_dir, "label_encoder.pkl")
        if os.path.exists(alt_label_path):
            label_path = alt_label_path
        else:
            raise_file_missing_exception(label_path)

    with open(label_path, "rb") as f:
        le = pickle.load(f)

    # Try to find crime_prediction_model.json in model_dir, then in script directory
    model_path = os.path.join(model_dir, "crime_prediction_model.json")
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_model_path = os.path.join(script_dir, "crime_prediction_model.json")
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
        else:
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

def predict_area(date: str, startTime: str, endTime: str, southWestLat: float, southWestLng: float, northEastLat: float, northEastLng: float, model_dir: str = "."):
    """
    Predicts crime categories for a rectangular area based on date, time, and bounding box coordinates.

    Args:
        start_time (str): Start Time in MM-DD format.
        end_time (str): End Time in HH:MM format.
        southWestLat (float): Southwest latitude coordinate.
        southWestLng (float): Southwest longitude coordinate.
        northEastLat (float): Northeast latitude coordinate.
        northEastLng (float): Northeast longitude coordinate.
        model_dir (str): Directory containing the model and label encoder.

    Returns:
        str: A human-readable prediction message for the area.
    """
    start = int(startTime.split(":")[0])
    end = int(endTime.split(":")[0])
    if end - start > 12:
       raise Exception("Time range should not exceed 12 hours.")

    resolution_horizontal = 8
    resolution_vertical = 6
    steps_lat = (northEastLat - southWestLat) / resolution_vertical
    steps_lng = (northEastLng - southWestLng) / resolution_horizontal
    model = XGBoostPredictionModel()

    positions = []
    for i in range(1, resolution_vertical - 1):
        for j in range(1, resolution_horizontal - 1):
            lat = southWestLat + steps_lat * i
            lng = southWestLng + steps_lng * j
            positions.append((lat, lng))

    predictions = []
    for time_val in range(start, end + 1):
        for position in positions:
            lat = position[0]
            lng = position[1]
            category, score = model.predict(date, str(time_val) + ":00", lat, lng)
            if score < 0.6:
                continue
            if category != "no-incident":
                predictions.append(
                    {
                        "time": time_val,
                        "lat": lat,
                        "lng": lng,
                        "incident_probability": 100 * score,
                        "category": category,
                    }
                )
    
    # create human friendly message for the predictions list
    if len(predictions) == 0:
        return f"No significant incidents predicted in the area ({southWestLat}, {southWestLng}) to ({northEastLat}, {northEastLng}) on {date} between {startTime} and {endTime}."
    else:
        messages = []
        for pred in predictions:
            datetime_str = f"{date} {pred['time']:02d}:00"
            try:
                human_friendly_message = generate_message(
                    prediction_label=pred["category"],
                    probability=pred["incident_probability"] / 100.0,
                    lat=pred["lat"],
                    lng=pred["lng"],
                    datetime_str=datetime_str,
                )
                messages.append(human_friendly_message)
            except Exception as e:
                messages.append(f"Predicted incident at {datetime_str} (lat: {pred['lat']}, lng: {pred['lng']}): {pred['category']} with estimated confidence {pred['incident_probability']:.2f}%\n(Could not generate human-friendly message: {e})")
        return "\n\n".join(messages)

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
