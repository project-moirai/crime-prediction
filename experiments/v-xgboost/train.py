import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from io import StringIO

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import json

file_path = "./data/data-gpt4o-v2.jsonl"

with open(file_path, "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)
print(df.head())

# Remove rows with empty values
df = df.dropna(subset=["date", "time"])
df = df[
    (df["date"].str.strip().str.lower() != "null")
    & (df["time"].str.strip().str.lower() != "null")
    & (df["date"].str.strip() != "")
    & (df["time"].str.strip() != "")
    & df["lat"].notna()
    & df["lng"].notna()
    & np.isfinite(df["lat"])
    & np.isfinite(df["lng"])
]
df["datetime"] = pd.to_datetime(
    "2021-" + df["date"] + " " + df["time"],
    format="%Y-%m-%d %H:%M",
    errors="coerce",  # Invalid ones become NaT
)

print(df.head())

# Drop rows that still failed
df = df.dropna(subset=["datetime"])

# Feature engineering
df["datetime"] = pd.to_datetime("2021-" + df["date"] + " " + df["time"])

df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["dow"] = df["datetime"].dt.weekday  # 0 = Monday, 1 = Tuesday etc.

# Cyclical encoding for time features
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Change precision of location
df["lat"] = df["lat"].round(2)
df["lng"] = df["lng"].round(2)

# We only take one class per row (multiclass classification) and ignore the rest here
df["category"] = df["category"].apply(lambda x: x[0] if isinstance(x, list) else x)

# We drop classes that appear extremely rarely in the dataset
categories_to_drop = [
    "animal-incident",
    "human-trafficking",
    "cybercrime",
]
df_filtered = df[~df["category"].isin(categories_to_drop)]

# We build now our feature table X - these are the columns used for predictions
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
X = df_filtered[features]

all_categories = df_filtered["category"].unique().tolist() + ["no-incident"]
le = LabelEncoder()
le.fit(all_categories)

print("Category distribution --------------------------")
print(df_filtered["category"].value_counts().sort_values())

y = le.transform(df_filtered["category"])

# Oversample with SMOTE so we have a more balanced dataset
sm = SMOTE(random_state=123)
X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

_, label_counts = np.unique(y_SMOTE, return_counts=True)

# We now introduce some random data at random dates, times and locations
# where no incident happened and give them the label "no-incident"
# The category is encoded as numbers and used as the target (y)
num_rows = label_counts[0]
rng = np.random.default_rng(seed=123)
days = rng.integers(0, 365, num_rows).astype("timedelta64[D]")
hours = rng.integers(0, 24, num_rows).astype("timedelta64[h]")

dates = pd.to_datetime(days + hours + np.datetime64("2021-01-01"))
dates = pd.Series(dates)

hours = dates.dt.hour
dows = dates.dt.dayofweek
months = dates.dt.month

X_new = pd.DataFrame(
    {
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "dow_sin": np.sin(2 * np.pi * dows / 7),
        "dow_cos": np.cos(2 * np.pi * dows / 7),
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12),
        "lat": rng.uniform(46, 55, num_rows).round(2),
        "lng": rng.uniform(5.5, 17.5, num_rows).round(2),
    }
)

print("X --------------------------------------------")
print(X.describe())
print("X with no-incident feature -------------------")
print(X_new.describe())

y_new = np.full(num_rows, le.transform(["no-incident"])[0], dtype=int)

X_aug = pd.concat([X_SMOTE, X_new], ignore_index=True)
y_aug = np.concatenate([y_SMOTE, y_new])

# Save the label encoder to be used in predict.py
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=123
)

print("Training features head ------------------------")
print(X_train.head())
print("Classes head ----------------------------------")
print(y_train[:5])

# Check how often labels appear now
label_dist_encoded = pd.Series(y_train).value_counts().sort_index()
label_names = le.inverse_transform(label_dist_encoded.index)
label_dist = pd.DataFrame(
    {
        "label_code": label_dist_encoded.index,
        "label_name": label_names,
        "count": label_dist_encoded.values,
    }
)
print("Category distribution after oversampling ------")
print(label_dist)

# Train the classification model to predict categories
model_category = xgb.XGBClassifier(eval_metric="mlogloss", n_estimators=120)
model_category.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

feature_important = model_category.get_booster().get_score(importance_type="weight")
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(
    by="score", ascending=False
)
data.nlargest(40, columns="score").plot(kind="barh", figsize=(20, 10))
model_category.save_model("crime_prediction_model.json")

# Evaluation of the model
y_pred = model_category.predict(X_test)
print("Evaluation -----------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
