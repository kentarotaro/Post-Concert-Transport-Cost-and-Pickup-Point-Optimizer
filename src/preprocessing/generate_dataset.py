import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Load dan merge data asli ─────────────────────────────────────────────────
rides   = pd.read_csv(DATA_DIR / "raw" / "cab_prices" / "cab_rides.csv")
weather = pd.read_csv(DATA_DIR / "raw" / "cab_prices" / "weather.csv")

rides["datetime"]   = pd.to_datetime(rides["time_stamp"] / 1000, unit="s")
weather["datetime"] = pd.to_datetime(weather["time_stamp"], unit="s")

rides["hour_key"]   = rides["datetime"].dt.floor("h")
weather["hour_key"] = weather["datetime"].dt.floor("h")
weather_agg = weather.groupby("hour_key")[["rain","clouds","temp"]].mean().reset_index()

df = rides.merge(weather_agg, on="hour_key", how="left").dropna()

# ── Encode fitur dasar ───────────────────────────────────────────────────────
def encode_weather(row):
    if row["rain"] > 0.2:   return "rain"
    elif row["clouds"] > 0.5: return "cloudy"
    else:                    return "clear"

df["weather"]      = df.apply(encode_weather, axis=1)
df["day_type"]     = df["datetime"].dt.dayofweek.apply(
    lambda x: "weekend" if x >= 5 else "weekday"
)
df["concert_end_hour"] = (df["datetime"].dt.hour % 6) + 19

# Weighted distribution: konser lebih sering selesai jam 21-23
hour_weights = {19: 0.05, 20: 0.10, 21: 0.25, 22: 0.35, 23: 0.20, 24: 0.05}
df["concert_end_hour"] = np.random.choice(
    list(hour_weights.keys()),
    size=len(df),
    p=list(hour_weights.values())
)

df["concert_size"] = np.random.choice(
    ["small", "medium", "large"],
    size=len(df),
    p=[0.20, 0.50, 0.30]
)
df["time_since_end_minutes"] = np.random.randint(0, 91, size=len(df))

PICKUP_POOL = {
    "Pintu_1_GBK": 120, "Pintu_7_GBK": 380,
    "Bundaran_Senayan": 450, "MRT_Istora": 650, "FX_Sudirman": 900
}
pickup_choice = np.random.choice(
    list(PICKUP_POOL.keys()), size=len(df),
    p=[0.15, 0.35, 0.20, 0.20, 0.10]
)
df["pickup_point"] = pickup_choice
df["distance_to_pickup_meters"] = df["pickup_point"].map(PICKUP_POOL)
df["venue_name"] = "GBK"

# ── Formula surge yang REALISTIS ─────────────────────────────────────────────
SIZE_FACTOR = {"small": 1.5, "medium": 2.2, "large": 2.8}
WEATHER_AMP = {"clear": 1.0, "cloudy": 1.1, "rain": 1.3}

def compute_surge(row):
    decay = max(0, 1 - (row["time_since_end_minutes"] / 90))
    sf    = SIZE_FACTOR[row["concert_size"]]
    base  = 1.0 + (sf - 1.0) * decay

    # Weather
    base *= WEATHER_AMP[row["weather"]]

    # Day type — efek diperbesar agar XGBoost bisa deteksi
    if row["day_type"] == "weekend":
        base *= 1.30  # naik dari 1.15 → 1.30

    # Concert end hour — efek diperbesar
    hour_factor = {
        19: 1.00, 20: 1.08, 21: 1.16,
        22: 1.25, 23: 1.35, 24: 1.45  # range lebih lebar
    }
    base *= hour_factor.get(row["concert_end_hour"], 1.15)

    # Distance — efek diperbesar
    dist_factor = max(0.60, 1.0 - (row["distance_to_pickup_meters"] / 2000))
    base *= dist_factor

    # Noise dikecilkan: ±5% saja (bukan ±10%)
    noise = np.random.normal(loc=0, scale=0.05)
    base *= (1 + noise)

    return round(float(np.clip(base, 1.0, 3.5)), 2)

print("Menghitung surge_multiplier baru...")
df["surge_multiplier"] = df.apply(compute_surge, axis=1)

# ── Pilih hanya kolom yang diperlukan ────────────────────────────────────────
KEEP = [
    "concert_end_hour", "day_type", "concert_size", "weather",
    "time_since_end_minutes", "distance_to_pickup_meters", "surge_multiplier"
]
df = df[KEEP].reset_index(drop=True)

print(f"Total rows: {len(df)}")
print(df["surge_multiplier"].describe())
print(f"High-surge (>3.0): {(df['surge_multiplier'] > 3.0).sum()}")

# ── Clean + Split ────────────────────────────────────────────────────────────
df = df.drop_duplicates(keep="first")

train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test   = train_test_split(temp, test_size=0.5, random_state=42)

# Oversample high-surge HANYA di train
high = train[train["surge_multiplier"] > 3.0]
if len(high) < 2000:
    needed = 2000 - len(high)
    extra  = high.sample(n=needed, replace=True, random_state=42)
    train  = pd.concat([train, extra]).sample(frac=1, random_state=42).reset_index(drop=True)

# Simpan
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "train").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "val").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "test").mkdir(parents=True, exist_ok=True)

df.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
train.to_csv(DATA_DIR / "train" / "train.csv", index=False)
val.to_csv(DATA_DIR / "val" / "val.csv", index=False)
test.to_csv(DATA_DIR / "test" / "test.csv", index=False)

print(f"\nSelesai — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
print(f"Surge distribution (train): mean={train['surge_multiplier'].mean():.3f}, std={train['surge_multiplier'].std():.3f}")