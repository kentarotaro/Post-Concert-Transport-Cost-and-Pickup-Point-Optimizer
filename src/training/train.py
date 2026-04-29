"""
train.py
XGBoost surge-multiplier regression pipeline for GBK concert transport.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parents[2]
DATA   = BASE / "data"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA / "train" / "train.csv"
VAL_PATH   = DATA / "val"   / "val.csv"
TEST_PATH  = DATA / "test"  / "test.csv"

ENCODER_PATH  = MODELS / "encoder.pkl"
SCALER_PATH   = MODELS / "scaler.pkl"
FEATURES_PATH = MODELS / "feature_columns.json"
EVAL_PATH     = MODELS / "eval_results.json"
MODEL_PATH    = MODELS / "surge_predictor.pkl"

# ── Feature schema ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = ["day_type", "concert_size", "weather"]
NUMERIC_COLS     = ["concert_end_hour", "time_since_end_minutes", "distance_to_pickup_meters"]
FEATURE_COLS     = NUMERIC_COLS + CATEGORICAL_COLS   # order after transform
TARGET           = "surge_multiplier"

# OrdinalEncoder category ordering (matters for interpretability, not correctness)
CAT_CATEGORIES = [
    ["weekday", "weekend"],          # day_type
    ["small",   "medium", "large"],  # concert_size
    ["clear",   "cloudy", "rain"],   # weather
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_split(path: Path):
    df = pd.read_csv(path)
    X  = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y  = df[TARGET].values
    return X, y


def apply_transforms(X: pd.DataFrame, encoder: OrdinalEncoder,
                     scaler: StandardScaler) -> np.ndarray:
    X = X.copy()
    X[CATEGORICAL_COLS] = encoder.transform(X[CATEGORICAL_COLS])
    X[NUMERIC_COLS]     = scaler.transform(X[NUMERIC_COLS])
    return X[NUMERIC_COLS + CATEGORICAL_COLS].values


def regression_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def bucket_accuracy(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    def bucket(v):
        return "low" if v < 1.5 else ("high" if v > 2.5 else "medium")

    buckets  = {"low": [], "medium": [], "high": []}
    for yt, yp in zip(y_true, y_pred):
        b = bucket(yt)
        buckets[b].append(abs(yt - yp))

    result = {}
    for name, errors in buckets.items():
        arr = np.array(errors)
        result[name] = {
            "count":          int(len(arr)),
            "mean_abs_error": round(float(arr.mean()), 4) if len(arr) else None,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# [1/6] Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
print("[1/6] Preprocessing...")

X_train, y_train = load_split(TRAIN_PATH)
X_val,   y_val   = load_split(VAL_PATH)
X_test,  y_test  = load_split(TEST_PATH)

# Fit encoder and scaler on train only
encoder = OrdinalEncoder(categories=CAT_CATEGORIES,
                         handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(X_train[CATEGORICAL_COLS])

scaler = StandardScaler()
scaler.fit(X_train[NUMERIC_COLS])

# Transform all splits
X_train_t = apply_transforms(X_train, encoder, scaler)
X_val_t   = apply_transforms(X_val,   encoder, scaler)
X_test_t  = apply_transforms(X_test,  encoder, scaler)

# Save artifacts
joblib.dump(encoder, ENCODER_PATH)
joblib.dump(scaler,  SCALER_PATH)

# Final ordered feature list (numeric first, then categorical — matches apply_transforms)
final_features = NUMERIC_COLS + CATEGORICAL_COLS
with open(FEATURES_PATH, "w") as f:
    json.dump(final_features, f, indent=2)

print(f"  train {X_train_t.shape} | val {X_val_t.shape} | test {X_test_t.shape}")
print(f"  Artifacts saved: encoder.pkl, scaler.pkl, feature_columns.json")


# ══════════════════════════════════════════════════════════════════════════════
# [2/6] Baseline training
# ══════════════════════════════════════════════════════════════════════════════
print("[2/6] Baseline training...")

baseline = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
baseline.fit(X_train_t, y_train)

val_pred_baseline = baseline.predict(X_val_t)
bm = regression_metrics(y_val, val_pred_baseline)
print(f"  Baseline val  ->  RMSE {bm['rmse']}  MAE {bm['mae']}  R2 {bm['r2']}")


# ══════════════════════════════════════════════════════════════════════════════
# [3/6] Hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════
print("[3/6] Hyperparameter tuning (GridSearchCV cv=3)...")

param_grid = {
    "max_depth":     [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators":  [300, 500],
    "subsample":     [0.8, 1.0],
}

# Base estimator shares fixed params not in the grid
base_xgb = XGBRegressor(
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    refit=True,
)
grid_search.fit(X_train_t, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_

print(f"  Best params : {best_params}")
print(f"  Best CV RMSE : {-grid_search.best_score_:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# [4/6] Evaluating on test
# ══════════════════════════════════════════════════════════════════════════════
print("[4/6] Evaluating on test...")

y_pred_test = best_model.predict(X_test_t)

test_metrics = regression_metrics(y_test, y_pred_test)
buckets      = bucket_accuracy(y_test, y_pred_test)

print(f"  Test RMSE {test_metrics['rmse']}  MAE {test_metrics['mae']}  R2 {test_metrics['r2']}")
print("  Surge bucket breakdown:")
for bucket_name, stats in buckets.items():
    print(f"    {bucket_name:6s}  count={stats['count']:6d}  mean_abs_error={stats['mean_abs_error']}")

eval_results = {
    "best_params":   best_params,
    "test_metrics":  test_metrics,
    "surge_buckets": buckets,
}
with open(EVAL_PATH, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"  Results saved: eval_results.json")


# ══════════════════════════════════════════════════════════════════════════════
# [5/6] Saving model
# ══════════════════════════════════════════════════════════════════════════════
print("[5/6] Saving model...")

joblib.dump(best_model, MODEL_PATH)
print(f"  Saved: surge_predictor.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# [6/6] Sanity check
# ══════════════════════════════════════════════════════════════════════════════
print("[6/6] Sanity check...")

# Load artifacts fresh to exercise the full inference path
_encoder = joblib.load(ENCODER_PATH)
_scaler  = joblib.load(SCALER_PATH)
_model   = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    _feature_cols = json.load(f)


def predict_one(end_hour, day_type, concert_size, weather, t_min, dist_m) -> float:
    row = pd.DataFrame([{
        "concert_end_hour":         end_hour,
        "time_since_end_minutes":   t_min,
        "distance_to_pickup_meters": dist_m,
        "day_type":                 day_type,
        "concert_size":             concert_size,
        "weather":                  weather,
    }])
    row[CATEGORICAL_COLS] = _encoder.transform(row[CATEGORICAL_COLS])
    row[NUMERIC_COLS]     = _scaler.transform(row[NUMERIC_COLS])
    return float(_model.predict(row[_feature_cols].values)[0])


pred_A = predict_one(20, "weekday", "small",  "clear",  80, 120)
pred_B = predict_one(22, "weekend", "medium", "cloudy", 30, 380)
pred_C = predict_one(23, "weekend", "large",  "rain",    5, 650)

print(f"  Case A (low)  -> {pred_A:.4f}")
print(f"  Case B (mid)  -> {pred_B:.4f}")
print(f"  Case C (high) -> {pred_C:.4f}")

if pred_A < pred_B < pred_C:
    print("  Assertion PASSED: A < B < C")
else:
    print("  WARNING — model direction incorrect: expected A < B < C")

print("Done.")
