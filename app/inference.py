import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from app.schemas import PredictRequest, PredictResponse, TransportOption

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

encoder         = joblib.load(MODEL_DIR / "encoder.pkl")
scaler          = joblib.load(MODEL_DIR / "scaler.pkl")
model           = joblib.load(MODEL_DIR / "surge_predictor.pkl")
feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())
DESTINATIONS    = json.loads((DATA_DIR / "destinations.json").read_text(encoding="utf-8"))

G = nx.Graph()

nodes = [
    "Pintu_1_GBK",
    "Pintu_7_GBK",
    "Bundaran_Senayan",
    "MRT_Istora",
    "FX_Sudirman",
]
G.add_nodes_from(nodes)

edges = [
    ("Pintu_1_GBK",      "Pintu_7_GBK",       260),
    ("Pintu_1_GBK",      "Bundaran_Senayan",   330),
    ("Pintu_7_GBK",      "MRT_Istora",         270),
    ("Pintu_7_GBK",      "Bundaran_Senayan",   200),
    ("Bundaran_Senayan",  "FX_Sudirman",        250),
    ("MRT_Istora",        "FX_Sudirman",        300),
]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

NODE_COORDS = {
    "Pintu_1_GBK":      (-6.2183, 106.8023),
    "Pintu_7_GBK":      (-6.2195, 106.7991),
    "Bundaran_Senayan": (-6.2201, 106.8001),
    "MRT_Istora":       (-6.2228, 106.8013),
    "FX_Sudirman":      (-6.2245, 106.7997),
}

def heuristic(u, v):
    lat1, lon1 = NODE_COORDS[u]
    lat2, lon2 = NODE_COORDS[v]
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111000

WALK_SPEED_MPS  = 1.2
OJOL_SPEED_KPH  = 25
TRANS_SPEED_KPH = 30
DEFAULT_DEST_KM = 8.0

OJOL_FLAG_FALL  = 9000
OJOL_PER_KM     = 2500
TRANS_FLAT_RATE = 3500


def predict_transport(request: PredictRequest) -> PredictResponse:
    dest_km = DESTINATIONS.get(request.destination_zone, DEFAULT_DEST_KM)

    cat_input = pd.DataFrame([[
        request.day_type,
        request.concert_size,
        request.weather,
    ]], columns=["day_type", "concert_size", "weather"])
    cat_encoded = encoder.transform(cat_input)

    try:
        walk_to_mrt = nx.astar_path_length(
            G, request.current_location, "MRT_Istora",
            heuristic=heuristic, weight="weight"
        )
    except nx.NetworkXNoPath:
        walk_to_mrt = 650

    num_input = pd.DataFrame([[
        request.concert_end_hour,
        request.time_since_end_minutes,
        int(walk_to_mrt),
    ]], columns=["concert_end_hour", "time_since_end_minutes", "distance_to_pickup_meters"])
    num_scaled = scaler.transform(num_input)

    feature_array = np.hstack([num_scaled, cat_encoded])

    surge = round(float(model.predict(feature_array)[0]), 2)
    surge = max(1.0, min(surge, 3.5))

    walk_a = 0 if request.current_location == "Pintu_1_GBK" else int(
        nx.astar_path_length(G, request.current_location, "Pintu_1_GBK",
                             heuristic=heuristic, weight="weight")
    )
    cost_a = int(OJOL_FLAG_FALL + (dest_km * OJOL_PER_KM * surge))
    twalka = round(walk_a / WALK_SPEED_MPS / 60, 1)
    tridea = round(dest_km / OJOL_SPEED_KPH * 60, 1)
    time_a = int(twalka + tridea)

    option_a = TransportOption(
        mode="ojol_langsung",
        pickup_point="Pintu 1 GBK",
        walk_distance_meters=walk_a,
        estimated_cost_idr=cost_a,
        estimated_time_minutes=time_a,
    )

    walk_b  = int(nx.astar_path_length(
        G, request.current_location, "Pintu_7_GBK",
        heuristic=heuristic, weight="weight"
    ))
    surge_b = round(max(surge - 0.4, 1.0), 2)
    cost_b  = int(OJOL_FLAG_FALL + (dest_km * OJOL_PER_KM * surge_b))
    twalkb  = round(walk_b / WALK_SPEED_MPS / 60, 1)
    trideb  = round(dest_km / OJOL_SPEED_KPH * 60, 1)
    time_b  = int(twalkb + trideb)

    option_b = TransportOption(
        mode="ojol_jalan_dulu",
        pickup_point="Pintu 7 GBK",
        walk_distance_meters=walk_b,
        estimated_cost_idr=cost_b,
        estimated_time_minutes=time_b,
    )

    walk_c  = int(walk_to_mrt)
    cost_c  = TRANS_FLAT_RATE
    twalkc  = round(walk_c / WALK_SPEED_MPS / 60, 1)
    tridec  = round(dest_km / TRANS_SPEED_KPH * 60, 1)
    time_c  = int(twalkc + tridec)

    option_c = TransportOption(
        mode="transjakarta",
        pickup_point="Stasiun MRT Istora",
        walk_distance_meters=walk_c,
        estimated_cost_idr=cost_c,
        estimated_time_minutes=time_c,
    )

    options = [option_a, option_b, option_c]
    best    = min(options, key=lambda x: x.estimated_cost_idr)
    savings = cost_a - best.estimated_cost_idr

    rec_text = (
        f"Surge saat ini {surge}x. "
        f"Jarak ke {request.destination_zone}: {dest_km} km. "
        f"Rekomendasi terbaik: {best.mode.replace('_', ' ').title()} — "
        f"jalan kaki {best.walk_distance_meters}m ke {best.pickup_point}, "
        f"estimasi biaya Rp {best.estimated_cost_idr:,} "
        f"dengan waktu tempuh sekitar {best.estimated_time_minutes} menit. "
        f"Hemat Rp {savings:,} dibanding naik ojol langsung dari venue."
    )

    return PredictResponse(
        surge_multiplier=surge,
        best_option=best.mode,
        recommendation_text=rec_text,
        options=options,
    )