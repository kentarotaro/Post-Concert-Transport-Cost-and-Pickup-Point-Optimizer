import json, joblib, numpy as np, pandas as pd, networkx as nx
from pathlib import Path
from app.schemas import PredictRequest, PredictResponse, TransportOption

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
encoder         = joblib.load(MODEL_DIR / "encoder.pkl")
scaler          = joblib.load(MODEL_DIR / "scaler.pkl")
model           = joblib.load(MODEL_DIR / "surge_predictor.pkl")
feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())

G = nx.Graph()
G.add_nodes_from(["Pintu_1_GBK","Pintu_7_GBK","Bundaran_Senayan","MRT_Istora","FX_Sudirman"])
for u,v,w in [
    ("Pintu_1_GBK","Pintu_7_GBK",260),
    ("Pintu_1_GBK","Bundaran_Senayan",330),
    ("Pintu_7_GBK","MRT_Istora",270),
    ("Pintu_7_GBK","Bundaran_Senayan",200),
    ("Bundaran_Senayan","FX_Sudirman",250),
    ("MRT_Istora","FX_Sudirman",300),
]: G.add_edge(u,v,weight=w)

NODE_COORDS = {
    "Pintu_1_GBK":(-6.2183,106.8023),
    "Pintu_7_GBK":(-6.2195,106.7991),
    "Bundaran_Senayan":(-6.2201,106.8001),
    "MRT_Istora":(-6.2228,106.8013),
    "FX_Sudirman":(-6.2245,106.7997),
}

def heuristic(u,v):
    la1,lo1=NODE_COORDS[u]; la2,lo2=NODE_COORDS[v]
    return ((la1-la2)**2+(lo1-lo2)**2)**0.5*111000

WALK_MPS=1.2; OJOL_KPH=25; TRANS_KPH=30; DIST_KM=8.0
OJOL_FLAG=9000; OJOL_KM=2500; TRANS_FLAT=3500

def predict_transport(request: PredictRequest) -> PredictResponse:
    cat_input = pd.DataFrame([[request.day_type,request.concert_size,request.weather]],
                              columns=["day_type","concert_size","weather"])
    cat_enc = encoder.transform(cat_input)
    try: walk_mrt = nx.astar_path_length(G,request.current_location,"MRT_Istora",heuristic=heuristic,weight="weight")
    except: walk_mrt = 650
    num_input = pd.DataFrame([[request.concert_end_hour,request.time_since_end_minutes,int(walk_mrt)]],
                              columns=["concert_end_hour","time_since_end_minutes","distance_to_pickup_meters"])
    num_sc = scaler.transform(num_input)
    feat   = np.hstack([num_sc, cat_enc])
    surge  = round(float(max(1.0,min(model.predict(feat)[0],3.5))),2)

    def walk_dist(src,dst):
        try: return int(nx.astar_path_length(G,src,dst,heuristic=heuristic,weight="weight"))
        except: return 0

    wA = walk_dist(request.current_location,"Pintu_1_GBK") if request.current_location!="Pintu_1_GBK" else 0
    cA = int(OJOL_FLAG+(DIST_KM*OJOL_KM*surge))
    tA = int(wA/WALK_MPS/60 + DIST_KM/OJOL_KPH*60)
    optA = TransportOption(mode="ojol_langsung",pickup_point="Pintu 1 GBK",walk_distance_meters=wA,estimated_cost_idr=cA,estimated_time_minutes=tA)

    wB = walk_dist(request.current_location,"Pintu_7_GBK")
    sB = round(max(surge-0.4,1.0),2)
    cB = int(OJOL_FLAG+(DIST_KM*OJOL_KM*sB))
    tB = int(wB/WALK_MPS/60 + DIST_KM/OJOL_KPH*60)
    optB = TransportOption(mode="ojol_jalan_dulu",pickup_point="Pintu 7 GBK",walk_distance_meters=wB,estimated_cost_idr=cB,estimated_time_minutes=tB)

    wC = int(walk_mrt)
    cC = TRANS_FLAT
    tC = int(wC/WALK_MPS/60 + DIST_KM/TRANS_KPH*60)
    optC = TransportOption(mode="transjakarta",pickup_point="Stasiun MRT Istora",walk_distance_meters=wC,estimated_cost_idr=cC,estimated_time_minutes=tC)

    options = [optA,optB,optC]
    best    = min(options,key=lambda x: x.estimated_cost_idr)
    savings = cA - best.estimated_cost_idr
    rec     = (f"Surge saat ini {surge}x. Rekomendasi: {best.mode.replace('_',' ').title()} - "
               f"jalan kaki {best.walk_distance_meters}m ke {best.pickup_point}, "
               f"estimasi biaya Rp {best.estimated_cost_idr:,} (~{best.estimated_time_minutes} menit). "
               f"Hemat Rp {savings:,} vs ojol langsung.")

    return PredictResponse(surge_multiplier=surge,best_option=best.mode,recommendation_text=rec,options=options)
