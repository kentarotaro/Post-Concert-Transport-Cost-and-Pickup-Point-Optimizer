import json
import joblib
import networkx as nx
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path


class IModelRepository(ABC):
    @abstractmethod
    def transform_features(self, cat_df: pd.DataFrame,
                           num_df: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def predict_surge_raw(self, features: np.ndarray) -> float:
        ...

    @abstractmethod
    def get_destination_distance(self, zone: str) -> float:
        ...

    @abstractmethod
    def get_astar_path_length(self, start: str, end: str) -> float:
        ...

    @abstractmethod
    def get_pickup_points(self, venue_name: str) -> list[dict]:
        ...


class ModelRepository(IModelRepository):
    def __init__(self):
        base = Path(__file__).resolve().parent.parent.parent
        self._model_dir = base / "models"
        self._data_dir  = base / "data" / "processed"

        # validasi direktori saat startup 
        if not self._model_dir.exists() or not self._data_dir.exists():
            raise FileNotFoundError(
                "Direktori models/ atau data/processed/ tidak ditemukan. "
                "Pastikan artifacts sudah ada sebelum server dijalankan."
            )

        self._encoder = joblib.load(self._model_dir / "encoder.pkl")
        self._scaler  = joblib.load(self._model_dir / "scaler.pkl")
        self._model   = joblib.load(self._model_dir / "surge_predictor.pkl")

        # load data statis JSON
        self._destinations: dict = json.loads(
            (self._data_dir / "destinations.json").read_text(encoding="utf-8")
        )
        self._pickups: dict = json.loads(
            (self._data_dir / "venue_pickup_points.json").read_text(encoding="utf-8")
        )

        # bangun graph A* 
        self._graph, self._node_coords = self._build_graph()

    def _build_graph(self) -> tuple[nx.Graph, dict]:
        G = nx.Graph()

        nodes = [
            "Pintu_1_GBK", "Pintu_7_GBK", "Bundaran_Senayan",
            "MRT_Istora",  "FX_Sudirman",
            "Pintu_1_JIS", "Pintu_2_JIS",
            "Gate_A_ICE",  "Gate_B_ICE",
        ]
        G.add_nodes_from(nodes)

        # bobot dalam meter
        edges = [
            ("Pintu_1_GBK",      "Pintu_7_GBK",        260),
            ("Pintu_1_GBK",      "Bundaran_Senayan",    330),
            ("Pintu_7_GBK",      "MRT_Istora",          270),
            ("Pintu_7_GBK",      "Bundaran_Senayan",    200),
            ("Bundaran_Senayan", "FX_Sudirman",         250),
            ("MRT_Istora",       "FX_Sudirman",         300),
            ("Pintu_1_JIS",      "Pintu_2_JIS",         180),
            ("Gate_A_ICE",       "Gate_B_ICE",          220),
        ]
        G.add_weighted_edges_from(edges)

        node_coords = {
            "Pintu_1_GBK":      (-6.2183, 106.8023),
            "Pintu_7_GBK":      (-6.2195, 106.7991),
            "Bundaran_Senayan": (-6.2201, 106.8001),
            "MRT_Istora":       (-6.2228, 106.8013),
            "FX_Sudirman":      (-6.2245, 106.7997),
            "Pintu_1_JIS":      (-6.1275, 106.8027),
            "Pintu_2_JIS":      (-6.1289, 106.8041),
            "Gate_A_ICE":       (-6.3013, 106.6533),
            "Gate_B_ICE":       (-6.3025, 106.6548),
        }
        return G, node_coords

    def _heuristic(self, u: str, v: str) -> float:
        # euclidean heuristic dari koordinat GPS untuk A*.
        lat1, lon1 = self._node_coords[u]
        lat2, lon2 = self._node_coords[v]
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5 * 111_000

    # metode interface

    def transform_features(self, cat_df: pd.DataFrame,
                           num_df: pd.DataFrame) -> np.ndarray:
  
        cat_encoded = self._encoder.transform(cat_df)
        num_scaled  = self._scaler.transform(num_df)
        return np.hstack([num_scaled, cat_encoded])

    def predict_surge_raw(self, features: np.ndarray) -> float:
        return float(self._model.predict(features)[0])

    def get_destination_distance(self, zone: str) -> float:
        return self._destinations.get(zone, 8.0)

    def get_astar_path_length(self, start: str, end: str) -> float:
        try:
            return nx.astar_path_length(
                self._graph, start, end,
                heuristic=self._heuristic,
                weight="weight",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 650.0  # fallback jika node tidak terhubung

    def get_pickup_points(self, venue_name: str) -> list[dict]:
        return self._pickups.get(venue_name, [])


class MockModelRepository(IModelRepository):

    def transform_features(self, cat_df, num_df) -> np.ndarray:
        return np.array([[0.5, 0.7, 0.3, 0.8, 0.4, 0.6]])

    def predict_surge_raw(self, features: np.ndarray) -> float:
        return 2.3  # nilai tetap, predictable untuk assertion

    def get_destination_distance(self, zone: str) -> float:
        return 10.0

    def get_astar_path_length(self, start: str, end: str) -> float:
        return 270.0

    def get_pickup_points(self, venue_name: str) -> list[dict]:
        return [{"name": f"Pintu Utama {venue_name}"}]