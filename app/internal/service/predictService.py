import pandas as pd
from abc import ABC, abstractmethod

from app.schemas import PredictRequest, PredictResponse, TransportOption
from app.internal.repository.modelRepository import IModelRepository

WALK_SPEED_MPS  = 1.2    # meter per detik
OJOL_SPEED_KPH  = 25     # km per jam saat macet pasca-konser
TRANS_SPEED_KPH = 30     # transjakarta/MRT lebih konsisten
OJOL_FLAG_FALL  = 9_000  # biaya buka pintu ojol (Rp)
OJOL_PER_KM     = 2_500  # biaya per km ojol normal (Rp)
TRANS_FLAT_RATE = 3_500  # tarif flat transjakarta (Rp)
SURGE_REDUCTION_WALK = 0.4  # asumsi surge turun jika user menjauh dari keramaian

SURGE_MIN = 1.0
SURGE_MAX = 3.5


class ITransportService(ABC):
    @abstractmethod
    def get_transport_recommendation(self, request: PredictRequest) -> PredictResponse:
        ...


class TransportServiceImpl(ITransportService):

    def __init__(self, repo: IModelRepository):
        self._repo = repo

    def get_transport_recommendation(self, request: PredictRequest) -> PredictResponse:
        # ambil jarak destinasi dari Repository
        dest_km = self._repo.get_destination_distance(request.destination_zone)

        # hitung jarak jalan kaki ke MRT via A* dari repository
        walk_to_mrt_m = self._repo.get_astar_path_length(
            request.current_location, "MRT_Istora"
        )

        # siapkan DataFrame untuk model ML 
        cat_input = pd.DataFrame([[
            request.day_type.value,
            request.concert_size.value,
            request.weather.value,
        ]], columns=["day_type", "concert_size", "weather"])

        num_input = pd.DataFrame([[
            request.concert_end_hour,
            request.time_since_end_minutes,
            int(walk_to_mrt_m),
        ]], columns=["concert_end_hour", "time_since_end_minutes",
                     "distance_to_pickup_meters"])

        # transformasi fitur via Repository
        features = self._repo.transform_features(cat_input, num_input)

        # prediksi surge 
        raw_surge = self._repo.predict_surge_raw(features)
        surge = round(max(SURGE_MIN, min(SURGE_MAX, raw_surge)), 2)

        # bangun ketiga opsi transportasi
        option_a = self._build_ojol_direct(request, dest_km, surge)
        option_b = self._build_ojol_walk(request, dest_km, surge)
        option_c = self._build_transjakarta(request, dest_km, walk_to_mrt_m)

        options = [option_a, option_b, option_c]

        # tentukan opsi terbaik berdasarkan biaya terendah
        best    = min(options, key=lambda x: x.estimated_cost_idr)
        savings = option_a.estimated_cost_idr - best.estimated_cost_idr

        rec_text = self._build_recommendation(surge, dest_km, best, savings,
                                              request.destination_zone)

        return PredictResponse(
            surge_multiplier=surge,
            options=options,
            best_option=best.mode,
            recommendation_text=rec_text,
        )

    # masing-masing opsi punya method sendiri 

    def _build_ojol_direct(
        self, request: PredictRequest, dest_km: float, surge: float
    ) -> TransportOption:
        # opsi A: ojol langsung dari pintu utama venue
        # hitung jarak jalan dari lokasi user ke Pintu 1
        if request.current_location == "Pintu_1_GBK":
            walk_m = 0
        else:
            walk_m = int(self._repo.get_astar_path_length(
                request.current_location, "Pintu_1_GBK"
            ))

        cost     = int(OJOL_FLAG_FALL + dest_km * OJOL_PER_KM * surge)
        walk_min = round(walk_m / WALK_SPEED_MPS / 60, 1)
        ride_min = round(dest_km / OJOL_SPEED_KPH * 60, 1)

        return TransportOption(
            mode="ojol_langsung",
            pickup_point="Pintu 1 GBK",
            walk_distance_meters=walk_m,
            estimated_cost_idr=cost,
            estimated_time_minutes=int(walk_min + ride_min),
        )

    def _build_ojol_walk(
        self, request: PredictRequest, dest_km: float, surge: float
    ) -> TransportOption:
        # opsi B: jalan dulu menjauh dari keramaian, surge lebih rendah
        walk_m = int(self._repo.get_astar_path_length(
            request.current_location, "Pintu_7_GBK"
        ))
        # asumsi surge turun saat user menjauh dari penumpukan massa
        surge_b = round(max(SURGE_MIN, surge - SURGE_REDUCTION_WALK), 2)
        cost     = int(OJOL_FLAG_FALL + dest_km * OJOL_PER_KM * surge_b)
        walk_min = round(walk_m / WALK_SPEED_MPS / 60, 1)
        ride_min = round(dest_km / OJOL_SPEED_KPH * 60, 1)

        return TransportOption(
            mode="ojol_jalan_dulu",
            pickup_point="Pintu 7 GBK",
            walk_distance_meters=walk_m,
            estimated_cost_idr=cost,
            estimated_time_minutes=int(walk_min + ride_min),
        )

    def _build_transjakarta(
        self, request: PredictRequest, dest_km: float, walk_to_mrt_m: float
    ) -> TransportOption:
        # opsi C: transjakarta/MRT tarif flat, tidak kena surge
        walk_m   = int(walk_to_mrt_m)
        walk_min = round(walk_m / WALK_SPEED_MPS / 60, 1)
        ride_min = round(dest_km / TRANS_SPEED_KPH * 60, 1)

        return TransportOption(
            mode="transjakarta",
            pickup_point="Stasiun MRT Istora",
            walk_distance_meters=walk_m,
            estimated_cost_idr=TRANS_FLAT_RATE,
            estimated_time_minutes=int(walk_min + ride_min),
        )

    def _build_recommendation(
        self, surge: float, dest_km: float,
        best: TransportOption, savings: int, destination: str,
    ) -> str:
        mode_label = best.mode.replace("_", " ").title()
        return (
            f"Surge saat ini {surge}x. "
            f"Jarak ke {destination}: {dest_km} km. "
            f"Rekomendasi terbaik: {mode_label} — "
            f"jalan kaki {best.walk_distance_meters}m ke {best.pickup_point}, "
            f"estimasi biaya Rp {best.estimated_cost_idr:,} "
            f"dengan waktu tempuh sekitar {best.estimated_time_minutes} menit. "
            f"Hemat Rp {savings:,} dibanding naik ojol langsung dari venue."
        )