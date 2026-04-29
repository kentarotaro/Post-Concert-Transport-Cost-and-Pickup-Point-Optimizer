# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal

class PredictRequest(BaseModel):
    venue_name: Literal["GBK"] = Field(
        default="GBK",
        description="Nama venue konser (saat ini hanya GBK)"
    )
    concert_end_hour: int = Field(
        ..., ge=19, le=24,
        description="Jam selesai konser (19-24)"
    )
    day_type: Literal["weekday", "weekend"] = Field(
        ..., description="Jenis hari pelaksanaan"
    )
    concert_size: Literal["small", "medium", "large"] = Field(
        ..., description="Kapasitas penonton"
    )
    weather: Literal["clear", "cloudy", "rain"] = Field(
        ..., description="Kondisi cuaca"
    )
    time_since_end_minutes: int = Field(
        ..., ge=0, le=90,
        description="Menit sejak konser selesai (0-90)"
    )
    destination_zone: str = Field(
        ..., description="Zona tujuan (contoh: Jakarta Selatan)"
    )
    current_location: Literal[
        "Pintu_1_GBK",
        "Pintu_7_GBK",
        "Bundaran_Senayan",
        "MRT_Istora",
        "FX_Sudirman"
    ] = Field(
        default="Pintu_1_GBK",
        description="Posisi pengguna saat ini di sekitar GBK"
    )

class TransportOption(BaseModel):
    mode: str
    pickup_point: str
    walk_distance_meters: int
    estimated_cost_idr: int
    estimated_time_minutes: int

class PredictResponse(BaseModel):
    surge_multiplier: float
    best_option: str
    recommendation_text: str
    options: List[TransportOption]