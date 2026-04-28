from pydantic import BaseModel, Field
from typing import List, Literal

class PredictRequest(BaseModel):
    venue_name: Literal["GBK"] = Field(default="GBK")
    concert_end_hour: int = Field(..., ge=19, le=24)
    day_type: Literal["weekday", "weekend"]
    concert_size: Literal["small", "medium", "large"]
    weather: Literal["clear", "cloudy", "rain"]
    time_since_end_minutes: int = Field(..., ge=0, le=90)
    destination_zone: str
    current_location: Literal["Pintu_1_GBK","Pintu_7_GBK","Bundaran_Senayan","MRT_Istora","FX_Sudirman"] = Field(default="Pintu_1_GBK")

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
