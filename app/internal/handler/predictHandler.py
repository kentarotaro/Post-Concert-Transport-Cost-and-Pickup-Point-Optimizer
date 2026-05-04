from fastapi import APIRouter
from app.schemas import PredictRequest, PredictResponse, HealthResponse
from app.internal.service.predictService import ITransportService

router = APIRouter()


class TransportHandler:
    def __init__(self, service: ITransportService):
        self._service = service

    def register_routes(self) -> APIRouter:
        router = APIRouter()  # ← pindahkan ke dalam method!
        service = self._service  # ← capture dulu sebelum closure

        @router.get("/", tags=["info"])
        async def root():
            return {"name": "Post-Concert Transport Optimizer", "version": "1.0.0"}

        @router.get("/health", tags=["info"])
        async def health():
            return {"status": "ok", "version": "1.0.0"}

        @router.post("/predict", response_model=PredictResponse, tags=["predict"])
        async def predict(req: PredictRequest):
            return service.get_transport_recommendation(req)  # ← pakai captured service

        return router