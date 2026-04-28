from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse
from app.inference import predict_transport

app = FastAPI(
    title="GBK Concert Transport Optimizer API",
    description="Prediksi surge price dan rekomendasi transportasi pasca konser GBK Jakarta.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": "surge_predictor_xgboost", "version": "1.0.0", "venue": "GBK Jakarta"}

@app.get("/health", tags=["Health"])
def health():
    try:
        from pathlib import Path
        import joblib
        model_path = Path(__file__).resolve().parent.parent / "models" / "surge_predictor.pkl"
        if not model_path.exists():
            raise FileNotFoundError("Model tidak ditemukan.")
        joblib.load(model_path)
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    try:
        return predict_transport(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)
