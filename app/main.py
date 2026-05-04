from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.schemas import PredictRequest, PredictResponse
from app.inference import predict_transport

app = FastAPI(
    title="GBK Concert Transport Optimizer API",
    description="Prediksi surge price dan rekomendasi transportasi pasca konser GBK Jakarta.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.hf.space", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model": "surge_predictor_xgboost", "version": "1.0.0", "venue": "GBK Jakarta"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": True, "version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    try:
        return predict_transport(request)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
@app.middleware("http")
async def limit_payload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 1024:
        return JSONResponse(status_code=413,
                            content={"status": 413, "error": "Request Entity Too Large"})
    return await call_next(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=False)