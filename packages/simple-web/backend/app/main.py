from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.analyze import router as analyze_router
from .routes.health import router as health_router

MODEL_PATH = Path(os.getenv("JDVP_MODEL_PATH", "/app/models/embedding"))
ONNX_MODEL_PATH = Path(os.getenv("JDVP_ONNX_MODEL_PATH", "/app/models/onnx"))
REGRESSORS_PATH = Path(os.getenv("JDVP_REGRESSORS_PATH", "/app/regressors.pkl"))
BACKEND = os.getenv("JDVP_BACKEND", "onnx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if BACKEND == "onnx":
        from .services.inference_onnx import InferenceServiceONNX
        app.state.inference = InferenceServiceONNX(
            onnx_model_dir=str(ONNX_MODEL_PATH),
            regressors_path=str(REGRESSORS_PATH),
        )
    else:
        from .services.inference import InferenceService
        app.state.inference = InferenceService(
            model_path=str(MODEL_PATH),
            regressors_path=str(REGRESSORS_PATH),
        )
    yield


app = FastAPI(title="JDVP Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")
app.include_router(health_router, prefix="/api")
