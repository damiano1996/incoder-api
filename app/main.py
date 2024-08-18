import os
import threading
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from configs import logger
from llm import FacebookInFiller

# Environment variables and configuration
BIG_MODEL: bool = os.getenv("BIG_MODEL", "False").lower() in ("true", "1", "t")
CUDA: bool = os.getenv("CUDA", "True").lower() in ("true", "1", "t")
CUDA_AVAILABLE: bool = torch.cuda.is_available()
HF_HOME: str = os.getenv("HF_HOME")
HF_HUB_CACHE: str = os.getenv("HF_HUB_CACHE")


class ModelStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    INITIALIZING = "INITIALIZING"
    DOWNLOADING = "DOWNLOADING"
    READY = "READY"
    ERROR = "ERROR"


# Placeholder for the model
model_status: ModelStatus = ModelStatus.UNKNOWN
model: Optional[FacebookInFiller] = None


def load_model_in_background():
    global model, model_status

    try:
        logger.info("Initializing the model in the background")
        model_status = ModelStatus.INITIALIZING
        model = FacebookInFiller(big_model=BIG_MODEL, cuda=CUDA)

        if not model.is_model_available():
            logger.info("Model is not available, the model will be downloaded")
            model_status = ModelStatus.DOWNLOADING

        logger.info("Starting model initialization")
        model.init_model()

        model_status = ModelStatus.READY
        logger.info("Model initialized")
    except Exception as e:
        logger.error(f"Unable to initialize the model: {e}")
        model_status = ModelStatus.ERROR


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Startup
    logger.info(f"Big model: {BIG_MODEL}")
    logger.info(f"Use CUDA: {CUDA}")
    logger.info(f"CUDA available: {CUDA_AVAILABLE}")
    logger.info(f"Hugging Face - Home: {HF_HOME}, Hub Cache: {HF_HUB_CACHE}")

    # Start loading the model in a separate thread
    thread = threading.Thread(target=load_model_in_background)
    thread.start()

    yield

    # Shutdown
    if thread.is_alive():
        thread.join()


app = FastAPI(lifespan=lifespan)


class Status(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


class HealthCheckResponse(BaseModel):
    status: str
    modelStatus: str
    envVariables: dict


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    env_variables = {
        "BIG_MODEL": BIG_MODEL,
        "CUDA": CUDA,
        "CUDA_AVAILABLE": CUDA_AVAILABLE,
        "HF_HOME": HF_HOME,
        "HF_HUB_CACHE": HF_HUB_CACHE,
    }

    return HealthCheckResponse(
        status=Status.UP.name,
        modelStatus=model_status.name,
        envVariables=env_variables,
    )


class AutocompleteRequest(BaseModel):
    leftContext: str
    rightContext: str
    language: Optional[str] = None
    ide: Optional[str] = None
    pluginVersion: Optional[str] = None


class AutocompleteResponse(BaseModel):
    prediction: str


@app.post("/api/v1/code/complete", response_model=AutocompleteResponse)
async def code_complete(
    request: AutocompleteRequest,
):
    logger.info(f"Received autocomplete request: {request.dict()}")

    logger.info(
        f"Generating completion given LEFT context:\n"
        f"{request.leftContext}\n"
        f"and RIGHT context:\n"
        f"{request.rightContext}"
    )

    prediction = model.infill(request.leftContext, request.rightContext)

    logger.info(f"Generated prediction: {prediction}")

    return AutocompleteResponse(
        prediction=prediction,
    )
