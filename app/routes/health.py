import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas.health import HealthResponse, StatusResponse
from config import DEVICE, DF_MAG7_RAW, LATEST_MODEL_PATH

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", description="Health check endpoint.", summary="health check")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    return {"status": "healthy"}


@router.get(
    "/status",
    description="Get system status and available models.",
    summary="system status",
)
async def get_status() -> StatusResponse:
    """Get system status and available models"""
    try:
        lstm_model_path = Path(LATEST_MODEL_PATH) / "lstm" / "weights.pth"
        fc_model_path = Path(LATEST_MODEL_PATH) / "fc" / "weights.pth"

        return {
            "system": "ok",
            "device": str(DEVICE),
            "latest_models": {
                "lstm": {
                    "available": lstm_model_path.exists(),
                    "path": str(lstm_model_path),
                },
                "fc": {
                    "available": fc_model_path.exists(),
                    "path": str(fc_model_path),
                },
            },
            "data": {
                "stocks": list(DF_MAG7_RAW.iloc[0][1:].unique()),
                "count": len(DF_MAG7_RAW.iloc[0][1:].unique()),
            },
        }
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
