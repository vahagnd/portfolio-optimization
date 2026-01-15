import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException

from app.schemas.models import ModelInfo, ModelInfoBase
from config import LATEST_MODEL_PATH

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Models"])


@router.get("/models", description="List all available models.", summary="list models")
async def list_models() -> dict[str, ModelInfoBase]:
    """List all available models and their info."""
    try:
        models_info = {}
        model_path = Path(LATEST_MODEL_PATH)

        for model_type in ["lstm", "fc"]:
            model_dir = model_path / model_type
            weights_file = model_dir / "weights.pth"
            info_file = model_dir / "info.json"

            if weights_file.exists():
                file_size = weights_file.stat().st_size / (1024 * 1024)

                info_data = {}
                if info_file.exists():
                    with open(info_file) as f:
                        info_data = json.load(f)

                models_info[model_type] = {
                    "available": True,
                    "size_mb": round(file_size, 2),
                    "info": info_data,
                    "path": str(weights_file),
                }
            else:
                models_info[model_type] = {"available": False}

        return models_info

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/models/{model_type}",
    description="Get information about a specific model.",
    summary="model info",
)
async def get_model_info(model_type: Literal["lstm", "fc"]) -> ModelInfo:
    """Get information about a specific model"""
    if model_type not in ["lstm", "fc"]:
        raise HTTPException(status_code=400, detail="model_type must be 'lstm' or 'fc'")

    try:
        model_dir = Path(LATEST_MODEL_PATH) / model_type
        weights_file = model_dir / "weights.pth"
        info_file = model_dir / "info.json"

        if not weights_file.exists():
            raise HTTPException(status_code=404, detail=f"{model_type} model not found")

        file_size = weights_file.stat().st_size / (1024 * 1024)

        info_data = {}
        if info_file.exists():
            with open(info_file) as f:
                info_data = json.load(f)

        return {
            "model_type": model_type,
            "available": True,
            "size_mb": round(file_size, 2),
            "info": info_data,
            "path": str(weights_file),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
