import logging

from fastapi import APIRouter, HTTPException

from app.schemas.data import DatasetInfo, StocksResponse
from config import DF_MAG7_RAW

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Data"])


@router.get(
    "/data/stocks",
    description="Get list of available stocks.",
    summary="stocks in dataset",
)
async def get_stocks() -> StocksResponse:
    """Get list of available stocks"""
    try:
        stocks = list(DF_MAG7_RAW.iloc[0][1:].unique())
        return {"stocks": stocks, "count": len(stocks)}
    except Exception as e:
        logger.error(f"Error getting stock info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/data/info", description="Get dataset information.", summary="dataset info"
)
async def get_data_info() -> DatasetInfo:
    """Get information about the dataset"""
    try:
        return {
            "total_records": len(DF_MAG7_RAW),
            "stocks": list(DF_MAG7_RAW.iloc[0][1:].unique()),
            "date_range": {
                "start": str(DF_MAG7_RAW.iloc[2, 0]),
                "end": str(DF_MAG7_RAW.iloc[-1, 0]),
            },
            "memory_usage_mb": round(
                DF_MAG7_RAW.memory_usage(deep=True).sum() / 1024**2, 2
            ).item(),
        }
    except Exception as e:
        logger.error(f"Error getting data info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
