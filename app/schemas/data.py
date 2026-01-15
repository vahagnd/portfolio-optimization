"""pydantic schemas for data"""
from pydantic import BaseModel


class StocksResponse(BaseModel):
    stocks: list[str]
    count: int


class DatasetInfo(BaseModel):
    total_records: int
    stocks: list[str]
    date_range: dict[str, str]
    memory_usage_mb: float
