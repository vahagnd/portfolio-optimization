from pydantic import BaseModel


class ModelInfoBase(BaseModel):
    available: bool
    size_mb: float | None = None
    info: dict | None = None
    path: str | None = None


class ModelInfo(ModelInfoBase):
    model_type: str
