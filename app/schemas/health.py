from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    system: str
    device: str
    latest_models: dict
    data: dict
