import logging
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import data, health, models

load_dotenv()

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8002))

logger = logging.getLogger(__name__)

app = FastAPI(
    title="portfolio-optimization api",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(data.router)
app.include_router(models.router)
# app.include_router(test.router)
# app.include_router(train.router)
app.include_router(health.router)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host=API_HOST, port=API_PORT, reload=True)
