import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "FastAPI Data Mining"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "dev-secret-change-me")
    DATASET_DIR: str = os.getenv("DATASET_DIR", "data/datasets")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "data/models")
    SEED_DIR: str = os.getenv("SEED_DIR", "data/seed")


settings = Settings()
