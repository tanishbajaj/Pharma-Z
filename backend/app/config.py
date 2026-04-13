"""
Pharma-Z Application Configuration

Loads settings from environment variables / .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    # --- Application ---
    app_name: str = "Pharma-Z"
    app_version: str = "1.0.0"
    debug: bool = True

    # --- Database ---
    database_url: str = "sqlite:///./pharma_z.db"

    # --- Blockchain (Ganache) ---
    blockchain_provider_url: str = "http://127.0.0.1:8545"
    contract_address: str = ""

    # --- Privacy ---
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0

    # --- ML Training Defaults ---
    default_batch_size: int = 64
    default_epochs: int = 50
    default_learning_rate: float = 0.001
    default_latent_dim: int = 128

    # --- Paths ---
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    saved_models_dir: str = "data/saved_models"
    logs_dir: str = "logs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance — import this everywhere
settings = Settings()
