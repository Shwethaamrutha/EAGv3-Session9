"""Centralized configuration — all settings validated at import time."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AWS Bedrock
    aws_profile: str = "bedrock"
    aws_region: str = "us-west-2"

    # Agent loop
    max_iterations: int = 25
    artifact_threshold_bytes: int = 8000
    attachment_budget_bytes: int = 50000
    fetch_content_limit: int = 80000

    # Memory
    memory_file: str = "state/memory.json"
    memory_max_items: int = 500
    memory_dedup_threshold: float = 0.8

    # Embeddings
    embed_dimension: int = 768
    faiss_index_file: str = "state/index.faiss"
    faiss_ids_file: str = "state/index_ids.json"

    # Artifacts
    artifacts_dir: str = "state/artifacts"
    artifact_ttl_hours: int = 72

    # Chatbot
    chatbot_port: int = 8000
    max_query_length: int = 10000

    # Logging
    log_format: str = "console"  # "console" or "json"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
