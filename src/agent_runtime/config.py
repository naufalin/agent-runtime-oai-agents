"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # PostgreSQL
    database_url: str = "postgresql://localhost:5432/agent_runtime"

    # TinyFish
    tinyfish_api_key: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
