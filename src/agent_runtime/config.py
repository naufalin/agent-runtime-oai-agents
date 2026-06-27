"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Agent
    max_turns: int = 20

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-5.4-mini"

    # Model provider
    agent_runtime_model_provider: str = "openai"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "z-ai/glm-5.2"
    openrouter_reasoning_effort: str = ""

    # PostgreSQL
    database_url: str = "postgresql://localhost:5432/agent_runtime"

    # TinyFish
    tinyfish_api_key: str = ""

    # Langfuse observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_tracing_environment: str = "default"

    # API auth
    agent_runtime_bearer_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
