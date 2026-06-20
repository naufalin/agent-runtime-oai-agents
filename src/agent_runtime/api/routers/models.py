"""Model metadata endpoints."""

from fastapi import APIRouter

from agent_runtime.agents.model_provider import supported_models_payload
from agent_runtime.api.schemas import ModelsOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsOut)
async def list_models():
    return ModelsOut(**supported_models_payload())
