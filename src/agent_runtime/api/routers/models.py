"""Model metadata endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from agent_runtime.agents.model_provider import supported_models_payload
from agent_runtime.api.deps import get_runtime_model_repo
from agent_runtime.api.schemas import (
    ModelsOut,
    RuntimeModelCreate,
    RuntimeModelOut,
    RuntimeModelUpdate,
)
from agent_runtime.db.models import RuntimeModel
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelsOut)
async def list_models(
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
):
    return ModelsOut(**await supported_models_payload(model_repo))


@router.post("", status_code=201, response_model=RuntimeModelOut)
async def create_model(
    body: RuntimeModelCreate,
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
):
    provider = body.provider.lower()
    if provider not in ("openai", "openrouter"):
        raise HTTPException(400, f"Unsupported model provider: {body.provider}")
    existing = await model_repo.get_by_provider_model(provider, body.model_id)
    if existing:
        raise HTTPException(409, f"Model '{provider}/{body.model_id}' already exists")
    model = await model_repo.create(
        provider=provider,
        model_id=body.model_id,
        name=body.name,
        enabled=body.enabled,
        supports_reasoning=body.supports_reasoning,
        sort_order=body.sort_order,
        config_json=body.config,
    )
    return _model_out(model)


@router.patch("/{model_row_id}", response_model=RuntimeModelOut)
async def update_model(
    model_row_id: int,
    body: RuntimeModelUpdate,
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
):
    update_data = body.model_dump(exclude_unset=True)
    model = await model_repo.update(
        model_row_id,
        name=body.name,
        enabled=body.enabled,
        supports_reasoning=body.supports_reasoning,
        sort_order=body.sort_order,
        config_json=body.config,
        replace_config="config" in update_data,
    )
    if model is None:
        raise HTTPException(404, "Model not found")
    return _model_out(model)


@router.delete("/{model_row_id}", status_code=204)
async def delete_model(
    model_row_id: int,
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
):
    deleted = await model_repo.delete(model_row_id)
    if not deleted:
        raise HTTPException(404, "Model not found")


def _model_out(model: RuntimeModel) -> RuntimeModelOut:
    return RuntimeModelOut(
        id=model.id,
        provider=model.provider,
        model_id=model.model_id,
        name=model.name,
        enabled=model.enabled,
        supports_reasoning=model.supports_reasoning,
        sort_order=model.sort_order,
        config=model.config_json,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )
