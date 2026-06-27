"""Prompt endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from agent_runtime.agents.runtime import switch_prompt
from agent_runtime.api.deps import get_prompt_repo, get_session_repo
from agent_runtime.api.schemas import (
    PromptCreate,
    PromptListOut,
    PromptOut,
    PromptSwitch,
)
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.session_repo import SessionRepo
from agent_runtime.ids import decode

router = APIRouter(tags=["prompts"])


@router.get("/prompts", response_model=PromptListOut)
async def list_prompts(
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
):
    prompts = await prompt_repo.list_all()
    return PromptListOut(
        prompts=[
            PromptOut(id=p.id, name=p.name, content=p.content, created_at=p.created_at)
            for p in prompts
        ]
    )


@router.post("/prompts", status_code=201, response_model=PromptOut)
async def create_prompt(
    body: PromptCreate,
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
):
    existing = await prompt_repo.get_by_name(body.name)
    if existing:
        raise HTTPException(409, f"Prompt '{body.name}' already exists")
    p = await prompt_repo.create(body.name, body.content)
    return PromptOut(id=p.id, name=p.name, content=p.content, created_at=p.created_at)


@router.post("/sessions/{encoded_id}/prompt")
async def switch_session_prompt(
    encoded_id: str,
    body: PromptSwitch,
    session_repo: SessionRepo = Depends(get_session_repo),
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
):
    try:
        sid = decode(encoded_id)
    except ValueError as e:
        raise HTTPException(404, "Invalid session ID") from e

    sess = await session_repo.get_session(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    prompt = await prompt_repo.get_by_name(body.name)
    if not prompt:
        raise HTTPException(404, f"Prompt '{body.name}' not found")

    try:
        await switch_prompt(
            encoded_id,
            body.name,
            session_repo=session_repo,
            prompt_repo=prompt_repo,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    return {"prompt": body.name}
