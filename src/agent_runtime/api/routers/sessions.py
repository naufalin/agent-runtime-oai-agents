"""Session endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from agent_runtime.agents.runtime import run_agent
from agent_runtime.api.deps import get_prompt_repo, get_session_repo
from agent_runtime.api.schemas import (
    ChatRequest,
    ChatResponse,
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionListOut,
    SessionOut,
)
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.session_repo import SessionRepo
from agent_runtime.ids import decode, encode

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _get_prompt_name(
    session_id: int,
    prompt_repo: SystemPromptRepo,
    session_repo: SessionRepo,
) -> str:
    sys_msg = await session_repo.get_latest_system_message(session_id)
    if sys_msg and sys_msg.system_prompt_id:
        p = await prompt_repo.get_by_id(sys_msg.system_prompt_id)
        if p:
            return p.name
    return "-"


def _msg_out(msg) -> MessageOut:
    return MessageOut(
        id=msg.id,
        role=msg.role,
        content=msg.content,
        tool_name=msg.tool_name,
        created_at=msg.created_at,
    )


@router.post("", status_code=201, response_model=SessionOut)
async def create_session(
    body: SessionCreate | None = None,
    session_repo: SessionRepo = Depends(get_session_repo),
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
):
    title = body.title if body else "New Session"
    sess = await session_repo.create_session(title)
    return SessionOut(
        id=encode(sess.id),
        title=sess.title,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
    )


@router.get("", response_model=SessionListOut)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    session_repo: SessionRepo = Depends(get_session_repo),
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
):
    sessions = await session_repo.list_sessions(limit=limit)
    items = []
    for s in sessions:
        prompt_name = await _get_prompt_name(s.id, prompt_repo, session_repo)
        items.append(
            SessionOut(
                id=encode(s.id),
                title=s.title,
                prompt=prompt_name,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
        )
    return SessionListOut(sessions=items, total=len(items))


@router.get("/{encoded_id}", response_model=SessionDetail)
async def get_session(
    encoded_id: str,
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

    messages = await session_repo.get_messages(sid)
    prompt_name = await _get_prompt_name(sid, prompt_repo, session_repo)

    return SessionDetail(
        id=encode(sess.id),
        title=sess.title,
        prompt=prompt_name,
        created_at=sess.created_at,
        messages=[_msg_out(m) for m in messages],
    )


@router.post("/{encoded_id}/chat", response_model=ChatResponse)
async def chat(
    encoded_id: str,
    body: ChatRequest,
    session_repo: SessionRepo = Depends(get_session_repo),
):
    try:
        sid = decode(encoded_id)
    except ValueError as e:
        raise HTTPException(404, "Invalid session ID") from e

    sess = await session_repo.get_session(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    result = await run_agent(body.message, session_id=encoded_id)

    # Fetch all messages to return full context
    internal_id = decode(result.session_id)
    messages = await session_repo.get_messages(internal_id)

    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
        messages=[_msg_out(m) for m in messages],
    )
