"""Session endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from agent_runtime.agents.model_provider import resolve_runtime_model
from agent_runtime.agents.runtime import (
    AgentFactory,
    resolve_tools,
    run_agent,
    run_agent_streamed,
)
from agent_runtime.api.deps import (
    get_agent_factory,
    get_prompt_repo,
    get_runtime_model_repo,
    get_session_repo,
)
from agent_runtime.api.schemas import (
    ChatRequest,
    ChatResponse,
    MessageOut,
    SessionCreate,
    SessionDetail,
    SessionListOut,
    SessionOut,
    SessionUpdate,
)
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo
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
        tool_call_id=msg.tool_call_id,
        tool_input=msg.tool_input,
        output_preview=msg.output_preview,
        provider=msg.provider,
        model=msg.model,
        usage=msg.usage_json,
        thinking=msg.thinking_json,
        created_at=msg.created_at,
    )


def _session_out(sess, prompt_name: str | None = None) -> SessionOut:
    return SessionOut(
        id=encode(sess.id),
        title=sess.title,
        prompt=prompt_name if prompt_name is not None else "-",
        tools=sess.tools_json,
        created_at=sess.created_at,
        updated_at=sess.updated_at,
    )


def _validate_tool_names(names: list[str]) -> None:
    """Raise HTTPException(400) if any tool name is not in the registry."""
    try:
        resolve_tools(names)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.post("", status_code=201, response_model=SessionOut)
async def create_session(
    body: SessionCreate | None = None,
    session_repo: SessionRepo = Depends(get_session_repo),
):
    title = body.title if body else "New Session"
    tools = body.tools if body else None
    if tools is not None:
        _validate_tool_names(tools)
    sess = await session_repo.create_session(title=title, tools=tools)
    return _session_out(sess)


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
        items.append(_session_out(s, prompt_name))
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
        tools=sess.tools_json,
        created_at=sess.created_at,
        messages=[_msg_out(m) for m in messages],
    )


@router.patch("/{encoded_id}", response_model=SessionOut)
async def patch_session(
    encoded_id: str,
    body: SessionUpdate,
    session_repo: SessionRepo = Depends(get_session_repo),
):
    """Partial update of session fields.

    Use `model_fields_set` to distinguish "field omitted" (do nothing)
    from "field explicitly set" (apply the value, including null).
    """
    try:
        sid = decode(encoded_id)
    except ValueError as e:
        raise HTTPException(404, "Invalid session ID") from e

    sess = await session_repo.get_session(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    fields_set = body.model_fields_set

    if "title" in fields_set and body.title is not None:
        await session_repo.update_title(sid, body.title)

    if "tools" in fields_set:
        # Validate first; fail fast with 400 before mutating the row.
        if body.tools is not None:
            _validate_tool_names(body.tools)
        await session_repo.update_tools(sid, body.tools)

    # Refresh from DB so we return the post-update row.
    sess = await session_repo.get_session(sid)
    return _session_out(sess)


@router.post("/{encoded_id}/chat", response_model=ChatResponse)
async def chat(
    encoded_id: str,
    body: ChatRequest,
    session_repo: SessionRepo = Depends(get_session_repo),
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    try:
        sid = decode(encoded_id)
    except ValueError as e:
        raise HTTPException(404, "Invalid session ID") from e

    sess = await session_repo.get_session(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    # Validate per-turn tools up front so an unknown name returns 400 immediately
    # instead of starting an LLM run that will fail later.
    if body.tools is not None:
        try:
            resolve_tools(body.tools)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    try:
        result = await run_agent(
            body.message,
            session_id=encoded_id,
            provider=body.provider,
            model=body.model,
            reasoning_effort=body.reasoning_effort,
            session_repo=session_repo,
            prompt_repo=prompt_repo,
            model_repo=model_repo,
            agent_factory=agent_factory,
            tools_override=body.tools,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    # Fetch all messages to return full context
    internal_id = decode(result.session_id)
    messages = await session_repo.get_messages(internal_id)

    return ChatResponse(
        response=result.response,
        session_id=result.session_id,
        provider=result.provider,
        model=result.model,
        usage=result.usage,
        thinking=result.thinking,
        messages=[_msg_out(m) for m in messages],
    )


@router.post("/{encoded_id}/chat/stream")
async def chat_stream(
    encoded_id: str,
    body: ChatRequest,
    session_repo: SessionRepo = Depends(get_session_repo),
    model_repo: RuntimeModelRepo = Depends(get_runtime_model_repo),
    prompt_repo: SystemPromptRepo = Depends(get_prompt_repo),
    agent_factory: AgentFactory = Depends(get_agent_factory),
):
    try:
        sid = decode(encoded_id)
    except ValueError as e:
        raise HTTPException(404, "Invalid session ID") from e

    sess = await session_repo.get_session(sid)
    if not sess:
        raise HTTPException(404, "Session not found")

    try:
        await resolve_runtime_model(
            model_repo=model_repo,
            provider=body.provider,
            model=body.model,
            reasoning_effort=body.reasoning_effort,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    # Validate tool names up front so we return 400 instead of starting a stream
    # that errors mid-flight.
    if body.tools is not None:
        try:
            resolve_tools(body.tools)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    async def event_generator():
        try:
            async for event in run_agent_streamed(
                body.message,
                session_id=encoded_id,
                provider=body.provider,
                model=body.model,
                reasoning_effort=body.reasoning_effort,
                session_repo=session_repo,
                prompt_repo=prompt_repo,
                model_repo=model_repo,
                agent_factory=agent_factory,
                tools_override=body.tools,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
