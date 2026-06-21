"""Main agent definition with all tools and persistence."""

import json
from dataclasses import dataclass
from typing import Any

from agents import Agent, ModelSettings, RunConfig, Runner
from agents.models.interface import Model

from agent_runtime.agents.hooks import PersistenceHooks, RunContext
from agent_runtime.agents.model_provider import (
    extract_thinking,
    resolve_runtime_model,
    serialize_usage,
)
from agent_runtime.config import Settings
from agent_runtime.db.connection import Database
from agent_runtime.db.prompt_repo import SystemPromptRepo
from agent_runtime.db.runtime_model_repo import RuntimeModelRepo
from agent_runtime.db.session_repo import SessionRepo
from agent_runtime.ids import decode, encode
from agent_runtime.tools.country import get_country_info
from agent_runtime.tools.currency import convert_currency
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.web_fetch import web_fetch
from agent_runtime.tools.web_search import web_search

# from agent_runtime.tools.web_agent import web_agent
# from agent_runtime.tools.web_browser import web_browser

_hooks = PersistenceHooks()

_db: Database | None = None


async def get_db() -> Database:
    """Get or create the database engine."""
    global _db
    if _db is None:
        settings = Settings()
        _db = Database(settings.database_url)
        _db.connect()
    return _db


def create_agent(
    system_prompt: str,
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent:
    """Create the configured runtime agent with the given system prompt."""
    settings = Settings()
    return Agent(
        name="MainAgent",
        instructions=system_prompt,
        model=model or settings.openai_model,
        model_settings=model_settings or ModelSettings(),
        tools=[
            web_search,
            web_fetch,
            # web_agent, # NOTE: for future use
            # web_browser, # NOTE: for future user
            get_weather,
            convert_currency,
            get_country_info,
        ],
    )


@dataclass
class AgentResponse:
    """Response from run_agent with the session ID for follow-up messages."""

    response: str
    session_id: str  # encoded ID for external use
    provider: str | None = None
    model: str | None = None
    usage: dict | None = None
    thinking: dict | None = None


async def run_agent(
    user_message: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> AgentResponse:
    """Run the agent with a user message, persist to DB, and return the response.

    Args:
        user_message: The user's message text.
        session_id: Encoded session ID (from ids.encode). None to start new.

    Returns:
        AgentResponse with the agent's reply and the session ID.
    """
    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)
    model_repo = RuntimeModelRepo(db)

    runtime_model = await resolve_runtime_model(
        model_repo=model_repo,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Resolve session: decode existing or create new
    internal_id: int | None = None
    if session_id is not None:
        try:
            internal_id = decode(session_id)
            existing = await repo.get_session(internal_id)
            if not existing:
                internal_id = None
        except ValueError:
            internal_id = None

    if internal_id is None:
        # New session — seed default prompt and insert system message
        default_prompt = await prompt_repo.seed_default()
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        sess = await repo.create_session(title)
        internal_id = sess.id
        await repo.add_message(
            internal_id,
            "system",
            default_prompt.content,
            system_prompt_id=default_prompt.id,
        )

    # Load the active system prompt from session history
    system_msg = await repo.get_latest_system_message(internal_id)
    system_prompt = system_msg.content if system_msg else ""

    # Save user message
    await repo.add_message(internal_id, "user", user_message)

    # Build conversation history for the agent
    history = await repo.get_messages(internal_id)
    input_items: list[Any] = []
    for msg in history:
        if msg.role in ("user", "assistant"):
            input_items.append({"role": msg.role, "content": msg.content})
        elif msg.role == "tool" and msg.tool_call_id:
            # Reconstruct function_call + function_call_output for the Responses API
            if msg.tool_input:
                input_items.append({
                    "type": "function_call",
                    "call_id": msg.tool_call_id,
                    "name": msg.tool_name or "unknown",
                    "arguments": json.dumps(msg.tool_input),
                })
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content,
            })

    # Run the agent with the active system prompt, full history, and persistence hooks
    agent = create_agent(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)
    result = await Runner.run(
        agent,
        input=input_items,
        context=run_context,
        hooks=_hooks,  # type: ignore[arg-type]
        run_config=RunConfig(tracing_disabled=runtime_model.tracing_disabled),
    )
    response = result.final_output
    usage = serialize_usage(result.context_wrapper.usage)
    thinking = extract_thinking(result.raw_responses)

    # Save agent response
    await repo.add_message(
        internal_id,
        "assistant",
        response,
        provider=runtime_model.provider,
        model=runtime_model.model_name,
        usage_json=usage,
        thinking_json=thinking,
    )

    return AgentResponse(
        response=response,
        session_id=encode(internal_id),
        provider=runtime_model.provider,
        model=runtime_model.model_name,
        usage=usage,
        thinking=thinking,
    )


async def run_agent_streamed(
    user_message: str,
    session_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
):
    """Stream agent events as async generator of dicts.

    Yields dicts with 'type' key: text_delta, thinking_delta, tool_start, tool_end,
    done, error.
    """
    from agents.items import ItemHelpers
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)
    model_repo = RuntimeModelRepo(db)

    runtime_model = await resolve_runtime_model(
        model_repo=model_repo,
        provider=provider,
        model=model,
        reasoning_effort=reasoning_effort,
    )

    # Resolve session (same logic as run_agent)
    internal_id: int | None = None
    if session_id is not None:
        try:
            internal_id = decode(session_id)
            existing = await repo.get_session(internal_id)
            if not existing:
                internal_id = None
        except ValueError:
            internal_id = None

    if internal_id is None:
        default_prompt = await prompt_repo.seed_default()
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        sess = await repo.create_session(title)
        internal_id = sess.id
        await repo.add_message(
            internal_id, "system", default_prompt.content,
            system_prompt_id=default_prompt.id,
        )

    system_msg = await repo.get_latest_system_message(internal_id)
    system_prompt = system_msg.content if system_msg else ""

    await repo.add_message(internal_id, "user", user_message)

    history = await repo.get_messages(internal_id)
    input_items: list[Any] = []
    for msg in history:
        if msg.role in ("user", "assistant"):
            input_items.append({"role": msg.role, "content": msg.content})
        elif msg.role == "tool" and msg.tool_call_id:
            if msg.tool_input:
                input_items.append({
                    "type": "function_call",
                    "call_id": msg.tool_call_id,
                    "name": msg.tool_name or "unknown",
                    "arguments": json.dumps(msg.tool_input),
                })
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": msg.content,
            })

    agent = create_agent(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)

    result = Runner.run_streamed(
        agent,
        input=input_items,
        context=run_context,
        hooks=_hooks,
        run_config=RunConfig(tracing_disabled=runtime_model.tracing_disabled),
    )

    full_text = ""
    _pending_tool_args: dict[str, dict] = {}  # call_id -> parsed args
    try:
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                event_type = getattr(data, "type", None)
                delta = getattr(data, "delta", None)
                if (
                    event_type == "response.output_text.delta"
                    and isinstance(delta, str)
                    and delta
                ):
                    full_text += delta
                    yield {"type": "text_delta", "delta": delta}
                elif (
                    event_type == "response.reasoning_text.delta"
                    and isinstance(delta, str)
                    and delta
                ):
                    yield {"type": "thinking_delta", "delta": delta, "kind": "reasoning"}
                elif (
                    event_type == "response.reasoning_summary_text.delta"
                    and isinstance(delta, str)
                    and delta
                ):
                    yield {"type": "thinking_delta", "delta": delta, "kind": "summary"}

            elif isinstance(event, RunItemStreamEvent):
                if event.name == "tool_called":
                    item = event.item
                    tool_name = getattr(item, "tool_name", None) or "tool"
                    call_id = getattr(item, "call_id", None)
                    raw = getattr(item, "raw_item", None)
                    args = None
                    if raw is not None:
                        args_str = getattr(raw, "arguments", None)
                        if args_str:
                            try:
                                args = json.loads(args_str)
                            except (json.JSONDecodeError, TypeError):
                                args = {"raw": args_str}
                    if call_id and args:
                        _pending_tool_args[call_id] = args
                    yield {
                        "type": "tool_start",
                        "tool": tool_name,
                        "call_id": call_id,
                        "args": args,
                    }

                elif event.name == "tool_output":
                    item = event.item
                    tool_name = getattr(item, "tool_name", None) or "tool"
                    call_id = getattr(item, "call_id", None)
                    output = getattr(item, "output", None)
                    raw = getattr(item, "raw_item", None)
                    output_str = (
                        str(output) if output is not None
                        else str(raw) if raw else ""
                    )
                    output_preview = output_str[:500] if output_str else None

                    # Persist tool message to DB
                    await repo.add_message(
                        internal_id,
                        "tool",
                        content=output_str,
                        tool_name=tool_name,
                        tool_call_id=call_id,
                        tool_input=_pending_tool_args.pop(call_id, None) if call_id else None,
                        tool_output=output if isinstance(output, dict) else {"raw": output_str},
                        output_preview=output_preview,
                    )

                    yield {
                        "type": "tool_end",
                        "tool": tool_name,
                        "call_id": call_id,
                    }
                elif event.name == "message_output_created":
                    text = ItemHelpers.text_message_output(event.item)
                    if text and text != full_text:
                        remaining = text[len(full_text):]
                        if remaining:
                            yield {"type": "text_delta", "delta": remaining}
                        full_text = text

        usage = serialize_usage(result.context_wrapper.usage)
        thinking = extract_thinking(result.raw_responses)

        # Save final assistant response
        if full_text:
            await repo.add_message(
                internal_id,
                "assistant",
                full_text,
                provider=runtime_model.provider,
                model=runtime_model.model_name,
                usage_json=usage,
                thinking_json=thinking,
            )

        yield {
            "type": "done",
            "session_id": encode(internal_id),
            "provider": runtime_model.provider,
            "model": runtime_model.model_name,
            "usage": usage,
            "thinking": thinking,
        }

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}


async def switch_prompt(session_id: str, prompt_name: str) -> str:
    """Switch the system prompt for an existing session.

    Inserts a new system message with the named prompt. Returns the prompt name.
    """
    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)

    internal_id = decode(session_id)
    sess = await repo.get_session(internal_id)
    if not sess:
        raise ValueError(f"Session not found: {session_id}")

    prompt = await prompt_repo.get_by_name(prompt_name)
    if not prompt:
        raise ValueError(f"Prompt not found: {prompt_name}")

    await repo.add_message(
        internal_id,
        "system",
        prompt.content,
        system_prompt_id=prompt.id,
    )
    return prompt.name
