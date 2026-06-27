"""Main agent definition with all tools and persistence."""

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
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
from agent_runtime.tools.visualization import generate_visualization
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.web_fetch import web_fetch
from agent_runtime.tools.web_search import web_search
from agent_runtime.tracing import (
    finish_generation,
    finish_tool_observation,
    is_langfuse_active,
    langfuse_trace,
    log_generation,
    mark_observation_error,
    start_generation,
    start_tool_observation,
)

# from agent_runtime.tools.web_agent import web_agent
# from agent_runtime.tools.web_browser import web_browser

_hooks = PersistenceHooks()


def _extract_model_parameters(model_settings: ModelSettings) -> dict[str, Any]:
    """Extract non-None model parameters for Langfuse tracing."""
    params: dict[str, Any] = {}
    if model_settings.temperature is not None:
        params["temperature"] = model_settings.temperature
    if model_settings.max_tokens is not None:
        params["max_tokens"] = model_settings.max_tokens
    if model_settings.top_p is not None:
        params["top_p"] = model_settings.top_p
    if model_settings.frequency_penalty is not None:
        params["frequency_penalty"] = model_settings.frequency_penalty
    if model_settings.presence_penalty is not None:
        params["presence_penalty"] = model_settings.presence_penalty
    return params

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
            get_weather,
            convert_currency,
            get_country_info,
            generate_visualization,
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
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": msg.tool_call_id,
                        "name": msg.tool_name or "unknown",
                        "arguments": json.dumps(msg.tool_input),
                    }
                )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                }
            )

    # Run the agent with the active system prompt, full history, and persistence hooks
    agent = create_agent(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)

    async with langfuse_trace(
        name="agent-run",
        input_data=user_message,
        session_id=str(internal_id),
        metadata={"provider": runtime_model.provider, "model": runtime_model.model_name},
    ) as span:
        _run_started = time.monotonic()
        _started_at_iso = datetime.now(UTC).isoformat()
        result = await Runner.run(
            agent,
            input=input_items,
            context=run_context,
            hooks=_hooks,  # type: ignore[arg-type]
            run_config=RunConfig(
                tracing_disabled=runtime_model.tracing_disabled or is_langfuse_active()
            ),
        )
        _run_finished = time.monotonic()
        response = result.final_output

        # Perf metrics
        _duration_s = _run_finished - _run_started
        _usage_raw = serialize_usage(result.context_wrapper.usage)
        _output_tokens = _usage_raw["output_tokens"] if _usage_raw else 0
        _perf = {
            "ttft_ms": round(_duration_s * 1000),
            "tps": round(_output_tokens / _duration_s, 1) if _duration_s > 0 else 0.0,
            "generation_duration_ms": round(_duration_s * 1000),
            "started_at": _started_at_iso,
        }
        usage = serialize_usage(result.context_wrapper.usage, perf=_perf)
        thinking = extract_thinking(result.raw_responses)

        # Log the LLM generation as a child of the current trace
        log_generation(
            name="agent-llm-call",
            model=runtime_model.model_name,
            input_data=input_items,
            output=response,
            usage=usage,
            metadata={
                "provider": runtime_model.provider,
                "session_id": internal_id,
                "thinking": thinking,
                "perf": _perf,
            },
            model_parameters=_extract_model_parameters(runtime_model.model_settings),
            start_time=_run_started,
            end_time=_run_finished,
        )

        if span:
            span.update(output=response)

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
            internal_id,
            "system",
            default_prompt.content,
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
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": msg.tool_call_id,
                        "name": msg.tool_name or "unknown",
                        "arguments": json.dumps(msg.tool_input),
                    }
                )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                }
            )

    agent = create_agent(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)

    async with langfuse_trace(
        name="agent-run-streamed",
        input_data=user_message,
        session_id=str(internal_id),
        metadata={"provider": runtime_model.provider, "model": runtime_model.model_name},
    ) as span:
        result = Runner.run_streamed(
            agent,
            input=input_items,
            context=run_context,
            hooks=_hooks,
            run_config=RunConfig(
                tracing_disabled=runtime_model.tracing_disabled or is_langfuse_active()
            ),
        )

        full_text = ""
        _pending_tool_args: dict[str, dict] = {}  # call_id -> parsed args
        _stream_started = time.monotonic()
        _first_token_at: float | None = None
        _last_token_at: float | None = None
        _output_delta_count = 0
        _started_at_iso = datetime.now(UTC).isoformat()
        generation = start_generation(
            name="agent-llm-call",
            model=runtime_model.model_name,
            input_data=input_items,
            metadata={
                "provider": runtime_model.provider,
                "session_id": internal_id,
                "streamed": True,
                "started_at": _started_at_iso,
            },
            model_parameters=_extract_model_parameters(runtime_model.model_settings),
        )
        _tool_observations: dict[str, Any] = {}
        _tool_keys_by_call_id: dict[str, str] = {}
        _tool_keys_without_call_id: list[str] = []
        _tool_sequence = 0
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
                        if _first_token_at is None:
                            _first_token_at = time.monotonic()
                        _last_token_at = time.monotonic()
                        _output_delta_count += 1
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
                        _tool_sequence += 1
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
                        tool_key = call_id or f"{tool_name}:{_tool_sequence}"
                        _tool_observations[tool_key] = start_tool_observation(
                            tool_name=tool_name,
                            input_data=args,
                            metadata={
                                "call_id": call_id,
                                "provider": runtime_model.provider,
                                "session_id": internal_id,
                            },
                        )
                        if call_id:
                            _tool_keys_by_call_id[call_id] = tool_key
                        else:
                            _tool_keys_without_call_id.append(tool_key)
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
                        output_str = str(output) if output is not None else str(raw) if raw else ""
                        output_preview = output_str[:500] if output_str else None
                        if call_id and call_id in _tool_keys_by_call_id:
                            tool_key = _tool_keys_by_call_id.pop(call_id)
                        else:
                            tool_key = next(
                                (
                                    key
                                    for key in _tool_keys_without_call_id
                                    if key.startswith(f"{tool_name}:")
                                ),
                                f"{tool_name}:output:{_tool_sequence}",
                            )
                            if tool_key in _tool_keys_without_call_id:
                                _tool_keys_without_call_id.remove(tool_key)
                        tool_observation = _tool_observations.pop(tool_key, None)

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
                        if tool_observation is None:
                            tool_observation = start_tool_observation(
                                tool_name=tool_name,
                                input_data=None,
                                metadata={
                                    "call_id": call_id,
                                    "provider": runtime_model.provider,
                                    "session_id": internal_id,
                                    "started_from_output": True,
                                },
                            )
                        finish_tool_observation(
                            tool_observation,
                            output=output if output is not None else output_str,
                            metadata={
                                "call_id": call_id,
                                "output_preview": output_preview,
                                "session_id": internal_id,
                            },
                        )

                        # Check if this is a visualization tool output
                        viz_html = None
                        if tool_name == "generate_visualization":
                            # Try to parse output as JSON and extract html field
                            parsed_output = None
                            if isinstance(output, dict):
                                parsed_output = output
                            elif isinstance(output, str):
                                try:
                                    parsed_output = json.loads(output)
                                except (json.JSONDecodeError, TypeError):
                                    parsed_output = None
                            if (
                                parsed_output
                                and isinstance(parsed_output, dict)
                                and "html" in parsed_output
                            ):
                                viz_html = parsed_output["html"]

                        yield {
                            "type": "tool_end",
                            "tool": tool_name,
                            "call_id": call_id,
                            "output_preview": output_preview,
                            "viz_html": viz_html,
                        }
                    elif event.name == "message_output_created":
                        text = ItemHelpers.text_message_output(event.item)
                        if text and text != full_text:
                            remaining = text[len(full_text) :]
                            if remaining:
                                yield {"type": "text_delta", "delta": remaining}
                            full_text = text

            # Compute perf metrics
            _now = time.monotonic()
            _perf = None
            if _first_token_at is not None:
                _ttft_ms = round((_first_token_at - _stream_started) * 1000)
                _duration_s = (_last_token_at or _now) - _first_token_at
                _usage_raw = serialize_usage(result.context_wrapper.usage)
                _real_output = _usage_raw["output_tokens"] if _usage_raw else _output_delta_count
                _tps = round(_real_output / _duration_s, 1) if _duration_s > 0 else 0.0
                _perf = {
                    "ttft_ms": _ttft_ms,
                    "tps": _tps,
                    "generation_duration_ms": round(_duration_s * 1000),
                    "started_at": _started_at_iso,
                }

            usage = serialize_usage(result.context_wrapper.usage, perf=_perf)
            thinking = extract_thinking(result.raw_responses)

            finish_generation(
                generation,
                output=full_text or None,
                usage=usage,
                metadata={
                    "provider": runtime_model.provider,
                    "session_id": internal_id,
                    "thinking": thinking,
                    "perf": _perf,
                    "output_delta_count": _output_delta_count,
                },
            )

            if span:
                span.update(output=full_text)

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
                "output_delta_count": _output_delta_count,
            }

        except Exception as exc:
            mark_observation_error(generation, exc, {"session_id": internal_id})
            for tool_observation in _tool_observations.values():
                mark_observation_error(tool_observation, exc, {"session_id": internal_id})
                finish_tool_observation(tool_observation)
            if span:
                mark_observation_error(span, exc, {"session_id": internal_id})
            finish_generation(generation)
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
