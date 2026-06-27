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
from agent_runtime.config import settings
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


def _looks_like_visualization_args(args: Any) -> bool:
    return (
        isinstance(args, dict)
        and isinstance(args.get("chart_type"), str)
        and isinstance(args.get("data"), str)
        and isinstance(args.get("title"), str)
    )


def _infer_known_tool_from_args(args: Any) -> str | None:
    if not isinstance(args, dict):
        return None
    if _looks_like_visualization_args(args):
        return "generate_visualization"
    if isinstance(args.get("query"), str):
        return "web_search"
    if isinstance(args.get("urls"), list):
        return "web_fetch"
    if isinstance(args.get("city"), str):
        return "get_weather"
    if (
        isinstance(args.get("amount"), int | float)
        and isinstance(args.get("from_currency"), str)
        and isinstance(args.get("to_currency"), str)
    ):
        return "convert_currency"
    if isinstance(args.get("country_name"), str):
        return "get_country_info"
    return None


def _infer_known_tool_from_output(output: Any) -> str | None:
    parsed_output = _parse_tool_output(output)
    if parsed_output is not None and isinstance(parsed_output.get("html"), str):
        return "generate_visualization"

    if not isinstance(output, str):
        return None
    if output.startswith("[") and "](http" in output:
        return "web_search"
    if output.startswith("## ") or "## Failed URLs" in output:
        return "web_fetch"
    if output.startswith("Weather in "):
        return "get_weather"
    if "\nRate: 1 " in output:
        return "convert_currency"
    if "\n  Capital: " in output and "\n  Population: " in output:
        return "get_country_info"
    return None


def _parse_tool_output(output: Any) -> dict[str, Any] | None:
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _infer_tool_name(tool_name: str, args: Any = None, output: Any = None) -> str:
    if tool_name != "tool":
        return tool_name
    inferred = _infer_known_tool_from_args(args) or _infer_known_tool_from_output(output)
    if inferred:
        return inferred
    return tool_name


def _extract_visualization_html(tool_name: str, args: Any, output: Any) -> str | None:
    parsed_output = _parse_tool_output(output)
    if parsed_output is None:
        return None
    html = parsed_output.get("html")
    if not isinstance(html, str) or not html:
        return None
    if tool_name == "generate_visualization" or _looks_like_visualization_args(args):
        return html
    return None


_DEFAULT_TOOLS = [
    web_search,
    web_fetch,
    get_weather,
    convert_currency,
    get_country_info,
    generate_visualization,
]


# Name -> function_tool instance. Built once at import from _DEFAULT_TOOLS.
# Names come from the SDK's `name_override` argument on each tool module.
TOOL_REGISTRY: dict[str, Any] = {t.name: t for t in _DEFAULT_TOOLS}


def register_tool(name: str, tool: Any) -> None:
    """Register a tool by name for use in ChatRequest.tools / SessionCreate.tools.

    Convenience for adding tools at runtime; the canonical registry is built
    from `_DEFAULT_TOOLS` at import. New tools should normally be added by
    appending to `_DEFAULT_TOOLS` and re-exporting here.
    """
    TOOL_REGISTRY[name] = tool


def available_tool_names() -> list[str]:
    """Sorted list of tool names exposed to clients via the API."""
    return sorted(TOOL_REGISTRY)


def resolve_tools(names: list[str] | None) -> list[Any]:
    """Resolve a list of tool names to function_tool instances.

    Semantics:
      * None  -> return a copy of the server default tool set.
      * []    -> return an empty list (agent has NO tools; pure chat).
      * [str] -> return the named tools, in order. Unknown names raise ValueError.

    Returns a fresh list (caller may mutate). Does not raise on empty result.
    """
    if names is None:
        return list(_DEFAULT_TOOLS)
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown tool(s): {unknown}. Available: {available_tool_names()}")
    return [TOOL_REGISTRY[n] for n in names]


class AgentFactory:
    """Creates Agent instances with a configurable tool set and default model."""

    def __init__(
        self,
        tools: list | None = None,
        default_model: str = "gpt-5.4-mini",
    ) -> None:
        self.tools = tools if tools is not None else _DEFAULT_TOOLS
        self.default_model = default_model

    def create(
        self,
        system_prompt: str,
        model: str | Model | None = None,
        model_settings: ModelSettings | None = None,
        tools: list | None = None,
    ) -> Agent:
        return Agent(
            name="MainAgent",
            instructions=system_prompt,
            model=model or self.default_model,
            model_settings=model_settings or ModelSettings(),
            tools=tools if tools is not None else self.tools,
        )


def create_agent(
    system_prompt: str,
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
) -> Agent:
    """Create the configured runtime agent with the given system prompt."""
    return AgentFactory(default_model=settings.openai_model).create(
        system_prompt, model=model, model_settings=model_settings
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
    *,
    session_repo: SessionRepo,
    prompt_repo: SystemPromptRepo,
    model_repo: RuntimeModelRepo,
    agent_factory: AgentFactory,
    tools_override: list[str] | None = None,
) -> AgentResponse:
    """Run the agent with a user message, persist to DB, and return the response.

    Args:
        user_message: The user's message text.
        session_id: Encoded session ID (from ids.encode). None to start new.
        tools_override: Per-turn tool allowlist. Resolution order is
            ``tools_override`` -> ``Session.tools_json`` -> server defaults.

    Returns:
        AgentResponse with the agent's reply and the session ID.
    """
    repo = session_repo
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

    # Resolve effective tool set for this turn.
    # Priority: tools_override (per-turn) > session.tools_json > server defaults.
    assert internal_id is not None  # guaranteed: either decoded or freshly created above
    session_tools = await repo.get_tools(internal_id)
    effective_tools = resolve_tools(tools_override if tools_override is not None else session_tools)

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
    agent = agent_factory.create(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
        tools=effective_tools,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)

    async with langfuse_trace(
        name="agent-run",
        input_data=user_message,
        session_id=str(internal_id),
        metadata={
            "provider": runtime_model.provider,
            "model": runtime_model.model_name,
            "tool_names": [t.name for t in effective_tools],
            "tool_source": (
                "override"
                if tools_override is not None
                else "session"
                if session_tools is not None
                else "defaults"
            ),
        },
    ) as span:
        _run_started = time.monotonic()
        _started_at_iso = datetime.now(UTC).isoformat()
        result = await Runner.run(
            agent,
            input=input_items,
            context=run_context,
            max_turns=settings.max_turns,
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
    *,
    session_repo: SessionRepo,
    prompt_repo: SystemPromptRepo,
    model_repo: RuntimeModelRepo,
    agent_factory: AgentFactory,
    tools_override: list[str] | None = None,
):
    """Stream agent events as async generator of dicts.

    Yields dicts with 'type' key: text_delta, thinking_delta, tool_start, tool_end,
    done, error.

    ``tools_override`` follows the same resolution priority as ``run_agent``:
    override > session.tools_json > server defaults.
    """
    from agents.items import ItemHelpers
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

    repo = session_repo
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

    # Resolve effective tool set for this turn.
    # Priority: tools_override (per-turn) > session.tools_json > server defaults.
    assert internal_id is not None  # guaranteed: either decoded or freshly created above
    session_tools = await repo.get_tools(internal_id)
    effective_tools = resolve_tools(tools_override if tools_override is not None else session_tools)

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

    agent = agent_factory.create(
        system_prompt,
        model=runtime_model.model,
        model_settings=runtime_model.model_settings,
        tools=effective_tools,
    )
    run_context = RunContext(session_id=internal_id, repo=repo)

    async with langfuse_trace(
        name="agent-run-streamed",
        input_data=user_message,
        session_id=str(internal_id),
        metadata={
            "provider": runtime_model.provider,
            "model": runtime_model.model_name,
            "tool_names": [t.name for t in effective_tools],
            "tool_source": (
                "override"
                if tools_override is not None
                else "session"
                if session_tools is not None
                else "defaults"
            ),
            "streamed": True,
        },
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
                        tool_name = _infer_tool_name(tool_name, args)
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
                        pending_args = _pending_tool_args.pop(call_id, None) if call_id else None
                        tool_name = _infer_tool_name(tool_name, pending_args, output)
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
                            tool_input=pending_args,
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

                        viz_html = _extract_visualization_html(
                            tool_name,
                            pending_args,
                            output,
                        )

                        yield {
                            "type": "tool_end",
                            "tool": tool_name,
                            "call_id": call_id,
                            "output_preview": output_preview,
                            "viz_html": viz_html,
                        }
                    elif event.name == "message_output_created":
                        text = ItemHelpers.text_message_output(event.item)
                        if text and not full_text:
                            if _first_token_at is None:
                                _first_token_at = time.monotonic()
                            _last_token_at = time.monotonic()
                            _output_delta_count += 1
                            full_text = text
                            yield {"type": "text_delta", "delta": text}
                        elif text and text.startswith(full_text):
                            remaining = text[len(full_text) :]
                            if remaining:
                                if _first_token_at is None:
                                    _first_token_at = time.monotonic()
                                _last_token_at = time.monotonic()
                                _output_delta_count += 1
                                yield {"type": "text_delta", "delta": remaining}
                            full_text = text
                        elif text:
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


async def switch_prompt(
    session_id: str,
    prompt_name: str,
    *,
    session_repo: SessionRepo,
    prompt_repo: SystemPromptRepo,
) -> str:
    """Switch the system prompt for an existing session.

    Inserts a new system message with the named prompt. Returns the prompt name.
    """
    repo = session_repo
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
