"""Langfuse tracing integration.

Uses the Langfuse v4 SDK which is OTEL-based. Observations are created
via `start_as_current_observation()` context managers which automatically
handle nesting (child observations link to their parent span).

Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST env vars.
When keys are not set, tracing is silently disabled.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from agent_runtime.config import Settings

logger = logging.getLogger(__name__)

_langfuse_initialized = False
_langfuse_available = False
_langfuse_client: Any = None


def init_langfuse() -> bool:
    """Initialize the Langfuse client.

    Returns True if Langfuse was successfully initialized, False otherwise.
    Safe to call multiple times — only initializes once.
    """
    global _langfuse_initialized, _langfuse_available, _langfuse_client

    if _langfuse_initialized:
        return _langfuse_available

    _langfuse_initialized = True

    settings = Settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info(
            "Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY not set"
        )
        return False

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            environment=settings.langfuse_tracing_environment,
        )
        _langfuse_available = True
        logger.info(
            "Langfuse tracing initialized (host=%s, env=%s)",
            settings.langfuse_host,
            settings.langfuse_tracing_environment,
        )
        return True

    except Exception:
        logger.exception("Failed to initialize Langfuse tracing")
        return False


def is_langfuse_active() -> bool:
    """Check if Langfuse tracing is active."""
    return _langfuse_available


def get_langfuse_client() -> Any:
    """Return the Langfuse client (or None if not initialized)."""
    return _langfuse_client


def _usage_details(usage: dict[str, Any] | None) -> dict[str, int]:
    """Map the API usage summary to Langfuse usage detail keys."""
    if not usage:
        return {}

    fields = {
        "input": usage.get("input_tokens"),
        "output": usage.get("output_tokens"),
        "total": usage.get("total_tokens"),
        "input_cached": usage.get("cached_tokens"),
        "output_reasoning": usage.get("reasoning_tokens"),
    }
    return {key: value for key, value in fields.items() if isinstance(value, int)}


def _safe_update(observation: Any, **kwargs: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**kwargs)
    except Exception:
        logger.debug("Failed to update Langfuse observation", exc_info=True)


def end_observation(observation: Any) -> None:
    """End a manually managed Langfuse observation."""
    if observation is None:
        return
    try:
        observation.end()
    except Exception:
        logger.debug("Failed to end Langfuse observation", exc_info=True)


def mark_observation_error(
    observation: Any,
    error: BaseException | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Mark an observation as errored without raising tracing failures."""
    message = str(error)
    _safe_update(
        observation,
        level="ERROR",
        metadata={
            **(metadata or {}),
            "error": message,
            "status": "error",
        },
    )


@asynccontextmanager
async def langfuse_trace(
    name: str,
    input_data: Any = None,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> AsyncGenerator[Any, None]:
    """Create a Langfuse trace as an async context manager.

    Creates a top-level span (trace root) using the v4 OTEL-based API.
    Any nested `start_as_current_observation()` calls inside this context
    will automatically become children of this span.

    Yields the observation object which has `.update()` and can create
    child observations via `.start_as_current_observation()`.

    Falls back to a no-op when Langfuse is not configured.
    """
    if not _langfuse_available or _langfuse_client is None:
        yield None
        return

    trace_context = {"trace_id": _langfuse_client.create_trace_id()}
    attributes: dict[str, Any] = {
        "trace_name": name,
        "metadata": metadata or {},
    }
    if session_id:
        attributes["session_id"] = session_id
    if user_id:
        attributes["user_id"] = user_id
    if tags:
        attributes["tags"] = tags

    from langfuse import propagate_attributes

    with (
        propagate_attributes(**attributes),
        _langfuse_client.start_as_current_observation(
            trace_context=trace_context,
            name=name,
            as_type="agent",
            input=input_data,
            metadata={
                **(metadata or {}),
                **({"session_id": session_id} if session_id else {}),
                **({"user_id": user_id} if user_id else {}),
                **({"tags": tags} if tags else {}),
            },
        ) as span,
    ):
        try:
            yield span
        except Exception as exc:
            mark_observation_error(span, exc)
            raise
        finally:
            _langfuse_client.flush()


def start_generation(
    name: str,
    model: str,
    input_data: Any = None,
    output: Any = None,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Any:
    """Start a manually managed generation child observation."""
    if not _langfuse_available or _langfuse_client is None:
        return None

    kwargs: dict[str, Any] = {
        "name": name,
        "as_type": "generation",
        "model": model,
        "input": input_data,
    }
    if output is not None:
        kwargs["output"] = output

    usage_details = _usage_details(usage)
    if usage_details:
        kwargs["usage_details"] = usage_details

    if metadata:
        kwargs["metadata"] = metadata

    if model_parameters:
        kwargs["model_parameters"] = model_parameters

    try:
        return _langfuse_client.start_observation(**kwargs)
    except Exception:
        logger.debug("Failed to start Langfuse generation", exc_info=True)
        return None


def finish_generation(
    observation: Any,
    output: Any = None,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update and end a manually managed generation observation."""
    update_kwargs: dict[str, Any] = {}
    if output is not None:
        update_kwargs["output"] = output
    usage_details = _usage_details(usage)
    if usage_details:
        update_kwargs["usage_details"] = usage_details
    if metadata:
        update_kwargs["metadata"] = metadata
    if update_kwargs:
        _safe_update(observation, **update_kwargs)
    end_observation(observation)


def start_tool_observation(
    tool_name: str,
    input_data: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Start a manually managed tool child observation."""
    if not _langfuse_available or _langfuse_client is None:
        return None

    try:
        return _langfuse_client.start_observation(
            name=tool_name,
            as_type="tool",
            input=input_data,
            metadata=metadata,
        )
    except Exception:
        logger.debug("Failed to start Langfuse tool observation", exc_info=True)
        return None


def finish_tool_observation(
    observation: Any,
    output: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Update and end a manually managed tool observation."""
    update_kwargs: dict[str, Any] = {}
    if output is not None:
        update_kwargs["output"] = output
    if metadata:
        update_kwargs["metadata"] = metadata
    if update_kwargs:
        _safe_update(observation, **update_kwargs)
    end_observation(observation)


def log_generation(
    name: str,
    model: str,
    input_data: Any = None,
    output: Any = None,
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> None:
    """Log a generation (LLM call) as a child of the current span.

    Must be called inside a `langfuse_trace()` or
    `start_as_current_observation()` context to be properly nested.
    """
    if not _langfuse_available or _langfuse_client is None:
        return

    generation = start_generation(
        name=name,
        model=model,
        input_data=input_data,
        output=output,
        usage=usage,
        metadata={
            **(metadata or {}),
            **({"started_at_monotonic": start_time} if start_time is not None else {}),
            **({"ended_at_monotonic": end_time} if end_time is not None else {}),
        },
        model_parameters=model_parameters,
    )
    finish_generation(generation, output=output, usage=usage)


def flush_langfuse() -> None:
    """Flush pending Langfuse events. Call on shutdown."""
    if _langfuse_available and _langfuse_client is not None:
        try:
            _langfuse_client.flush()
        except Exception:
            logger.debug("Failed to flush Langfuse events", exc_info=True)
