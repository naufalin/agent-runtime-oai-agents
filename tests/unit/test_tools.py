"""Tests for the tool registry and resolution helpers."""

import pytest

from agent_runtime.agents.runtime import (
    _DEFAULT_TOOLS,
    TOOL_REGISTRY,
    AgentFactory,
    available_tool_names,
    resolve_tools,
)
from agent_runtime.tools.country import get_country_info
from agent_runtime.tools.currency import convert_currency
from agent_runtime.tools.visualization import generate_visualization
from agent_runtime.tools.weather import get_weather
from agent_runtime.tools.web_search import web_search

EXPECTED_NAMES = {
    "convert_currency",
    "generate_visualization",
    "get_country_info",
    "get_weather",
    "web_fetch",
    "web_search",
}


def test_tool_registry_contains_all_defaults():
    assert set(TOOL_REGISTRY) == EXPECTED_NAMES
    assert set(TOOL_REGISTRY) == {t.name for t in _DEFAULT_TOOLS}


def test_tool_registry_names_are_clean():
    """SDK without name_override prepends an underscore. We use name_override to
    expose clean public names. This guard catches accidental regressions when a
    new tool is added without the override kwarg."""
    for name in TOOL_REGISTRY:
        assert not name.startswith("_"), f"Tool name {name!r} has unexpected underscore prefix"


def test_available_tool_names_is_sorted():
    assert available_tool_names() == sorted(EXPECTED_NAMES)


def test_resolve_tools_none_returns_defaults_copy():
    resolved = resolve_tools(None)
    assert [t.name for t in resolved] == [t.name for t in _DEFAULT_TOOLS]
    # Returns a copy — caller mutation does not affect the module-level default.
    resolved.clear()
    assert len(_DEFAULT_TOOLS) == 6


def test_resolve_tools_empty_returns_empty_list():
    assert resolve_tools([]) == []


def test_resolve_tools_single_name():
    resolved = resolve_tools(["web_search"])
    assert len(resolved) == 1
    assert resolved[0].name == "web_search"
    assert resolved[0] is web_search


def test_resolve_tools_preserves_order_and_duplicates():
    resolved = resolve_tools(["web_search", "web_fetch", "web_search"])
    assert [t.name for t in resolved] == ["web_search", "web_fetch", "web_search"]


def test_resolve_tools_unknown_name_lists_available():
    with pytest.raises(ValueError) as exc:
        resolve_tools(["nope"])
    assert "Unknown tool(s): ['nope']" in str(exc.value)
    assert "Available:" in str(exc.value)
    for name in EXPECTED_NAMES:
        assert name in str(exc.value)


def test_resolve_tools_partial_unknown_lists_only_bad():
    with pytest.raises(ValueError) as exc:
        resolve_tools(["web_search", "bad_one", "also_bad"])
    msg = str(exc.value)
    assert "bad_one" in msg
    assert "also_bad" in msg
    assert "web_search" not in msg.split("Available:")[0]


def test_resolve_tools_empty_string_treated_as_name():
    """Empty strings are not valid tool names; the registry does not contain one."""
    with pytest.raises(ValueError):
        resolve_tools([""])


def test_agent_factory_default_tools():
    factory = AgentFactory()
    assert factory.tools == _DEFAULT_TOOLS


def test_agent_factory_custom_tools():
    subset = [web_search, get_weather]
    factory = AgentFactory(tools=subset)
    agent = factory.create("sys", tools=subset)
    assert agent.tools == subset


def test_agent_factory_create_tools_override_per_call():
    """create(..., tools=[...]) overrides the factory-level tool set."""
    factory = AgentFactory(tools=[convert_currency])
    agent = factory.create("sys", tools=[web_search, get_country_info])
    assert [t.name for t in agent.tools] == ["web_search", "get_country_info"]


def test_agent_factory_default_model_is_readable():
    factory = AgentFactory(default_model="gpt-test")
    assert factory.default_model == "gpt-test"


def test_agent_factory_create_uses_factory_default_model():
    factory = AgentFactory(default_model="custom-model")
    agent = factory.create("sys", tools=[generate_visualization])
    assert agent.model == "custom-model"
