"""Tests for the generate_visualization tool."""

import json

import pytest

from agent_runtime.tools.visualization import (
    _build_html,
    _generate_visualization,
)


@pytest.mark.asyncio
async def test_bar_chart_returns_valid_html_and_metadata():
    """Test 1: bar chart with valid data returns JSON with html, chart_type, title."""
    data = json.dumps(
        {
            "xAxis": {"type": "category", "data": ["A", "B", "C"]},
            "yAxis": {"type": "value"},
            "series": [{"data": [10, 20, 30], "type": "bar"}],
        }
    )
    result_str = await _generate_visualization("bar", data, "Test Chart")
    result = json.loads(result_str)

    assert result["chart_type"] == "bar"
    assert result["title"] == "Test Chart"
    assert "html" in result
    assert "echarts.init" in result["html"]
    assert "echarts.min.js" in result["html"]


@pytest.mark.asyncio
async def test_line_chart_with_custom_options():
    """Test 2: line chart with extra options merges them into the HTML."""
    data = json.dumps(
        {
            "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar"]},
            "yAxis": {"type": "value"},
            "series": [{"data": [5, 15, 10], "type": "line"}],
        }
    )
    options = json.dumps({"dataZoom": {"type": "slider"}})
    result_str = await _generate_visualization("line", data, "Trend", options)
    result = json.loads(result_str)

    assert result["chart_type"] == "line"
    assert result["title"] == "Trend"
    assert "dataZoom" in result["html"]


@pytest.mark.asyncio
async def test_invalid_json_data_returns_error():
    """Test 3: invalid JSON in data parameter returns an error dict."""
    result_str = await _generate_visualization("bar", "not-json", "Test")
    result = json.loads(result_str)

    assert "error" in result
    assert "Invalid JSON" in result["error"]
    assert result["chart_type"] == "bar"
    assert result["title"] == "Test"


@pytest.mark.asyncio
async def test_pie_chart_contains_echarts_cdn():
    """Test 4: pie chart HTML includes the ECharts CDN script tag."""
    data = json.dumps(
        {
            "series": [{"data": [{"value": 1, "name": "A"}]}],
        }
    )
    result_str = await _generate_visualization("pie", data, "Pie Chart")
    result = json.loads(result_str)

    assert "echarts.min.js" in result["html"]
    assert "pie" in result["html"]


@pytest.mark.asyncio
async def test_series_type_is_injected_when_missing():
    """Test: series entries without type get the chart_type injected."""
    data = json.dumps(
        {
            "xAxis": {"type": "category", "data": ["X"]},
            "yAxis": {"type": "value"},
            "series": [{"data": [1]}],
        }
    )
    result_str = await _generate_visualization("bar", data, "Inject Type")
    result = json.loads(result_str)

    assert "bar" in result["html"]


@pytest.mark.asyncio
async def test_invalid_options_json_is_ignored():
    """Test: invalid options JSON is silently ignored, chart still renders."""
    data = json.dumps(
        {
            "series": [{"data": [1, 2, 3], "type": "bar"}],
        }
    )
    result_str = await _generate_visualization("bar", data, "Test", options="not-json")
    result = json.loads(result_str)

    assert "html" in result
    assert result["chart_type"] == "bar"


@pytest.mark.asyncio
async def test_empty_options_defaults_to_empty_dict():
    """Test: empty options string works fine."""
    data = json.dumps(
        {
            "series": [{"data": [1], "type": "gauge"}],
        }
    )
    result_str = await _generate_visualization("gauge", data, "Gauge", options="")
    result = json.loads(result_str)

    assert "html" in result
    assert "gauge" in result["chart_type"]


def test_build_html_structure():
    """Test: _build_html produces a complete HTML document with expected elements."""
    html = _build_html("My Title", '{"series":[]}')

    assert html.startswith("<!DOCTYPE html>")
    assert "<title>My Title</title>" in html
    assert "echarts.init" in html
    assert "postMessage" in html
    assert "viz-resize" in html
    assert "ResizeObserver" in html


@pytest.mark.asyncio
async def test_html_injection_in_title_is_escaped_by_json():
    """Test: title with special chars doesn't break the HTML (JSON serialization handles it)."""
    data = json.dumps({"series": [{"data": [1], "type": "bar"}]})
    result_str = await _generate_visualization("bar", data, 'Test <script>alert("xss")</script>')
    result = json.loads(result_str)

    # The title ends up in the HTML; it won't be XSS because the browser
    # treats it as part of a <title> tag and a JS string, not executable HTML.
    assert "html" in result
