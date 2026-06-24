"""Visualization tool — generates self-contained HTML/JS charts via Apache ECharts."""

import json

from agents import function_tool


async def _generate_visualization(
    chart_type: str,
    data: str,
    title: str,
    options: str = "{}",
) -> str:
    """Generate an interactive chart as a self-contained HTML page using Apache ECharts.

    Args:
        chart_type: ECharts series type (bar, line, pie, scatter, radar, heatmap,
            treemap, funnel, gauge, candlestick, boxplot, sankey, sunburst, parallel).
        data: JSON string of the ECharts option object. Must include the relevant
            config for the chart type (xAxis/yAxis for cartesian, series with type
            and data, etc.).
        title: Chart title displayed at the top.
        options: JSON string of additional ECharts option overrides (tooltip, legend,
            toolbox, dataZoom, color palette, etc.) merged into the base config.
    """
    try:
        data_obj = json.loads(data)
    except json.JSONDecodeError:
        return json.dumps(
            {"error": "Invalid JSON in 'data' parameter", "chart_type": chart_type, "title": title}
        )

    try:
        options_obj = json.loads(options) if options else {}
    except json.JSONDecodeError:
        options_obj = {}

    # Build ECharts option: merge data config + user overrides + title + defaults
    echarts_option = {
        **data_obj,
        "title": {
            "text": title,
            "left": "center",
            "top": 8,
            "textStyle": {"fontSize": 16, "fontWeight": "bold"},
        },
        "tooltip": {"trigger": "axis"},
        "toolbox": {
            "feature": {
                "saveAsImage": {"title": "Save"},
                "dataView": {"title": "Data", "readOnly": True},
            },
            "right": 16,
            "top": 8,
        },
        **options_obj,
    }

    # Ensure series entries have the correct type
    if "series" in echarts_option and isinstance(echarts_option["series"], list):
        for s in echarts_option["series"]:
            if "type" not in s:
                s["type"] = chart_type

    html = _build_html(title, json.dumps(echarts_option))

    return json.dumps(
        {
            "html": html,
            "chart_type": chart_type,
            "title": title,
        }
    )


def _build_html(title: str, echarts_option_json: str) -> str:
    """Wrap ECharts option in a complete HTML page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #FFFCF6;
    padding: 12px;
  }}
  #chart {{
    width: 100%;
    height: 420px;
  }}
</style>
</head>
<body>
<div id="chart"></div>
<script>
  var chart = echarts.init(document.getElementById('chart'));
  var option = {echarts_option_json};
  chart.setOption(option);

  // Resize chart when container changes
  var ro = new ResizeObserver(function() {{
    chart.resize();
  }});
  ro.observe(document.getElementById('chart'));

  // Auto-resize iframe: tell parent our height
  function reportHeight() {{
    var h = document.documentElement.scrollHeight;
    window.parent.postMessage({{ type: 'viz-resize', height: h }}, '*');
  }}
  reportHeight();
  window.addEventListener('resize', reportHeight);
  // Also report after chart finishes initial render
  setTimeout(reportHeight, 300);
</script>
</body>
</html>"""


generate_visualization = function_tool(_generate_visualization)
