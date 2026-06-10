"""Web agent tool powered by TinyFish Agent API (uses credits).

Endpoint: POST https://agent.tinyfish.ai/v1/automation/run
Docs: https://docs.tinyfish.ai/agent-api
Timeout: 120s. Typical latency: 15-60s.

NOTE: This tool uses TinyFish credits. Prefer web_search or web_fetch
for simple lookups. This is intended for future use when multi-step
browser automation workflows are needed.
"""

import json

from agents import function_tool

from agent_runtime.tools._tinyfish_common import tinyfish_request

TINYFISH_AGENT_URL = "https://agent.tinyfish.ai/v1/automation/run"

# Words in the result that indicate the goal likely failed
_BLOCK_INDICATORS = {"captcha", "blocked", "access denied", "forbidden", "bot detection"}


async def _web_agent(url: str, goal: str) -> str:
    """Run an automated web agent task on a specific URL.

    The agent navigates the page and executes the goal using browser automation.
    It can click, scroll, fill forms, and extract data — all from a natural-language
    instruction.

    ⚠️  Uses TinyFish credits. Prefer web_search or web_fetch when possible.

    Args:
        url: The target URL to automate on.
        goal: Natural-language task description. Be specific about what to extract
              and the desired output format.
              Good: "Extract all product names and prices. Return as a JSON array."
              Bad: "Get the data."
    """
    data = await tinyfish_request(
        "POST",
        TINYFISH_AGENT_URL,
        json={"url": url, "goal": goal, "browser_profile": "lite"},
        timeout=120.0,
    )

    status = data.get("status", "UNKNOWN")
    run_id = data.get("run_id", "")
    result = data.get("result")
    error = data.get("error")

    if status == "FAILED":
        error_msg = "Unknown error"
        if error:
            code = error.get("code", "")
            message = error.get("message", "")
            error_msg = f"{code}: {message}" if code else message
        return f"Agent task failed (run {run_id}): {error_msg}"

    if status == "CANCELLED":
        return f"Agent task was cancelled (run {run_id})."

    if status == "COMPLETED":
        if result is None:
            return f"Agent task completed (run {run_id}) but returned no result."

        # Check for block/captcha indicators in the result
        result_str = json.dumps(result) if isinstance(result, dict) else str(result)
        result_lower = result_str.lower()
        for indicator in _BLOCK_INDICATORS:
            if indicator in result_lower:
                return f"Agent task completed but may have been blocked.\nResult: {result_str}"

        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        return str(result)

    return f"Agent task ended with unexpected status '{status}' (run {run_id})."


web_agent = function_tool(_web_agent)
