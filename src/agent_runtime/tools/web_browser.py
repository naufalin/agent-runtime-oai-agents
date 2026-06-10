"""Web browser tool powered by TinyFish Browser API (uses credits).

Endpoint: POST https://api.browser.tinyfish.ai
Docs: https://docs.tinyfish.ai/browser-api
Session creation: 10-30s. Inactivity timeout: 1 hour.

NOTE: This tool uses TinyFish credits and creates a remote browser session.
The session returns a CDP URL for Playwright/CDP control. This is intended
for future use — the runtime would need to connect to the CDP session via
Playwright to actually interact with pages.
"""

from agents import function_tool

from agent_runtime.tools._tinyfish_common import tinyfish_request

TINYFISH_BROWSER_URL = "https://api.browser.tinyfish.ai"


async def _web_browser(url: str) -> str:
    """Create a remote browser session for direct Playwright/CDP control.

    Returns a session ID and CDP WebSocket URL. The browser navigates to the
    given URL on startup. Sessions auto-terminate after 1 hour of inactivity.

    ⚠️  Uses TinyFish credits. This tool creates a session but does not
    interact with pages — use web_agent for automated tasks, or connect
    to the returned CDP URL with Playwright for direct control.

    Args:
        url: The URL to navigate to on session start.
    """
    data = await tinyfish_request(
        "POST",
        TINYFISH_BROWSER_URL,
        json={"url": url},
        timeout=60.0,
    )

    session_id = data.get("session_id", "")
    cdp_url = data.get("cdp_url", "")
    base_url = data.get("base_url", "")

    if not session_id:
        return "Browser session creation returned no session ID."

    return (
        f"Browser session created.\n\n"
        f"  Session ID: {session_id}\n"
        f"  CDP URL:    {cdp_url}\n"
        f"  Base URL:   {base_url}\n\n"
        f"Connect with Playwright using the CDP URL above.\n"
        f"Session auto-terminates after 1 hour of inactivity."
    )


web_browser = function_tool(_web_browser)
