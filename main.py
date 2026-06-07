"""Agent runtime entry point — dispatches to CLI or API server."""

import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "api":
        import uvicorn

        uvicorn.run("agent_runtime.api.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        from agent_runtime.cli import cli_main

        cli_main()


if __name__ == "__main__":
    main()
