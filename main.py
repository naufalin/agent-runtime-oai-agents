"""Agent Runtime — API server is the primary interface."""

import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "cli":
        # CLI is the secondary interface: python main.py cli [command]
        from agent_runtime.cli import cli_main

        sys.argv = [sys.argv[0]] + args[1:]  # strip "cli" from argv
        cli_main()
    else:
        # Default: start the API server
        import uvicorn

        uvicorn.run("agent_runtime.api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
