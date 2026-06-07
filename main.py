"""Agent Runtime — API server is the primary interface."""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # CLI is the secondary interface: python main.py cli [command]
        from agent_runtime.cli import cli_main

        sys.argv = [sys.argv[0]] + sys.argv[2:]  # strip "cli" from argv
        cli_main()
        return

    # Default: start the API server with configurable args
    parser = argparse.ArgumentParser(
        prog="agent-runtime",
        description="Agent Runtime API Server",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "agent_runtime.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
