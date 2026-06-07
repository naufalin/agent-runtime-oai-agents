"""CLI chat interface for the agent runtime."""

import asyncio
import uuid

from dotenv import load_dotenv

load_dotenv()  # must run before importing agent modules (OpenAI SDK reads env at import)

from agent_runtime.agents.runtime import run_agent  # noqa: E402


async def chat_loop() -> None:
    conversation_id = str(uuid.uuid4())
    print(f"Agent Runtime — conversation {conversation_id[:8]}...")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()  # noqa: ASYNC250
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        response = await run_agent(user_input, conversation_id=conversation_id)
        print(f"\nAgent: {response}\n")


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
