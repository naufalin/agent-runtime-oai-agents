"""CLI chat interface for the agent runtime."""

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()  # must run before importing agent modules (OpenAI SDK reads env at import)

from agent_runtime.agents.runtime import get_db, run_agent  # noqa: E402
from agent_runtime.db.conversation_repo import ConversationRepo  # noqa: E402
from agent_runtime.ids import encode  # noqa: E402


async def chat_loop(conversation_id: str | None = None) -> None:
    db = await get_db()
    repo = ConversationRepo(db)

    # Track internal ID for history lookups
    internal_id: int | None = None

    # Resume or create conversation
    if conversation_id:
        from agent_runtime.ids import decode

        try:
            internal_id = decode(conversation_id)
            existing = await repo.get_conversation(internal_id)
            if existing:
                print(f"Resuming: {existing.title}")
                messages = await repo.get_messages(internal_id)
                for msg in messages:
                    role = "You" if msg.role == "user" else "Agent"
                    print(f"  [{role}]: {msg.content[:80]}...")
                print()
            else:
                print(f"Conversation {conversation_id} not found, starting new one.")
                conversation_id = None
                internal_id = None
        except ValueError:
            print(f"Invalid conversation ID: {conversation_id}")
            conversation_id = None

    if not conversation_id:
        print("Starting new conversation...")

    print("Commands: 'quit' to exit, 'history' to show chat history\n")

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
        if user_input.lower() == "history":
            if internal_id:
                messages = await repo.get_messages(internal_id)
                for msg in messages:
                    role = "You" if msg.role == "user" else "Agent"
                    print(f"  [{role}]: {msg.content[:200]}")
            else:
                print("  No messages yet.")
            print()
            continue

        result = await run_agent(user_input, conversation_id=conversation_id)
        conversation_id = result.conversation_id
        if internal_id is None:
            from agent_runtime.ids import decode

            internal_id = decode(conversation_id)
        print(f"\nAgent: {result.response}")
        print(f"  [id: {conversation_id}]\n")


async def list_conversations() -> None:
    db = await get_db()
    repo = ConversationRepo(db)
    convos = await repo.list_conversations(limit=10)
    if not convos:
        print("No conversations yet.")
        return
    print(f"{'ID':<12} {'Title':<50} {'Updated'}")
    print("-" * 80)
    for c in convos:
        encoded = encode(c.id)
        ts = c.updated_at.strftime("%Y-%m-%d %H:%M") if c.updated_at else ""
        print(f"{encoded:<12} {c.title:<50} {ts}")


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "list":
        asyncio.run(list_conversations())
    elif args and args[0] == "resume":
        cid = args[1] if len(args) > 1 else None
        if not cid:
            print("Usage: python main.py resume <conversation_id>")
            sys.exit(1)
        asyncio.run(chat_loop(cid))
    else:
        asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
