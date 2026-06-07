"""CLI chat interface for the agent runtime."""

import asyncio
import sys

from dotenv import load_dotenv

load_dotenv()  # must run before importing agent modules (OpenAI SDK reads env at import)

from agent_runtime.agents.runtime import get_db, run_agent, switch_prompt  # noqa: E402
from agent_runtime.db.prompt_repo import SystemPromptRepo  # noqa: E402
from agent_runtime.db.session_repo import SessionRepo  # noqa: E402
from agent_runtime.ids import decode, encode  # noqa: E402


async def chat_loop(session_id: str | None = None) -> None:
    db = await get_db()
    repo = SessionRepo(db)

    # Track internal ID for history lookups
    internal_id: int | None = None

    # Resume or create session
    if session_id:
        try:
            internal_id = decode(session_id)
            existing = await repo.get_session(internal_id)
            if existing:
                # Show active prompt
                sys_msg = await repo.get_latest_system_message(internal_id)
                prompt_name = "custom"
                if sys_msg and sys_msg.system_prompt_id:
                    prompt_repo = SystemPromptRepo(db)
                    p = await prompt_repo.get_by_id(sys_msg.system_prompt_id)
                    if p:
                        prompt_name = p.name
                print(f"Resuming: {existing.title} [prompt: {prompt_name}]")
                messages = await repo.get_messages(internal_id)
                for msg in messages:
                    if msg.role == "system":
                        continue
                    role = "You" if msg.role == "user" else "Agent"
                    print(f"  [{role}]: {msg.content[:80]}...")
                print()
            else:
                print(f"Session {session_id} not found, starting new one.")
                session_id = None
                internal_id = None
        except ValueError:
            print(f"Invalid session ID: {session_id}")
            session_id = None

    if not session_id:
        print("Starting new session...")

    print("Commands: 'quit', 'history', 'prompts', 'switch <name>'\n")

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
                    label = msg.role.capitalize()
                    if msg.role == "system":
                        label = f"System [prompt_id={msg.system_prompt_id}]"
                    content = msg.content[:200].replace("\n", " ")
                    print(f"  [{label}]: {content}")
            else:
                print("  No messages yet.")
            print()
            continue
        if user_input.lower() == "prompts":
            await _list_prompts()
            continue
        if user_input.lower().startswith("switch "):
            name = user_input[7:].strip()
            if session_id:
                try:
                    await switch_prompt(session_id, name)
                    print(f"  Switched to prompt: {name}\n")
                except ValueError as e:
                    print(f"  Error: {e}\n")
            else:
                print("  Start a session first.\n")
            continue

        result = await run_agent(user_input, session_id=session_id)
        session_id = result.session_id
        if internal_id is None:
            internal_id = decode(session_id)
        print(f"\nAgent: {result.response}")
        print(f"  [id: {session_id}]\n")


async def _list_prompts() -> None:
    db = await get_db()
    repo = SystemPromptRepo(db)
    prompts = await repo.list_all()
    if not prompts:
        print("  No prompts found.")
        return
    for p in prompts:
        preview = p.content[:80].replace("\n", " ")
        print(f"  {p.name:<20} {preview}...")
    print()


async def list_sessions() -> None:
    db = await get_db()
    repo = SessionRepo(db)
    prompt_repo = SystemPromptRepo(db)
    sessions = await repo.list_sessions(limit=10)
    if not sessions:
        print("No sessions yet.")
        return
    print(f"{'ID':<12} {'Title':<40} {'Prompt':<12} {'Updated'}")
    print("-" * 80)
    for s in sessions:
        encoded = encode(s.id)
        # Get active prompt name
        sys_msg = await repo.get_latest_system_message(s.id)
        prompt_name = "-"
        if sys_msg and sys_msg.system_prompt_id:
            p = await prompt_repo.get_by_id(sys_msg.system_prompt_id)
            if p:
                prompt_name = p.name
        ts = s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else ""
        print(f"{encoded:<12} {s.title:<40} {prompt_name:<12} {ts}")


async def create_prompt(name: str, content: str) -> None:
    db = await get_db()
    repo = SystemPromptRepo(db)
    existing = await repo.get_by_name(name)
    if existing:
        print(f"Prompt '{name}' already exists.")
        return
    await repo.create(name, content)
    print(f"Created prompt: {name}")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        asyncio.run(chat_loop())
    elif args[0] == "list":
        asyncio.run(list_sessions())
    elif args[0] == "resume":
        sid = args[1] if len(args) > 1 else None
        if not sid:
            print("Usage: python main.py resume <session_id>")
            sys.exit(1)
        asyncio.run(chat_loop(sid))
    elif args[0] == "prompts":
        asyncio.run(_list_prompts())
    elif args[0] == "create-prompt":
        if len(args) < 3:
            print("Usage: python main.py create-prompt <name> <content>")
            sys.exit(1)
        asyncio.run(create_prompt(args[1], " ".join(args[2:])))
    else:
        print(f"Unknown command: {args[0]}")
        print("Commands: list, resume <id>, prompts, create-prompt <name> <content>")
        sys.exit(1)


if __name__ == "__main__":
    main()
