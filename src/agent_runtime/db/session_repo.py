"""Session and message repository using SQLAlchemy ORM."""

from sqlalchemy import select

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Message, Session


class SessionRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_session(
        self,
        title: str = "New Session",
        tools: list[str] | None = None,
    ) -> Session:
        """Create a new session. Returns the ORM object with the auto-generated ID.

        `tools` is the per-session tool allowlist. None means "use server defaults"
        (tools_json column is left NULL); [] means "no tools"; ["web_search"]
        means "subset". See api/schemas.py::SessionCreate for the full contract.
        """
        async with self.db.session() as s:
            sess = Session(title=title, tools_json=tools)
            s.add(sess)
            await s.flush()
            return sess

    async def update_title(self, session_id: int, title: str) -> None:
        """Update a session's title."""
        async with self.db.session() as s:
            sess = await s.get(Session, session_id)
            if sess:
                sess.title = title

    async def update_tools(self, session_id: int, tools: list[str] | None) -> None:
        """Update a session's tool allowlist.

        Pass None to reset to server defaults (sets tools_json to NULL).
        Pass [] to disable all tools for this session.
        Pass a non-empty list to scope the session to those tools.
        """
        async with self.db.session() as s:
            sess = await s.get(Session, session_id)
            if sess:
                sess.tools_json = tools

    async def get_tools(self, session_id: int) -> list[str] | None:
        """Read a session's persisted tool allowlist.

        Returns None when the row's tools_json is NULL (i.e. server defaults).
        """
        async with self.db.session() as s:
            sess = await s.get(Session, session_id)
            if sess is None:
                return None
            return sess.tools_json

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        system_prompt_id: int | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
        output_preview: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        usage_json: dict | None = None,
        thinking_json: dict | None = None,
    ) -> Message:
        async with self.db.session() as s:
            msg = Message(
                session_id=session_id,
                role=role,
                content=content,
                system_prompt_id=system_prompt_id,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                tool_input=tool_input,
                tool_output=tool_output,
                output_preview=output_preview,
                provider=provider,
                model=model,
                usage_json=usage_json,
                thinking_json=thinking_json,
            )
            s.add(msg)
            await s.flush()
            return msg

    async def get_latest_system_message(self, session_id: int) -> Message | None:
        """Get the most recent system message in a session."""
        async with self.db.session() as s:
            result = await s.execute(
                select(Message)
                .where(
                    Message.session_id == session_id,
                    Message.role == "system",
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_messages(self, session_id: int) -> list[Message]:
        async with self.db.session() as s:
            result = await s.execute(
                select(Message).where(Message.session_id == session_id).order_by(Message.created_at)
            )
            return list(result.scalars().all())

    async def list_sessions(self, limit: int = 20) -> list[Session]:
        async with self.db.session() as s:
            result = await s.execute(
                select(Session).order_by(Session.updated_at.desc()).limit(limit)
            )
            return list(result.scalars().all())

    async def get_session(self, session_id: int) -> Session | None:
        async with self.db.session() as s:
            return await s.get(Session, session_id)
