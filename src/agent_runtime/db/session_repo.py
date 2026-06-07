"""Session and message repository using SQLAlchemy ORM."""

from sqlalchemy import select

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Message, Session


class SessionRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_session(self, title: str = "New Session") -> Session:
        """Create a new session. Returns the ORM object with the auto-generated ID."""
        async with self.db.session() as s:
            sess = Session(title=title)
            s.add(sess)
            await s.flush()
            return sess

    async def update_title(self, session_id: int, title: str) -> None:
        """Update a session's title."""
        async with self.db.session() as s:
            sess = await s.get(Session, session_id)
            if sess:
                sess.title = title

    async def add_message(
        self,
        session_id: int,
        role: str,
        content: str,
        system_prompt_id: int | None = None,
    ) -> Message:
        async with self.db.session() as s:
            msg = Message(
                session_id=session_id,
                role=role,
                content=content,
                system_prompt_id=system_prompt_id,
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
