"""Conversation and message repository using SQLAlchemy ORM."""

from sqlalchemy import select

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Conversation, Message


class ConversationRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_conversation(self, title: str = "New Conversation") -> Conversation:
        """Create a new conversation. Returns the ORM object with the auto-generated ID."""
        async with self.db.session() as session:
            conv = Conversation(title=title)
            session.add(conv)
            await session.flush()
            return conv

    async def update_title(self, conversation_id: int, title: str) -> None:
        """Update a conversation's title."""
        async with self.db.session() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv:
                conv.title = title

    async def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        system_prompt_id: int | None = None,
    ) -> Message:
        async with self.db.session() as session:
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                system_prompt_id=system_prompt_id,
            )
            session.add(msg)
            await session.flush()
            return msg

    async def get_latest_system_message(self, conversation_id: int) -> Message | None:
        """Get the most recent system message in a conversation."""
        async with self.db.session() as session:
            result = await session.execute(
                select(Message)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.role == "system",
                )
                .order_by(Message.id.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def get_messages(self, conversation_id: int) -> list[Message]:
        async with self.db.session() as session:
            result = await session.execute(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            )
            return list(result.scalars().all())

    async def list_conversations(self, limit: int = 20) -> list[Conversation]:
        async with self.db.session() as session:
            result = await session.execute(
                select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            )
            return list(result.scalars().all())

    async def get_conversation(self, conversation_id: int) -> Conversation | None:
        async with self.db.session() as session:
            return await session.get(Conversation, conversation_id)
