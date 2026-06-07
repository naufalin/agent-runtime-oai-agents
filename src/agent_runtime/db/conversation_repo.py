"""Conversation and message repository using SQLAlchemy ORM."""

from sqlalchemy import select

from agent_runtime.db.connection import Database
from agent_runtime.db.models import Conversation, Message


class ConversationRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_conversation(
        self, conversation_id: str, title: str = "New Conversation"
    ) -> Conversation:
        async with self.db.session() as session:
            conv = Conversation(id=conversation_id, title=title)
            session.add(conv)
            await session.flush()
            return conv

    async def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> Message:
        async with self.db.session() as session:
            msg = Message(
                conversation_id=conversation_id, role=role, content=content
            )
            session.add(msg)
            await session.flush()
            return msg

    async def get_messages(self, conversation_id: str) -> list[Message]:
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

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        async with self.db.session() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            return result.scalar_one_or_none()
