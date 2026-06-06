"""Repository for conversation and message CRUD."""

from agent_runtime.db.connection import Database


class ConversationRepo:
    def __init__(self, db: Database):
        self.db = db

    async def create_conversation(self, conversation_id: str, title: str = "New Conversation"):
        return await self.db.fetchrow(
            "INSERT INTO conversations (id, title) VALUES ($1, $2) RETURNING id, title, created_at",
            conversation_id,
            title,
        )

    async def add_message(self, conversation_id: str, role: str, content: str):
        return await self.db.fetchrow(
            "INSERT INTO messages (conversation_id, role, content) "
            "VALUES ($1, $2, $3) RETURNING id, role, content, created_at",
            conversation_id,
            role,
            content,
        )

    async def get_messages(self, conversation_id: str) -> list[dict]:
        rows = await self.db.fetch(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = $1 ORDER BY created_at",
            conversation_id,
        )
        return [dict(row) for row in rows]

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        rows = await self.db.fetch(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT $1",
            limit,
        )
        return [dict(row) for row in rows]

    async def get_conversation(self, conversation_id: str):
        return await self.db.fetchrow(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = $1",
            conversation_id,
        )
