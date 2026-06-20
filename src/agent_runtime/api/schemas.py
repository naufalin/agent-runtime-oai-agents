"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# --- Sessions ---


class SessionCreate(BaseModel):
    title: str = "New Session"


class SessionOut(BaseModel):
    id: str
    title: str
    prompt: str = "-"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SessionListOut(BaseModel):
    sessions: list[SessionOut]
    total: int


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    tool_name: str | None = None
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    created_at: datetime | None = None


class SessionDetail(BaseModel):
    id: str
    title: str
    prompt: str = "-"
    created_at: datetime | None = None
    messages: list[MessageOut]


# --- Chat ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    response: str
    session_id: str
    provider: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    messages: list[MessageOut]


class ModelsOut(BaseModel):
    default_provider: str
    openai: dict[str, Any]
    openrouter: dict[str, Any]


# --- Prompts ---


class PromptCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1)


class PromptOut(BaseModel):
    id: int
    name: str
    content: str
    created_at: datetime | None = None


class PromptListOut(BaseModel):
    prompts: list[PromptOut]


class PromptSwitch(BaseModel):
    name: str = Field(..., min_length=1)
