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


class RuntimeModelOut(BaseModel):
    id: int
    provider: str
    model_id: str
    name: str
    enabled: bool
    supports_reasoning: bool
    sort_order: int
    config: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RuntimeModelCreate(BaseModel):
    provider: str = Field(..., min_length=1, max_length=50)
    model_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(..., min_length=1, max_length=200)
    enabled: bool = True
    supports_reasoning: bool = False
    sort_order: int = 0
    config: dict[str, Any] | None = None


class RuntimeModelUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    enabled: bool | None = None
    supports_reasoning: bool | None = None
    sort_order: int | None = None
    config: dict[str, Any] | None = None


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
