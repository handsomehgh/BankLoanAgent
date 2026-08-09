"""
API request/response Pydantic models for FastAPI endpoints.
"""
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# ==================== Chat ====================

class ChatRequest(BaseModel):
    """POST /api/chat"""
    message: str = Field(default="", description="User message (empty when resuming)")
    thread_id: Optional[str] = Field(None, description="Conversation thread ID (auto-generated if not provided)")
    user_id: Optional[str] = Field(None, description="User ID (optional)")
    resume: Optional[Dict[str, Any]] = Field(
        None,
        description="Resume value for interrupted flow, e.g. {'action': 'reply', 'content': '...'}",
    )


class ChatResponse(BaseModel):
    """Response header for /api/chat (actual content via SSE stream)"""
    thread_id: str
    status: str = "streaming"


# ==================== Session ====================

class SessionResponse(BaseModel):
    """POST /api/session"""
    thread_id: str


# ==================== History ====================

class HistoryResponse(BaseModel):
    """GET /api/history/{thread_id}"""
    thread_id: str
    messages: List[Dict[str, str]]


# ==================== Memory ====================

class MemoryResponse(BaseModel):
    """GET /api/memory/{user_id}"""
    user_id: str
    memories: List[Dict[str, str]]


# ==================== Handoff ====================

class HandoffResolveRequest(BaseModel):
    """POST /api/handoff/resolve"""
    thread_id: str
    action: str = Field(..., description="Resolution action: 'reply' | 'escalate' | 'close'")
    content: str = Field("", description="Reply content (required when action='reply')")


class HandoffResolveResponse(BaseModel):
    """POST /api/handoff/resolve response"""
    thread_id: str
    status: str = "resolved"
