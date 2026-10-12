"""Pydantic models for the Smart Customer Support Chatbot."""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class IntentCategory(str, Enum):
    BILLING       = "billing"
    TECHNICAL     = "technical"
    RETURNS       = "returns"
    SHIPPING      = "shipping"
    ACCOUNT       = "account"
    PRODUCT       = "product"
    ESCALATION    = "escalation"
    GENERAL       = "general"
    OUT_OF_SCOPE  = "out_of_scope"


class SentimentLabel(str, Enum):
    POSITIVE  = "positive"
    NEUTRAL   = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY     = "angry"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str  = Field(..., description="Session ID for conversation continuity")
    message: str     = Field(..., description="Customer message")
    channel: str     = Field("web", description="Channel: web | mobile | api")
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "cust_001_sess_1",
                "message": "My order hasn't arrived yet and it's been 2 weeks.",
                "channel": "web"
            }
        }
    }


class SourceDoc(BaseModel):
    title: str
    snippet: str
    relevance_score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: IntentCategory
    sentiment: SentimentLabel
    confidence: str                     # high | medium | low | no_context
    sources: list[SourceDoc]
    is_escalated: bool                  # True if routed to human agent
    escalation_reason: Optional[str]
    follow_up_suggestions: list[str]
    workflow_triggered: Optional[str]   # e.g. "create_ticket", "refund_initiated"


class IngestRequest(BaseModel):
    doc_id: str
    title: str
    content: str
    category: Optional[str] = "general"


class IngestResponse(BaseModel):
    doc_id: str
    chunks_created: int
    status: str


class EscalationTicket(BaseModel):
    ticket_id: str
    session_id: str
    customer_message: str
    intent: str
    sentiment: str
    priority: str
    assigned_to: str
    status: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_sessions: int
    escalations_today: int
    embedding_model: str
    llm_model: str
