"""Tests for the Smart Customer Support Chatbot System."""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.chatbot import (
    intent_sentiment_agent, _score_confidence,
    get_history, add_to_history, clear_session,
    _sessions
)
from app.ingestion import _recursive_chunk
from app.security import TokenBucketLimiter
from app.models import IntentCategory, SentimentLabel

# Use demo API key for all tests
HEADERS = {"Authorization": "Bearer demo-key-12345"}
client = TestClient(app)


# ── Unit: chunking ────────────────────────────────────────────────
def test_chunk_short_text():
    chunks = _recursive_chunk("Short text.")
    assert len(chunks) == 1

def test_chunk_empty():
    assert _recursive_chunk("") == []

def test_chunk_long_text_produces_multiple():
    text = "A" * 5000
    chunks = _recursive_chunk(text)
    assert len(chunks) >= 2

def test_chunk_preserves_content():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = _recursive_chunk(text)
    combined = " ".join(chunks)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined


# ── Unit: rate limiter ────────────────────────────────────────────
def test_rate_limiter_allows_within_limit():
    limiter = TokenBucketLimiter(rate=5)
    for _ in range(5):
        allowed, _ = limiter.is_allowed("test_key")
        assert allowed is True

def test_rate_limiter_blocks_over_limit():
    limiter = TokenBucketLimiter(rate=2)
    limiter.is_allowed("key1")
    limiter.is_allowed("key1")
    allowed, remaining = limiter.is_allowed("key1")
    assert allowed is False
    assert remaining < 1

def test_rate_limiter_different_keys_independent():
    limiter = TokenBucketLimiter(rate=1)
    limiter.is_allowed("key_a")
    # key_a is depleted but key_b should still work
    allowed_b, _ = limiter.is_allowed("key_b")
    assert allowed_b is True


# ── Unit: confidence scoring ──────────────────────────────────────
def test_confidence_no_docs():
    assert _score_confidence([]) == "no_context"

def test_confidence_high():
    docs = [{"score": 0.92}, {"score": 0.88}]
    assert _score_confidence(docs) == "high"

def test_confidence_medium():
    docs = [{"score": 0.78}]
    assert _score_confidence(docs) == "medium"

def test_confidence_low():
    docs = [{"score": 0.65}]
    assert _score_confidence(docs) == "low"


# ── Unit: session management ──────────────────────────────────────
def test_session_add_and_get():
    clear_session("test_sess_1")
    add_to_history("test_sess_1", "user", "Hello")
    add_to_history("test_sess_1", "assistant", "Hi there!")
    history = get_history("test_sess_1")
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].content == "Hi there!"

def test_session_clear():
    add_to_history("test_sess_2", "user", "Test message")
    clear_session("test_sess_2")
    assert get_history("test_sess_2") == []

def test_session_new_is_empty():
    history = get_history("brand_new_session_xyz_987")
    assert history == []


# ── API endpoint tests ────────────────────────────────────────────
def test_health_no_auth():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_requires_auth():
    r = client.post("/chat", json={"session_id":"s1","message":"hello"})
    assert r.status_code == 401

def test_chat_invalid_key():
    r = client.post("/chat",
        json={"session_id":"s1","message":"hello"},
        headers={"Authorization": "Bearer wrong-key"})
    assert r.status_code == 401

def test_chat_empty_message():
    r = client.post("/chat",
        json={"session_id":"s1","message":""},
        headers=HEADERS)
    assert r.status_code == 400

def test_chat_too_long_message():
    r = client.post("/chat",
        json={"session_id":"s1","message":"x"*2001},
        headers=HEADERS)
    assert r.status_code == 400

MOCK_CHAT_RESPONSE = {
    "session_id": "test_session",
    "answer": "Your order typically arrives in 3-5 business days. [Source: Support KB]",
    "intent": "shipping",
    "sentiment": "neutral",
    "confidence": "high",
    "sources": [{"title":"Support KB","snippet":"Standard shipping 3-5 days","relevance_score":0.91}],
    "is_escalated": False,
    "escalation_reason": None,
    "follow_up_suggestions": ["Can I track my order?","How do I change delivery address?"],
    "workflow_triggered": "shipment_lookup_triggered",
}

@patch("app.main.process_message")
def test_chat_success(mock_process):
    from app.models import ChatResponse, SourceDoc
    mock_process.return_value = ChatResponse(
        session_id="test_session",
        answer="Your order arrives in 3-5 days.",
        intent=IntentCategory.SHIPPING,
        sentiment=SentimentLabel.NEUTRAL,
        confidence="high",
        sources=[SourceDoc(title="Support KB", snippet="3-5 days", relevance_score=0.91)],
        is_escalated=False,
        escalation_reason=None,
        follow_up_suggestions=["Can I track my order?"],
        workflow_triggered="shipment_lookup_triggered",
    )
    r = client.post("/chat",
        json={"session_id":"test_session","message":"How long does shipping take?"},
        headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["intent"] == "shipping"
    assert data["is_escalated"] is False
    assert "workflow_triggered" in data

@patch("app.main.process_message")
def test_chat_escalation_triggered(mock_process):
    from app.models import ChatResponse, SourceDoc
    mock_process.return_value = ChatResponse(
        session_id="angry_sess",
        answer="I'm escalating this to our specialist team.",
        intent=IntentCategory.ESCALATION,
        sentiment=SentimentLabel.ANGRY,
        confidence="medium",
        sources=[],
        is_escalated=True,
        escalation_reason="Customer requested escalation",
        follow_up_suggestions=[],
        workflow_triggered="escalation_ticket:TKT00001",
    )
    r = client.post("/chat",
        json={"session_id":"angry_sess","message":"This is unacceptable! I want to speak to a manager!"},
        headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert data["is_escalated"] is True
    assert data["sentiment"] == "angry"

def test_get_session_empty():
    r = client.get("/sessions/brand_new_session_abc123", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["message_count"] == 0

def test_delete_session():
    r = client.delete("/sessions/some_session_to_clear", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"

def test_stats_requires_auth():
    r = client.get("/stats")
    assert r.status_code == 401

def test_stats_with_auth():
    r = client.get("/stats", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    assert "total_chunks" in data
    assert "total_sessions" in data
    assert "escalations_today" in data

@patch("app.main.ingest_text", return_value=5)
def test_ingest_text(mock_ingest):
    r = client.post("/ingest/text",
        json={"doc_id":"test_001","title":"Test Doc","content":"Content " * 50,"category":"support"},
        headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["chunks_created"] == 5
