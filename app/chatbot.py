"""
Smart Customer Support Chatbot — Multi-Agent Engine.

Three-agent architecture:
  1. IntentSentimentAgent  — classifies intent and detects customer sentiment
  2. RAGResponseAgent      — retrieves relevant docs and generates grounded answer
  3. WorkflowAgent         — triggers backend workflows (tickets, refunds, escalations)

Session memory with conversation history.
Rate-limited per session.
Escalation routing for frustrated/angry customers or out-of-scope queries.
"""
import json, re, logging, time
from collections import defaultdict
from typing import Optional
from openai import OpenAI

from app.config import settings
from app.models import (
    IntentCategory, SentimentLabel, ChatMessage,
    ChatResponse, SourceDoc, EscalationTicket
)
from app.ingestion import retrieve

logger = logging.getLogger(__name__)

# ── Session store ─────────────────────────────────────────────────
_sessions: dict[str, list[ChatMessage]] = defaultdict(list)
_escalations: list[dict] = []
_ticket_counter = [1]


def get_history(session_id: str) -> list[ChatMessage]:
    msgs = _sessions[session_id]
    max_msgs = settings.MAX_HISTORY_TURNS * 2
    return msgs[-max_msgs:]


def add_to_history(session_id: str, role: str, content: str):
    _sessions[session_id].append(ChatMessage(role=role, content=content))


def get_all_sessions() -> list[str]:
    return list(_sessions.keys())


def clear_session(session_id: str):
    _sessions.pop(session_id, None)


# ── LLM caller ────────────────────────────────────────────────────
def _llm(messages: list[dict], model: str = None, temperature: float = None) -> str:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def _parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


# ══════════════════════════════════════════════════════════════════
# Agent 1: Intent & Sentiment Classification
# ══════════════════════════════════════════════════════════════════
def intent_sentiment_agent(message: str, history: list[ChatMessage]) -> dict:
    """
    Classifies:
    - Intent: billing | technical | returns | shipping | account | product | escalation | general | out_of_scope
    - Sentiment: positive | neutral | frustrated | angry
    - Needs escalation: bool
    - Reformulated query: standalone version of the message for RAG retrieval
    """
    history_text = "\n".join(f"{m.role}: {m.content}" for m in history[-4:]) if history else ""
    # Built outside the f-string: a backslash inside an f-string expression is a
    # SyntaxError before Python 3.12, and "\\n" there was a literal backslash-n anyway.
    history_block = f"Conversation history:\n{history_text}\n\n" if history_text else ""

    prompt = f"""Classify this customer support message.
{history_block}
Current message: {message}

Return JSON only:
{{
  "intent": one of [billing|technical|returns|shipping|account|product|escalation|general|out_of_scope],
  "sentiment": one of [positive|neutral|frustrated|angry],
  "needs_escalation": boolean (true if angry, mentions legal/complaint, or asks for manager),
  "reformulated_query": standalone version of the question for document search,
  "urgency": one of [low|medium|high]
}}"""

    try:
        raw = _llm([{"role": "user", "content": prompt}], temperature=0.0)
        data = _parse_json(raw)
        return {
            "intent":              data.get("intent", "general"),
            "sentiment":           data.get("sentiment", "neutral"),
            "needs_escalation":    bool(data.get("needs_escalation", False)),
            "reformulated_query":  data.get("reformulated_query", message),
            "urgency":             data.get("urgency", "medium"),
        }
    except Exception as e:
        logger.warning(f"Intent agent parse error: {e}")
        return {"intent":"general","sentiment":"neutral","needs_escalation":False,
                "reformulated_query":message,"urgency":"medium"}


# ══════════════════════════════════════════════════════════════════
# Agent 2: RAG Response Generation
# ══════════════════════════════════════════════════════════════════
GROUNDING_SYSTEM = """You are a helpful, professional customer support agent.

RULES:
1. Answer ONLY from the Context provided below. Do not use prior knowledge.
2. If Context does not contain enough information, say: "I don't have specific information about that. Let me connect you with a specialist who can help."
3. Be empathetic and professional, especially if the customer is frustrated.
4. For billing/account issues, acknowledge the inconvenience before explaining the process.
5. Add [Source: title] citations at the end of factual claims.
6. Never promise things that aren't in the documentation.
7. Keep responses concise — 3-5 sentences maximum unless a step-by-step process is needed."""


def rag_response_agent(
    message: str,
    reformulated_query: str,
    intent: str,
    sentiment: str,
    history: list[ChatMessage],
) -> tuple[str, list[dict], str]:
    """
    Retrieves relevant documentation and generates a grounded response.
    Returns: (answer, source_docs, confidence)
    """
    # Retrieve relevant docs
    docs = retrieve(reformulated_query)
    confidence = _score_confidence(docs)

    # Build context
    if docs:
        context_parts = []
        for doc in docs:
            title = doc["metadata"].get("title", "Support Document")
            context_parts.append(f"[Source: {title}]\n{doc['text']}")
        context = "\n\n---\n\n".join(context_parts)
        is_grounded = True
    else:
        context = "No relevant documentation found for this query."
        is_grounded = False

    # Tone adjustment based on sentiment
    tone_instruction = {
        "frustrated": "The customer is frustrated. Start by acknowledging their frustration empathetically.",
        "angry":      "The customer is angry. Start with a sincere apology and focus on resolution.",
        "positive":   "The customer is in a good mood. Be warm and helpful.",
        "neutral":    "",
    }.get(sentiment, "")

    # Build messages with history
    messages = [{"role": "system", "content": GROUNDING_SYSTEM}]
    for msg in history[-6:]:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({
        "role": "user",
        "content": (
            f"{tone_instruction + chr(10) if tone_instruction else ''}"
            f"Context:\n{context}\n\n"
            f"Customer question: {message}"
        )
    })

    answer = _llm(messages)
    return answer, docs, confidence


def _score_confidence(docs: list[dict]) -> str:
    if not docs:
        return "no_context"
    top = docs[0]["score"]
    if top >= 0.85 and len(docs) >= 2:
        return "high"
    elif top >= 0.75:
        return "medium"
    return "low"


# ══════════════════════════════════════════════════════════════════
# Agent 3: Workflow Triggering
# ══════════════════════════════════════════════════════════════════
def workflow_agent(
    session_id: str,
    message: str,
    intent: str,
    sentiment: str,
    needs_escalation: bool,
    answer: str,
    urgency: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Decides whether to trigger backend workflows:
    - Escalation ticket creation for angry customers or unresolved issues
    - Workflow actions (create_ticket, initiate_refund, etc.)

    Returns: (is_escalated, escalation_reason, workflow_triggered)
    """
    workflow_triggered = None
    is_escalated = False
    escalation_reason = None

    # Auto-escalation conditions
    should_escalate = (
        needs_escalation or
        sentiment == "angry" or
        intent == "escalation" or
        (urgency == "high" and sentiment == "frustrated")
    )

    if should_escalate:
        is_escalated = True
        priority = "high" if sentiment == "angry" else "medium"
        escalation_reason = (
            "Customer requested escalation" if intent == "escalation"
            else "High frustration/anger detected"
            if sentiment in ("angry", "frustrated")
            else "High urgency unresolved issue"
        )
        ticket_id = f"TKT{_ticket_counter[0]:05d}"
        _ticket_counter[0] += 1
        ticket = {
            "ticket_id": ticket_id,
            "session_id": session_id,
            "customer_message": message[:200],
            "intent": intent,
            "sentiment": sentiment,
            "priority": priority,
            "assigned_to": "Support Team Queue",
            "status": "open",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        _escalations.append(ticket)
        logger.info(f"[WorkflowAgent] Created escalation ticket {ticket_id} (priority={priority})")
        workflow_triggered = f"escalation_ticket:{ticket_id}"

    # Intent-based workflow triggers (non-escalation)
    elif intent == "returns":
        workflow_triggered = "returns_process_initiated"
    elif intent == "billing":
        workflow_triggered = "billing_review_queued"
    elif intent == "shipping":
        workflow_triggered = "shipment_lookup_triggered"

    return is_escalated, escalation_reason, workflow_triggered


# ══════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════
def follow_up_suggestions(message: str, intent: str, answer: str) -> list[str]:
    """Generate relevant follow-up question suggestions."""
    try:
        prompt = f"""Based on this support Q&A, suggest 2-3 short follow-up questions the customer might ask.
Intent: {intent}
Q: {message}
A: {answer[:200]}
Return JSON list of strings. Max 12 words each."""
        raw = _llm([{"role": "user", "content": prompt}], temperature=0.5)
        raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        raw = re.sub(r'\s*```$', '', raw)
        suggestions = json.loads(raw.strip())
        return [s for s in suggestions if isinstance(s, str)][:3]
    except Exception:
        return []


def process_message(session_id: str, message: str, channel: str = "web") -> ChatResponse:
    """
    Full multi-agent pipeline:
    1. IntentSentimentAgent  → classify message
    2. RAGResponseAgent      → retrieve docs + generate answer
    3. WorkflowAgent         → trigger escalation/workflows
    4. Follow-up suggestions → enhance UX
    """
    logger.info(f"[Chatbot] Session={session_id} Message={message[:60]}...")

    history = get_history(session_id)

    # Agent 1: Intent & sentiment
    classification = intent_sentiment_agent(message, history)
    intent    = classification["intent"]
    sentiment = classification["sentiment"]
    needs_esc = classification["needs_escalation"]
    ref_query = classification["reformulated_query"]
    urgency   = classification["urgency"]

    logger.info(f"[Chatbot] Intent={intent} Sentiment={sentiment} Escalate={needs_esc}")

    # Agent 2: RAG response
    answer, docs, confidence = rag_response_agent(
        message, ref_query, intent, sentiment, history
    )

    # Agent 3: Workflow
    is_escalated, esc_reason, workflow = workflow_agent(
        session_id, message, intent, sentiment, needs_esc, answer, urgency
    )

    # Append escalation note to answer if escalated
    if is_escalated:
        answer += (
            "\n\nI've flagged your case for our specialist team who will reach out to you shortly. "
            f"Your ticket reference is: {workflow.split(':')[-1] if workflow else 'pending'}."
        )

    # Follow-up suggestions
    suggestions = follow_up_suggestions(message, intent, answer) if not is_escalated else []

    # Update session history
    add_to_history(session_id, "user", message)
    add_to_history(session_id, "assistant", answer)

    # Build source docs
    sources = [
        SourceDoc(
            title=doc["metadata"].get("title", "Support Document"),
            snippet=doc["text"][:200],
            relevance_score=doc["score"],
        )
        for doc in docs
    ]

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        intent=IntentCategory(intent) if intent in [e.value for e in IntentCategory] else IntentCategory.GENERAL,
        sentiment=SentimentLabel(sentiment) if sentiment in [e.value for e in SentimentLabel] else SentimentLabel.NEUTRAL,
        confidence=confidence,
        sources=sources,
        is_escalated=is_escalated,
        escalation_reason=esc_reason,
        follow_up_suggestions=suggestions,
        workflow_triggered=workflow,
    )


def get_session_messages(session_id: str) -> list[dict]:
    return [m.model_dump() for m in _sessions.get(session_id, [])]


def get_escalation_count() -> int:
    return len(_escalations)
