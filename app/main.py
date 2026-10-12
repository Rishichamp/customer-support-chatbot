"""
Smart Customer Support Chatbot System — FastAPI Application
===========================================================
Production-ready features:
  - Multi-agent pipeline (Intent + RAG + Workflow)
  - Bearer token authentication
  - Token bucket rate limiting
  - Streaming SSE for real-time responses
  - Session management
  - Escalation ticketing

Endpoints:
  POST /chat               — send message (authenticated)
  GET  /chat/stream        — streaming version
  GET  /sessions/{id}      — conversation history
  DELETE /sessions/{id}    — clear session
  POST /ingest/text        — ingest a document
  POST /ingest/directory   — ingest all docs from DOCS_PATH
  GET  /stats              — system statistics
  GET  /health             — health check (no auth)
"""
import logging, json
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
import os

from app.config import settings
from app.models import (
    ChatRequest, ChatResponse, IngestRequest,
    IngestResponse, StatsResponse,
)
from app.chatbot import (
    process_message, get_session_messages,
    clear_session, get_all_sessions, get_escalation_count,
)
from app.ingestion import ingest_text, ingest_directory, get_stats
from app.security import verify_api_key, rate_limit_middleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Customer Support Chatbot",
    description=(
        "Production-ready customer support chatbot with multi-agent pipeline "
        "(Intent classification + RAG response + Workflow triggering), "
        "rate limiting, authentication, and escalation routing."
    ),
    version="1.0.0",
)

# ── Middleware ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
app.middleware("http")(rate_limit_middleware)


# ── Root ──────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>💬 Smart Customer Support Chatbot</h1>
    <p>Multi-agent: Intent Classification + RAG Response + Workflow Triggering</p>
    <p><a href="/docs">📖 Swagger UI</a></p>
    <h3>Authentication:</h3>
    <p>Use header: <code>Authorization: Bearer demo-key-12345</code></p>
    </body></html>"""


@app.get("/health")
async def health():
    """Health check — no authentication required."""
    stats = get_stats()
    return {"status": "ok", "llm_model": settings.LLM_MODEL, **stats}


# ── Chat endpoints ────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Process a customer message through the multi-agent pipeline.

    Pipeline:
    1. IntentSentimentAgent  — classify intent (billing/technical/returns/etc.) and sentiment
    2. RAGResponseAgent      — retrieve relevant docs + generate grounded answer
    3. WorkflowAgent         — trigger escalation ticket or backend workflows if needed

    Requires: Authorization: Bearer <api-key>
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    if len(req.message) > 2000:
        raise HTTPException(400, "Message too long (max 2000 chars).")
    try:
        return process_message(req.session_id, req.message, req.channel)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, f"Chat processing failed: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Streaming version of /chat. Returns tokens as Server-Sent Events.
    Run intent/RAG pipeline first, then stream the LLM generation token-by-token.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    # Run classification synchronously first
    from app.chatbot import intent_sentiment_agent, get_history, retrieve
    history = get_history(req.session_id)
    classification = intent_sentiment_agent(req.message, history)
    docs = retrieve(classification["reformulated_query"])
    context = "\n\n".join(d["text"] for d in docs[:3]) if docs else "No relevant docs found."

    async def token_stream():
        async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        try:
            async with async_client.chat.completions.stream(
                model=settings.LLM_MODEL,
                messages=[
                    {"role":"system","content":"You are a helpful customer support agent. Answer from context only."},
                    {"role":"user","content":f"Context:\n{context}\n\nQuestion: {req.message}"},
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=512,
            ) as stream:
                async for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        yield f"data: {json.dumps({'token': token, 'intent': classification['intent']})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"},
    )


# ── Session management ────────────────────────────────────────────
@app.get("/sessions/{session_id}")
async def get_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Get conversation history for a session."""
    msgs = get_session_messages(session_id)
    return {"session_id": session_id, "message_count": len(msgs), "messages": msgs}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, api_key: str = Depends(verify_api_key)):
    """Clear conversation history for a session."""
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/sessions")
async def list_sessions(api_key: str = Depends(verify_api_key)):
    return {"sessions": get_all_sessions()}


# ── Ingestion ─────────────────────────────────────────────────────
@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(
    req: IngestRequest,
    api_key: str = Depends(verify_api_key),
):
    """Ingest a document from raw text content."""
    try:
        n = ingest_text(req.content, req.doc_id, req.title, req.category or "support")
        return IngestResponse(doc_id=req.doc_id, chunks_created=n,
                              status="ingested" if n > 0 else "empty")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/directory")
async def ingest_dir(
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Ingest all documents from the configured DOCS_PATH.
    Runs as a background task and returns immediately with a job confirmation.
    """
    def _do_ingest():
        results = ingest_directory()
        logger.info(f"Directory ingest complete: {len(results['success'])} files")

    background_tasks.add_task(_do_ingest)
    return {"status": "ingestion_started", "docs_path": settings.DOCS_PATH,
            "message": "Check /stats in a few seconds to see ingested chunks."}


# ── Stats ─────────────────────────────────────────────────────────
@app.get("/stats", response_model=StatsResponse)
async def stats(api_key: str = Depends(verify_api_key)):
    s = get_stats()
    return StatsResponse(
        total_chunks=s["total_chunks"],
        total_sessions=len(get_all_sessions()),
        escalations_today=get_escalation_count(),
        embedding_model=s["embedding_model"],
        llm_model=s["llm_model"],
    )
