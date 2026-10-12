# Customer Support Chatbot

[![Test, Build, Deploy](https://github.com/Rishichamp/customer-support-chatbot/actions/workflows/deploy.yml/badge.svg)](https://github.com/Rishichamp/customer-support-chatbot/actions/workflows/deploy.yml)

A production-style customer support chatbot. Three agents classify the request, answer from the knowledge base, and trigger the right workflow (ticket, refund or escalation to a human). It sits behind API-key authentication and rate limiting, and ships as a Docker image built by CI.

## How it works

```
message ──► IntentSentimentAgent ──► RAGResponseAgent ──► WorkflowAgent
            intent, sentiment,        grounded answer +     tickets, refunds,
            urgency, escalate?        confidence score      escalations
```

Angry customers, legal or complaint wording, and requests for a manager are escalated automatically.

## Features

- **Intent and sentiment classification**, with a reformulated standalone query for retrieval.
- **RAG answers** over `sample_docs/`, with a high/medium/low confidence score from retrieval quality.
- **Workflow agent** that opens tickets and escalations and suggests follow-up questions.
- **Bearer-token API keys** and a **token-bucket rate limiter**, applied per key or IP.
- **Server-sent events streaming** (`POST /chat/stream`) for token-by-token replies.
- **Session memory** for multi-turn conversations.
- **Docker** multi-stage image, run as a non-root user, with a health check.
- **CI/CD:** tests, then build and push the image to GHCR, then an optional Render deploy hook.

## Tech stack

Python 3.11 · FastAPI · OpenAI · ChromaDB · Docker · GitHub Actions · pytest

## Project structure

```
app/
  chatbot.py     the three agents, sessions and escalation logic
  ingestion.py   document chunking, embedding and retrieval
  security.py    API-key auth, rate limiting, request logging
  main.py        FastAPI app and endpoints
  models.py      request/response schemas
  config.py      settings from .env
sample_docs/     support knowledge base
tests/           26 tests
Dockerfile
.github/workflows/deploy.yml
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # add OPENAI_API_KEY and set API_KEYS to your own keys
python run.py
```

Send requests with `Authorization: Bearer <one of API_KEYS>`.

With Docker:

```bash
docker build -t customer-support-chatbot .
docker run -p 8001:8001 --env-file .env customer-support-chatbot
```

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/chat` | ✓ | Send a message |
| POST | `/chat/stream` | ✓ | Streaming reply (SSE) |
| GET | `/sessions` | ✓ | List sessions |
| GET | `/sessions/{session_id}` | ✓ | Conversation history |
| DELETE | `/sessions/{session_id}` | ✓ | Clear a session |
| POST | `/ingest/text` | ✓ | Ingest a document from text |
| POST | `/ingest/directory` | ✓ | Ingest every document in `DOCS_PATH` |
| GET | `/stats` | ✓ | Statistics |
| GET | `/health` | – | Health check |

## Tests

```bash
pytest tests/ -q
```

26 tests covering chunking, the rate limiter, confidence scoring, sessions, authentication, escalation and the API, with the LLM mocked.

## Deployment

The workflow pushes `ghcr.io/rishichamp/customer-support-chatbot:latest` on every push to `main`. To auto-deploy to Render, add a `RENDER_DEPLOY_HOOK_URL` repository secret; without it, the deploy step is skipped.

## Notes

- Sessions and rate-limit buckets live in memory, per worker process. Use Redis to share them across workers or instances.
- Built as part of Module 4 of a Generative AI engineering programme.
