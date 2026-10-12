"""Entry point. Usage: python run.py"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"\n💬  Smart Customer Support Chatbot")
    print(f"   LLM model   : {settings.LLM_MODEL}")
    print(f"   Embed model : {settings.EMBEDDING_MODEL}")
    print(f"   Rate limit  : {settings.RATE_LIMIT_PER_MINUTE} req/min")
    print(f"   Auth        : Bearer token required")
    print(f"   Docs        : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   Demo API key: demo-key-12345")
    print(f"   Quick start:")
    print(f"   1) POST /ingest/directory  — load knowledge base")
    print(f"   2) POST /chat  with Authorization: Bearer demo-key-12345\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT,
                reload=settings.DEBUG, log_level="info")
