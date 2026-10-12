"""
Security middleware for the Customer Support Chatbot.
Implements:
  - Bearer token API key authentication
  - Token bucket rate limiting per API key / IP
  - Request/response logging middleware
"""
import time, logging
from collections import defaultdict
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

from app.config import settings

logger = logging.getLogger(__name__)

# ── API key auth ──────────────────────────────────────────────────
_security = HTTPBearer(auto_error=False)
VALID_KEYS = set(k.strip() for k in settings.API_KEYS.split(",") if k.strip())


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(_security)
) -> str:
    """Verify Bearer token. Returns the key if valid."""
    if not credentials:
        raise HTTPException(401, "Authorization header required. Use: Bearer <api-key>")
    if credentials.credentials not in VALID_KEYS:
        raise HTTPException(401, "Invalid API key.")
    return credentials.credentials


# ── Token bucket rate limiter ─────────────────────────────────────
class TokenBucketLimiter:
    """
    Token bucket algorithm:
    - Each key starts with `rate` tokens
    - Tokens refill at `rate` per minute
    - Each request consumes 1 token
    - Empty bucket → 429 response
    """
    def __init__(self, rate: int):
        self.rate    = rate
        self.buckets = defaultdict(lambda: {"tokens": float(rate), "last": time.time()})

    def is_allowed(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, tokens_remaining)."""
        b   = self.buckets[key]
        now = time.time()
        # Refill proportional to elapsed time
        elapsed = now - b["last"]
        b["tokens"] = min(float(self.rate), b["tokens"] + elapsed * self.rate / 60.0)
        b["last"] = now
        if b["tokens"] < 1.0:
            return False, b["tokens"]
        b["tokens"] -= 1.0
        return True, b["tokens"]


_limiter = TokenBucketLimiter(rate=settings.RATE_LIMIT_PER_MINUTE)


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.
    Identifies callers by API key (from Authorization header) or IP.
    """
    # Skip rate limiting for health check and docs
    if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc", "/"):
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    key  = auth.replace("Bearer ", "").strip() if auth else request.client.host

    allowed, remaining = _limiter.is_allowed(key)
    if not allowed:
        from fastapi.responses import JSONResponse
        logger.warning(f"Rate limit exceeded for key={key[:8]}...")
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please wait 60 seconds."},
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(int(remaining))
    return response
