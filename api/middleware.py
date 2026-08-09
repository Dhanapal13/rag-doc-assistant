import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from metrics import RAG_QUERIES_TOTAL

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        if not request.url.path.startswith(("/metrics", "/docs", "/openapi", "/redocs")):
            RAG_QUERIES_TOTAL.inc()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response