"""
Middleware Components

Provides middleware for CORS, rate limiting, request logging,
and global exception handling.
"""

import time
import logging
from typing import Callable
from uuid import uuid4
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from collections import defaultdict
from datetime import datetime, timedelta

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all API requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request details and processing time.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Generate request ID
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""
    
    def __init__(self, app, requests_per_window: int = 100, window_seconds: int = 3600):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(list)
        self.ai_endpoints = ["/api/v1/recognition", "/api/v1/translate"]
        self.ai_rate_limit = settings.ai_rate_limit_requests
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits and process request.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response or rate limit error
        """
        # Get client identifier (IP or user ID)
        client_id = self._get_client_id(request)
        
        # Determine rate limit based on endpoint
        is_ai_endpoint = any(
            request.url.path.startswith(endpoint)
            for endpoint in self.ai_endpoints
        )
        rate_limit = self.ai_rate_limit if is_ai_endpoint else self.requests_per_window
        
        # Check rate limit
        if not self._check_rate_limit(client_id, rate_limit):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_code": "RATE_001",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": self.window_seconds
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_id, rate_limit)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(
            int(time.time()) + self.window_seconds
        )
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get user ID from token
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # In production, decode token to get user ID
            return auth_header
        
        # Fall back to IP address
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if client has exceeded rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.request_counts[client_id]) >= limit:
            return False
        
        # Add current request
        self.request_counts[client_id].append(now)
        return True
    
    def _get_remaining_requests(self, client_id: str, limit: int) -> int:
        """Get remaining requests for client."""
        return max(0, limit - len(self.request_counts[client_id]))


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global exception handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle exceptions and return formatted error responses.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response or error response
        """
        try:
            return await call_next(request)
            
        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error_code": "VAL_001",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except PermissionError as e:
            logger.warning(f"Permission denied: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error_code": "AUTH_002",
                    "message": "Permission denied",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except FileNotFoundError as e:
            logger.warning(f"Resource not found: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error_code": "RES_001",
                    "message": "Resource not found",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except TimeoutError as e:
            logger.error(f"Request timeout: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error_code": "TIME_001",
                    "message": "Request processing timeout",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error_code": "SYS_001",
                    "message": "An unexpected error occurred. Please try again later.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


def setup_middleware(app):
    """
    Configure all middleware for the application.
    
    Args:
        app: FastAPI application instance
    """
    # CORS middleware (should be first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )
    
    # Error handling middleware
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window
    )
    
    # Request logging middleware (should be last)
    app.add_middleware(RequestLoggingMiddleware)