"""
Custom Exceptions

Defines custom exception classes for the GSL Backend application
with specific error codes and messages.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class GSLBackendException(Exception):
    """Base exception for GSL Backend."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# Authentication Exceptions
class AuthenticationError(GSLBackendException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AUTH_001",
            status_code=401,
            details=details
        )


class InvalidTokenError(GSLBackendException):
    """Raised when JWT token is invalid or expired."""
    
    def __init__(self, message: str = "Invalid or expired token", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AUTH_002",
            status_code=401,
            details=details
        )


class PermissionDeniedError(GSLBackendException):
    """Raised when user lacks required permissions."""
    
    def __init__(self, message: str = "Permission denied", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="AUTH_003",
            status_code=403,
            details=details
        )


# Recognition Exceptions
class GestureRecognitionError(GSLBackendException):
    """Raised when gesture recognition fails."""
    
    def __init__(self, message: str = "Gesture recognition failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RECOG_001",
            status_code=422,
            details=details
        )


class LowConfidenceError(GSLBackendException):
    """Raised when recognition confidence is too low."""
    
    def __init__(self, message: str = "Recognition confidence too low", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RECOG_002",
            status_code=422,
            details=details
        )


class ModelUnavailableError(GSLBackendException):
    """Raised when AI model is unavailable."""
    
    def __init__(self, message: str = "AI model unavailable", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RECOG_003",
            status_code=503,
            details=details
        )


# Translation Exceptions
class TranslationError(GSLBackendException):
    """Raised when translation fails."""
    
    def __init__(self, message: str = "Translation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="TRANS_001",
            status_code=422,
            details=details
        )


class SpeechRecognitionError(GSLBackendException):
    """Raised when speech-to-text conversion fails."""
    
    def __init__(self, message: str = "Speech recognition failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="TRANS_002",
            status_code=422,
            details=details
        )


# Media Exceptions
class InvalidMediaError(GSLBackendException):
    """Raised when media file is invalid."""
    
    def __init__(self, message: str = "Invalid media file", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="MEDIA_001",
            status_code=400,
            details=details
        )


class FileSizeExceededError(GSLBackendException):
    """Raised when file size exceeds limit."""
    
    def __init__(self, message: str = "File size exceeds maximum allowed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="MEDIA_002",
            status_code=413,
            details=details
        )


class MediaProcessingError(GSLBackendException):
    """Raised when media processing fails."""
    
    def __init__(self, message: str = "Media processing failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="MEDIA_003",
            status_code=422,
            details=details
        )


# Resource Exceptions
class ResourceNotFoundError(GSLBackendException):
    """Raised when requested resource is not found."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RES_001",
            status_code=404,
            details=details
        )


class ResourceAlreadyExistsError(GSLBackendException):
    """Raised when resource already exists."""
    
    def __init__(self, message: str = "Resource already exists", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RES_002",
            status_code=409,
            details=details
        )


# Rate Limiting Exceptions
class RateLimitExceededError(GSLBackendException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="RATE_001",
            status_code=429,
            details=details
        )


# Validation Exceptions
class ValidationError(GSLBackendException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str = "Validation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="VAL_001",
            status_code=400,
            details=details
        )


# Database Exceptions
class DatabaseError(GSLBackendException):
    """Raised when database operation fails."""
    
    def __init__(self, message: str = "Database operation failed", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="DB_001",
            status_code=500,
            details=details
        )


# System Exceptions
class SystemError(GSLBackendException):
    """Raised for general system errors."""
    
    def __init__(self, message: str = "System error occurred", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="SYS_001",
            status_code=500,
            details=details
        )


class TimeoutError(GSLBackendException):
    """Raised when operation times out."""
    
    def __init__(self, message: str = "Operation timed out", details: Optional[Dict] = None):
        super().__init__(
            message=message,
            error_code="TIME_001",
            status_code=504,
            details=details
        )