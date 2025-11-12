"""
Core Utilities and Infrastructure Package

This package contains core infrastructure components including database management,
security utilities, middleware, exception handling, and dependency injection.
"""

from app.core.database import (
    Base,
    engine,
    SessionLocal,
    get_db,
    init_db,
    DatabaseManager
)
from app.core.supabase_client import (
    SupabaseManager,
    get_supabase,
    get_supabase_client
)
from app.core.security import (
    SecurityManager,
    get_current_user,
    get_current_active_user,
    require_role,
    PermissionChecker
)
from app.core.middleware import (
    RequestLoggingMiddleware,
    RateLimitMiddleware,
    ErrorHandlingMiddleware,
    setup_middleware
)
from app.core.exceptions import (
    GSLBackendException,
    AuthenticationError,
    InvalidTokenError,
    PermissionDeniedError,
    GestureRecognitionError,
    LowConfidenceError,
    ModelUnavailableError,
    TranslationError,
    SpeechRecognitionError,
    InvalidMediaError,
    FileSizeExceededError,
    MediaProcessingError,
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    RateLimitExceededError,
    ValidationError,
    DatabaseError,
    SystemError,
    TimeoutError
)
from app.core.dependencies import (
    get_settings_cached,
    get_current_user as get_user_dependency,
    get_user_service,
    get_recognition_service,
    get_translation_service,
    get_learning_service,
    get_media_service
)

__all__ = [
    # Database
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "DatabaseManager",
    
    # Supabase
    "SupabaseManager",
    "get_supabase",
    "get_supabase_client",
    
    # Security
    "SecurityManager",
    "get_current_user",
    "get_current_active_user",
    "require_role",
    "PermissionChecker",
    
    # Middleware
    "RequestLoggingMiddleware",
    "RateLimitMiddleware",
    "ErrorHandlingMiddleware",
    "setup_middleware",
    
    # Exceptions
    "GSLBackendException",
    "AuthenticationError",
    "InvalidTokenError",
    "PermissionDeniedError",
    "GestureRecognitionError",
    "LowConfidenceError",
    "ModelUnavailableError",
    "TranslationError",
    "SpeechRecognitionError",
    "InvalidMediaError",
    "FileSizeExceededError",
    "MediaProcessingError",
    "ResourceNotFoundError",
    "ResourceAlreadyExistsError",
    "RateLimitExceededError",
    "ValidationError",
    "DatabaseError",
    "SystemError",
    "TimeoutError",
    
    # Dependencies
    "get_settings_cached",
    "get_user_dependency",
    "get_user_service",
    "get_recognition_service",
    "get_translation_service",
    "get_learning_service",
    "get_media_service",
]