"""
Common Schemas

Shared Pydantic models used across multiple API endpoints.
Includes pagination, error responses, and generic API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum


class StatusEnum(str, Enum):
    """API response status."""
    success = "success"
    error = "error"
    warning = "warning"
    info = "info"


class ErrorCode(str, Enum):
    """Standard error codes."""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INVALID_INPUT = "INVALID_INPUT"
    PROCESSING_ERROR = "PROCESSING_ERROR"


# Generic type for paginated responses
T = TypeVar('T')


class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    page: int = Field(1, ge=1, description="Page number (starts at 1)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset from page and page_size."""
        return (self.page - 1) * self.page_size
    
    @property
    def limit(self) -> int:
        """Get limit (same as page_size)."""
        return self.page_size


class PaginationMeta(BaseModel):
    """Schema for pagination metadata."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic schema for paginated responses."""
    items: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    
    class Config:
        from_attributes = True


class ErrorDetail(BaseModel):
    """Schema for detailed error information."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    status: StatusEnum = StatusEnum.error
    error_code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class SuccessResponse(BaseModel):
    """Schema for generic success responses."""
    status: StatusEnum = StatusEnum.success
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Schema for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict, description="Status of dependent services")


class FileUploadResponse(BaseModel):
    """Schema for file upload responses."""
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_url: str = Field(..., description="URL to access the file")
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="MIME type of the file")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class BulkOperationRequest(BaseModel):
    """Schema for bulk operation requests."""
    operation: str = Field(..., description="Operation to perform")
    items: List[Dict[str, Any]] = Field(..., min_length=1, max_length=100, description="Items to process")
    options: Optional[Dict[str, Any]] = Field(None, description="Operation options")


class BulkOperationResponse(BaseModel):
    """Schema for bulk operation responses."""
    total_items: int = Field(..., description="Total items processed")
    successful: int = Field(..., description="Successfully processed items")
    failed: int = Field(..., description="Failed items")
    errors: List[ErrorDetail] = Field(default_factory=list, description="Errors encountered")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Operation results")


class SearchRequest(BaseModel):
    """Schema for generic search requests."""
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", pattern="^(asc|desc)$", description="Sort order")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class SearchResponse(BaseModel, Generic[T]):
    """Generic schema for search responses."""
    query: str = Field(..., description="Original search query")
    results: List[T] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total matching items")
    search_time: float = Field(..., description="Search time in seconds")
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


class RateLimitInfo(BaseModel):
    """Schema for rate limit information."""
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_at: datetime = Field(..., description="When the limit resets")


class MetadataResponse(BaseModel):
    """Schema for metadata responses."""
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = Field(1, description="Version number")
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class BatchRequest(BaseModel):
    """Schema for batch requests."""
    requests: List[Dict[str, Any]] = Field(..., min_length=1, max_length=50, description="Batch of requests")


class BatchResponse(BaseModel):
    """Schema for batch responses."""
    responses: List[Dict[str, Any]] = Field(..., description="Batch of responses")
    total: int = Field(..., description="Total requests processed")
    successful: int = Field(..., description="Successful requests")
    failed: int = Field(..., description="Failed requests")
