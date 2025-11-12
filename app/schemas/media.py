"""
Media Schemas

Pydantic models for media-related API requests and responses.
Handles validation for video, image, and audio uploads and processing.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class MediaType(str, Enum):
    """Media file types."""
    video = "video"
    image = "image"
    audio = "audio"


class VideoFormat(str, Enum):
    """Supported video formats."""
    mp4 = "mp4"
    webm = "webm"
    avi = "avi"
    mov = "mov"


class ImageFormat(str, Enum):
    """Supported image formats."""
    jpg = "jpg"
    jpeg = "jpeg"
    png = "png"
    gif = "gif"
    webp = "webp"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    mp3 = "mp3"
    wav = "wav"
    ogg = "ogg"
    m4a = "m4a"


class ProcessingStatus(str, Enum):
    """Media processing status."""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class MediaQuality(str, Enum):
    """Media quality options."""
    low = "low"
    medium = "medium"
    high = "high"
    original = "original"


# Request Schemas

class MediaUploadRequest(BaseModel):
    """Schema for media upload metadata."""
    media_type: MediaType = Field(..., description="Type of media")
    filename: str = Field(..., min_length=1, max_length=255, description="Original filename")
    description: Optional[str] = Field(None, max_length=500, description="Media description")
    tags: List[str] = Field(default_factory=list, description="Media tags")
    is_public: bool = Field(False, description="Whether media is publicly accessible")
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename format."""
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename')
        return v


class VideoUploadRequest(MediaUploadRequest):
    """Schema for video upload."""
    media_type: MediaType = MediaType.video
    duration: Optional[int] = Field(None, ge=0, description="Video duration in seconds")
    resolution: Optional[str] = Field(None, description="Video resolution (e.g., 1920x1080)")
    fps: Optional[int] = Field(None, ge=1, le=120, description="Frames per second")


class ImageUploadRequest(MediaUploadRequest):
    """Schema for image upload."""
    media_type: MediaType = MediaType.image
    width: Optional[int] = Field(None, ge=1, description="Image width in pixels")
    height: Optional[int] = Field(None, ge=1, description="Image height in pixels")


class AudioUploadRequest(MediaUploadRequest):
    """Schema for audio upload."""
    media_type: MediaType = MediaType.audio
    duration: Optional[int] = Field(None, ge=0, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Sample rate in Hz")


class MediaProcessingRequest(BaseModel):
    """Schema for media processing request."""
    media_id: UUID = Field(..., description="Media ID to process")
    operations: List[str] = Field(..., min_length=1, description="Processing operations")
    quality: MediaQuality = Field(MediaQuality.medium, description="Output quality")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")


class VideoTrimRequest(BaseModel):
    """Schema for video trimming request."""
    video_id: UUID = Field(..., description="Video ID")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., gt=0, description="End time in seconds")
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        """Validate end time is after start time."""
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('End time must be after start time')
        return v


class ImageResizeRequest(BaseModel):
    """Schema for image resizing request."""
    image_id: UUID = Field(..., description="Image ID")
    width: Optional[int] = Field(None, ge=1, le=4096, description="Target width")
    height: Optional[int] = Field(None, ge=1, le=4096, description="Target height")
    maintain_aspect_ratio: bool = Field(True, description="Maintain aspect ratio")


class ThumbnailGenerateRequest(BaseModel):
    """Schema for thumbnail generation."""
    media_id: UUID = Field(..., description="Media ID")
    timestamp: Optional[float] = Field(None, ge=0, description="Timestamp for video thumbnail")
    width: int = Field(320, ge=50, le=1920, description="Thumbnail width")
    height: int = Field(240, ge=50, le=1080, description="Thumbnail height")


# Response Schemas

class MediaResponse(BaseModel):
    """Schema for media information."""
    id: UUID
    media_type: MediaType
    filename: str
    file_url: str
    thumbnail_url: Optional[str]
    file_size: int = Field(..., description="File size in bytes")
    mime_type: str
    description: Optional[str]
    tags: List[str]
    is_public: bool
    uploaded_by: UUID
    uploaded_at: datetime
    processing_status: ProcessingStatus
    
    class Config:
        from_attributes = True


class VideoResponse(MediaResponse):
    """Schema for video information."""
    media_type: MediaType = MediaType.video
    duration: Optional[int] = Field(None, description="Duration in seconds")
    resolution: Optional[str]
    fps: Optional[int]
    codec: Optional[str]
    bitrate: Optional[int]


class ImageResponse(MediaResponse):
    """Schema for image information."""
    media_type: MediaType = MediaType.image
    width: int
    height: int
    format: str


class AudioResponse(MediaResponse):
    """Schema for audio information."""
    media_type: MediaType = MediaType.audio
    duration: Optional[int]
    sample_rate: Optional[int]
    channels: Optional[int]
    bitrate: Optional[int]


class MediaProcessingResponse(BaseModel):
    """Schema for media processing result."""
    processing_id: UUID
    media_id: UUID
    status: ProcessingStatus
    progress: float = Field(..., ge=0.0, le=100.0, description="Processing progress percentage")
    operations_completed: List[str]
    output_url: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class MediaListResponse(BaseModel):
    """Schema for media list."""
    items: List[MediaResponse]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class MediaUploadUrlResponse(BaseModel):
    """Schema for pre-signed upload URL."""
    upload_url: str = Field(..., description="Pre-signed URL for upload")
    media_id: UUID = Field(..., description="Media ID for tracking")
    expires_at: datetime = Field(..., description="URL expiration time")
    max_file_size: int = Field(..., description="Maximum file size in bytes")
    allowed_types: List[str] = Field(..., description="Allowed MIME types")


class MediaAnalysisResponse(BaseModel):
    """Schema for media analysis results."""
    media_id: UUID
    analysis_type: str
    results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    analyzed_at: datetime


class MediaStatistics(BaseModel):
    """Schema for media statistics."""
    total_media: int
    total_videos: int
    total_images: int
    total_audio: int
    total_storage_bytes: int
    most_used_tags: List[Dict[str, Any]]
    recent_uploads: List[MediaResponse]
