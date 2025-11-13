"""
Media Database Models

SQLAlchemy models for media files, including videos, images, and audio
for the GSL learning platform.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


class MediaType(str, enum.Enum):
    """Media file types."""
    video = "video"
    image = "image"
    audio = "audio"


class ProcessingStatus(str, enum.Enum):
    """Media processing status."""
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class MediaFile(Base):
    """Media file model for storing uploaded videos, images, and audio."""
    
    __tablename__ = "media_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_url = Column(String(500), nullable=False)
    thumbnail_url = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    media_type = Column(SQLEnum(MediaType), nullable=False, default=MediaType.video)
    
    # Media metadata
    duration = Column(Integer, nullable=True)  # Duration in seconds for video/audio
    width = Column(Integer, nullable=True)  # Width for images/videos
    height = Column(Integer, nullable=True)  # Height for images/videos
    fps = Column(Integer, nullable=True)  # FPS for videos
    
    # Processing
    processing_status = Column(
        SQLEnum(ProcessingStatus),
        nullable=False,
        default=ProcessingStatus.completed
    )
    
    # Ownership and metadata
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="media_files")
    
    def __repr__(self):
        return f"<MediaFile(id={self.id}, filename={self.filename}, type={self.media_type})>"
