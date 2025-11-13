"""
Recognition Database Models

SQLAlchemy models for sign recognition results and tracking.
"""

from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from app.core.database import Base


class SignRecognition(Base):
    """Sign recognition result model."""
    
    __tablename__ = "sign_recognitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    recognized_sign_id = Column(UUID(as_uuid=True), ForeignKey("gsl_signs.id"), nullable=True)
    
    # Recognition results
    confidence_score = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)  # in seconds
    status = Column(String(50), nullable=False)  # success, low_confidence, failed
    
    # Optional metadata
    media_type = Column(String(20), nullable=True)  # video, image
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SignRecognition(id={self.id}, status={self.status}, confidence={self.confidence_score})>"
