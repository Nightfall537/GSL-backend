"""
GSL Database Models

SQLAlchemy models for GSL signs and categories.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from datetime import datetime
import uuid

from app.core.database import Base


class SignCategory(Base):
    """Sign category model."""
    
    __tablename__ = "sign_categories"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    parent_category_id = Column(UUID(as_uuid=True), ForeignKey("sign_categories.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GSLSign(Base):
    """GSL sign model."""
    
    __tablename__ = "gsl_signs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sign_name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=False)
    
    # Category and difficulty
    category_id = Column(UUID(as_uuid=True), ForeignKey("sign_categories.id"), nullable=False)
    difficulty_level = Column(Integer, default=1)  # 1=beginner, 2=intermediate, 3=advanced
    
    # Media
    video_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    
    # Additional data
    usage_examples = Column(ARRAY(String), default=list)
    related_signs = Column(ARRAY(UUID(as_uuid=True)), default=list)
    extra_data = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<GSLSign(id={self.id}, name={self.sign_name})>"
