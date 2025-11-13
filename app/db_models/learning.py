"""
Learning Database Models

SQLAlchemy models for lessons, achievements, and practice sessions.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.core.database import Base


class Lesson(Base):
    """Lesson model."""
    
    __tablename__ = "lessons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Classification
    level = Column(Integer, default=1)  # 1=beginner, 2=intermediate, 3=advanced
    category = Column(String(100), nullable=False)
    sequence_order = Column(Integer, default=0)
    
    # Content
    signs_covered = Column(ARRAY(UUID(as_uuid=True)), default=list)
    learning_objectives = Column(ARRAY(String), default=list)
    estimated_duration = Column(Integer, default=15)  # minutes
    
    # Prerequisites
    prerequisites = Column(ARRAY(UUID(as_uuid=True)), default=list)
    
    # Extra data
    extra_data = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Achievement(Base):
    """Achievement model."""
    
    __tablename__ = "achievements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # Classification
    type = Column(String(50), nullable=False)  # lesson_completion, streak, accuracy, etc.
    points = Column(Integer, default=10)
    
    # Criteria
    criteria = Column(JSONB, default=dict)
    icon_url = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class PracticeSession(Base):
    """Practice session model."""
    
    __tablename__ = "practice_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session details
    session_type = Column(String(50), nullable=False)  # free_practice, lesson_practice, etc.
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("lessons.id"), nullable=True)
    
    # Performance
    signs_practiced = Column(ARRAY(UUID(as_uuid=True)), default=list)
    duration_seconds = Column(Integer, nullable=False)
    accuracy_score = Column(Float, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    extra_data = Column(JSONB, default=dict)
    
    # Timestamps
    completed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="practice_sessions")
    lesson = relationship("Lesson")
