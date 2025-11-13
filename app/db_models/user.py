"""
User Database Model

SQLAlchemy model for users table.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Integer, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.core.database import Base


class UserRole(str, enum.Enum):
    """User role enumeration."""
    learner = "learner"
    teacher = "teacher"
    admin = "admin"


class AgeGroup(str, enum.Enum):
    """Age group enumeration."""
    child = "child"
    teen = "teen"
    adult = "adult"
    senior = "senior"


class LearningLevel(str, enum.Enum):
    """Learning level enumeration."""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class User(Base):
    """User model for authentication and profile management."""
    
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    
    # Profile fields
    age_group = Column(SQLEnum(AgeGroup), default=AgeGroup.adult)
    learning_level = Column(SQLEnum(LearningLevel), default=LearningLevel.beginner)
    preferred_language = Column(String(50), default="english")
    accessibility_needs = Column(ARRAY(String), default=list)
    
    # Status fields
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(SQLEnum(UserRole), default=UserRole.learner)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    learning_progress = relationship("LearningProgress", back_populates="user", uselist=False)
    practice_sessions = relationship("PracticeSession", back_populates="user")
    media_files = relationship("MediaFile", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"


class LearningProgress(Base):
    """Learning progress model for tracking user advancement."""
    
    __tablename__ = "learning_progress"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True)
    
    # Progress metrics
    total_lessons_completed = Column(Integer, default=0)
    current_level = Column(Integer, default=1)
    experience_points = Column(Integer, default=0)
    signs_learned = Column(Integer, default=0)
    
    # Performance metrics
    accuracy_rate = Column(Float, default=0.0)
    practice_time_minutes = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    
    # Activity tracking
    last_activity = Column(DateTime, nullable=True)
    days_active = Column(Integer, default=0)
    
    # Additional data
    achievements = Column(ARRAY(UUID(as_uuid=True)), default=list)
    completed_lessons = Column(ARRAY(UUID(as_uuid=True)), default=list)
    extra_data = Column(JSONB, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="learning_progress")
    
    def __repr__(self):
        return f"<LearningProgress(user_id={self.user_id}, level={self.current_level})>"
