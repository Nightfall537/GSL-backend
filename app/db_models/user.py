"""
User Database Model

SQLAlchemy model for users table.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
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
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, email={self.email})>"
