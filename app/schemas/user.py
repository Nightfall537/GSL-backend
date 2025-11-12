"""
User Schemas

Pydantic models for user-related API requests and responses.
Handles validation for user registration, authentication, and profile management.
"""

from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class AgeGroup(str, Enum):
    """Age group options for learners."""
    child = "child"
    teen = "teen"
    adult = "adult"
    senior = "senior"


class LearningLevel(str, Enum):
    """Learning level options."""
    beginner = "beginner"
    intermediate = "intermediate"
    advanced = "advanced"


class UserRole(str, Enum):
    """User role options."""
    learner = "learner"
    teacher = "teacher"
    admin = "admin"


# Request Schemas

class UserCreate(BaseModel):
    """Schema for user registration."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    age_group: Optional[AgeGroup] = Field(AgeGroup.adult, description="Age group")
    learning_level: Optional[LearningLevel] = Field(LearningLevel.beginner, description="Learning level")
    preferred_language: Optional[str] = Field("english", description="Preferred language")
    accessibility_needs: Optional[List[str]] = Field(default_factory=list, description="Accessibility requirements")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v.lower()


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    full_name: Optional[str] = Field(None, max_length=100)
    age_group: Optional[AgeGroup] = None
    learning_level: Optional[LearningLevel] = None
    preferred_language: Optional[str] = None
    accessibility_needs: Optional[List[str]] = None


class PasswordChange(BaseModel):
    """Schema for changing password."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


# Response Schemas

class UserResponse(BaseModel):
    """Schema for user data in responses (excludes sensitive data)."""
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    age_group: AgeGroup
    learning_level: LearningLevel
    preferred_language: str
    accessibility_needs: List[str]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool = True
    role: UserRole = UserRole.learner
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserProfile(BaseModel):
    """Schema for detailed user profile."""
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    age_group: AgeGroup
    learning_level: LearningLevel
    preferred_language: str
    accessibility_needs: List[str]
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class UserLoginResponse(BaseModel):
    """Schema for login response."""
    user: UserResponse
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours


class UserRegistrationResponse(BaseModel):
    """Schema for registration response."""
    user: UserResponse
    access_token: str
    message: str = "Registration successful"
    token_type: str = "bearer"


class UserStatistics(BaseModel):
    """Schema for user statistics."""
    total_lessons_completed: int
    current_level: int
    achievements_count: int
    practice_sessions: int
    days_active: int
    signs_learned: int
    accuracy_rate: float
    last_activity: Optional[datetime]
    
    class Config:
        from_attributes = True


class TokenRefresh(BaseModel):
    """Schema for token refresh request."""
    refresh_token: str = Field(..., description="Refresh token")


class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
