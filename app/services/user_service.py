"""
User Service

Handles user management, authentication, profile management, and learning progress tracking
for the GSL learning platform.
"""

from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config.settings import get_settings
from app.db_models.user import User
# Note: LearnerProfile and LearningProgress will be handled via Supabase
from app.schemas.user import (
    UserCreate, UserLogin, UserResponse, UserUpdate,
    ProfileResponse, ProgressResponse
)

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserService:
    """Service for user management and authentication operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: UUID, username: str) -> str:
        """
        Create JWT access token for authenticated user.
        
        Args:
            user_id: User's unique identifier
            username: User's username
            
        Returns:
            JWT token string
        """
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
        to_encode = {
            "sub": str(user_id),
            "username": username,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        encoded_jwt = jwt.encode(
            to_encode,
            settings.secret_key,
            algorithm=settings.jwt_algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=[settings.jwt_algorithm]
            )
            return payload
        except JWTError:
            return None
    
    def register_user(self, user_data: UserCreate) -> User:
        """
        Register a new learner account.
        
        Args:
            user_data: User registration data
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If username or email already exists
        """
        # Check if user already exists
        existing_user = self.db.query(User).filter(
            (User.username == user_data.username) | (User.email == user_data.email)
        ).first()
        
        if existing_user:
            if existing_user.username == user_data.username:
                raise ValueError("Username already exists")
            if existing_user.email == user_data.email:
                raise ValueError("Email already exists")
        
        # Create new user with hashed password
        hashed_password = self.hash_password(user_data.password)
        
        new_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            full_name=user_data.full_name
        )
        
        self.db.add(new_user)
        self.db.commit()
        self.db.refresh(new_user)
        
        # Create default profile and progress
        self._create_default_profile(new_user.id)
        self._create_default_progress(new_user.id)
        
        return new_user
    
    def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """
        Authenticate user with username/email and password.
        
        Args:
            login_data: Login credentials
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.db.query(User).filter(
            (User.username == login_data.username) | (User.email == login_data.username)
        ).first()
        
        if not user:
            return None
        
        if not self.verify_password(login_data.password, user.password_hash):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        return user
    
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(User.username == username).first()
    
    def update_user_profile(self, user_id: UUID, profile_data: UserUpdate) -> User:
        """
        Update user profile information.
        
        Args:
            user_id: User's unique identifier
            profile_data: Updated profile data
            
        Returns:
            Updated user object
            
        Raises:
            ValueError: If user not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Update user fields
        if profile_data.full_name is not None:
            user.full_name = profile_data.full_name
        if profile_data.email is not None:
            user.email = profile_data.email
        
        # Update profile if exists
        if user.profile:
            if profile_data.age_group is not None:
                user.profile.age_group = profile_data.age_group
            if profile_data.learning_level is not None:
                user.profile.learning_level = profile_data.learning_level
            if profile_data.preferred_language is not None:
                user.profile.preferred_language = profile_data.preferred_language
            if profile_data.accessibility_needs is not None:
                user.profile.accessibility_needs = profile_data.accessibility_needs
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def get_user_progress(self, user_id: UUID) -> Optional[LearningProgress]:
        """Get user's learning progress."""
        return self.db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id
        ).first()
    
    def update_learning_progress(
        self,
        user_id: UUID,
        lessons_completed: Optional[int] = None,
        current_level: Optional[int] = None
    ) -> LearningProgress:
        """
        Update user's learning progress.
        
        Args:
            user_id: User's unique identifier
            lessons_completed: Number of lessons completed
            current_level: Current learning level
            
        Returns:
            Updated learning progress object
        """
        progress = self.get_user_progress(user_id)
        if not progress:
            progress = self._create_default_progress(user_id)
        
        if lessons_completed is not None:
            progress.total_lessons_completed = lessons_completed
        if current_level is not None:
            progress.current_level = current_level
        
        progress.last_activity = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(progress)
        
        return progress
    
    def _create_default_profile(self, user_id: UUID) -> LearnerProfile:
        """Create default learner profile for new user."""
        profile = LearnerProfile(
            user_id=user_id,
            age_group="adult",
            learning_level="beginner",
            preferred_language="english",
            accessibility_needs=[]
        )
        self.db.add(profile)
        self.db.commit()
        return profile
    
    def _create_default_progress(self, user_id: UUID) -> LearningProgress:
        """Create default learning progress for new user."""
        progress = LearningProgress(
            user_id=user_id,
            total_lessons_completed=0,
            current_level=1,
            last_activity=datetime.utcnow()
        )
        self.db.add(progress)
        self.db.commit()
        return progress
    
    def add_achievement(self, user_id: UUID, achievement_id: UUID) -> None:
        """Add achievement to user's progress."""
        progress = self.get_user_progress(user_id)
        if progress:
            # TODO: Implement achievement tracking
            pass
    
    def get_user_statistics(self, user_id: UUID) -> dict:
        """
        Get comprehensive user statistics.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            Dictionary with user statistics
        """
        progress = self.get_user_progress(user_id)
        if not progress:
            return {
                "total_lessons_completed": 0,
                "current_level": 1,
                "achievements_count": 0,
                "practice_sessions": 0,
                "days_active": 0
            }
        
        return {
            "total_lessons_completed": progress.total_lessons_completed,
            "current_level": progress.current_level,
            "achievements_count": len(progress.achievements) if progress.achievements else 0,
            "practice_sessions": len(progress.practice_sessions) if progress.practice_sessions else 0,
            "last_activity": progress.last_activity,
            "days_active": self._calculate_days_active(user_id)
        }
    
    def _calculate_days_active(self, user_id: UUID) -> int:
        """Calculate number of days user has been active."""
        # TODO: Implement based on practice session tracking
        return 0