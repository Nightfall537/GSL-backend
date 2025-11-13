"""
User Repository

Repository classes for user-related database operations.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.core.repository import BaseRepository
from app.db_models.user import User, LearningProgress
from app.core.security import SecurityManager


class UserRepository(BaseRepository[User]):
    """Repository for User model operations."""
    
    def __init__(self, db: Session):
        super().__init__(User, db)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: User's email address
            
        Returns:
            User instance or None
        """
        return self.db.query(User).filter(User.email == email.lower()).first()
    
    def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: User's username
            
        Returns:
            User instance or None
        """
        return self.db.query(User).filter(User.username == username.lower()).first()
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """
        Create a new user with hashed password.
        
        Args:
            user_data: User data including plain password
            
        Returns:
            Created user instance
            
        Raises:
            HTTPException: If user creation fails
        """
        # Check if email already exists
        if self.get_by_email(user_data["email"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username already exists
        if self.get_by_username(user_data["username"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Hash password
        user_data_copy = user_data.copy()
        user_data_copy["hashed_password"] = SecurityManager.hash_password(
            user_data_copy.pop("password")
        )
        
        # Normalize email and username
        user_data_copy["email"] = user_data_copy["email"].lower()
        user_data_copy["username"] = user_data_copy["username"].lower()
        
        try:
            # Create user
            user = self.create(user_data_copy)
            
            # Create associated learning progress
            progress_repo = LearningProgressRepository(self.db)
            progress_repo.create_for_user(user.id)
            
            return user
        except IntegrityError as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user account"
            )
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate user with email and password.
        
        Args:
            email: User's email address
            password: Plain text password
            
        Returns:
            User instance if authentication successful, None otherwise
        """
        user = self.get_by_email(email)
        if not user:
            return None
        
        if not SecurityManager.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    def update_last_login(self, user_id: UUID) -> None:
        """
        Update user's last login timestamp.
        
        Args:
            user_id: User's ID
        """
        from datetime import datetime
        self.update(user_id, {"last_login": datetime.utcnow()})
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Get active users.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of active users
        """
        return self.get_multi(skip=skip, limit=limit, filters={"is_active": True})
    
    def deactivate_user(self, user_id: UUID) -> bool:
        """
        Deactivate a user account.
        
        Args:
            user_id: User's ID
            
        Returns:
            True if successful
        """
        result = self.update(user_id, {"is_active": False})
        return result is not None
    
    def change_password(self, user_id: UUID, new_password: str) -> bool:
        """
        Change user's password.
        
        Args:
            user_id: User's ID
            new_password: New plain text password
            
        Returns:
            True if successful
        """
        hashed_password = SecurityManager.hash_password(new_password)
        result = self.update(user_id, {"hashed_password": hashed_password})
        return result is not None


class LearningProgressRepository(BaseRepository[LearningProgress]):
    """Repository for LearningProgress model operations."""
    
    def __init__(self, db: Session):
        super().__init__(LearningProgress, db)
    
    def get_by_user_id(self, user_id: UUID) -> Optional[LearningProgress]:
        """
        Get learning progress by user ID.
        
        Args:
            user_id: User's ID
            
        Returns:
            LearningProgress instance or None
        """
        return self.db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id
        ).first()
    
    def create_for_user(self, user_id: UUID) -> LearningProgress:
        """
        Create initial learning progress for a user.
        
        Args:
            user_id: User's ID
            
        Returns:
            Created LearningProgress instance
        """
        progress_data = {
            "user_id": user_id,
            "total_lessons_completed": 0,
            "current_level": 1,
            "experience_points": 0,
            "signs_learned": 0,
            "accuracy_rate": 0.0,
            "practice_time_minutes": 0,
            "current_streak": 0,
            "longest_streak": 0,
            "days_active": 0,
            "achievements": [],
            "completed_lessons": [],
            "extra_data": {}
        }
        
        return self.create(progress_data)
    
    def update_lesson_completion(
        self,
        user_id: UUID,
        lesson_id: UUID,
        score: float = None
    ) -> Optional[LearningProgress]:
        """
        Update progress when a lesson is completed.
        
        Args:
            user_id: User's ID
            lesson_id: Completed lesson ID
            score: Optional lesson score
            
        Returns:
            Updated LearningProgress instance
        """
        progress = self.get_by_user_id(user_id)
        if not progress:
            return None
        
        # Add lesson to completed list if not already there
        completed_lessons = progress.completed_lessons or []
        if lesson_id not in completed_lessons:
            completed_lessons.append(lesson_id)
            
            # Update progress metrics
            updates = {
                "completed_lessons": completed_lessons,
                "total_lessons_completed": len(completed_lessons),
                "experience_points": progress.experience_points + 100,  # Base XP per lesson
                "last_activity": progress.updated_at
            }
            
            # Update level based on lessons completed
            new_level = min(10, (len(completed_lessons) // 5) + 1)  # Level up every 5 lessons
            if new_level > progress.current_level:
                updates["current_level"] = new_level
            
            return self.update(progress.id, updates)
        
        return progress
    
    def update_practice_session(
        self,
        user_id: UUID,
        duration_minutes: int,
        accuracy: float = None,
        signs_practiced: int = 0
    ) -> Optional[LearningProgress]:
        """
        Update progress after a practice session.
        
        Args:
            user_id: User's ID
            duration_minutes: Practice session duration
            accuracy: Session accuracy score
            signs_practiced: Number of signs practiced
            
        Returns:
            Updated LearningProgress instance
        """
        progress = self.get_by_user_id(user_id)
        if not progress:
            return None
        
        updates = {
            "practice_time_minutes": progress.practice_time_minutes + duration_minutes,
            "signs_learned": progress.signs_learned + signs_practiced,
            "last_activity": progress.updated_at
        }
        
        # Update accuracy rate (weighted average)
        if accuracy is not None:
            current_accuracy = progress.accuracy_rate or 0.0
            total_sessions = progress.practice_time_minutes // 15  # Estimate sessions
            if total_sessions > 0:
                updates["accuracy_rate"] = (
                    (current_accuracy * total_sessions + accuracy) / (total_sessions + 1)
                )
            else:
                updates["accuracy_rate"] = accuracy
        
        return self.update(progress.id, updates)
    
    def add_achievement(self, user_id: UUID, achievement_id: UUID) -> Optional[LearningProgress]:
        """
        Add an achievement to user's progress.
        
        Args:
            user_id: User's ID
            achievement_id: Achievement ID to add
            
        Returns:
            Updated LearningProgress instance
        """
        progress = self.get_by_user_id(user_id)
        if not progress:
            return None
        
        achievements = progress.achievements or []
        if achievement_id not in achievements:
            achievements.append(achievement_id)
            return self.update(progress.id, {"achievements": achievements})
        
        return progress
    
    def update_streak(self, user_id: UUID, increment: bool = True) -> Optional[LearningProgress]:
        """
        Update user's learning streak.
        
        Args:
            user_id: User's ID
            increment: Whether to increment or reset streak
            
        Returns:
            Updated LearningProgress instance
        """
        progress = self.get_by_user_id(user_id)
        if not progress:
            return None
        
        if increment:
            new_streak = progress.current_streak + 1
            updates = {
                "current_streak": new_streak,
                "longest_streak": max(progress.longest_streak, new_streak)
            }
        else:
            updates = {"current_streak": 0}
        
        return self.update(progress.id, updates)