"""
Unit Tests for Data Models and Authentication

Tests user model validation, password hashing, JWT token generation/validation,
and database operations with relationships.
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from sqlalchemy.exc import IntegrityError

from app.db_models.user import User, LearningProgress, UserRole, AgeGroup, LearningLevel
from app.core.security import SecurityManager
from app.repositories.user_repository import UserRepository, LearningProgressRepository
from app.schemas.user import UserCreate


class TestUserModel:
    """Test cases for User model validation."""
    
    def test_user_model_creation(self, test_db):
        """Test creating a user model with valid data."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User"
        )
        
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.role == UserRole.learner
        assert user.learning_level == LearningLevel.beginner
        assert user.created_at is not None
    
    def test_user_unique_username_constraint(self, test_db):
        """Test that username must be unique."""
        user1 = User(
            username="testuser",
            email="test1@example.com",
            hashed_password="hash1"
        )
        test_db.add(user1)
        test_db.commit()
        
        user2 = User(
            username="testuser",
            email="test2@example.com",
            hashed_password="hash2"
        )
        test_db.add(user2)
        
        with pytest.raises(IntegrityError):
            test_db.commit()
    
    def test_user_unique_email_constraint(self, test_db):
        """Test that email must be unique."""
        user1 = User(
            username="user1",
            email="test@example.com",
            hashed_password="hash1"
        )
        test_db.add(user1)
        test_db.commit()
        
        user2 = User(
            username="user2",
            email="test@example.com",
            hashed_password="hash2"
        )
        test_db.add(user2)
        
        with pytest.raises(IntegrityError):
            test_db.commit()
    
    def test_user_default_values(self, test_db):
        """Test user model default values."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        assert user.age_group == AgeGroup.adult
        assert user.learning_level == LearningLevel.beginner
        assert user.preferred_language == "english"
        assert user.accessibility_needs == []
        assert user.is_active is True
        assert user.is_verified is False
        assert user.role == UserRole.learner
    
    def test_user_repr(self, test_db):
        """Test user model string representation."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        repr_str = repr(user)
        assert "testuser" in repr_str
        assert "test@example.com" in repr_str


class TestLearningProgressModel:
    """Test cases for LearningProgress model."""
    
    def test_learning_progress_creation(self, test_db):
        """Test creating learning progress with valid data."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        progress = LearningProgress(
            user_id=user.id,
            total_lessons_completed=5,
            current_level=2
        )
        
        test_db.add(progress)
        test_db.commit()
        test_db.refresh(progress)
        
        assert progress.id is not None
        assert progress.user_id == user.id
        assert progress.total_lessons_completed == 5
        assert progress.current_level == 2
        assert progress.experience_points == 0
        assert progress.signs_learned == 0
    
    def test_learning_progress_default_values(self, test_db):
        """Test learning progress default values."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        progress = LearningProgress(user_id=user.id)
        test_db.add(progress)
        test_db.commit()
        test_db.refresh(progress)
        
        assert progress.total_lessons_completed == 0
        assert progress.current_level == 1
        assert progress.experience_points == 0
        assert progress.signs_learned == 0
        assert progress.accuracy_rate == 0.0
        assert progress.practice_time_minutes == 0
        assert progress.current_streak == 0
        assert progress.longest_streak == 0
        assert progress.achievements == []
        assert progress.completed_lessons == []
    
    def test_learning_progress_user_relationship(self, test_db):
        """Test relationship between user and learning progress."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        test_db.refresh(user)
        
        progress = LearningProgress(user_id=user.id)
        test_db.add(progress)
        test_db.commit()
        test_db.refresh(progress)
        
        # Test relationship access
        assert progress.user.username == "testuser"
        assert user.learning_progress.id == progress.id
    
    def test_learning_progress_unique_user_constraint(self, test_db):
        """Test that each user can have only one learning progress record."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        progress1 = LearningProgress(user_id=user.id)
        test_db.add(progress1)
        test_db.commit()
        
        progress2 = LearningProgress(user_id=user.id)
        test_db.add(progress2)
        
        with pytest.raises(IntegrityError):
            test_db.commit()


class TestPasswordHashing:
    """Test cases for password hashing functionality."""
    
    def test_hash_password(self):
        """Test password hashing produces different hash from plain text."""
        password = "SecurePass123!"
        hashed = SecurityManager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert hashed.startswith("$2b$")  # bcrypt hash prefix
    
    def test_hash_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "SecurePass123!"
        hash1 = SecurityManager.hash_password(password)
        hash2 = SecurityManager.hash_password(password)
        
        assert hash1 != hash2
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        password = "SecurePass123!"
        hashed = SecurityManager.hash_password(password)
        
        assert SecurityManager.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        password = "SecurePass123!"
        hashed = SecurityManager.hash_password(password)
        
        assert SecurityManager.verify_password("WrongPassword", hashed) is False
    
    def test_verify_password_case_sensitive(self):
        """Test that password verification is case sensitive."""
        password = "SecurePass123!"
        hashed = SecurityManager.hash_password(password)
        
        assert SecurityManager.verify_password("securepass123!", hashed) is False
    
    def test_hash_empty_password(self):
        """Test hashing empty password."""
        password = ""
        hashed = SecurityManager.hash_password(password)
        
        assert len(hashed) > 0
        assert SecurityManager.verify_password("", hashed) is True


class TestJWTTokens:
    """Test cases for JWT token generation and validation."""
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        user_id = uuid4()
        username = "testuser"
        
        data = {"sub": str(user_id), "username": username}
        token = SecurityManager.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert token.count('.') == 2  # JWT has 3 parts separated by dots
    
    def test_decode_valid_token(self):
        """Test decoding valid JWT token."""
        user_id = uuid4()
        username = "testuser"
        
        data = {"sub": str(user_id), "username": username}
        token = SecurityManager.create_access_token(data)
        payload = SecurityManager.decode_token(token)
        
        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["username"] == username
        assert "exp" in payload
        assert "iat" in payload
    
    def test_decode_invalid_token(self):
        """Test decoding invalid JWT token."""
        payload = SecurityManager.decode_token("invalid.token.here")
        
        assert payload is None
    
    def test_decode_expired_token(self):
        """Test decoding expired JWT token."""
        user_id = uuid4()
        data = {"sub": str(user_id), "username": "testuser"}
        
        # Create token with negative expiration (already expired)
        expired_delta = timedelta(seconds=-1)
        token = SecurityManager.create_access_token(data, expired_delta)
        
        payload = SecurityManager.decode_token(token)
        assert payload is None
    
    def test_token_expiration_time(self):
        """Test that token contains correct expiration time."""
        user_id = uuid4()
        data = {"sub": str(user_id), "username": "testuser"}
        
        expires_delta = timedelta(minutes=30)
        token = SecurityManager.create_access_token(data, expires_delta)
        payload = SecurityManager.decode_token(token)
        
        assert payload is not None
        exp_time = datetime.fromtimestamp(payload["exp"])
        iat_time = datetime.fromtimestamp(payload["iat"])
        
        # Check expiration is approximately 30 minutes from issued time
        time_diff = (exp_time - iat_time).total_seconds()
        assert 1790 <= time_diff <= 1810  # Allow 10 second tolerance
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        user_id = uuid4()
        token = SecurityManager.create_refresh_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        payload = SecurityManager.decode_token(token)
        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["type"] == "refresh"


class TestUserRepository:
    """Test cases for UserRepository database operations."""
    
    def test_create_user(self, test_db):
        """Test creating user through repository."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!",
            "full_name": "Test User"
        }
        
        user = repo.create_user(user_data)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password != "SecurePass123!"
        assert user.learning_progress is not None
    
    def test_get_by_email(self, test_db):
        """Test retrieving user by email."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        created_user = repo.create_user(user_data)
        
        found_user = repo.get_by_email("test@example.com")
        
        assert found_user is not None
        assert found_user.id == created_user.id
        assert found_user.email == "test@example.com"
    
    def test_get_by_username(self, test_db):
        """Test retrieving user by username."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        created_user = repo.create_user(user_data)
        
        found_user = repo.get_by_username("testuser")
        
        assert found_user is not None
        assert found_user.id == created_user.id
        assert found_user.username == "testuser"
    
    def test_authenticate_user_success(self, test_db):
        """Test successful user authentication."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        repo.create_user(user_data)
        
        authenticated_user = repo.authenticate_user("test@example.com", "SecurePass123!")
        
        assert authenticated_user is not None
        assert authenticated_user.email == "test@example.com"
    
    def test_authenticate_user_wrong_password(self, test_db):
        """Test authentication with wrong password."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        repo.create_user(user_data)
        
        authenticated_user = repo.authenticate_user("test@example.com", "WrongPassword")
        
        assert authenticated_user is None
    
    def test_authenticate_user_nonexistent(self, test_db):
        """Test authentication with nonexistent user."""
        repo = UserRepository(test_db)
        
        authenticated_user = repo.authenticate_user("nonexistent@example.com", "password")
        
        assert authenticated_user is None
    
    def test_update_last_login(self, test_db):
        """Test updating user's last login timestamp."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123!"
        }
        user = repo.create_user(user_data)
        
        assert user.last_login is None
        
        repo.update_last_login(user.id)
        test_db.refresh(user)
        
        assert user.last_login is not None
        assert isinstance(user.last_login, datetime)
    
    def test_change_password(self, test_db):
        """Test changing user password."""
        repo = UserRepository(test_db)
        
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "OldPassword123!"
        }
        user = repo.create_user(user_data)
        old_hash = user.hashed_password
        
        result = repo.change_password(user.id, "NewPassword123!")
        
        assert result is True
        test_db.refresh(user)
        assert user.hashed_password != old_hash
        
        # Verify new password works
        authenticated = repo.authenticate_user("test@example.com", "NewPassword123!")
        assert authenticated is not None


class TestLearningProgressRepository:
    """Test cases for LearningProgressRepository database operations."""
    
    def test_create_for_user(self, test_db):
        """Test creating learning progress for user."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        progress = repo.create_for_user(user.id)
        
        assert progress.id is not None
        assert progress.user_id == user.id
        assert progress.total_lessons_completed == 0
        assert progress.current_level == 1
    
    def test_get_by_user_id(self, test_db):
        """Test retrieving learning progress by user ID."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        created_progress = repo.create_for_user(user.id)
        
        found_progress = repo.get_by_user_id(user.id)
        
        assert found_progress is not None
        assert found_progress.id == created_progress.id
        assert found_progress.user_id == user.id
    
    def test_update_lesson_completion(self, test_db):
        """Test updating progress when lesson is completed."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        progress = repo.create_for_user(user.id)
        
        lesson_id = uuid4()
        updated_progress = repo.update_lesson_completion(user.id, lesson_id)
        
        assert updated_progress is not None
        assert lesson_id in updated_progress.completed_lessons
        assert updated_progress.total_lessons_completed == 1
        assert updated_progress.experience_points == 100
    
    def test_update_practice_session(self, test_db):
        """Test updating progress after practice session."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        progress = repo.create_for_user(user.id)
        
        updated_progress = repo.update_practice_session(
            user.id,
            duration_minutes=30,
            accuracy=0.85,
            signs_practiced=10
        )
        
        assert updated_progress is not None
        assert updated_progress.practice_time_minutes == 30
        assert updated_progress.signs_learned == 10
        assert updated_progress.accuracy_rate == 0.85
    
    def test_add_achievement(self, test_db):
        """Test adding achievement to user progress."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        progress = repo.create_for_user(user.id)
        
        achievement_id = uuid4()
        updated_progress = repo.add_achievement(user.id, achievement_id)
        
        assert updated_progress is not None
        assert achievement_id in updated_progress.achievements
    
    def test_update_streak(self, test_db):
        """Test updating user's learning streak."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hash"
        )
        test_db.add(user)
        test_db.commit()
        
        repo = LearningProgressRepository(test_db)
        progress = repo.create_for_user(user.id)
        
        # Increment streak
        updated_progress = repo.update_streak(user.id, increment=True)
        assert updated_progress.current_streak == 1
        assert updated_progress.longest_streak == 1
        
        # Increment again
        updated_progress = repo.update_streak(user.id, increment=True)
        assert updated_progress.current_streak == 2
        assert updated_progress.longest_streak == 2
        
        # Reset streak
        updated_progress = repo.update_streak(user.id, increment=False)
        assert updated_progress.current_streak == 0
        assert updated_progress.longest_streak == 2  # Longest should remain
