"""
Unit Tests for User Service

Tests user management, authentication, and progress tracking functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID

from app.services.user_service import UserService


class TestUserService:
    """Test cases for UserService."""
    
    @pytest.fixture
    def user_service(self, test_db):
        """Create UserService instance."""
        return UserService(test_db)
    
    def test_hash_password(self, user_service):
        """Test password hashing."""
        password = "SecurePass123!"
        hashed = user_service.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert user_service.verify_password(password, hashed)
    
    def test_verify_password_correct(self, user_service):
        """Test password verification with correct password."""
        password = "SecurePass123!"
        hashed = user_service.hash_password(password)
        
        assert user_service.verify_password(password, hashed) is True
    
    def test_verify_password_incorrect(self, user_service):
        """Test password verification with incorrect password."""
        password = "SecurePass123!"
        hashed = user_service.hash_password(password)
        
        assert user_service.verify_password("WrongPassword", hashed) is False
    
    def test_create_access_token(self, user_service):
        """Test JWT token creation."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        username = "testuser"
        
        token = user_service.create_access_token(user_id, username)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_verify_token_valid(self, user_service):
        """Test token verification with valid token."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        username = "testuser"
        
        token = user_service.create_access_token(user_id, username)
        payload = user_service.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == str(user_id)
        assert payload["username"] == username
    
    def test_verify_token_invalid(self, user_service):
        """Test token verification with invalid token."""
        payload = user_service.verify_token("invalid-token")
        
        assert payload is None
    
    @patch('app.services.user_service.User')
    def test_register_user_success(self, mock_user, user_service, sample_user_data):
        """Test successful user registration."""
        from app.schemas.user import UserCreate
        
        user_data = UserCreate(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            password=sample_user_data["password"],
            full_name=sample_user_data["full_name"]
        )
        
        # Mock database query to return None (user doesn't exist)
        user_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=None)))
        ))
        
        with patch.object(user_service, '_create_default_profile'):
            with patch.object(user_service, '_create_default_progress'):
                user = user_service.register_user(user_data)
                
                assert user is not None
    
    def test_register_user_duplicate_username(self, user_service, sample_user_data):
        """Test user registration with duplicate username."""
        from app.schemas.user import UserCreate
        
        user_data = UserCreate(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
            password=sample_user_data["password"]
        )
        
        # Mock existing user
        existing_user = Mock(username=sample_user_data["username"])
        user_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=existing_user)))
        ))
        
        with pytest.raises(ValueError, match="Username already exists"):
            user_service.register_user(user_data)
    
    def test_get_user_statistics(self, user_service):
        """Test getting user statistics."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        
        # Mock progress
        mock_progress = Mock(
            total_lessons_completed=5,
            current_level=2,
            achievements=[],
            practice_sessions=[]
        )
        
        with patch.object(user_service, 'get_user_progress', return_value=mock_progress):
            stats = user_service.get_user_statistics(user_id)
            
            assert stats["total_lessons_completed"] == 5
            assert stats["current_level"] == 2
            assert "achievements_count" in stats
    
    def test_get_user_statistics_no_progress(self, user_service):
        """Test getting statistics for user with no progress."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        
        with patch.object(user_service, 'get_user_progress', return_value=None):
            stats = user_service.get_user_statistics(user_id)
            
            assert stats["total_lessons_completed"] == 0
            assert stats["current_level"] == 1