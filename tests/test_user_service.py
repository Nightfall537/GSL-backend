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


class TestAchievementService:
    """Test cases for AchievementService."""
    
    @pytest.fixture
    def achievement_service(self, test_db):
        """Create AchievementService instance."""
        from app.services.achievement_service import AchievementService
        return AchievementService(test_db)
    
    def test_check_lesson_completion_achievement(self, achievement_service):
        """Test checking lesson completion achievement."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        # Mock progress with 5 lessons completed
        mock_progress = Mock(
            total_lessons_completed=5,
            current_level=1,
            achievements=[],
            accuracy_rate=0.8,
            practice_time_minutes=100,
            signs_learned=20,
            current_streak=3
        )
        
        # Mock achievement
        mock_achievement = Mock(
            id=uuid4(),
            type="lesson_completion",
            criteria={"lessons_required": 5}
        )
        
        with patch.object(achievement_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(achievement_service.achievement_repo, 'get_all', return_value=[mock_achievement]):
                with patch.object(achievement_service.progress_repo, 'add_achievement'):
                    newly_awarded = achievement_service.check_and_award_achievements(user_id)
                    
                    assert len(newly_awarded) == 1
                    assert newly_awarded[0].type == "lesson_completion"
    
    def test_check_streak_achievement(self, achievement_service):
        """Test checking streak achievement."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        # Mock progress with 7-day streak
        mock_progress = Mock(
            total_lessons_completed=10,
            current_level=2,
            achievements=[],
            accuracy_rate=0.85,
            practice_time_minutes=200,
            signs_learned=30,
            current_streak=7
        )
        
        # Mock achievement
        mock_achievement = Mock(
            id=uuid4(),
            type="streak",
            criteria={"days_required": 7}
        )
        
        with patch.object(achievement_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(achievement_service.achievement_repo, 'get_all', return_value=[mock_achievement]):
                with patch.object(achievement_service.progress_repo, 'add_achievement'):
                    newly_awarded = achievement_service.check_and_award_achievements(user_id)
                    
                    assert len(newly_awarded) == 1
                    assert newly_awarded[0].type == "streak"
    
    def test_get_user_achievements(self, achievement_service):
        """Test getting user's earned achievements."""
        from uuid import uuid4
        
        user_id = uuid4()
        achievement_ids = [uuid4(), uuid4()]
        
        mock_progress = Mock(achievements=achievement_ids)
        mock_achievements = [
            Mock(id=achievement_ids[0], name="First Lesson"),
            Mock(id=achievement_ids[1], name="Week Streak")
        ]
        
        with patch.object(achievement_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(achievement_service.achievement_repo, 'get_multi_by_ids', return_value=mock_achievements):
                achievements = achievement_service.get_user_achievements(user_id)
                
                assert len(achievements) == 2
                assert achievements[0].name == "First Lesson"
    
    def test_calculate_achievement_progress(self, achievement_service):
        """Test calculating progress towards achievement."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        mock_progress = Mock(
            total_lessons_completed=3,
            current_level=1,
            accuracy_rate=0.6,
            practice_time_minutes=50,
            signs_learned=15,
            current_streak=2
        )
        
        mock_achievement = Mock(
            type="lesson_completion",
            criteria={"lessons_required": 10}
        )
        
        progress_pct = achievement_service._calculate_achievement_progress(
            mock_achievement, mock_progress, user_id
        )
        
        assert progress_pct == 30.0  # 3/10 * 100


class TestAnalyticsService:
    """Test cases for AnalyticsService."""
    
    @pytest.fixture
    def analytics_service(self, test_db):
        """Create AnalyticsService instance."""
        from app.services.analytics_service import AnalyticsService
        return AnalyticsService(test_db)
    
    def test_get_user_analytics(self, analytics_service):
        """Test getting comprehensive user analytics."""
        from uuid import uuid4
        from datetime import datetime
        
        user_id = uuid4()
        
        mock_progress = Mock(
            total_lessons_completed=10,
            current_level=2,
            experience_points=1000,
            signs_learned=50,
            accuracy_rate=0.85,
            practice_time_minutes=300,
            current_streak=5,
            longest_streak=7,
            days_active=15
        )
        
        mock_session_stats = {
            "total_sessions": 20,
            "total_time": 18000,
            "average_accuracy": 0.85,
            "signs_practiced": 50
        }
        
        with patch.object(analytics_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(analytics_service.session_repo, 'get_user_session_stats', return_value=mock_session_stats):
                with patch.object(analytics_service.session_repo, 'get_recent_sessions', return_value=[]):
                    with patch.object(analytics_service, '_calculate_learning_velocity', return_value=10.0):
                        with patch.object(analytics_service, '_analyze_strengths_weaknesses', return_value={"strengths": [], "weaknesses": []}):
                            analytics = analytics_service.get_user_analytics(user_id)
                            
                            assert analytics["overview"]["total_lessons_completed"] == 10
                            assert analytics["overview"]["current_level"] == 2
                            assert analytics["streaks"]["current_streak"] == 5
                            assert analytics["practice_stats"]["total_sessions"] == 20
    
    def test_get_learning_patterns(self, analytics_service):
        """Test analyzing learning patterns."""
        from uuid import uuid4
        from datetime import datetime
        
        user_id = uuid4()
        
        mock_sessions = [
            Mock(completed_at=datetime(2024, 1, 1, 10, 0), duration_seconds=900),
            Mock(completed_at=datetime(2024, 1, 2, 10, 30), duration_seconds=1200),
            Mock(completed_at=datetime(2024, 1, 3, 11, 0), duration_seconds=800),
        ]
        
        with patch.object(analytics_service.session_repo, 'get_by_user', return_value=mock_sessions):
            with patch.object(analytics_service, '_calculate_consistency_score', return_value=0.7):
                with patch.object(analytics_service, '_get_most_practiced_signs', return_value=[]):
                    with patch.object(analytics_service, '_get_weekly_activity', return_value=[]):
                        patterns = analytics_service.get_learning_patterns(user_id)
                        
                        assert patterns["preferred_time"] == "morning"
                        assert patterns["average_session_duration"] == 900
                        assert patterns["consistency_score"] == 0.7
    
    def test_get_recommendations(self, analytics_service):
        """Test getting personalized recommendations."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        # Mock progress with low streak
        mock_progress = Mock(
            current_streak=1,
            accuracy_rate=0.65,
            total_lessons_completed=15,
            current_level=2
        )
        
        with patch.object(analytics_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            recommendations = analytics_service.get_recommendations(user_id)
            
            assert len(recommendations) > 0
            assert any(r["type"] == "consistency" for r in recommendations)
    
    def test_calculate_learning_velocity(self, analytics_service):
        """Test calculating learning velocity."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        mock_progress = Mock(
            signs_learned=70,
            days_active=14  # 2 weeks
        )
        
        with patch.object(analytics_service.progress_repo, 'get_by_user_id', return_value=mock_progress):
            velocity = analytics_service._calculate_learning_velocity(user_id)
            
            assert velocity == 35.0  # 70 signs / 2 weeks
    
    def test_find_preferred_practice_time(self, analytics_service):
        """Test finding preferred practice time."""
        from datetime import datetime
        
        # Morning sessions
        morning_sessions = [
            Mock(completed_at=datetime(2024, 1, 1, 9, 0)),
            Mock(completed_at=datetime(2024, 1, 2, 10, 0)),
            Mock(completed_at=datetime(2024, 1, 3, 8, 30)),
        ]
        
        preferred_time = analytics_service._find_preferred_practice_time(morning_sessions)
        assert preferred_time == "morning"
        
        # Evening sessions
        evening_sessions = [
            Mock(completed_at=datetime(2024, 1, 1, 19, 0)),
            Mock(completed_at=datetime(2024, 1, 2, 20, 0)),
        ]
        
        preferred_time = analytics_service._find_preferred_practice_time(evening_sessions)
        assert preferred_time == "evening"


class TestLearningProgressTracking:
    """Test cases for learning progress tracking."""
    
    @pytest.fixture
    def progress_repo(self, test_db):
        """Create LearningProgressRepository instance."""
        from app.repositories.user_repository import LearningProgressRepository
        return LearningProgressRepository(test_db)
    
    def test_update_lesson_completion(self, progress_repo):
        """Test updating progress when lesson is completed."""
        from uuid import uuid4
        
        user_id = uuid4()
        lesson_id = uuid4()
        
        mock_progress = Mock(
            id=uuid4(),
            completed_lessons=[],
            total_lessons_completed=0,
            experience_points=0,
            current_level=1,
            updated_at=None
        )
        
        with patch.object(progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(progress_repo, 'update') as mock_update:
                progress_repo.update_lesson_completion(user_id, lesson_id)
                
                # Verify update was called with correct data
                mock_update.assert_called_once()
                call_args = mock_update.call_args[0]
                assert call_args[0] == mock_progress.id
    
    def test_update_practice_session(self, progress_repo):
        """Test updating progress after practice session."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        mock_progress = Mock(
            id=uuid4(),
            practice_time_minutes=100,
            signs_learned=20,
            accuracy_rate=0.8,
            updated_at=None
        )
        
        with patch.object(progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(progress_repo, 'update') as mock_update:
                progress_repo.update_practice_session(
                    user_id,
                    duration_minutes=30,
                    accuracy=0.85,
                    signs_practiced=5
                )
                
                mock_update.assert_called_once()
    
    def test_add_achievement(self, progress_repo):
        """Test adding achievement to user progress."""
        from uuid import uuid4
        
        user_id = uuid4()
        achievement_id = uuid4()
        
        mock_progress = Mock(
            id=uuid4(),
            achievements=[]
        )
        
        with patch.object(progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(progress_repo, 'update') as mock_update:
                progress_repo.add_achievement(user_id, achievement_id)
                
                mock_update.assert_called_once()
                call_args = mock_update.call_args[0]
                assert achievement_id in call_args[1]["achievements"]
    
    def test_update_streak_increment(self, progress_repo):
        """Test incrementing user's learning streak."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        mock_progress = Mock(
            id=uuid4(),
            current_streak=5,
            longest_streak=7
        )
        
        with patch.object(progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(progress_repo, 'update') as mock_update:
                progress_repo.update_streak(user_id, increment=True)
                
                mock_update.assert_called_once()
                call_args = mock_update.call_args[0]
                assert call_args[1]["current_streak"] == 6
    
    def test_update_streak_reset(self, progress_repo):
        """Test resetting user's learning streak."""
        from uuid import uuid4
        
        user_id = uuid4()
        
        mock_progress = Mock(
            id=uuid4(),
            current_streak=5,
            longest_streak=7
        )
        
        with patch.object(progress_repo, 'get_by_user_id', return_value=mock_progress):
            with patch.object(progress_repo, 'update') as mock_update:
                progress_repo.update_streak(user_id, increment=False)
                
                mock_update.assert_called_once()
                call_args = mock_update.call_args[0]
                assert call_args[1]["current_streak"] == 0
