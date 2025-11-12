"""
Unit Tests for Learning Service

Tests lesson management, progress tracking, and dictionary search.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4

from app.services.learning_service import LearningService


class TestLearningService:
    """Test cases for LearningService."""
    
    @pytest.fixture
    def learning_service(self, test_db):
        """Create LearningService instance."""
        return LearningService(test_db)
    
    @pytest.mark.asyncio
    async def test_get_lessons(self, learning_service):
        """Test getting lessons list."""
        # Mock lessons
        mock_lessons = [
            Mock(
                id=uuid4(),
                title="Lesson 1",
                level=1,
                category="greetings",
                sequence_order=1,
                estimated_duration=15,
                signs_covered=["sign1", "sign2"]
            )
        ]
        
        learning_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                order_by=Mock(return_value=Mock(
                    limit=Mock(return_value=Mock(
                        offset=Mock(return_value=Mock(all=Mock(return_value=mock_lessons)))
                    ))
                ))
            ))
        ))
        
        result = await learning_service.get_lessons()
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_get_lessons_with_filters(self, learning_service):
        """Test getting lessons with level and category filters."""
        learning_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                filter=Mock(return_value=Mock(
                    order_by=Mock(return_value=Mock(
                        limit=Mock(return_value=Mock(
                            offset=Mock(return_value=Mock(all=Mock(return_value=[])))
                        ))
                    ))
                ))
            ))
        ))
        
        result = await learning_service.get_lessons(level=1, category="greetings")
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_lesson_details(self, learning_service, sample_lesson):
        """Test getting detailed lesson information."""
        lesson_id = UUID(sample_lesson["id"])
        
        mock_lesson = Mock(**sample_lesson)
        mock_steps = [
            Mock(step_number=1, instruction="Step 1"),
            Mock(step_number=2, instruction="Step 2")
        ]
        mock_signs = [Mock(id="sign1"), Mock(id="sign2")]
        
        # Mock lesson query
        learning_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_lesson)))
        ))
        
        with patch.object(learning_service.db, 'query') as mock_query:
            # Setup different returns for different queries
            mock_query.return_value.filter.return_value.order_by.return_value.all.return_value = mock_steps
            mock_query.return_value.filter.return_value.all.return_value = mock_signs
            
            result = await learning_service.get_lesson_details(lesson_id)
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_update_lesson_progress(self, learning_service):
        """Test updating lesson completion progress."""
        user_id = uuid4()
        lesson_id = uuid4()
        
        mock_progress = Mock(
            user_id=user_id,
            total_lessons_completed=0,
            completed_lessons=[]
        )
        
        learning_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_progress)))
        ))
        
        with patch.object(learning_service, '_check_and_award_achievements'):
            result = await learning_service.update_lesson_progress(
                user_id,
                lesson_id,
                completed=True
            )
            
            assert result["completed"] is True
            assert "total_lessons_completed" in result
    
    @pytest.mark.asyncio
    async def test_get_user_achievements(self, learning_service):
        """Test getting user achievements."""
        user_id = uuid4()
        achievement_ids = [uuid4(), uuid4()]
        
        mock_progress = Mock(achievements=achievement_ids)
        mock_achievements = [
            Mock(id=achievement_ids[0], name="First Lesson", earned_at=None),
            Mock(id=achievement_ids[1], name="Five Lessons", earned_at=None)
        ]
        
        learning_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=mock_progress),
                all=Mock(return_value=mock_achievements)
            ))
        ))
        
        result = await learning_service.get_user_achievements(user_id)
        
        assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_search_dictionary(self, learning_service):
        """Test searching GSL dictionary."""
        query = "hello"
        
        mock_signs = [
            Mock(id=uuid4(), sign_name="hello", description="Greeting"),
            Mock(id=uuid4(), sign_name="hello_morning", description="Morning greeting")
        ]
        
        with patch.object(learning_service.cache, 'get', return_value=None):
            with patch.object(learning_service.cache, 'set', return_value=True):
                learning_service.db.query = Mock(return_value=Mock(
                    filter=Mock(return_value=Mock(
                        order_by=Mock(return_value=Mock(
                            limit=Mock(return_value=Mock(
                                offset=Mock(return_value=Mock(all=Mock(return_value=mock_signs)))
                            ))
                        ))
                    ))
                ))
                
                result = await learning_service.search_dictionary(query)
                
                assert isinstance(result, list)
                assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_search_dictionary_cached(self, learning_service):
        """Test dictionary search with cached results."""
        query = "hello"
        cached_results = [Mock(sign_name="hello")]
        
        with patch.object(learning_service.cache, 'get', return_value=cached_results):
            result = await learning_service.search_dictionary(query)
            
            assert result == cached_results
    
    @pytest.mark.asyncio
    async def test_record_practice_session(self, learning_service):
        """Test recording a practice session."""
        user_id = uuid4()
        lesson_id = uuid4()
        signs_practiced = [uuid4(), uuid4()]
        
        result = await learning_service.record_practice_session(
            user_id=user_id,
            lesson_id=lesson_id,
            signs_practiced=signs_practiced,
            duration_seconds=300
        )
        
        assert result is not None
    
    def test_is_lesson_locked_no_progress(self, learning_service):
        """Test lesson lock status with no user progress."""
        lesson = Mock(level=2, sequence_order=1, prerequisites=[])
        
        is_locked = learning_service._is_lesson_locked(lesson, None)
        
        assert is_locked is True  # Level 2 should be locked without progress
    
    def test_is_lesson_locked_sufficient_level(self, learning_service):
        """Test lesson lock status with sufficient user level."""
        lesson = Mock(level=2, sequence_order=1, prerequisites=[])
        progress = Mock(current_level=3, completed_lessons=[])
        
        is_locked = learning_service._is_lesson_locked(lesson, progress)
        
        assert is_locked is False  # User level 3 can access level 2