"""
Unit Tests for Recognition Service

Tests gesture recognition, validation, and similar gesture suggestions.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4

from app.services.recognition_service import RecognitionService


class TestRecognitionService:
    """Test cases for RecognitionService."""
    
    @pytest.fixture
    def recognition_service(self, test_db):
        """Create RecognitionService instance."""
        return RecognitionService(test_db)
    
    @pytest.mark.asyncio
    async def test_recognize_gesture_high_confidence(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """Test gesture recognition with high confidence."""
        # Mock CV model predictions
        mock_predictions = [
            {'label': 'hello', 'confidence': 0.95},
            {'label': 'thank_you', 'confidence': 0.03}
        ]
        
        with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
            with patch.object(recognition_service, '_get_sign_by_label', return_value=Mock(id=uuid4())):
                result = await recognition_service.recognize_gesture(
                    sample_video_bytes,
                    "video",
                    UUID("12345678-1234-5678-1234-567812345678")
                )
                
                assert result.status == "success"
                assert result.confidence_score >= 0.7
                assert result.recognized_sign is not None
    
    @pytest.mark.asyncio
    async def test_recognize_gesture_low_confidence(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """Test gesture recognition with low confidence."""
        # Mock CV model predictions with low confidence
        mock_predictions = [
            {'label': 'hello', 'confidence': 0.45},
            {'label': 'thank_you', 'confidence': 0.35}
        ]
        
        with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
            with patch.object(recognition_service, '_find_similar_gestures', return_value=[]):
                result = await recognition_service.recognize_gesture(
                    sample_video_bytes,
                    "video"
                )
                
                assert result.status == "low_confidence"
                assert result.confidence_score < 0.7
    
    @pytest.mark.asyncio
    async def test_validate_gesture_correct(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """Test gesture validation with correct gesture."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        expected_sign_id = UUID("87654321-4321-8765-4321-876543218765")
        
        # Mock recognition result
        mock_result = Mock(
            recognized_sign=Mock(id=expected_sign_id),
            confidence_score=0.85,
            status="success"
        )
        
        with patch.object(recognition_service, 'recognize_gesture', return_value=mock_result):
            result = await recognition_service.validate_gesture(
                sample_video_bytes,
                expected_sign_id,
                user_id
            )
            
            assert result["is_correct"] is True
            assert result["confidence_score"] >= 0.7
    
    @pytest.mark.asyncio
    async def test_validate_gesture_incorrect(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """Test gesture validation with incorrect gesture."""
        user_id = UUID("12345678-1234-5678-1234-567812345678")
        expected_sign_id = UUID("87654321-4321-8765-4321-876543218765")
        different_sign_id = UUID("11111111-1111-1111-1111-111111111111")
        
        # Mock recognition result with different sign
        mock_result = Mock(
            recognized_sign=Mock(id=different_sign_id),
            confidence_score=0.85,
            status="success"
        )
        
        with patch.object(recognition_service, 'recognize_gesture', return_value=mock_result):
            with patch.object(recognition_service, '_generate_validation_feedback', return_value="Try again"):
                result = await recognition_service.validate_gesture(
                    sample_video_bytes,
                    expected_sign_id,
                    user_id
                )
                
                assert result["is_correct"] is False
    
    @pytest.mark.asyncio
    async def test_get_similar_gestures(self, recognition_service):
        """Test getting similar gestures."""
        gesture_name = "hello"
        
        # Mock database query
        mock_sign = Mock(
            id=uuid4(),
            sign_name="hello",
            category_id=uuid4()
        )
        
        recognition_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_sign)))
        ))
        
        with patch.object(recognition_service.cache, 'get', return_value=None):
            with patch.object(recognition_service.cache, 'set', return_value=True):
                result = await recognition_service.get_similar_gestures(gesture_name, limit=5)
                
                assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_get_confidence_scores(self, recognition_service):
        """Test getting confidence scores for recognition."""
        recognition_id = UUID("12345678-1234-5678-1234-567812345678")
        
        # Mock recognition record
        mock_recognition = Mock(
            id=recognition_id,
            confidence_score=0.85,
            status="success",
            processing_time=1.5
        )
        
        recognition_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_recognition)))
        ))
        
        result = await recognition_service.get_confidence_scores(recognition_id)
        
        assert result is not None
        assert result["overall_confidence"] == 0.85
        assert result["status"] == "success"
    
    def test_calculate_similarity(self, recognition_service):
        """Test similarity calculation between signs."""
        sign1 = Mock(
            category_id=uuid4(),
            difficulty_level=1,
            related_signs=[]
        )
        
        sign2 = Mock(
            category_id=sign1.category_id,
            difficulty_level=1,
            related_signs=[]
        )
        
        similarity = recognition_service._calculate_similarity(sign1, sign2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Same category and difficulty