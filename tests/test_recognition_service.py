"""
Unit Tests for Recognition Service

Tests gesture recognition, validation, and similar gesture suggestions.
Requirements: 2.1, 2.2, 2.3
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import UUID, uuid4
from datetime import datetime

from app.services.recognition_service import RecognitionService
from app.db_models.gsl import GSLSign, SignCategory
from app.db_models.recognition import SignRecognition
from app.schemas.gsl import RecognitionStatus


class TestRecognitionService:
    """Test cases for RecognitionService."""
    
    @pytest.fixture
    def recognition_service(self, test_db):
        """Create RecognitionService instance."""
        return RecognitionService(test_db)
    
    @pytest.fixture
    def sample_sign(self, test_db):
        """Create a sample GSL sign in the database."""
        category = SignCategory(
            id=uuid4(),
            name="greetings",
            description="Greeting signs"
        )
        test_db.add(category)
        test_db.commit()
        
        sign = GSLSign(
            id=uuid4(),
            sign_name="hello",
            description="Greeting sign",
            category_id=category.id,
            difficulty_level=1,
            video_url="https://example.com/hello.mp4",
            thumbnail_url="https://example.com/hello_thumb.jpg",
            usage_examples=["Hello, how are you?"],
            related_signs=[]
        )
        test_db.add(sign)
        test_db.commit()
        test_db.refresh(sign)
        return sign
    
    @pytest.mark.asyncio
    async def test_recognize_gesture_high_confidence(
        self,
        recognition_service,
        sample_video_bytes,
        sample_sign
    ):
        """
        Test gesture recognition with high confidence.
        Requirements: 2.1, 2.2
        """
        # Mock CV model predictions
        mock_predictions = [
            {'label': 'hello', 'confidence': 0.95},
            {'label': 'thank_you', 'confidence': 0.03}
        ]
        
        # Mock preprocessing
        with patch.object(recognition_service, '_preprocess_media', return_value=AsyncMock()):
            with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
                with patch.object(recognition_service, '_get_sign_by_label', return_value=sample_sign):
                    with patch.object(recognition_service, '_get_alternative_matches', return_value=[]):
                        result = await recognition_service.recognize_gesture(
                            sample_video_bytes,
                            "video",
                            UUID("12345678-1234-5678-1234-567812345678")
                        )
                        
                        assert result.status == RecognitionStatus.success
                        assert result.confidence_score >= 0.7
                        assert result.recognized_sign is not None
                        assert result.processing_time > 0
                        assert result.message == "Gesture recognized successfully"
    
    @pytest.mark.asyncio
    async def test_recognize_gesture_low_confidence(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """
        Test gesture recognition with low confidence.
        Requirements: 2.2, 2.3
        """
        # Mock CV model predictions with low confidence
        mock_predictions = [
            {'label': 'hello', 'confidence': 0.45},
            {'label': 'thank_you', 'confidence': 0.35}
        ]
        
        with patch.object(recognition_service, '_preprocess_media', return_value=AsyncMock()):
            with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
                with patch.object(recognition_service, '_find_similar_gestures', return_value=[]):
                    result = await recognition_service.recognize_gesture(
                        sample_video_bytes,
                        "video"
                    )
                    
                    assert result.status == RecognitionStatus.low_confidence
                    assert result.confidence_score < 0.7
                    assert result.recognized_sign is None
                    assert "not recognized with high confidence" in result.message
    
    @pytest.mark.asyncio
    async def test_validate_gesture_correct(
        self,
        recognition_service,
        sample_video_bytes,
        sample_sign
    ):
        """
        Test gesture validation with correct gesture.
        Requirements: 2.2, 2.3
        """
        user_id = uuid4()
        expected_sign_id = sample_sign.id
        
        # Mock recognition result
        mock_result = Mock(
            recognized_sign=Mock(id=expected_sign_id),
            confidence_score=0.85,
            status=RecognitionStatus.success
        )
        
        with patch.object(recognition_service, 'recognize_gesture', return_value=mock_result):
            with patch.object(recognition_service, '_generate_validation_feedback', return_value="Great job!"):
                result = await recognition_service.validate_gesture(
                    sample_video_bytes,
                    expected_sign_id,
                    user_id
                )
                
                assert result["is_correct"] is True
                assert result["confidence_score"] >= 0.7
                assert "Great job" in result["feedback"]
    
    @pytest.mark.asyncio
    async def test_validate_gesture_incorrect(
        self,
        recognition_service,
        sample_video_bytes
    ):
        """
        Test gesture validation with incorrect gesture.
        Requirements: 2.2, 2.3
        """
        user_id = uuid4()
        expected_sign_id = uuid4()
        different_sign_id = uuid4()
        
        # Mock recognition result with different sign
        mock_result = Mock(
            recognized_sign=Mock(id=different_sign_id),
            confidence_score=0.85,
            status=RecognitionStatus.success
        )
        
        with patch.object(recognition_service, 'recognize_gesture', return_value=mock_result):
            with patch.object(recognition_service, '_generate_validation_feedback', return_value="Try again"):
                result = await recognition_service.validate_gesture(
                    sample_video_bytes,
                    expected_sign_id,
                    user_id
                )
                
                assert result["is_correct"] is False
                assert result["confidence_score"] == 0.85
                assert "Try again" in result["feedback"]
    
    @pytest.mark.asyncio
    async def test_get_similar_gestures(self, recognition_service, sample_sign, test_db):
        """
        Test getting similar gestures for failed recognition.
        Requirements: 2.3
        """
        gesture_name = "hello"
        
        # Create another similar sign
        similar_sign = GSLSign(
            id=uuid4(),
            sign_name="hi",
            description="Another greeting",
            category_id=sample_sign.category_id,
            difficulty_level=1,
            video_url="https://example.com/hi.mp4",
            thumbnail_url="https://example.com/hi_thumb.jpg",
            usage_examples=["Hi there!"],
            related_signs=[]
        )
        test_db.add(similar_sign)
        test_db.commit()
        
        with patch.object(recognition_service.cache, 'get', return_value=None):
            with patch.object(recognition_service.cache, 'set', return_value=True):
                result = await recognition_service.get_similar_gestures(gesture_name, limit=5)
                
                assert isinstance(result, list)
                assert len(result) > 0
                # Verify similar gestures have similarity scores
                for similar in result:
                    assert hasattr(similar, 'similarity_score')
                    assert 0.0 <= similar.similarity_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_confidence_scores(self, recognition_service, test_db, sample_sign):
        """
        Test getting confidence scores for recognition.
        Requirements: 2.2
        """
        recognition_id = uuid4()
        user_id = uuid4()
        
        # Create recognition record
        recognition = SignRecognition(
            id=recognition_id,
            user_id=user_id,
            recognized_sign_id=sample_sign.id,
            confidence_score=0.85,
            status="success",
            processing_time=1.5,
            media_type="video"
        )
        test_db.add(recognition)
        test_db.commit()
        
        result = await recognition_service.get_confidence_scores(recognition_id)
        
        assert result is not None
        assert result["overall_confidence"] == 0.85
        assert result["status"] == "success"
        assert result["processing_time"] == 1.5
    
    def test_calculate_similarity(self, recognition_service):
        """
        Test similarity calculation between signs.
        Requirements: 2.3
        """
        category_id = uuid4()
        
        sign1 = Mock(
            category_id=category_id,
            difficulty_level=1,
            related_signs=[]
        )
        
        sign2 = Mock(
            category_id=category_id,
            difficulty_level=1,
            related_signs=[]
        )
        
        similarity = recognition_service._calculate_similarity(sign1, sign2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0  # Same category and difficulty
        assert similarity >= 0.8  # Should be high for same category and difficulty
    
    @pytest.mark.asyncio
    async def test_recognize_gesture_processing_time(
        self,
        recognition_service,
        sample_video_bytes,
        sample_sign
    ):
        """
        Test that gesture recognition completes within acceptable time.
        Requirements: 2.2 (3-second response time requirement)
        """
        mock_predictions = [
            {'label': 'hello', 'confidence': 0.95}
        ]
        
        with patch.object(recognition_service, '_preprocess_media', return_value=AsyncMock()):
            with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
                with patch.object(recognition_service, '_get_sign_by_label', return_value=sample_sign):
                    with patch.object(recognition_service, '_get_alternative_matches', return_value=[]):
                        result = await recognition_service.recognize_gesture(
                            sample_video_bytes,
                            "video"
                        )
                        
                        # Verify processing time is recorded and reasonable
                        assert result.processing_time > 0
                        assert result.processing_time < 3.0  # Should be under 3 seconds
    
    @pytest.mark.asyncio
    async def test_fallback_handling_unrecognized_gesture(
        self,
        recognition_service,
        sample_video_bytes,
        sample_sign
    ):
        """
        Test fallback handling for unrecognized gestures with similar sign suggestions.
        Requirements: 2.3
        """
        # Mock predictions with no high confidence match
        mock_predictions = [
            {'label': 'unknown_sign', 'confidence': 0.35},
            {'label': 'hello', 'confidence': 0.25}
        ]
        
        # Mock similar gesture
        from app.schemas.gsl import SimilarGesture, GSLSignResponse
        mock_similar = [
            SimilarGesture(
                sign=GSLSignResponse(
                    id=sample_sign.id,
                    sign_name=sample_sign.sign_name,
                    description=sample_sign.description,
                    category="greetings",
                    difficulty_level=1,
                    video_url=sample_sign.video_url,
                    thumbnail_url=sample_sign.thumbnail_url,
                    usage_examples=[],
                    related_signs=[],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                ),
                similarity_score=0.35,
                reason="Possible match"
            )
        ]
        
        with patch.object(recognition_service, '_preprocess_media', return_value=AsyncMock()):
            with patch.object(recognition_service.cv_model, 'predict', return_value=mock_predictions):
                with patch.object(recognition_service, '_find_similar_gestures', return_value=mock_similar):
                    result = await recognition_service.recognize_gesture(
                        sample_video_bytes,
                        "video"
                    )
                    
                    assert result.status == RecognitionStatus.low_confidence
                    assert len(result.alternative_matches) > 0
                    assert "not recognized with high confidence" in result.message