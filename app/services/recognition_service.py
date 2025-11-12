"""
Sign Recognition Service

Handles video/image processing for GSL gesture recognition using AI models,
confidence scoring, and similar gesture suggestions.
"""

from typing import Optional, List, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.db_models.gsl import GSLSign
# Note: SignRecognition will be handled via Supabase
from app.schemas.gsl import (
    RecognitionRequest, RecognitionResponse,
    ConfidenceScore, SimilarGesture
)
from app.ai.computer_vision import ComputerVisionModel
from app.utils.cache import CacheManager

settings = get_settings()


class RecognitionService:
    """Service for AI-powered sign recognition operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.cv_model = ComputerVisionModel()
        self.cache = CacheManager()
        self.confidence_threshold = 0.7
        self.max_similar_suggestions = 5
    
    async def recognize_gesture(
        self,
        media_data: bytes,
        media_type: str,
        user_id: Optional[UUID] = None
    ) -> RecognitionResponse:
        """
        Recognize GSL gesture from video or image data.
        
        Args:
            media_data: Raw media file bytes
            media_type: Type of media (video/image)
            user_id: Optional user ID for tracking
            
        Returns:
            Recognition result with confidence scores
        """
        start_time = datetime.utcnow()
        
        # Preprocess media data
        processed_frames = await self._preprocess_media(media_data, media_type)
        
        # Run AI model inference
        predictions = await self.cv_model.predict(processed_frames)
        
        # Get top prediction
        top_prediction = predictions[0] if predictions else None
        
        if not top_prediction or top_prediction['confidence'] < self.confidence_threshold:
            # Low confidence - provide similar gestures
            similar_gestures = await self._find_similar_gestures(predictions)
            
            recognition = SignRecognition(
                id=uuid4(),
                user_id=user_id,
                recognized_sign_id=None,
                confidence_score=top_prediction['confidence'] if top_prediction else 0.0,
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                status="low_confidence"
            )
            self.db.add(recognition)
            self.db.commit()
            
            return RecognitionResponse(
                recognition_id=recognition.id,
                recognized_sign=None,
                confidence_score=recognition.confidence_score,
                alternative_matches=similar_gestures,
                processing_time=recognition.processing_time,
                status="low_confidence",
                message="Gesture not recognized with high confidence. See suggestions."
            )
        
        # High confidence - get sign details
        sign = await self._get_sign_by_label(top_prediction['label'])
        
        # Get alternative matches
        alternative_matches = await self._get_alternative_matches(predictions[1:5])
        
        # Save recognition result
        recognition = SignRecognition(
            id=uuid4(),
            user_id=user_id,
            recognized_sign_id=sign.id if sign else None,
            confidence_score=top_prediction['confidence'],
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
            status="success"
        )
        self.db.add(recognition)
        self.db.commit()
        
        return RecognitionResponse(
            recognition_id=recognition.id,
            recognized_sign=sign,
            confidence_score=recognition.confidence_score,
            alternative_matches=alternative_matches,
            processing_time=recognition.processing_time,
            status="success",
            message="Gesture recognized successfully"
        )
    
    async def validate_gesture(
        self,
        user_gesture_data: bytes,
        expected_sign_id: UUID,
        user_id: UUID
    ) -> dict:
        """
        Validate user's gesture attempt against expected sign.
        
        Args:
            user_gesture_data: User's gesture video/image
            expected_sign_id: Expected sign ID
            user_id: User performing the gesture
            
        Returns:
            Validation result with feedback
        """
        # Recognize the user's gesture
        recognition = await self.recognize_gesture(
            user_gesture_data,
            "video",
            user_id
        )
        
        # Check if recognized sign matches expected
        is_correct = (
            recognition.recognized_sign and
            recognition.recognized_sign.id == expected_sign_id and
            recognition.confidence_score >= self.confidence_threshold
        )
        
        # Generate feedback
        feedback = await self._generate_validation_feedback(
            recognition,
            expected_sign_id,
            is_correct
        )
        
        return {
            "is_correct": is_correct,
            "confidence_score": recognition.confidence_score,
            "feedback": feedback,
            "recognized_sign": recognition.recognized_sign,
            "expected_sign_id": expected_sign_id
        }
    
    async def get_confidence_scores(self, recognition_id: UUID) -> Optional[dict]:
        """
        Get detailed confidence scores for a recognition result.
        
        Args:
            recognition_id: Recognition result ID
            
        Returns:
            Detailed confidence breakdown
        """
        recognition = self.db.query(SignRecognition).filter(
            SignRecognition.id == recognition_id
        ).first()
        
        if not recognition:
            return None
        
        return {
            "recognition_id": recognition.id,
            "overall_confidence": recognition.confidence_score,
            "status": recognition.status,
            "processing_time": recognition.processing_time,
            "timestamp": recognition.created_at
        }
    
    async def get_similar_gestures(
        self,
        gesture_name: str,
        limit: int = 5
    ) -> List[SimilarGesture]:
        """
        Get similar gestures for a given gesture name.
        
        Args:
            gesture_name: Name of the gesture
            limit: Maximum number of similar gestures to return
            
        Returns:
            List of similar gestures with similarity scores
        """
        # Check cache first
        cache_key = f"similar_gestures:{gesture_name}:{limit}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Find the target sign
        target_sign = self.db.query(GSLSign).filter(
            GSLSign.sign_name.ilike(f"%{gesture_name}%")
        ).first()
        
        if not target_sign:
            return []
        
        # Get similar signs based on category and difficulty
        similar_signs = self.db.query(GSLSign).filter(
            GSLSign.id != target_sign.id,
            GSLSign.category_id == target_sign.category_id
        ).limit(limit).all()
        
        # Calculate similarity scores (simplified)
        results = []
        for sign in similar_signs:
            similarity = self._calculate_similarity(target_sign, sign)
            results.append(SimilarGesture(
                sign=sign,
                similarity_score=similarity,
                reason=f"Same category: {sign.category.name if sign.category else 'Unknown'}"
            ))
        
        # Sort by similarity
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Cache the results
        await self.cache.set(cache_key, results, ttl=3600)
        
        return results
    
    async def _preprocess_media(self, media_data: bytes, media_type: str) -> np.ndarray:
        """Preprocess media data for AI model input."""
        # TODO: Implement video/image preprocessing
        # - Extract frames from video
        # - Resize and normalize
        # - Convert to model input format
        return np.array([])
    
    async def _get_sign_by_label(self, label: str) -> Optional[GSLSign]:
        """Get GSL sign by model prediction label."""
        return self.db.query(GSLSign).filter(
            GSLSign.sign_name == label
        ).first()
    
    async def _get_alternative_matches(
        self,
        predictions: List[dict]
    ) -> List[Tuple[GSLSign, float]]:
        """Get alternative sign matches from predictions."""
        alternatives = []
        for pred in predictions:
            sign = await self._get_sign_by_label(pred['label'])
            if sign:
                alternatives.append((sign, pred['confidence']))
        return alternatives
    
    async def _find_similar_gestures(
        self,
        predictions: List[dict]
    ) -> List[SimilarGesture]:
        """Find similar gestures when recognition confidence is low."""
        similar = []
        for pred in predictions[:self.max_similar_suggestions]:
            sign = await self._get_sign_by_label(pred['label'])
            if sign:
                similar.append(SimilarGesture(
                    sign=sign,
                    similarity_score=pred['confidence'],
                    reason="Possible match based on gesture features"
                ))
        return similar
    
    async def _generate_validation_feedback(
        self,
        recognition: RecognitionResponse,
        expected_sign_id: UUID,
        is_correct: bool
    ) -> str:
        """Generate feedback for gesture validation."""
        if is_correct:
            return "Great job! Your gesture matches the expected sign."
        
        if recognition.confidence_score < self.confidence_threshold:
            return "Gesture not clearly recognized. Try again with better lighting and hand positioning."
        
        expected_sign = self.db.query(GSLSign).filter(
            GSLSign.id == expected_sign_id
        ).first()
        
        if expected_sign and recognition.recognized_sign:
            return f"You performed '{recognition.recognized_sign.sign_name}' but expected '{expected_sign.sign_name}'. Review the tutorial and try again."
        
        return "Gesture not recognized. Please review the tutorial and try again."
    
    def _calculate_similarity(self, sign1: GSLSign, sign2: GSLSign) -> float:
        """Calculate similarity score between two signs."""
        score = 0.0
        
        # Same category
        if sign1.category_id == sign2.category_id:
            score += 0.5
        
        # Similar difficulty
        if abs(sign1.difficulty_level - sign2.difficulty_level) <= 1:
            score += 0.3
        
        # Related signs
        if sign2.id in (sign1.related_signs or []):
            score += 0.2
        
        return min(score, 1.0)