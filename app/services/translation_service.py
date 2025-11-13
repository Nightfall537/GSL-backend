"""
Translation Service

Handles bidirectional translation between speech, text, and GSL signs
with support for Ghanaian English accents and local phrases.
"""

from typing import List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from app.config.settings import get_settings
from app.db_models.gsl import GSLSign
from app.db_models.translation import Translation, SignSequence
from app.schemas.gsl import (
    SpeechToSignRequest, TextToSignRequest,
    SignToTextRequest, TranslationResponse, GSLSignResponse,
    SignSequenceResponse
)
from app.ai.speech_to_text import SpeechToTextModel
from app.ai.nlp_processor import NLPProcessor
from app.utils.cache import CacheManager

settings = get_settings()


class TranslationService:
    """Service for translation operations between speech, text, and GSL."""
    
    def __init__(self, db: Session):
        self.db = db
        self.stt_model = SpeechToTextModel()
        self.nlp_processor = NLPProcessor()
        self.cache = CacheManager()
    
    async def speech_to_sign(
        self,
        audio_data: bytes,
        user_id: Optional[UUID] = None
    ) -> TranslationResponse:
        """
        Convert speech audio to GSL signs.
        
        Args:
            audio_data: Audio file bytes
            user_id: Optional user ID for tracking
            
        Returns:
            Translation result with GSL signs
        """
        # Convert speech to text
        transcribed_text = await self.stt_model.transcribe(
            audio_data,
            language="en",
            accent="ghanaian"
        )
        
        if not transcribed_text:
            return TranslationResponse(
                source_type="speech",
                target_type="sign",
                source_content=None,
                translated_signs=[],
                confidence_score=0.0,
                message="Could not transcribe audio"
            )
        
        # Convert text to signs
        return await self.text_to_sign(transcribed_text, user_id)
    
    async def text_to_sign(
        self,
        text: str,
        user_id: Optional[UUID] = None
    ) -> TranslationResponse:
        """
        Convert text to GSL signs.
        
        Args:
            text: Input text to translate
            user_id: Optional user ID for tracking
            
        Returns:
            Translation result with GSL signs
        """
        # Check cache
        cache_key = f"text_to_sign:{text.lower()}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Process text with NLP
        processed_text = await self.nlp_processor.process_text(text)
        
        # Extract key words and phrases
        keywords = await self.nlp_processor.extract_keywords(processed_text)
        
        # Map keywords to GSL signs
        signs = await self._map_keywords_to_signs(keywords)
        
        # Handle Ghanaian phrases and idioms
        signs = await self._handle_local_phrases(text, signs)
        
        # Prioritize Harmonized GSL versions
        signs = await self._prioritize_harmonized_signs(signs)
        
        # Save translation
        translation = Translation(
            user_id=user_id,
            source_type="text",
            target_type="sign",
            source_content=text,
            confidence_score=self._calculate_translation_confidence(signs)
        )
        self.db.add(translation)
        self.db.commit()
        
        result = TranslationResponse(
            translation_id=translation.id,
            source_type="text",
            target_type="sign",
            source_content=text,
            translated_signs=signs,
            confidence_score=translation.confidence_score,
            message="Translation successful"
        )
        
        # Cache the result
        await self.cache.set(cache_key, result, ttl=3600)
        
        return result
    
    async def sign_to_text(
        self,
        sign_ids: List[UUID],
        user_id: Optional[UUID] = None
    ) -> TranslationResponse:
        """
        Convert GSL signs to text.
        
        Args:
            sign_ids: List of GSL sign IDs
            user_id: Optional user ID for tracking
            
        Returns:
            Translation result with text
        """
        # Get signs from database
        signs = self.db.query(GSLSign).filter(
            GSLSign.id.in_(sign_ids)
        ).all()
        
        if not signs:
            return TranslationResponse(
                source_type="sign",
                target_type="text",
                source_content=None,
                translated_text="",
                confidence_score=0.0,
                message="No signs found"
            )
        
        # Convert signs to text
        text = await self._signs_to_sentence(signs)
        
        # Save translation
        translation = Translation(
            user_id=user_id,
            source_type="sign",
            target_type="text",
            source_content=",".join(str(sid) for sid in sign_ids),
            confidence_score=1.0
        )
        self.db.add(translation)
        self.db.commit()
        
        return TranslationResponse(
            translation_id=translation.id,
            source_type="sign",
            target_type="text",
            source_content=[sign.sign_name for sign in signs],
            translated_text=text,
            confidence_score=1.0,
            message="Translation successful"
        )
    
    async def get_sign_video(self, sign_id: UUID) -> Optional[GSLSignResponse]:
        """
        Get sign demonstration video.
        
        Args:
            sign_id: GSL sign ID
            
        Returns:
            Sign video information
        """
        # Check cache first
        cache_key = f"sign_video:{sign_id}"
        cached_sign = await self.cache.get(cache_key)
        if cached_sign:
            return cached_sign
        
        sign = self.db.query(GSLSign).filter(GSLSign.id == sign_id).first()
        
        if not sign:
            return None
        
        sign_response = GSLSignResponse.from_orm(sign)
        
        # Cache the result
        await self.cache.set(cache_key, sign_response, ttl=7200)
        
        return sign_response
    
    async def get_sign_variations(self, sign_id: UUID) -> List[GSLSignResponse]:
        """
        Get variations of a sign, prioritizing Harmonized GSL.
        
        Args:
            sign_id: GSL sign ID
            
        Returns:
            List of sign variations
        """
        sign = self.db.query(GSLSign).filter(GSLSign.id == sign_id).first()
        
        if not sign:
            return []
        
        # Get related signs
        variations = []
        if sign.related_signs:
            related = self.db.query(GSLSign).filter(
                GSLSign.id.in_(sign.related_signs)
            ).all()
            variations = [GSLSignResponse.from_orm(s) for s in related]
        
        # Sort by harmonized status (if we have that field)
        # Harmonized GSL versions should come first
        # For now, return as-is
        return variations
    
    async def create_sign_sequence(
        self,
        text: str,
        signs: List[GSLSign],
        user_id: Optional[UUID] = None
    ) -> SignSequenceResponse:
        """
        Create an animated sign sequence with timing and transitions.
        
        Args:
            text: Original text
            signs: List of GSL signs
            user_id: Optional user ID
            
        Returns:
            Sign sequence with timing information
        """
        if not signs:
            return SignSequenceResponse(
                original_text=text,
                signs=[],
                total_signs=0,
                estimated_duration=0.0,
                transitions=[],
                grammar_applied=False
            )
        
        # Calculate timing for each sign (average 2 seconds per sign)
        sign_duration = 2.0
        transition_duration = 0.5
        
        transitions = []
        current_time = 0.0
        
        for i, sign in enumerate(signs):
            transition = {
                "sign_index": i,
                "start_time": current_time,
                "duration": sign_duration,
                "transition_type": "smooth" if i > 0 else "none",
                "transition_duration": transition_duration if i > 0 else 0.0
            }
            transitions.append(transition)
            current_time += sign_duration + (transition_duration if i < len(signs) - 1 else 0)
        
        estimated_duration = current_time
        
        # Save sequence to database
        sign_sequence = SignSequence(
            original_text=text,
            sign_ids=[str(sign.id) for sign in signs],
            transitions=transitions,
            estimated_duration=estimated_duration,
            grammar_applied="harmonized_gsl"
        )
        self.db.add(sign_sequence)
        self.db.commit()
        self.db.refresh(sign_sequence)
        
        # Convert signs to response format
        sign_responses = [GSLSignResponse.from_orm(sign) for sign in signs]
        
        return SignSequenceResponse(
            original_text=text,
            signs=sign_responses,
            total_signs=len(signs),
            estimated_duration=estimated_duration,
            transitions=transitions,
            grammar_applied=True
        )
    
    async def get_multiple_sign_videos(
        self,
        sign_ids: List[UUID]
    ) -> List[GSLSignResponse]:
        """
        Get multiple sign demonstration videos.
        
        Args:
            sign_ids: List of GSL sign IDs
            
        Returns:
            List of sign video information
        """
        signs = self.db.query(GSLSign).filter(
            GSLSign.id.in_(sign_ids)
        ).all()
        
        return [GSLSignResponse.from_orm(sign) for sign in signs]
    
    async def _map_keywords_to_signs(self, keywords: List[str]) -> List[GSLSign]:
        """Map extracted keywords to GSL signs."""
        signs = []
        
        for keyword in keywords:
            # Search for exact match first
            sign = self.db.query(GSLSign).filter(
                GSLSign.sign_name.ilike(keyword)
            ).first()
            
            if sign:
                signs.append(sign)
                continue
            
            # Search in descriptions and usage examples
            sign = self.db.query(GSLSign).filter(
                (GSLSign.description.ilike(f"%{keyword}%")) |
                (GSLSign.usage_examples.contains([keyword]))
            ).first()
            
            if sign:
                signs.append(sign)
        
        return signs
    
    async def _handle_local_phrases(
        self,
        text: str,
        signs: List[GSLSign]
    ) -> List[GSLSign]:
        """Handle Ghanaian English phrases and local idioms."""
        # Common Ghanaian phrases mapping
        ghanaian_phrases = {
            "how far": ["greeting", "hello"],
            "chale": ["friend", "buddy"],
            "small small": ["gradually", "slowly"],
            "by force": ["mandatory", "must"],
            "i beg": ["please", "excuse me"]
        }
        
        text_lower = text.lower()
        for phrase, keywords in ghanaian_phrases.items():
            if phrase in text_lower:
                # Add signs for the phrase
                for keyword in keywords:
                    sign = self.db.query(GSLSign).filter(
                        GSLSign.sign_name.ilike(keyword)
                    ).first()
                    if sign and sign not in signs:
                        signs.append(sign)
        
        return signs
    
    async def _prioritize_harmonized_signs(
        self,
        signs: List[GSLSign]
    ) -> List[GSLSign]:
        """Prioritize Harmonized GSL versions over regional variations."""
        # TODO: Implement harmonized sign prioritization
        # This would check if a sign has a 'harmonized' flag or similar
        return signs
    
    async def _signs_to_sentence(self, signs: List[GSLSign]) -> str:
        """Convert list of signs to a coherent sentence."""
        if not signs:
            return ""
        
        # Simple concatenation for now
        # TODO: Implement proper grammar and sentence structure
        words = [sign.sign_name for sign in signs]
        return " ".join(words).capitalize() + "."
    
    def _calculate_translation_confidence(self, signs: List[GSLSign]) -> float:
        """Calculate confidence score for translation."""
        if not signs:
            return 0.0
        
        # Simple confidence based on number of signs found
        # TODO: Implement more sophisticated confidence calculation
        return min(len(signs) / 10.0, 1.0)