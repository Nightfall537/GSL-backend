"""
Unit Tests for Translation Service

Tests speech-to-sign, text-to-sign, and sign-to-text translation.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4

from app.services.translation_service import TranslationService


class TestTranslationService:
    """Test cases for TranslationService."""
    
    @pytest.fixture
    def translation_service(self, test_db):
        """Create TranslationService instance."""
        return TranslationService(test_db)
    
    @pytest.mark.asyncio
    async def test_speech_to_sign(
        self,
        translation_service,
        sample_audio_bytes
    ):
        """Test speech-to-sign translation."""
        # Mock speech-to-text
        with patch.object(translation_service.stt_model, 'transcribe', return_value="Hello"):
            with patch.object(translation_service, 'text_to_sign') as mock_text_to_sign:
                mock_text_to_sign.return_value = Mock(
                    source_type="text",
                    target_type="sign",
                    translated_signs=[Mock(sign_name="hello")]
                )
                
                result = await translation_service.speech_to_sign(sample_audio_bytes)
                
                assert result is not None
                mock_text_to_sign.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_speech_to_sign_no_transcription(
        self,
        translation_service,
        sample_audio_bytes
    ):
        """Test speech-to-sign with failed transcription."""
        with patch.object(translation_service.stt_model, 'transcribe', return_value=None):
            result = await translation_service.speech_to_sign(sample_audio_bytes)
            
            assert result.source_type == "speech"
            assert result.confidence_score == 0.0
            assert len(result.translated_signs) == 0
    
    @pytest.mark.asyncio
    async def test_text_to_sign(self, translation_service):
        """Test text-to-sign translation."""
        text = "Hello, how are you?"
        
        # Mock NLP processing
        with patch.object(translation_service.nlp_processor, 'process_text', return_value=text.lower()):
            with patch.object(translation_service.nlp_processor, 'extract_keywords', return_value=["hello", "how", "are", "you"]):
                with patch.object(translation_service, '_map_keywords_to_signs', return_value=[Mock(sign_name="hello")]):
                    with patch.object(translation_service.cache, 'get', return_value=None):
                        with patch.object(translation_service.cache, 'set', return_value=True):
                            result = await translation_service.text_to_sign(text)
                            
                            assert result.source_type == "text"
                            assert result.target_type == "sign"
                            assert len(result.translated_signs) > 0
    
    @pytest.mark.asyncio
    async def test_text_to_sign_cached(self, translation_service):
        """Test text-to-sign with cached result."""
        text = "Hello"
        cached_result = Mock(
            source_type="text",
            translated_signs=[Mock(sign_name="hello")]
        )
        
        with patch.object(translation_service.cache, 'get', return_value=cached_result):
            result = await translation_service.text_to_sign(text)
            
            assert result == cached_result
    
    @pytest.mark.asyncio
    async def test_sign_to_text(self, translation_service):
        """Test sign-to-text translation."""
        sign_ids = [uuid4(), uuid4()]
        
        # Mock signs
        mock_signs = [
            Mock(id=sign_ids[0], sign_name="hello"),
            Mock(id=sign_ids[1], sign_name="thank_you")
        ]
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(all=Mock(return_value=mock_signs)))
        ))
        
        with patch.object(translation_service, '_signs_to_sentence', return_value="Hello thank you."):
            result = await translation_service.sign_to_text(sign_ids)
            
            assert result.source_type == "sign"
            assert result.target_type == "text"
            assert result.translated_text == "Hello thank you."
    
    @pytest.mark.asyncio
    async def test_sign_to_text_no_signs(self, translation_service):
        """Test sign-to-text with no signs found."""
        sign_ids = [uuid4()]
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(all=Mock(return_value=[])))
        ))
        
        result = await translation_service.sign_to_text(sign_ids)
        
        assert result.confidence_score == 0.0
        assert result.translated_text == ""
    
    @pytest.mark.asyncio
    async def test_get_sign_video(self, translation_service):
        """Test getting sign demonstration video."""
        sign_id = uuid4()
        
        mock_sign = Mock(
            id=sign_id,
            sign_name="hello",
            video_url="https://example.com/hello.mp4",
            thumbnail_url="https://example.com/hello_thumb.jpg",
            description="Greeting sign",
            usage_examples=["Hello, how are you?"]
        )
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_sign)))
        ))
        
        result = await translation_service.get_sign_video(sign_id)
        
        assert result is not None
        assert result["sign_name"] == "hello"
        assert "video_url" in result
    
    @pytest.mark.asyncio
    async def test_handle_local_phrases(self, translation_service):
        """Test handling of Ghanaian local phrases."""
        text = "how far chale"
        signs = []
        
        # Mock database queries for phrase keywords
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=Mock(sign_name="greeting"))))
        ))
        
        result = await translation_service._handle_local_phrases(text, signs)
        
        assert isinstance(result, list)
    
    def test_calculate_translation_confidence(self, translation_service):
        """Test translation confidence calculation."""
        signs = [Mock(), Mock(), Mock()]
        
        confidence = translation_service._calculate_translation_confidence(signs)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_translation_confidence_empty(self, translation_service):
        """Test confidence calculation with no signs."""
        confidence = translation_service._calculate_translation_confidence([])
        
        assert confidence == 0.0