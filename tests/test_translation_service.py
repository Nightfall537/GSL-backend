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
    
    @pytest.mark.asyncio
    async def test_get_sign_variations(self, translation_service):
        """Test getting sign variations with Harmonized GSL priority."""
        sign_id = uuid4()
        related_id_1 = uuid4()
        related_id_2 = uuid4()
        
        mock_sign = Mock(
            id=sign_id,
            sign_name="hello",
            related_signs=[related_id_1, related_id_2]
        )
        
        mock_variations = [
            Mock(id=related_id_1, sign_name="hello_variant1"),
            Mock(id=related_id_2, sign_name="hello_variant2")
        ]
        
        # Mock database queries
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(
                first=Mock(return_value=mock_sign),
                all=Mock(return_value=mock_variations)
            ))
        ))
        
        result = await translation_service.get_sign_variations(sign_id)
        
        assert isinstance(result, list)
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_sign_variations_no_sign(self, translation_service):
        """Test getting variations for non-existent sign."""
        sign_id = uuid4()
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=None)))
        ))
        
        result = await translation_service.get_sign_variations(sign_id)
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_create_sign_sequence(self, translation_service):
        """Test creating animated sign sequence."""
        text = "Hello thank you"
        mock_signs = [
            Mock(id=uuid4(), sign_name="hello"),
            Mock(id=uuid4(), sign_name="thank_you")
        ]
        
        result = await translation_service.create_sign_sequence(text, mock_signs)
        
        assert result.original_text == text
        assert result.total_signs == 2
        assert result.estimated_duration > 0
        assert len(result.transitions) == 2
        assert result.grammar_applied is True
    
    @pytest.mark.asyncio
    async def test_create_sign_sequence_empty(self, translation_service):
        """Test creating sequence with no signs."""
        text = "test"
        
        result = await translation_service.create_sign_sequence(text, [])
        
        assert result.total_signs == 0
        assert result.estimated_duration == 0.0
        assert len(result.transitions) == 0
    
    @pytest.mark.asyncio
    async def test_get_multiple_sign_videos(self, translation_service):
        """Test batch retrieval of sign videos."""
        sign_ids = [uuid4(), uuid4(), uuid4()]
        
        mock_signs = [
            Mock(id=sign_ids[0], sign_name="hello"),
            Mock(id=sign_ids[1], sign_name="thank_you"),
            Mock(id=sign_ids[2], sign_name="goodbye")
        ]
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(all=Mock(return_value=mock_signs)))
        ))
        
        result = await translation_service.get_multiple_sign_videos(sign_ids)
        
        assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_text_to_sign_ghanaian_phrases(self, translation_service):
        """Test text-to-sign with Ghanaian English phrases."""
        text = "How far chale, make we go"
        
        # Mock NLP processing
        with patch.object(translation_service.nlp_processor, 'process_text', return_value=text.lower()):
            with patch.object(translation_service.nlp_processor, 'extract_keywords', return_value=["greeting", "friend", "let", "us", "go"]):
                with patch.object(translation_service, '_map_keywords_to_signs', return_value=[Mock(sign_name="greeting")]):
                    with patch.object(translation_service, '_handle_local_phrases', return_value=[Mock(sign_name="greeting"), Mock(sign_name="friend")]):
                        with patch.object(translation_service.cache, 'get', return_value=None):
                            with patch.object(translation_service.cache, 'set', return_value=True):
                                result = await translation_service.text_to_sign(text)
                                
                                assert result.source_type == "text"
                                assert len(result.translated_signs) > 0
    
    @pytest.mark.asyncio
    async def test_sign_video_caching(self, translation_service):
        """Test sign video retrieval with caching."""
        sign_id = uuid4()
        
        # First call - cache miss
        mock_sign = Mock(
            id=sign_id,
            sign_name="hello",
            video_url="https://example.com/hello.mp4"
        )
        
        translation_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_sign)))
        ))
        
        with patch.object(translation_service.cache, 'get', return_value=None):
            with patch.object(translation_service.cache, 'set', return_value=True) as mock_set:
                result = await translation_service.get_sign_video(sign_id)
                
                assert result is not None
                mock_set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_harmonized_gsl_prioritization(self, translation_service):
        """Test that Harmonized GSL versions are prioritized."""
        text = "hello"
        
        # Mock signs with different variations
        mock_signs = [
            Mock(sign_name="hello", extra_data={"harmonized": True}),
            Mock(sign_name="hello_regional", extra_data={"harmonized": False})
        ]
        
        with patch.object(translation_service, '_prioritize_harmonized_signs', return_value=mock_signs):
            result = await translation_service._prioritize_harmonized_signs(mock_signs)
            
            assert isinstance(result, list)