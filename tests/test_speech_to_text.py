"""
Unit Tests for Speech-to-Text Module

Tests audio transcription with Ghanaian English support.
"""

import pytest
from unittest.mock import Mock, patch

from app.ai.speech_to_text import SpeechToTextModel


class TestSpeechToTextModel:
    """Test cases for SpeechToTextModel."""
    
    @pytest.fixture
    def stt_model(self):
        """Create SpeechToTextModel instance."""
        return SpeechToTextModel()
    
    def test_initialization(self, stt_model):
        """Test model initialization."""
        assert stt_model.sample_rate == 16000
        assert stt_model.model_name is not None
    
    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self, stt_model, sample_audio_bytes):
        """Test that transcribe returns text."""
        result = await stt_model.transcribe(sample_audio_bytes)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, stt_model, sample_audio_bytes):
        """Test transcription with language parameter."""
        result = await stt_model.transcribe(
            sample_audio_bytes,
            language="en"
        )
        
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_transcribe_with_ghanaian_accent(
        self,
        stt_model,
        sample_audio_bytes
    ):
        """Test transcription optimized for Ghanaian accent."""
        result = await stt_model.transcribe(
            sample_audio_bytes,
            language="en",
            accent="ghanaian"
        )
        
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_transcribe_streaming(self, stt_model):
        """Test streaming transcription."""
        audio_stream = Mock()
        
        result = await stt_model.transcribe_streaming(audio_stream)
        
        assert isinstance(result, str)
    
    def test_optimize_for_accent(self, stt_model):
        """Test accent-specific text optimization."""
        text = "how far chale"
        
        optimized = stt_model._optimize_for_accent(text, "ghanaian")
        
        assert isinstance(optimized, str)
    
    def test_detect_language(self, stt_model, sample_audio_bytes):
        """Test language detection."""
        language = stt_model.detect_language(sample_audio_bytes)
        
        assert isinstance(language, str)
        assert len(language) == 2  # Language code like 'en'
    
    def test_get_confidence_score(self, stt_model):
        """Test confidence score extraction."""
        transcription_result = {"text": "Hello", "confidence": 0.85}
        
        confidence = stt_model.get_confidence_score(transcription_result)
        
        assert 0.0 <= confidence <= 1.0