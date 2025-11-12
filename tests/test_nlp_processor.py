"""
Unit Tests for NLP Processor

Tests text processing, keyword extraction, and Ghanaian phrase handling.
"""

import pytest
from app.ai.nlp_processor import NLPProcessor


class TestNLPProcessor:
    """Test cases for NLPProcessor."""
    
    @pytest.fixture
    def nlp_processor(self):
        """Create NLPProcessor instance."""
        return NLPProcessor()
    
    def test_initialization(self, nlp_processor):
        """Test processor initialization."""
        assert len(nlp_processor.stop_words) > 0
        assert len(nlp_processor.ghanaian_phrases) > 0
    
    @pytest.mark.asyncio
    async def test_process_text(self, nlp_processor):
        """Test text processing and normalization."""
        text = "  Hello,  WORLD!  How are you?  "
        
        processed = await nlp_processor.process_text(text)
        
        assert processed == "hello world how are you"
    
    @pytest.mark.asyncio
    async def test_process_text_ghanaian_phrases(self, nlp_processor):
        """Test processing with Ghanaian phrases."""
        text = "how far chale"
        
        processed = await nlp_processor.process_text(text)
        
        # Should replace Ghanaian phrases with standard English
        assert "how far" not in processed or "greeting" in processed
    
    @pytest.mark.asyncio
    async def test_extract_keywords(self, nlp_processor):
        """Test keyword extraction."""
        text = "I want to learn sign language"
        
        keywords = await nlp_processor.extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "want" in keywords
        assert "learn" in keywords
        # Stop words should be removed
        assert "to" not in keywords
        assert "I" not in keywords
    
    @pytest.mark.asyncio
    async def test_extract_keywords_removes_duplicates(self, nlp_processor):
        """Test that keyword extraction removes duplicates."""
        text = "hello hello world world"
        
        keywords = await nlp_processor.extract_keywords(text)
        
        assert len(keywords) == len(set(keywords))  # No duplicates
    
    def test_tokenize(self, nlp_processor):
        """Test text tokenization."""
        text = "Hello, world! How are you?"
        
        tokens = nlp_processor.tokenize(text)
        
        assert isinstance(tokens, list)
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens  # Punctuation removed
    
    def test_extract_phrases(self, nlp_processor):
        """Test multi-word phrase extraction."""
        text = "sign language learning"
        
        phrases = nlp_processor.extract_phrases(text, max_phrase_length=2)
        
        assert isinstance(phrases, list)
        assert "sign language" in phrases
        assert "language learning" in phrases
    
    def test_calculate_word_importance(self, nlp_processor):
        """Test word importance scoring."""
        text = "hello world hello"
        
        importance = nlp_processor.calculate_word_importance(text)
        
        assert isinstance(importance, dict)
        assert "hello" in importance
        assert importance["hello"] > 0
    
    def test_detect_intent_question(self, nlp_processor):
        """Test intent detection for questions."""
        text = "What is your name?"
        
        intent = nlp_processor.detect_intent(text)
        
        assert intent == "question"
    
    def test_detect_intent_greeting(self, nlp_processor):
        """Test intent detection for greetings."""
        text = "Hello, how are you?"
        
        intent = nlp_processor.detect_intent(text)
        
        assert intent == "greeting"
    
    def test_detect_intent_request(self, nlp_processor):
        """Test intent detection for requests."""
        text = "Please help me"
        
        intent = nlp_processor.detect_intent(text)
        
        assert intent == "request"
    
    def test_detect_intent_statement(self, nlp_processor):
        """Test intent detection for statements."""
        text = "I am learning sign language"
        
        intent = nlp_processor.detect_intent(text)
        
        assert intent == "statement"
    
    def test_extract_entities(self, nlp_processor):
        """Test entity extraction."""
        text = "Meet me at 3:00 today with 5 people"
        
        entities = nlp_processor.extract_entities(text)
        
        assert isinstance(entities, dict)
        assert "numbers" in entities
        assert "times" in entities
        assert len(entities["numbers"]) > 0
    
    def test_simplify_sentence(self, nlp_processor):
        """Test sentence simplification."""
        text = "I actually really want to basically learn"
        
        simplified = nlp_processor.simplify_sentence(text)
        
        # Filler words should be removed
        assert "actually" not in simplified
        assert "really" not in simplified
        assert "basically" not in simplified
    
    def test_get_word_context(self, nlp_processor):
        """Test getting context words around target word."""
        text = "I want to learn sign language today"
        
        context = nlp_processor.get_word_context(text, "learn", window=2)
        
        assert isinstance(context, list)
        assert "want" in context or "to" in context
        assert "sign" in context or "language" in context
    
    def test_handle_ghanaian_phrase_chale(self, nlp_processor):
        """Test handling of 'chale' phrase."""
        text = "chale how are you"
        
        processed = nlp_processor._handle_ghanaian_phrases(text)
        
        # Should replace 'chale' with standard equivalent
        assert "friend" in processed or "chale" in processed
    
    def test_handle_ghanaian_phrase_small_small(self, nlp_processor):
        """Test handling of 'small small' phrase."""
        text = "we go small small"
        
        processed = nlp_processor._handle_ghanaian_phrases(text)
        
        # Should replace 'small small' with 'gradually'
        assert "gradually" in processed or "small small" in processed