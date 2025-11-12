"""
NLP Processor

Handles natural language processing for text analysis, keyword extraction,
and context-aware sign mapping.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import Counter


class NLPProcessor:
    """Natural language processor for text-to-sign translation."""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.ghanaian_phrases = self._load_ghanaian_phrases()
    
    def _load_stop_words(self) -> Set[str]:
        """Load stop words for filtering."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        }
    
    def _load_ghanaian_phrases(self) -> Dict[str, List[str]]:
        """Load Ghanaian English phrases and their meanings."""
        return {
            "how far": ["greeting", "hello", "how are you"],
            "chale": ["friend", "buddy", "mate"],
            "small small": ["gradually", "slowly", "little by little"],
            "by force": ["mandatory", "must", "compulsory"],
            "i beg": ["please", "excuse me", "pardon"],
            "abi": ["right", "isn't it", "correct"],
            "wey": ["which", "that", "who"],
            "dey": ["is", "are", "be"],
            "make we": ["let us", "let's"],
            "no wahala": ["no problem", "no worries"],
            "sharp sharp": ["quickly", "fast", "hurry"],
            "plenty": ["many", "much", "a lot"]
        }
    
    async def process_text(self, text: str) -> str:
        """
        Process and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        # Convert to lowercase
        processed = text.lower()
        
        # Handle Ghanaian phrases
        processed = self._handle_ghanaian_phrases(processed)
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Remove special characters but keep apostrophes
        processed = re.sub(r'[^\w\s\']', '', processed)
        
        return processed
    
    async def extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Tokenize
        words = text.lower().split()
        
        # Remove stop words
        keywords = [
            word for word in words
            if word not in self.stop_words and len(word) > 2
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _handle_ghanaian_phrases(self, text: str) -> str:
        """Replace Ghanaian phrases with standard English equivalents."""
        for phrase, meanings in self.ghanaian_phrases.items():
            if phrase in text:
                # Use the first meaning as replacement
                text = text.replace(phrase, meanings[0])
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def extract_phrases(self, text: str, max_phrase_length: int = 3) -> List[str]:
        """
        Extract multi-word phrases.
        
        Args:
            text: Input text
            max_phrase_length: Maximum words in a phrase
            
        Returns:
            List of phrases
        """
        words = self.tokenize(text)
        phrases = []
        
        for length in range(2, max_phrase_length + 1):
            for i in range(len(words) - length + 1):
                phrase = ' '.join(words[i:i + length])
                phrases.append(phrase)
        
        return phrases
    
    def calculate_word_importance(self, text: str) -> Dict[str, float]:
        """
        Calculate importance score for each word.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of word importance scores
        """
        words = self.tokenize(text)
        word_freq = Counter(words)
        
        # Calculate TF (term frequency)
        total_words = len(words)
        importance = {}
        
        for word, freq in word_freq.items():
            if word not in self.stop_words:
                # Simple importance based on frequency and length
                tf = freq / total_words
                length_bonus = min(len(word) / 10, 0.5)
                importance[word] = tf + length_bonus
        
        return importance
    
    def detect_intent(self, text: str) -> str:
        """
        Detect user intent from text.
        
        Args:
            text: Input text
            
        Returns:
            Detected intent
        """
        text_lower = text.lower()
        
        # Question detection
        if any(word in text_lower for word in ['what', 'where', 'when', 'who', 'why', 'how']):
            return "question"
        
        # Greeting detection
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "greeting"
        
        # Request detection
        if any(word in text_lower for word in ['please', 'can you', 'could you', 'would you']):
            return "request"
        
        # Statement
        return "statement"
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and values
        """
        entities = {
            "persons": [],
            "locations": [],
            "times": [],
            "numbers": []
        }
        
        # Simple pattern-based entity extraction
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        entities["numbers"] = numbers
        
        # Time expressions
        time_patterns = [
            r'\b\d{1,2}:\d{2}\b',
            r'\b(morning|afternoon|evening|night)\b',
            r'\b(today|tomorrow|yesterday)\b'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, text.lower())
            entities["times"].extend(matches)
        
        # TODO: Implement more sophisticated NER
        
        return entities
    
    def simplify_sentence(self, text: str) -> str:
        """
        Simplify complex sentences for easier sign mapping.
        
        Args:
            text: Input text
            
        Returns:
            Simplified text
        """
        # Remove filler words
        fillers = ['actually', 'basically', 'literally', 'really', 'very']
        words = text.split()
        simplified_words = [w for w in words if w.lower() not in fillers]
        
        # Join back
        simplified = ' '.join(simplified_words)
        
        return simplified
    
    def get_word_context(self, text: str, target_word: str, window: int = 2) -> List[str]:
        """
        Get context words around a target word.
        
        Args:
            text: Input text
            target_word: Word to find context for
            window: Number of words before and after
            
        Returns:
            List of context words
        """
        words = self.tokenize(text)
        context = []
        
        for i, word in enumerate(words):
            if word == target_word.lower():
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                context.extend(words[start:i] + words[i+1:end])
        
        return context