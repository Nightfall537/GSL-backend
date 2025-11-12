"""
Text-to-Sign Model

Converts text input into GSL (Ghanaian Sign Language) sign demonstrations.
Maps text to corresponding sign videos/animations from the GSL dictionary.
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

from app.ai.nlp_processor import NLPProcessor


class TextToSignModel:
    """Model for converting text to GSL sign language."""
    
    def __init__(self, signs_data_path: Optional[str] = None):
        """
        Initialize text-to-sign model.
        
        Args:
            signs_data_path: Path to GSL signs data JSON file
        """
        self.nlp = NLPProcessor()
        self.signs_data_path = signs_data_path or "app/models/colors_signs_data.json"
        self.signs_mapping = self._load_signs_mapping()
        self.phrase_mappings = self._load_phrase_mappings()
    
    def _load_signs_mapping(self) -> Dict[str, Dict]:
        """Load GSL signs mapping from JSON file."""
        try:
            signs_path = Path(self.signs_data_path)
            if signs_path.exists():
                with open(signs_path, 'r') as f:
                    data = json.load(f)
                    return {sign['word'].lower(): sign for sign in data.get('signs', [])}
            return {}
        except Exception as e:
            print(f"Error loading signs mapping: {e}")
            return {}
    
    def _load_phrase_mappings(self) -> Dict[str, List[str]]:
        """Load common phrase to signs mappings."""
        return {
            # Greetings
            "hello": ["hello"],
            "hi": ["hello"],
            "good morning": ["good", "morning"],
            "good afternoon": ["good", "afternoon"],
            "good evening": ["good", "evening"],
            "good night": ["good", "night"],
            
            # Common phrases
            "how are you": ["how", "you"],
            "thank you": ["thank_you"],
            "thanks": ["thank_you"],
            "please": ["please"],
            "sorry": ["sorry"],
            "excuse me": ["excuse", "me"],
            
            # Ghanaian phrases
            "how far": ["hello", "how", "you"],
            "chale": ["friend"],
            "small small": ["slowly", "gradually"],
            "by force": ["must", "mandatory"],
            "i beg": ["please"],
            
            # Questions
            "what is your name": ["what", "your", "name"],
            "where are you from": ["where", "you", "from"],
            "how old are you": ["how", "old", "you"],
            
            # Basic needs
            "i need help": ["i", "need", "help"],
            "i want water": ["i", "want", "water"],
            "i am hungry": ["i", "hungry"],
            "i am thirsty": ["i", "thirsty"],
        }
    
    async def convert_text_to_signs(
        self,
        text: str,
        include_fingerspelling: bool = False
    ) -> List[Dict]:
        """
        Convert text to GSL signs.
        
        Args:
            text: Input text to convert
            include_fingerspelling: Whether to include fingerspelling for unknown words
            
        Returns:
            List of sign dictionaries with video URLs and metadata
        """
        # Normalize text
        normalized_text = await self.nlp.process_text(text)
        
        # Check for exact phrase matches first
        signs = await self._match_phrases(normalized_text)
        
        if not signs:
            # Extract keywords and map to signs
            keywords = await self.nlp.extract_keywords(text)
            signs = await self._map_keywords_to_signs(keywords)
        
        # Add fingerspelling for unknown words if requested
        if include_fingerspelling:
            signs = await self._add_fingerspelling(signs, text)
        
        return signs
    
    async def _match_phrases(self, text: str) -> List[Dict]:
        """Match complete phrases to sign sequences."""
        text_lower = text.lower()
        
        # Check for phrase matches (longest first)
        sorted_phrases = sorted(
            self.phrase_mappings.keys(),
            key=len,
            reverse=True
        )
        
        for phrase in sorted_phrases:
            if phrase in text_lower:
                # Get signs for this phrase
                sign_words = self.phrase_mappings[phrase]
                signs = []
                
                for word in sign_words:
                    sign_data = self.signs_mapping.get(word)
                    if sign_data:
                        signs.append(sign_data)
                    else:
                        # Create placeholder for missing sign
                        signs.append({
                            "word": word,
                            "video_url": None,
                            "description": f"Sign for '{word}'",
                            "category": "unknown"
                        })
                
                return signs
        
        return []
    
    async def _map_keywords_to_signs(self, keywords: List[str]) -> List[Dict]:
        """Map keywords to GSL signs."""
        signs = []
        
        for keyword in keywords:
            # Direct match
            sign_data = self.signs_mapping.get(keyword.lower())
            
            if sign_data:
                signs.append(sign_data)
            else:
                # Try variations
                sign_data = await self._find_sign_variation(keyword)
                if sign_data:
                    signs.append(sign_data)
                else:
                    # Create placeholder
                    signs.append({
                        "word": keyword,
                        "video_url": None,
                        "description": f"Sign for '{keyword}' not found",
                        "category": "unknown",
                        "needs_fingerspelling": True
                    })
        
        return signs
    
    async def _find_sign_variation(self, word: str) -> Optional[Dict]:
        """Find sign variations (plural, tense, etc.)."""
        # Try common variations
        variations = [
            word.rstrip('s'),  # Remove plural
            word.rstrip('ing'),  # Remove -ing
            word.rstrip('ed'),  # Remove -ed
            word.rstrip('ly'),  # Remove -ly
        ]
        
        for variation in variations:
            if variation in self.signs_mapping:
                return self.signs_mapping[variation]
        
        return None
    
    async def _add_fingerspelling(
        self,
        signs: List[Dict],
        original_text: str
    ) -> List[Dict]:
        """Add fingerspelling for unknown words."""
        enhanced_signs = []
        
        for sign in signs:
            enhanced_signs.append(sign)
            
            # If sign not found, add fingerspelling
            if sign.get('needs_fingerspelling'):
                word = sign['word']
                for letter in word:
                    if letter.isalpha():
                        enhanced_signs.append({
                            "word": letter.upper(),
                            "type": "fingerspelling",
                            "video_url": f"/static/fingerspelling/{letter.lower()}.mp4",
                            "description": f"Fingerspell letter '{letter.upper()}'",
                            "category": "alphabet"
                        })
        
        return enhanced_signs
    
    def get_sign_by_word(self, word: str) -> Optional[Dict]:
        """Get sign data for a specific word."""
        return self.signs_mapping.get(word.lower())
    
    def get_available_signs(self) -> List[str]:
        """Get list of all available sign words."""
        return list(self.signs_mapping.keys())
    
    def get_signs_by_category(self, category: str) -> List[Dict]:
        """Get all signs in a specific category."""
        return [
            sign for sign in self.signs_mapping.values()
            if sign.get('category', '').lower() == category.lower()
        ]
    
    async def generate_sign_sequence(
        self,
        text: str,
        grammar_rules: bool = True
    ) -> Dict:
        """
        Generate complete sign sequence with timing and transitions.
        
        Args:
            text: Input text
            grammar_rules: Apply GSL grammar rules
            
        Returns:
            Dictionary with sign sequence and metadata
        """
        signs = await self.convert_text_to_signs(text)
        
        # Apply GSL grammar rules if requested
        if grammar_rules:
            signs = await self._apply_gsl_grammar(signs)
        
        # Calculate timing
        sequence = {
            "original_text": text,
            "signs": signs,
            "total_signs": len(signs),
            "estimated_duration": len(signs) * 2,  # 2 seconds per sign
            "transitions": await self._generate_transitions(signs)
        }
        
        return sequence
    
    async def _apply_gsl_grammar(self, signs: List[Dict]) -> List[Dict]:
        """
        Apply Ghanaian Sign Language grammar rules.
        
        GSL grammar may differ from English grammar:
        - Topic-comment structure
        - Time indicators at beginning
        - Question markers
        """
        # TODO: Implement GSL-specific grammar rules
        # For now, return signs as-is
        return signs
    
    async def _generate_transitions(self, signs: List[Dict]) -> List[Dict]:
        """Generate smooth transitions between signs."""
        transitions = []
        
        for i in range(len(signs) - 1):
            current_sign = signs[i]
            next_sign = signs[i + 1]
            
            transitions.append({
                "from": current_sign['word'],
                "to": next_sign['word'],
                "duration": 0.5,  # 500ms transition
                "type": "smooth"
            })
        
        return transitions
    
    def add_custom_sign(
        self,
        word: str,
        video_url: str,
        description: str,
        category: str = "custom"
    ) -> None:
        """Add a custom sign to the mapping."""
        self.signs_mapping[word.lower()] = {
            "word": word,
            "video_url": video_url,
            "description": description,
            "category": category,
            "custom": True
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about available signs."""
        categories = {}
        for sign in self.signs_mapping.values():
            category = sign.get('category', 'uncategorized')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_signs": len(self.signs_mapping),
            "categories": categories,
            "phrases": len(self.phrase_mappings)
        }


# Singleton instance
_text_to_sign_model = None


def get_text_to_sign_model() -> TextToSignModel:
    """Get singleton instance of TextToSignModel."""
    global _text_to_sign_model
    if _text_to_sign_model is None:
        _text_to_sign_model = TextToSignModel()
    return _text_to_sign_model