"""
AI/ML Integration Package

This package contains AI and machine learning model integrations including
computer vision for gesture recognition, speech-to-text processing,
text-to-sign conversion, text-to-speech synthesis, natural language processing,
and Gemma 2:2b LLM for advanced text generation and assistance.

Enhanced models use Gemma for improved results across all AI tasks.
"""

from app.ai.computer_vision import ComputerVisionModel
from app.ai.speech_to_text import SpeechToTextModel
from app.ai.text_to_sign import TextToSignModel, get_text_to_sign_model
from app.ai.text_to_speech import TextToSpeechModel, get_tts_model
from app.ai.nlp_processor import NLPProcessor
from app.ai.gemma_model import GemmaModel, get_gemma_model
from app.ai.enhanced_models import (
    EnhancedNLPProcessor,
    EnhancedTranslationHelper,
    EnhancedLearningHelper,
    EnhancedConversationalAssistant,
    enhance_text_for_signing,
    explain_sign_simply,
    generate_lesson,
    chat_with_assistant
)

__all__ = [
    # Base Models
    "ComputerVisionModel",
    "SpeechToTextModel",
    "TextToSignModel",
    "TextToSpeechModel",
    "NLPProcessor",
    "GemmaModel",
    
    # Model Getters
    "get_text_to_sign_model",
    "get_tts_model",
    "get_gemma_model",
    
    # Enhanced Models (with Gemma)
    "EnhancedNLPProcessor",
    "EnhancedTranslationHelper",
    "EnhancedLearningHelper",
    "EnhancedConversationalAssistant",
    
    # Quick Helper Functions
    "enhance_text_for_signing",
    "explain_sign_simply",
    "generate_lesson",
    "chat_with_assistant",
]