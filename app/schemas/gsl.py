"""
GSL (Ghanaian Sign Language) Schemas

Pydantic models for GSL-related API requests and responses.
Handles validation for sign recognition, translation, and GSL dictionary operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class SignCategory(str, Enum):
    """GSL sign categories."""
    greetings = "greetings"
    family = "family"
    colors = "colors"
    animals = "animals"
    food = "food"
    numbers = "numbers"
    emotions = "emotions"
    actions = "actions"
    objects = "objects"
    places = "places"
    time = "time"
    weather = "weather"
    clothing = "clothing"
    body_parts = "body_parts"
    grammar = "grammar"
    custom = "custom"


class DifficultyLevel(int, Enum):
    """Sign difficulty levels."""
    beginner = 1
    intermediate = 2
    advanced = 3


class RecognitionStatus(str, Enum):
    """Recognition result status."""
    success = "success"
    low_confidence = "low_confidence"
    failed = "failed"
    processing = "processing"


class TranslationType(str, Enum):
    """Translation types."""
    speech_to_sign = "speech_to_sign"
    text_to_sign = "text_to_sign"
    sign_to_text = "sign_to_text"


# Request Schemas

class SignRecognitionRequest(BaseModel):
    """Schema for sign recognition request."""
    media_type: str = Field(..., pattern="^(video|image)$", description="Media type: video or image")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_alternatives: bool = Field(True, description="Include alternative matches")
    max_alternatives: int = Field(5, ge=1, le=10, description="Maximum alternative matches")


class GestureValidationRequest(BaseModel):
    """Schema for gesture validation request."""
    expected_sign_id: UUID = Field(..., description="Expected sign ID")
    media_type: str = Field(..., pattern="^(video|image)$")
    confidence_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0)


class SpeechToSignRequest(BaseModel):
    """Schema for speech-to-sign translation."""
    language: str = Field("en", description="Audio language code")
    accent: str = Field("ghanaian", description="Accent type")
    include_fingerspelling: bool = Field(False, description="Include fingerspelling for unknown words")


class TextToSignRequest(BaseModel):
    """Schema for text-to-sign translation."""
    text: str = Field(..., min_length=1, max_length=500, description="Text to translate")
    include_fingerspelling: bool = Field(False, description="Include fingerspelling")
    grammar_rules: bool = Field(True, description="Apply GSL grammar rules")
    simplify_text: bool = Field(True, description="Simplify text before translation")


class SignToTextRequest(BaseModel):
    """Schema for sign-to-text translation."""
    sign_ids: List[UUID] = Field(..., min_length=1, description="List of sign IDs")
    include_grammar: bool = Field(True, description="Apply proper grammar")


class DictionarySearchRequest(BaseModel):
    """Schema for GSL dictionary search."""
    query: Optional[str] = Field(None, max_length=100, description="Search query")
    category: Optional[SignCategory] = Field(None, description="Filter by category")
    difficulty_level: Optional[DifficultyLevel] = Field(None, description="Filter by difficulty")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class SignCreateRequest(BaseModel):
    """Schema for creating a new sign."""
    sign_name: str = Field(..., min_length=1, max_length=100, description="Sign name")
    description: str = Field(..., min_length=1, max_length=500, description="Sign description")
    category: SignCategory = Field(..., description="Sign category")
    difficulty_level: DifficultyLevel = Field(DifficultyLevel.beginner, description="Difficulty level")
    video_url: Optional[str] = Field(None, description="Video demonstration URL")
    thumbnail_url: Optional[str] = Field(None, description="Thumbnail image URL")
    usage_examples: List[str] = Field(default_factory=list, description="Usage examples")
    related_signs: List[UUID] = Field(default_factory=list, description="Related sign IDs")
    
    @validator('sign_name')
    def validate_sign_name(cls, v):
        """Validate sign name format."""
        return v.strip().upper()


# Response Schemas

class GSLSignResponse(BaseModel):
    """Schema for GSL sign data."""
    id: UUID
    sign_name: str
    description: str
    category: SignCategory
    difficulty_level: DifficultyLevel
    video_url: Optional[str]
    thumbnail_url: Optional[str]
    usage_examples: List[str]
    related_signs: List[UUID]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class SignRecognitionResponse(BaseModel):
    """Schema for sign recognition result."""
    recognition_id: UUID
    recognized_sign: Optional[GSLSignResponse]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    alternative_matches: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float = Field(..., description="Processing time in seconds")
    status: RecognitionStatus
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class GestureValidationResponse(BaseModel):
    """Schema for gesture validation result."""
    is_correct: bool
    confidence_score: float
    feedback: str
    recognized_sign: Optional[GSLSignResponse]
    expected_sign_id: UUID
    suggestions: List[str] = Field(default_factory=list)


class TranslationResponse(BaseModel):
    """Schema for translation results."""
    translation_id: Optional[UUID] = None
    source_type: TranslationType
    target_type: str
    source_content: Union[str, List[str], None]
    translated_signs: Optional[List[GSLSignResponse]] = None
    translated_text: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time: Optional[float] = None
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SignSequenceResponse(BaseModel):
    """Schema for sign sequence with timing."""
    original_text: str
    signs: List[GSLSignResponse]
    total_signs: int
    estimated_duration: float = Field(..., description="Duration in seconds")
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    grammar_applied: bool = False


class DictionarySearchResponse(BaseModel):
    """Schema for dictionary search results."""
    query: Optional[str]
    results: List[GSLSignResponse]
    total_count: int
    page_info: Dict[str, Any] = Field(default_factory=dict)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)


class BatchRecognitionRequest(BaseModel):
    """Schema for batch sign recognition."""
    media_items: List[Dict[str, Any]] = Field(..., min_length=1, max_length=10)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    include_alternatives: bool = True


class BatchRecognitionResponse(BaseModel):
    """Schema for batch recognition results."""
    results: List[SignRecognitionResponse]
    total_processed: int
    successful_recognitions: int
    average_confidence: float
    processing_time: float
