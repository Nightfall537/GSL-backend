"""
Learning Schemas

Pydantic models for learning-related API requests and responses.
Handles validation for lessons, achievements, practice sessions, and progress tracking.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from enum import Enum


class LessonLevel(int, Enum):
    """Lesson difficulty levels."""
    beginner = 1
    intermediate = 2
    advanced = 3


class LessonCategory(str, Enum):
    """Lesson categories."""
    greetings = "greetings"
    family = "family"
    daily_life = "daily_life"
    emotions = "emotions"
    numbers = "numbers"
    colors = "colors"
    food = "food"
    animals = "animals"
    places = "places"
    actions = "actions"
    grammar = "grammar"
    conversation = "conversation"


class ExerciseType(str, Enum):
    """Types of learning exercises."""
    recognition = "recognition"
    translation = "translation"
    practice = "practice"
    quiz = "quiz"
    matching = "matching"
    sequence = "sequence"


class AchievementType(str, Enum):
    """Types of achievements."""
    lesson_completion = "lesson_completion"
    streak = "streak"
    accuracy = "accuracy"
    practice_time = "practice_time"
    sign_mastery = "sign_mastery"
    level_up = "level_up"


class PracticeSessionType(str, Enum):
    """Types of practice sessions."""
    free_practice = "free_practice"
    lesson_practice = "lesson_practice"
    review_session = "review_session"
    challenge = "challenge"


# Request Schemas

class LessonSearchRequest(BaseModel):
    """Schema for lesson search and filtering."""
    level: Optional[LessonLevel] = Field(None, description="Filter by difficulty level")
    category: Optional[LessonCategory] = Field(None, description="Filter by category")
    completed: Optional[bool] = Field(None, description="Filter by completion status")
    limit: int = Field(20, ge=1, le=100, description="Maximum results")
    offset: int = Field(0, ge=0, description="Pagination offset")


class LessonProgressUpdate(BaseModel):
    """Schema for updating lesson progress."""
    lesson_id: UUID = Field(..., description="Lesson ID")
    completed: bool = Field(..., description="Completion status")
    score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Lesson score (0-100)")
    time_spent: Optional[int] = Field(None, ge=0, description="Time spent in seconds")
    exercises_completed: List[UUID] = Field(default_factory=list, description="Completed exercise IDs")


class PracticeSessionCreate(BaseModel):
    """Schema for creating a practice session."""
    session_type: PracticeSessionType = Field(..., description="Type of practice session")
    lesson_id: Optional[UUID] = Field(None, description="Associated lesson ID")
    signs_practiced: List[UUID] = Field(default_factory=list, description="Signs practiced in session")
    duration_seconds: int = Field(..., ge=0, description="Session duration")
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=100.0, description="Overall accuracy")
    notes: Optional[str] = Field(None, max_length=500, description="Session notes")


class ExerciseSubmission(BaseModel):
    """Schema for exercise submission."""
    exercise_id: UUID = Field(..., description="Exercise ID")
    user_answer: Any = Field(..., description="User's answer (format depends on exercise type)")
    time_taken: int = Field(..., ge=0, description="Time taken in seconds")
    attempts: int = Field(1, ge=1, description="Number of attempts")


class QuizSubmission(BaseModel):
    """Schema for quiz submission."""
    quiz_id: UUID = Field(..., description="Quiz ID")
    answers: List[Dict[str, Any]] = Field(..., description="List of question answers")
    time_taken: int = Field(..., ge=0, description="Total time taken")


class LessonCreateRequest(BaseModel):
    """Schema for creating a new lesson (admin/teacher)."""
    title: str = Field(..., min_length=1, max_length=200, description="Lesson title")
    description: str = Field(..., min_length=1, max_length=1000, description="Lesson description")
    level: LessonLevel = Field(..., description="Difficulty level")
    category: LessonCategory = Field(..., description="Lesson category")
    signs_covered: List[UUID] = Field(..., min_length=1, description="Signs covered in lesson")
    estimated_duration: int = Field(..., ge=1, description="Estimated duration in minutes")
    prerequisites: List[UUID] = Field(default_factory=list, description="Prerequisite lesson IDs")
    learning_objectives: List[str] = Field(..., min_length=1, description="Learning objectives")


# Response Schemas

class ExerciseResponse(BaseModel):
    """Schema for exercise information."""
    id: UUID
    type: ExerciseType
    title: str
    prompt: str
    expected_answer: Optional[Any] = None
    validation_criteria: Dict[str, Any]
    points: int = Field(default=10, description="Points awarded for completion")
    time_limit: Optional[int] = Field(None, description="Time limit in seconds")
    
    class Config:
        from_attributes = True


class LessonResponse(BaseModel):
    """Schema for lesson information."""
    id: UUID
    title: str
    description: str
    level: LessonLevel
    category: LessonCategory
    estimated_duration: int
    signs_count: int
    is_completed: bool = False
    is_locked: bool = False
    completion_percentage: float = Field(0.0, ge=0.0, le=100.0)
    user_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class LessonProgressResponse(BaseModel):
    """Schema for lesson progress information."""
    lesson_id: UUID
    user_id: UUID
    completed: bool = False
    score: Optional[float] = None
    time_spent: int = 0
    exercises_completed: List[UUID] = Field(default_factory=list)
    last_accessed: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    attempts: int = 0
    
    class Config:
        from_attributes = True


class AchievementResponse(BaseModel):
    """Schema for achievement information."""
    id: UUID
    name: str
    description: str
    type: AchievementType
    icon_url: Optional[str]
    points: int
    criteria: Dict[str, Any]
    earned_at: Optional[datetime] = None
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress towards achievement")
    
    class Config:
        from_attributes = True


class PracticeSessionResponse(BaseModel):
    """Schema for practice session information."""
    id: UUID
    user_id: UUID
    session_type: PracticeSessionType
    lesson_id: Optional[UUID]
    signs_practiced: List[UUID]
    duration_seconds: int
    accuracy_score: Optional[float]
    completed_at: datetime
    notes: Optional[str]
    
    class Config:
        from_attributes = True


class QuizResponse(BaseModel):
    """Schema for quiz information."""
    id: UUID
    title: str
    description: str
    lesson_id: Optional[UUID]
    questions: List[Dict[str, Any]]
    time_limit: Optional[int]
    passing_score: float = Field(70.0, description="Minimum score to pass")
    max_attempts: int = Field(3, description="Maximum attempts allowed")
    
    class Config:
        from_attributes = True


class QuizResultResponse(BaseModel):
    """Schema for quiz result information."""
    quiz_id: UUID
    user_id: UUID
    score: float = Field(..., ge=0.0, le=100.0)
    passed: bool
    time_taken: int
    answers: List[Dict[str, Any]]
    completed_at: datetime
    attempt_number: int
    
    class Config:
        from_attributes = True


class LearningAnalytics(BaseModel):
    """Schema for learning analytics and insights."""
    user_id: UUID
    total_lessons: int
    completed_lessons: int
    completion_rate: float = Field(..., ge=0.0, le=100.0)
    average_score: float
    total_practice_time: int
    signs_mastered: int
    current_streak: int
    longest_streak: int
    achievements_earned: int
    weekly_activity: List[Dict[str, Any]] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ProgressSummary(BaseModel):
    """Schema for overall progress summary."""
    user_id: UUID
    level: int
    experience_points: int
    next_level_points: int
    completion_percentage: float
    lessons_completed: int
    total_lessons: int
    achievements: List[AchievementResponse]
    recent_activity: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True
