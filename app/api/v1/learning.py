"""
Learning Service API Endpoints

Handles lesson content, tutorial progression, achievements, and GSL dictionary
for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from uuid import UUID

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.learning_service import LearningService
from app.schemas.learning import (
    LessonSearchRequest, LessonResponse, LessonProgressUpdate,
    LessonProgressResponse, AchievementResponse, PracticeSessionCreate,
    PracticeSessionResponse, LearningAnalytics, ProgressSummary,
    ExerciseSubmission, QuizSubmission, QuizResultResponse
)
from app.schemas.gsl import DictionarySearchRequest, DictionarySearchResponse, GSLSignResponse
from app.schemas.common import PaginatedResponse, SuccessResponse

router = APIRouter()


@router.get("/lessons", response_model=List[LessonResponse])
async def get_lessons(
    level: Optional[int] = Query(None, ge=1, le=3, description="Filter by difficulty level"),
    category: Optional[str] = Query(None, description="Filter by category"),
    completed: Optional[bool] = Query(None, description="Filter by completion status"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get structured lesson content with filtering.
    
    - **level**: Filter by difficulty (1=beginner, 2=intermediate, 3=advanced)
    - **category**: Filter by category (greetings, family, daily_life, etc.)
    - **completed**: Show only completed or incomplete lessons
    - **limit**: Maximum number of lessons to return
    - **offset**: Pagination offset
    
    Returns lessons with progress tracking and lock status.
    """
    learning_service = LearningService(db)
    
    try:
        lessons = await learning_service.get_lessons(
            user_id=current_user.id,
            level=level,
            category=category,
            limit=limit,
            offset=offset
        )
        return lessons
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve lessons: {str(e)}"
        )


@router.get("/lessons/{lesson_id}")
async def get_lesson_details(
    lesson_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed lesson information.
    
    Includes tutorial steps, signs covered, exercises, and user progress.
    
    - **lesson_id**: UUID of the lesson
    """
    learning_service = LearningService(db)
    
    lesson = await learning_service.get_lesson_details(lesson_id, current_user.id)
    if not lesson:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Lesson not found"
        )
    
    return lesson


@router.post("/progress", response_model=LessonProgressResponse)
async def update_lesson_progress(
    progress_data: LessonProgressUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update lesson completion and progress.
    
    - **lesson_id**: Lesson UUID
    - **completed**: Whether lesson is completed
    - **score**: Optional score (0-100)
    - **time_spent**: Time spent in seconds
    - **exercises_completed**: List of completed exercise IDs
    
    Automatically checks for and awards achievements.
    """
    learning_service = LearningService(db)
    
    try:
        result = await learning_service.update_lesson_progress(
            current_user.id,
            progress_data.lesson_id,
            progress_data.completed
        )
        return LessonProgressResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update progress: {str(e)}"
        )


@router.get("/achievements", response_model=List[AchievementResponse])
async def get_user_achievements(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's earned achievements and badges.
    
    Returns all achievements with earned status and progress.
    Includes gamification elements like streaks and milestones.
    """
    learning_service = LearningService(db)
    
    try:
        achievements = await learning_service.get_user_achievements(current_user.id)
        return achievements
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve achievements: {str(e)}"
        )


@router.get("/dictionary", response_model=DictionarySearchResponse)
async def search_gsl_dictionary(
    query: Optional[str] = Query(None, max_length=100, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    difficulty_level: Optional[int] = Query(None, ge=1, le=3, description="Filter by difficulty"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search GSL dictionary.
    
    Comprehensive search across sign names, descriptions, and usage examples.
    
    - **query**: Search text (searches name, description, examples)
    - **category**: Filter by category (greetings, family, colors, etc.)
    - **difficulty_level**: Filter by difficulty (1-3)
    - **limit**: Maximum results
    - **offset**: Pagination offset
    
    Returns matching signs with videos, thumbnails, and metadata.
    """
    learning_service = LearningService(db)
    
    try:
        signs = await learning_service.search_dictionary(
            query=query or "",
            category_id=None,  # TODO: Map category string to UUID
            difficulty_level=difficulty_level,
            limit=limit,
            offset=offset
        )
        
        return DictionarySearchResponse(
            query=query,
            results=[GSLSignResponse.from_orm(sign) for sign in signs],
            total_count=len(signs),
            filters_applied={
                "category": category,
                "difficulty_level": difficulty_level
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dictionary search failed: {str(e)}"
        )


@router.get("/categories")
async def get_sign_categories(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all GSL sign categories.
    
    Returns list of categories for filtering and browsing.
    """
    learning_service = LearningService(db)
    
    try:
        categories = await learning_service.get_sign_categories()
        return {
            "categories": categories,
            "count": len(categories)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve categories: {str(e)}"
        )


@router.post("/practice-session", response_model=PracticeSessionResponse)
async def record_practice_session(
    session_data: PracticeSessionCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Record a practice session for analytics.
    
    - **session_type**: Type of practice (free_practice, lesson_practice, etc.)
    - **lesson_id**: Optional associated lesson
    - **signs_practiced**: List of sign IDs practiced
    - **duration_seconds**: Session duration
    - **accuracy_score**: Optional accuracy score
    
    Used for tracking learning patterns and progress analytics.
    """
    learning_service = LearningService(db)
    
    try:
        session = await learning_service.record_practice_session(
            user_id=current_user.id,
            lesson_id=session_data.lesson_id,
            signs_practiced=session_data.signs_practiced,
            duration_seconds=session_data.duration_seconds
        )
        return PracticeSessionResponse.from_orm(session)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record practice session: {str(e)}"
        )


@router.get("/analytics", response_model=LearningAnalytics)
async def get_learning_analytics(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive learning analytics and insights.
    
    Includes:
    - Completion rates and average scores
    - Practice time and patterns
    - Strengths and areas for improvement
    - Weekly activity breakdown
    - Streak tracking
    """
    learning_service = LearningService(db)
    
    try:
        # TODO: Implement comprehensive analytics
        return LearningAnalytics(
            user_id=current_user.id,
            total_lessons=0,
            completed_lessons=0,
            completion_rate=0.0,
            average_score=0.0,
            total_practice_time=0,
            signs_mastered=0,
            current_streak=0,
            longest_streak=0,
            achievements_earned=0
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@router.get("/progress-summary", response_model=ProgressSummary)
async def get_progress_summary(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get overall progress summary.
    
    Dashboard view with level, XP, achievements, and recent activity.
    """
    learning_service = LearningService(db)
    
    try:
        # TODO: Implement progress summary
        achievements = await learning_service.get_user_achievements(current_user.id)
        
        return ProgressSummary(
            user_id=current_user.id,
            level=1,
            experience_points=0,
            next_level_points=100,
            completion_percentage=0.0,
            lessons_completed=0,
            total_lessons=0,
            achievements=achievements,
            recent_activity=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve progress summary: {str(e)}"
        )