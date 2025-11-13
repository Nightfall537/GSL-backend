"""
Learning Service API Endpoints

Handles lesson content, tutorial progression, achievements, and GSL dictionary
for the GSL learning platform.
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_active_user
from app.repositories.gsl_repository import GSLSignRepository, SignCategoryRepository
from app.repositories.learning_repository import LessonRepository, AchievementRepository, PracticeSessionRepository
from app.repositories.user_repository import LearningProgressRepository
from app.schemas.gsl import (
    GSLSignResponse, DictionarySearchRequest, DictionarySearchResponse,
    SignCategory as SignCategoryEnum, DifficultyLevel
)
from app.schemas.learning import (
    LessonResponse, LessonSearchRequest, LessonProgressUpdate,
    AchievementResponse, PracticeSessionCreate, PracticeSessionResponse
)
from app.db_models.user import User

router = APIRouter()


@router.get("/lessons")
async def get_lessons(
    level: Optional[int] = Query(None, ge=1, le=3, description="Filter by difficulty level"),
    category: Optional[str] = Query(None, description="Filter by lesson category"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get structured lesson content with filtering options.
    
    Returns lessons filtered by level and category with user progress information.
    """
    lesson_repo = LessonRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    # Get user's learning progress
    user_progress = progress_repo.get_by_user_id(current_user.id)
    completed_lessons = user_progress.completed_lessons if user_progress else []
    
    # Search lessons
    lessons = lesson_repo.search_lessons(
        level=level,
        category=category,
        skip=offset,
        limit=limit
    )
    
    # Convert to response with completion status
    lesson_responses = []
    for lesson in lessons:
        lesson_data = LessonResponse.from_orm(lesson)
        lesson_data.is_completed = lesson.id in completed_lessons
        lesson_data.signs_count = len(lesson.signs_covered or [])
        lesson_responses.append(lesson_data)
    
    return {
        "lessons": lesson_responses,
        "total_count": lesson_repo.count(),
        "filters_applied": {
            "level": level,
            "category": category
        }
    }


@router.get("/lessons/{lesson_id}", response_model=LessonResponse)
async def get_lesson_details(
    lesson_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get specific lesson details with user progress.
    
    Returns comprehensive lesson information including signs covered and user completion status.
    """
    lesson_repo = LessonRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    lesson = lesson_repo.get_or_404(lesson_id)
    
    # Get user progress
    user_progress = progress_repo.get_by_user_id(current_user.id)
    completed_lessons = user_progress.completed_lessons if user_progress else []
    
    # Convert to response
    lesson_response = LessonResponse.from_orm(lesson)
    lesson_response.is_completed = lesson.id in completed_lessons
    lesson_response.signs_count = len(lesson.signs_covered or [])
    
    return lesson_response


@router.post("/progress")
async def update_lesson_progress(
    progress_data: LessonProgressUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update lesson completion progress.
    
    Records lesson completion and updates user's learning progress metrics.
    """
    progress_repo = LearningProgressRepository(db)
    
    # Update lesson completion
    updated_progress = progress_repo.update_lesson_completion(
        user_id=current_user.id,
        lesson_id=progress_data.lesson_id,
        score=progress_data.score
    )
    
    if not updated_progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User progress not found"
        )
    
    return {
        "message": "Progress updated successfully",
        "lesson_id": progress_data.lesson_id,
        "completed": progress_data.completed,
        "total_lessons_completed": updated_progress.total_lessons_completed,
        "current_level": updated_progress.current_level
    }


@router.get("/achievements")
async def get_user_achievements(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user badges and achievements.
    
    Returns earned achievements and available achievements to unlock.
    """
    achievement_repo = AchievementRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    # Get user progress
    user_progress = progress_repo.get_by_user_id(current_user.id)
    earned_achievement_ids = user_progress.achievements if user_progress else []
    
    # Get all achievements
    all_achievements = achievement_repo.get_multi()
    
    earned_achievements = []
    available_achievements = []
    
    for achievement in all_achievements:
        achievement_response = AchievementResponse.from_orm(achievement)
        
        if achievement.id in earned_achievement_ids:
            achievement_response.earned_at = user_progress.updated_at  # Placeholder
            earned_achievements.append(achievement_response)
        else:
            # Calculate progress towards achievement (simplified)
            achievement_response.progress = 0.0  # Would need proper calculation
            available_achievements.append(achievement_response)
    
    return {
        "earned_achievements": earned_achievements,
        "available_achievements": available_achievements,
        "total_earned": len(earned_achievements),
        "total_available": len(available_achievements)
    }


@router.post("/practice-sessions", response_model=PracticeSessionResponse)
async def create_practice_session(
    session_data: PracticeSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Record a practice session.
    
    Creates a new practice session record and updates user progress metrics.
    """
    session_repo = PracticeSessionRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    # Create practice session
    session_dict = session_data.dict()
    session_dict["user_id"] = current_user.id
    
    practice_session = session_repo.create(session_dict)
    
    # Update user progress
    progress_repo.update_practice_session(
        user_id=current_user.id,
        duration_minutes=session_data.duration_seconds // 60,
        accuracy=session_data.accuracy_score,
        signs_practiced=len(session_data.signs_practiced)
    )
    
    return PracticeSessionResponse.from_orm(practice_session)


@router.get("/dictionary", response_model=DictionarySearchResponse)
async def search_gsl_dictionary(
    query: Optional[str] = Query(None, description="Search query for sign name or description"),
    category: Optional[str] = Query(None, description="Filter by category name"),
    difficulty_level: Optional[int] = Query(None, ge=1, le=3, description="Filter by difficulty level (1-3)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db)
):
    """
    Search GSL dictionary with filtering options.
    
    Provides comprehensive search functionality for GSL signs with support for:
    - Text search in sign names and descriptions
    - Category-based filtering
    - Difficulty level filtering
    - Pagination support
    """
    sign_repo = GSLSignRepository(db)
    category_repo = SignCategoryRepository(db)
    
    # Convert category name to ID if provided
    category_id = None
    if category:
        category_obj = category_repo.get_by_name(category)
        if category_obj:
            category_id = category_obj.id
        else:
            # Return empty results if category doesn't exist
            return DictionarySearchResponse(
                query=query,
                results=[],
                total_count=0,
                page_info={
                    "limit": limit,
                    "offset": offset,
                    "has_more": False
                },
                filters_applied={
                    "category": category,
                    "difficulty_level": difficulty_level
                }
            )
    
    # Search signs
    signs = sign_repo.search_signs(
        query=query,
        category_id=category_id,
        difficulty_level=difficulty_level,
        skip=offset,
        limit=limit
    )
    
    # Get total count for pagination
    total_count = sign_repo.count(filters={
        k: v for k, v in {
            "category_id": category_id,
            "difficulty_level": difficulty_level
        }.items() if v is not None
    })
    
    # Convert to response models
    sign_responses = [GSLSignResponse.from_orm(sign) for sign in signs]
    
    return DictionarySearchResponse(
        query=query,
        results=sign_responses,
        total_count=total_count,
        page_info={
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(signs) < total_count
        },
        filters_applied={
            "category": category,
            "difficulty_level": difficulty_level
        }
    )


@router.get("/dictionary/categories")
async def get_sign_categories(
    db: Session = Depends(get_db)
):
    """
    Get all available sign categories.
    
    Returns a hierarchical list of sign categories for filtering purposes.
    """
    category_repo = SignCategoryRepository(db)
    
    # Get root categories
    root_categories = category_repo.get_root_categories()
    
    categories_data = []
    for category in root_categories:
        # Get subcategories
        subcategories = category_repo.get_subcategories(category.id)
        
        category_data = {
            "id": category.id,
            "name": category.name,
            "description": category.description,
            "subcategories": [
                {
                    "id": sub.id,
                    "name": sub.name,
                    "description": sub.description
                }
                for sub in subcategories
            ]
        }
        categories_data.append(category_data)
    
    return {"categories": categories_data}


@router.get("/dictionary/signs/{sign_id}", response_model=GSLSignResponse)
async def get_sign_details(
    sign_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific GSL sign.
    
    Returns comprehensive sign data including related signs and usage examples.
    """
    sign_repo = GSLSignRepository(db)
    
    sign = sign_repo.get_or_404(sign_id)
    
    return GSLSignResponse.from_orm(sign)


@router.get("/dictionary/signs/{sign_id}/related")
async def get_related_signs(
    sign_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get signs related to a specific sign.
    
    Returns signs that are semantically or contextually related to the given sign.
    """
    sign_repo = GSLSignRepository(db)
    
    # Verify sign exists
    sign_repo.get_or_404(sign_id)
    
    # Get related signs
    related_signs = sign_repo.get_related_signs(sign_id)
    
    return {
        "sign_id": sign_id,
        "related_signs": [GSLSignResponse.from_orm(sign) for sign in related_signs]
    }


@router.get("/dictionary/popular")
async def get_popular_signs(
    limit: int = Query(10, ge=1, le=50, description="Number of popular signs to return"),
    db: Session = Depends(get_db)
):
    """
    Get popular GSL signs.
    
    Returns commonly used or beginner-friendly signs for quick access.
    """
    sign_repo = GSLSignRepository(db)
    
    popular_signs = sign_repo.get_popular_signs(limit=limit)
    
    return {
        "popular_signs": [GSLSignResponse.from_orm(sign) for sign in popular_signs]
    }