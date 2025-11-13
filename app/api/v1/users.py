"""
User Management API Endpoints

Handles user registration, authentication, profile management, and learning progress tracking
for the GSL learning platform.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import SecurityManager, get_current_user, get_current_active_user
from app.core.rbac import require_admin, require_teacher_or_admin
from app.repositories.user_repository import UserRepository, LearningProgressRepository
from app.schemas.user import (
    UserCreate, UserLogin, UserUpdate, PasswordChange,
    UserResponse, UserLoginResponse, UserRegistrationResponse, UserStatistics
)
from app.db_models.user import User
from app.config.settings import get_settings

settings = get_settings()
router = APIRouter()


@router.post("/register", response_model=UserRegistrationResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Create new learner account.
    
    Creates a new user account with hashed password and initializes
    learning progress tracking.
    """
    user_repo = UserRepository(db)
    
    try:
        # Create user with hashed password
        user = user_repo.create_user(user_data.dict())
        
        # Generate access token
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(minutes=settings.jwt_expire_minutes)
        )
        
        return UserRegistrationResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token,
            message="Registration successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return token.
    
    Validates user credentials and returns JWT access token
    for authenticated requests.
    """
    user_repo = UserRepository(db)
    
    # Authenticate user
    user = user_repo.authenticate_user(login_data.email, login_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    
    # Update last login
    user_repo.update_last_login(user.id)
    
    # Generate access token
    access_token = SecurityManager.create_access_token(
        data={"sub": str(user.id)},
        expires_delta=timedelta(minutes=settings.jwt_expire_minutes)
    )
    
    return UserLoginResponse(
        user=UserResponse.from_orm(user),
        access_token=access_token,
        expires_in=settings.jwt_expire_minutes * 60
    )


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get user profile and basic information.
    
    Returns the current user's profile data excluding sensitive information.
    """
    return UserResponse.from_orm(current_user)


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    profile_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile information.
    
    Updates user profile fields like name, learning level, and preferences.
    """
    user_repo = UserRepository(db)
    
    # Filter out None values
    update_data = {k: v for k, v in profile_data.dict().items() if v is not None}
    
    if not update_data:
        return UserResponse.from_orm(current_user)
    
    updated_user = user_repo.update(current_user.id, update_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(updated_user)


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    
    Validates current password and updates to new password.
    """
    user_repo = UserRepository(db)
    
    # Verify current password
    if not SecurityManager.verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    success = user_repo.change_password(current_user.id, password_data.new_password)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update password"
        )
    
    return {"message": "Password updated successfully"}


@router.get("/progress", response_model=UserStatistics)
async def get_learning_progress(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed learning analytics and progress.
    
    Returns comprehensive learning statistics including lessons completed,
    achievements, practice time, and accuracy metrics.
    """
    progress_repo = LearningProgressRepository(db)
    
    # Get learning progress
    progress = progress_repo.get_by_user_id(current_user.id)
    
    if not progress:
        # Create initial progress if it doesn't exist
        progress = progress_repo.create_for_user(current_user.id)
    
    # Get practice session stats
    from app.repositories.learning_repository import PracticeSessionRepository
    session_repo = PracticeSessionRepository(db)
    session_stats = session_repo.get_user_session_stats(current_user.id)
    
    return UserStatistics(
        total_lessons_completed=progress.total_lessons_completed,
        current_level=progress.current_level,
        achievements_count=len(progress.achievements or []),
        practice_sessions=session_stats["total_sessions"],
        days_active=progress.days_active,
        signs_learned=progress.signs_learned,
        accuracy_rate=progress.accuracy_rate,
        last_activity=progress.last_activity
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user information.
    
    Alias for /profile endpoint for convenience.
    """
    return UserResponse.from_orm(current_user)


@router.get("/achievements")
async def get_user_achievements(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user's earned achievements and available achievements.
    
    Returns all achievements earned by the user and progress towards
    achievements that can still be earned.
    """
    from app.services.achievement_service import AchievementService
    
    achievement_service = AchievementService(db)
    
    # Check for newly earned achievements
    newly_awarded = achievement_service.check_and_award_achievements(current_user.id)
    
    # Get earned achievements
    earned = achievement_service.get_user_achievements(current_user.id)
    
    # Get available achievements with progress
    available = achievement_service.get_available_achievements(current_user.id)
    
    return {
        "earned_achievements": [
            {
                "id": str(a.id),
                "name": a.name,
                "description": a.description,
                "type": a.type,
                "points": a.points,
                "icon_url": a.icon_url,
            }
            for a in earned
        ],
        "available_achievements": [
            {
                "id": str(a["achievement"].id),
                "name": a["achievement"].name,
                "description": a["achievement"].description,
                "type": a["achievement"].type,
                "points": a["achievement"].points,
                "progress_percentage": a["progress_percentage"],
            }
            for a in available
        ],
        "newly_awarded": [
            {
                "id": str(a.id),
                "name": a.name,
                "description": a.description,
            }
            for a in newly_awarded
        ],
    }


@router.get("/analytics")
async def get_user_analytics(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive learning analytics and insights.
    
    Returns detailed analytics including performance trends,
    learning patterns, and personalized recommendations.
    """
    from app.services.analytics_service import AnalyticsService
    
    analytics_service = AnalyticsService(db)
    
    analytics = analytics_service.get_user_analytics(current_user.id)
    patterns = analytics_service.get_learning_patterns(current_user.id)
    trends = analytics_service.get_performance_trends(current_user.id, days=30)
    recommendations = analytics_service.get_recommendations(current_user.id)
    
    return {
        "analytics": analytics,
        "learning_patterns": patterns,
        "performance_trends": trends,
        "recommendations": recommendations,
    }


@router.post("/practice-session")
async def record_practice_session(
    session_data: dict,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Record a completed practice session.
    
    Updates user's learning progress and checks for newly earned achievements.
    """
    from app.repositories.learning_repository import PracticeSessionRepository
    from app.services.achievement_service import AchievementService
    
    session_repo = PracticeSessionRepository(db)
    progress_repo = LearningProgressRepository(db)
    achievement_service = AchievementService(db)
    
    # Create practice session record
    session = session_repo.create({
        "user_id": current_user.id,
        "session_type": session_data.get("session_type", "free_practice"),
        "lesson_id": session_data.get("lesson_id"),
        "signs_practiced": session_data.get("signs_practiced", []),
        "duration_seconds": session_data.get("duration_seconds", 0),
        "accuracy_score": session_data.get("accuracy_score"),
        "notes": session_data.get("notes"),
    })
    
    # Update learning progress
    progress_repo.update_practice_session(
        current_user.id,
        duration_minutes=session_data.get("duration_seconds", 0) // 60,
        accuracy=session_data.get("accuracy_score"),
        signs_practiced=len(session_data.get("signs_practiced", [])),
    )
    
    # Check for achievements
    newly_awarded = achievement_service.check_and_award_achievements(current_user.id)
    
    return {
        "session_id": str(session.id),
        "message": "Practice session recorded successfully",
        "newly_awarded_achievements": [
            {"id": str(a.id), "name": a.name}
            for a in newly_awarded
        ],
    }


@router.post("/lesson-complete/{lesson_id}")
async def complete_lesson(
    lesson_id: str,
    score: float = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Mark a lesson as completed.
    
    Updates user's learning progress and awards experience points.
    """
    from uuid import UUID
    from app.services.achievement_service import AchievementService
    
    progress_repo = LearningProgressRepository(db)
    achievement_service = AchievementService(db)
    
    # Update lesson completion
    progress = progress_repo.update_lesson_completion(
        current_user.id,
        UUID(lesson_id),
        score=score,
    )
    
    if not progress:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Learning progress not found"
        )
    
    # Check for achievements
    newly_awarded = achievement_service.check_and_award_achievements(current_user.id)
    
    return {
        "message": "Lesson completed successfully",
        "experience_points": progress.experience_points,
        "current_level": progress.current_level,
        "newly_awarded_achievements": [
            {"id": str(a.id), "name": a.name}
            for a in newly_awarded
        ],
    }



# Admin endpoints

@router.get("/admin/users", dependencies=[Depends(require_admin)])
async def list_all_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all users (admin only).
    
    Returns paginated list of all users in the system.
    """
    user_repo = UserRepository(db)
    users = user_repo.get_multi(skip=skip, limit=limit)
    
    return {
        "users": [UserResponse.from_orm(user) for user in users],
        "total": user_repo.count(),
        "skip": skip,
        "limit": limit,
    }


@router.get("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def get_user_by_id(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Get user details by ID (admin only).
    
    Returns detailed user information including progress and statistics.
    """
    from uuid import UUID
    
    user_repo = UserRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    user = user_repo.get(UUID(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    progress = progress_repo.get_by_user_id(UUID(user_id))
    
    return {
        "user": UserResponse.from_orm(user),
        "progress": {
            "total_lessons_completed": progress.total_lessons_completed if progress else 0,
            "current_level": progress.current_level if progress else 1,
            "experience_points": progress.experience_points if progress else 0,
            "signs_learned": progress.signs_learned if progress else 0,
        } if progress else None,
    }


@router.put("/admin/users/{user_id}/role", dependencies=[Depends(require_admin)])
async def update_user_role(
    user_id: str,
    role_data: dict,
    db: Session = Depends(get_db)
):
    """
    Update user role (admin only).
    
    Allows admin to change user roles (learner, teacher, admin).
    """
    from uuid import UUID
    from app.db_models.user import UserRole
    
    user_repo = UserRepository(db)
    
    try:
        new_role = UserRole(role_data.get("role"))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid role. Must be: learner, teacher, or admin"
        )
    
    user = user_repo.update(UUID(user_id), {"role": new_role})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "message": "User role updated successfully",
        "user": UserResponse.from_orm(user),
    }


@router.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def deactivate_user(
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    Deactivate user account (admin only).
    
    Soft deletes user by setting is_active to False.
    """
    from uuid import UUID
    
    user_repo = UserRepository(db)
    
    success = user_repo.deactivate_user(UUID(user_id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {"message": "User deactivated successfully"}


# Teacher endpoints

@router.get("/teacher/students", dependencies=[Depends(require_teacher_or_admin)])
async def get_students_progress(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get students' learning progress (teacher/admin only).
    
    Returns list of learners with their progress statistics.
    """
    from app.db_models.user import UserRole
    
    user_repo = UserRepository(db)
    progress_repo = LearningProgressRepository(db)
    
    # Get all learners
    learners = user_repo.get_multi(
        skip=skip,
        limit=limit,
        filters={"role": UserRole.learner, "is_active": True}
    )
    
    students_data = []
    for learner in learners:
        progress = progress_repo.get_by_user_id(learner.id)
        students_data.append({
            "user": UserResponse.from_orm(learner),
            "progress": {
                "total_lessons_completed": progress.total_lessons_completed if progress else 0,
                "current_level": progress.current_level if progress else 1,
                "accuracy_rate": progress.accuracy_rate if progress else 0.0,
                "days_active": progress.days_active if progress else 0,
                "last_activity": progress.last_activity if progress else None,
            } if progress else None,
        })
    
    return {
        "students": students_data,
        "total": user_repo.count(filters={"role": UserRole.learner}),
    }
