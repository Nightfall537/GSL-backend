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