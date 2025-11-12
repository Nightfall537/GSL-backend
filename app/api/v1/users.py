"""
User Management API Endpoints

Handles user registration, authentication, profile management, and learning progress tracking
for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.user_service import UserService
from app.schemas.user import (
    UserCreate, UserLogin, UserUpdate, PasswordChange,
    UserResponse, UserLoginResponse, UserRegistrationResponse,
    UserProfile, UserStatistics
)
from app.schemas.common import SuccessResponse, ErrorResponse

router = APIRouter()


@router.post("/register", response_model=UserRegistrationResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Create new learner account.
    
    - **username**: Unique username (3-50 characters)
    - **email**: Valid email address
    - **password**: Strong password (min 8 characters, must include uppercase, lowercase, and digit)
    - **full_name**: Optional full name
    - **age_group**: Age group (child, teen, adult, senior)
    - **learning_level**: Initial learning level (beginner, intermediate, advanced)
    """
    user_service = UserService(db)
    
    try:
        user = user_service.register_user(user_data)
        access_token = user_service.create_access_token(user.id, user.username)
        
        return UserRegistrationResponse(
            user=UserResponse.from_orm(user),
            access_token=access_token,
            message="Registration successful"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=UserLoginResponse)
async def login_user(
    login_data: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token.
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns access token for authenticated requests.
    """
    user_service = UserService(db)
    
    user = user_service.authenticate_user(login_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    access_token = user_service.create_access_token(user.id, user.username)
    
    return UserLoginResponse(
        user=UserResponse.from_orm(user),
        access_token=access_token
    )


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user's profile and learning progress.
    
    Requires authentication.
    """
    user_service = UserService(db)
    user = user_service.get_user_by_id(current_user.id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserProfile.from_orm(user)


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    profile_data: UserUpdate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update user profile information.
    
    Requires authentication. Users can only update their own profile.
    """
    user_service = UserService(db)
    
    try:
        updated_user = user_service.update_user_profile(current_user.id, profile_data)
        return UserResponse.from_orm(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/statistics", response_model=UserStatistics)
async def get_user_statistics(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive user learning statistics.
    
    Includes lessons completed, achievements, practice sessions, and more.
    """
    user_service = UserService(db)
    stats = user_service.get_user_statistics(current_user.id)
    
    return UserStatistics(**stats)


@router.post("/password/change", response_model=SuccessResponse)
async def change_password(
    password_data: PasswordChange,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    
    Requires current password for verification.
    """
    user_service = UserService(db)
    user = user_service.get_user_by_id(current_user.id)
    
    if not user_service.verify_password(password_data.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    user.password_hash = user_service.hash_password(password_data.new_password)
    db.commit()
    
    return SuccessResponse(message="Password changed successfully")