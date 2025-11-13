"""
Sign Recognition API Endpoints

Handles video/image upload for gesture recognition, confidence scoring,
and similar gesture suggestions for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import tempfile
from pathlib import Path

from app.core.database import get_db
from app.services.recognition_service import RecognitionService
from app.schemas.gsl import (
    SignRecognitionRequest,
    SignRecognitionResponse,
    GestureValidationRequest,
    GestureValidationResponse
)
from app.utils.file_handler import FileHandler

router = APIRouter()


@router.post("/recognize", response_model=SignRecognitionResponse)
async def recognize_gesture(
    file: UploadFile = File(..., description="Video or image file for gesture recognition"),
    media_type: str = Form(..., pattern="^(video|image)$"),
    confidence_threshold: Optional[float] = Form(0.7),
    include_alternatives: bool = Form(True),
    user_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload video/image for gesture recognition.
    
    Processes uploaded media through AI model to identify GSL gestures.
    Returns recognized sign with confidence score and alternative matches.
    
    Requirements: 2.1, 2.2, 2.3, 2.5
    """
    # Validate file type
    file_handler = FileHandler()
    if not file_handler.validate_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {file_handler.allowed_types}"
        )
    
    # Read file data
    file_data = await file.read()
    
    # Validate file size
    if not file_handler.validate_file_size(len(file_data)):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {file_handler.max_file_size} bytes"
        )
    
    # Parse user_id if provided
    parsed_user_id = None
    if user_id:
        try:
            parsed_user_id = UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user_id format"
            )
    
    # Create recognition service
    recognition_service = RecognitionService(db)
    recognition_service.confidence_threshold = confidence_threshold
    
    try:
        # Recognize gesture
        result = await recognition_service.recognize_gesture(
            media_data=file_data,
            media_type=media_type,
            user_id=parsed_user_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing gesture recognition: {str(e)}"
        )


@router.get("/confidence/{recognition_id}")
async def get_confidence_scores(
    recognition_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get detailed confidence scores for a recognition result.
    
    Returns breakdown of confidence scores and processing metadata
    for a completed recognition operation.
    
    Requirements: 2.2
    """
    recognition_service = RecognitionService(db)
    
    try:
        result = await recognition_service.get_confidence_scores(recognition_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recognition result not found: {recognition_id}"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving confidence scores: {str(e)}"
        )


@router.post("/validate", response_model=GestureValidationResponse)
async def validate_gesture(
    file: UploadFile = File(..., description="User's gesture video/image"),
    expected_sign_id: str = Form(..., description="Expected sign ID"),
    user_id: str = Form(..., description="User ID"),
    media_type: str = Form(..., pattern="^(video|image)$"),
    confidence_threshold: Optional[float] = Form(0.7),
    db: Session = Depends(get_db)
):
    """
    Validate user's gesture attempt against expected sign.
    
    Compares user's performed gesture with the expected sign and provides
    feedback for learning purposes. Returns validation result with suggestions.
    
    Requirements: 2.2, 2.3
    """
    # Validate file type
    file_handler = FileHandler()
    if not file_handler.validate_file_type(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {file_handler.allowed_types}"
        )
    
    # Read file data
    file_data = await file.read()
    
    # Validate file size
    if not file_handler.validate_file_size(len(file_data)):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {file_handler.max_file_size} bytes"
        )
    
    # Parse UUIDs
    try:
        parsed_user_id = UUID(user_id)
        parsed_expected_sign_id = UUID(expected_sign_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid UUID format for user_id or expected_sign_id"
        )
    
    # Create recognition service
    recognition_service = RecognitionService(db)
    recognition_service.confidence_threshold = confidence_threshold
    
    try:
        # Validate gesture
        result = await recognition_service.validate_gesture(
            user_gesture_data=file_data,
            expected_sign_id=parsed_expected_sign_id,
            user_id=parsed_user_id
        )
        
        # Convert to response model
        return GestureValidationResponse(
            is_correct=result["is_correct"],
            confidence_score=result["confidence_score"],
            feedback=result["feedback"],
            recognized_sign=result.get("recognized_sign"),
            expected_sign_id=parsed_expected_sign_id,
            suggestions=[]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating gesture: {str(e)}"
        )


@router.get("/similar/{gesture}")
async def get_similar_gestures(
    gesture: str,
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get similar gestures for a given gesture name.
    
    Returns list of similar signs based on category, difficulty,
    and visual features. Useful for providing suggestions when
    recognition fails or for learning related signs.
    
    Requirements: 2.3
    """
    if limit < 1 or limit > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 20"
        )
    
    recognition_service = RecognitionService(db)
    
    try:
        similar_gestures = await recognition_service.get_similar_gestures(
            gesture_name=gesture,
            limit=limit
        )
        
        return {
            "gesture": gesture,
            "similar_gestures": similar_gestures,
            "count": len(similar_gestures)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding similar gestures: {str(e)}"
        )