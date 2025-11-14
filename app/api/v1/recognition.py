"""
Sign Recognition API Endpoints

Handles video/image upload for gesture recognition, confidence scoring,
and similar gesture suggestions for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.recognition_service import RecognitionService
from app.schemas.gsl import (
    SignRecognitionRequest, SignRecognitionResponse,
    GestureValidationRequest, GestureValidationResponse,
    BatchRecognitionRequest, BatchRecognitionResponse
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.post("/recognize", response_model=SignRecognitionResponse)
async def recognize_gesture(
    file: UploadFile = File(..., description="Video or image file for gesture recognition"),
    confidence_threshold: Optional[float] = Form(0.7, ge=0.0, le=1.0),
    include_alternatives: bool = Form(True),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload video/image for AI-powered gesture recognition.
    
    - **file**: Video (mp4, mov, avi) or image (jpg, png) file
    - **confidence_threshold**: Minimum confidence score (0.0-1.0)
    - **include_alternatives**: Include alternative gesture matches
    
    Returns recognized sign with confidence score and alternatives.
    Processing time target: < 3 seconds.
    """
    # Validate file type
    if not file.content_type.startswith(('video/', 'image/')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video or image"
        )
    
    # Read file data
    file_data = await file.read()
    
    # Determine media type
    media_type = "video" if file.content_type.startswith('video/') else "image"
    
    # Perform recognition
    recognition_service = RecognitionService(db)
    recognition_service.confidence_threshold = confidence_threshold
    
    try:
        result = await recognition_service.recognize_gesture(
            file_data,
            media_type,
            current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recognition failed: {str(e)}"
        )


@router.get("/confidence/{recognition_id}", response_model=dict)
async def get_confidence_scores(
    recognition_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed confidence scores for a recognition result.
    
    - **recognition_id**: UUID of the recognition result
    
    Returns detailed confidence breakdown and processing metadata.
    """
    recognition_service = RecognitionService(db)
    
    result = await recognition_service.get_confidence_scores(recognition_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recognition result not found"
        )
    
    return result


@router.post("/validate", response_model=GestureValidationResponse)
async def validate_gesture(
    file: UploadFile = File(..., description="User's gesture video/image"),
    expected_sign_id: UUID = Form(..., description="Expected sign ID"),
    confidence_threshold: Optional[float] = Form(0.7, ge=0.0, le=1.0),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Validate user's gesture attempt against expected sign.
    
    Used in learning exercises to check if user performed the correct gesture.
    
    - **file**: User's gesture video/image
    - **expected_sign_id**: The sign the user was supposed to perform
    - **confidence_threshold**: Minimum confidence for validation
    
    Returns validation result with feedback and suggestions.
    """
    # Read file data
    file_data = await file.read()
    
    # Validate gesture
    recognition_service = RecognitionService(db)
    recognition_service.confidence_threshold = confidence_threshold
    
    try:
        result = await recognition_service.validate_gesture(
            file_data,
            expected_sign_id,
            current_user.id
        )
        return GestureValidationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.get("/similar/{gesture}")
async def get_similar_gestures(
    gesture: str,
    limit: int = 5,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get similar gestures for a given gesture name.
    
    Useful for providing suggestions when recognition fails or
    for exploring related signs.
    
    - **gesture**: Name of the gesture
    - **limit**: Maximum number of similar gestures (1-10)
    """
    if limit < 1 or limit > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 10"
        )
    
    recognition_service = RecognitionService(db)
    
    try:
        similar = await recognition_service.get_similar_gestures(gesture, limit)
        return {
            "gesture": gesture,
            "similar_gestures": similar,
            "count": len(similar)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar gestures: {str(e)}"
        )