"""
Translation Service API Endpoints

Handles speech-to-sign, text-to-sign, and sign-to-text conversion
for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import time

from app.core.database import get_db
from app.core.security import get_optional_user
from app.db_models.user import User
from app.services.translation_service import TranslationService
from app.schemas.gsl import (
    SpeechToSignRequest,
    TextToSignRequest,
    SignToTextRequest,
    TranslationResponse,
    GSLSignResponse,
    SignSequenceResponse
)

router = APIRouter()


@router.post("/speech-to-sign", response_model=TranslationResponse)
async def convert_speech_to_sign(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    language: str = Query("en", description="Audio language code"),
    accent: str = Query("ghanaian", description="Accent type"),
    include_fingerspelling: bool = Query(False, description="Include fingerspelling"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Convert speech audio to GSL signs.
    
    Supports Ghanaian English accents and local language phrases.
    """
    start_time = time.time()
    
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        if not audio_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty audio file"
            )
        
        # Create translation service
        translation_service = TranslationService(db)
        
        # Convert speech to sign
        user_id = current_user.id if current_user else None
        result = await translation_service.speech_to_sign(audio_data, user_id)
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech-to-sign conversion failed: {str(e)}"
        )


@router.post("/text-to-sign", response_model=TranslationResponse)
async def convert_text_to_sign(
    request: TextToSignRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Convert text to GSL signs.
    
    Handles Ghanaian English phrases and applies GSL grammar rules.
    """
    start_time = time.time()
    
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # Convert text to sign
        user_id = current_user.id if current_user else None
        result = await translation_service.text_to_sign(request.text, user_id)
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text-to-sign conversion failed: {str(e)}"
        )


@router.post("/sign-to-text", response_model=TranslationResponse)
async def convert_sign_to_text(
    request: SignToTextRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Convert GSL signs to text.
    
    Takes a sequence of sign IDs and generates natural language text.
    """
    start_time = time.time()
    
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # Convert sign to text
        user_id = current_user.id if current_user else None
        result = await translation_service.sign_to_text(request.sign_ids, user_id)
        
        # Add processing time
        result.processing_time = time.time() - start_time
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sign-to-text conversion failed: {str(e)}"
        )


@router.get("/sign-video/{sign_id}", response_model=GSLSignResponse)
async def get_sign_video(
    sign_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Retrieve sign demonstration video.
    
    Returns video URL, thumbnail, and sign details.
    """
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # Get sign video
        sign = await translation_service.get_sign_video(sign_id)
        
        if not sign:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sign with ID {sign_id} not found"
            )
        
        return sign
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sign video: {str(e)}"
        )


@router.get("/sign-variations/{sign_id}", response_model=List[GSLSignResponse])
async def get_sign_variations(
    sign_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get variations of a sign with Harmonized GSL priority.
    
    Returns related signs and regional variations.
    """
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # Get sign variations
        variations = await translation_service.get_sign_variations(sign_id)
        
        return variations
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sign variations: {str(e)}"
        )


@router.post("/sign-sequence", response_model=SignSequenceResponse)
async def create_sign_sequence(
    request: TextToSignRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Create animated sign sequence with timing and transitions.
    
    Generates a complete sequence for demonstrating multiple signs.
    """
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # First convert text to signs
        user_id = current_user.id if current_user else None
        translation_result = await translation_service.text_to_sign(request.text, user_id)
        
        if not translation_result.translated_signs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No signs found for the given text"
            )
        
        # Get full sign objects
        sign_ids = [sign.id for sign in translation_result.translated_signs]
        signs = await translation_service.get_multiple_sign_videos(sign_ids)
        
        # Create sequence
        from app.db_models.gsl import GSLSign
        sign_objects = translation_service.db.query(GSLSign).filter(
            GSLSign.id.in_(sign_ids)
        ).all()
        
        sequence = await translation_service.create_sign_sequence(
            request.text,
            sign_objects,
            user_id
        )
        
        return sequence
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create sign sequence: {str(e)}"
        )


@router.post("/batch-sign-videos", response_model=List[GSLSignResponse])
async def get_batch_sign_videos(
    sign_ids: List[UUID],
    db: Session = Depends(get_db)
):
    """
    Get multiple sign demonstration videos in a single request.
    
    Optimized for retrieving sign sequences.
    """
    try:
        # Create translation service
        translation_service = TranslationService(db)
        
        # Get multiple sign videos
        signs = await translation_service.get_multiple_sign_videos(sign_ids)
        
        return signs
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sign videos: {str(e)}"
        )