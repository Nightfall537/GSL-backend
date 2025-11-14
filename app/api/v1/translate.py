"""
Translation Service API Endpoints

Handles speech-to-sign, text-to-sign, and sign-to-text conversion
for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.translation_service import TranslationService
from app.schemas.gsl import (
    SpeechToSignRequest, TextToSignRequest, SignToTextRequest,
    TranslationResponse, SignSequenceResponse
)

router = APIRouter()


@router.post("/speech-to-sign", response_model=TranslationResponse)
async def convert_speech_to_sign(
    audio_file: UploadFile = File(..., description="Audio file (mp3, wav, m4a)"),
    language: str = Form("en", description="Audio language code"),
    accent: str = Form("ghanaian", description="Accent type"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Convert speech audio to GSL signs.
    
    Supports Ghanaian English accents and local phrases.
    
    - **audio_file**: Audio file containing speech
    - **language**: Language code (default: en)
    - **accent**: Accent type (ghanaian, standard, etc.)
    
    Process:
    1. Speech-to-text transcription (Whisper model)
    2. Text processing for Ghanaian phrases
    3. Text-to-sign mapping with GSL dictionary
    
    Returns sequence of GSL signs with demonstration videos.
    """
    # Validate audio file
    if not audio_file.content_type.startswith('audio/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an audio file"
        )
    
    # Read audio data
    audio_data = await audio_file.read()
    
    # Perform translation
    translation_service = TranslationService(db)
    
    try:
        result = await translation_service.speech_to_sign(
            audio_data,
            current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/text-to-sign", response_model=TranslationResponse)
async def convert_text_to_sign(
    request: TextToSignRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Convert text to GSL signs.
    
    Handles Ghanaian English phrases and local idioms.
    Prioritizes Harmonized GSL versions.
    
    - **text**: Text to translate (max 500 characters)
    - **include_fingerspelling**: Include fingerspelling for unknown words
    - **grammar_rules**: Apply GSL grammar rules
    - **simplify_text**: Simplify text before translation
    
    Examples of Ghanaian phrases handled:
    - "how far" → greeting signs
    - "chale" → friend sign
    - "small small" → gradually sign
    
    Returns sequence of GSL signs with videos and timing.
    """
    translation_service = TranslationService(db)
    
    try:
        result = await translation_service.text_to_sign(
            request.text,
            current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/sign-to-text", response_model=TranslationResponse)
async def convert_sign_to_text(
    request: SignToTextRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Convert sequence of GSL signs to text.
    
    - **sign_ids**: List of GSL sign IDs in sequence
    - **include_grammar**: Apply proper English grammar
    
    Converts sign sequence to natural English text with proper grammar.
    """
    if not request.sign_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one sign ID is required"
        )
    
    translation_service = TranslationService(db)
    
    try:
        result = await translation_service.sign_to_text(
            request.sign_ids,
            current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.get("/sign-video/{sign_id}")
async def get_sign_video(
    sign_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve sign demonstration video.
    
    - **sign_id**: GSL sign UUID
    
    Returns video URL, thumbnail, description, and usage examples.
    Optimized for low-bandwidth with multiple quality options.
    """
    translation_service = TranslationService(db)
    
    video_info = await translation_service.get_sign_video(sign_id)
    if not video_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sign not found"
        )
    
    return video_info


@router.get("/sign-variations/{sign_id}")
async def get_sign_variations(
    sign_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get variations of a sign.
    
    Returns different ways to perform the same sign,
    with Harmonized GSL versions prioritized.
    
    - **sign_id**: GSL sign UUID
    """
    translation_service = TranslationService(db)
    
    try:
        variations = await translation_service.get_sign_variations(sign_id)
        return {
            "sign_id": sign_id,
            "variations": variations,
            "count": len(variations)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve variations: {str(e)}"
        )