"""
Media Service API Endpoints

Handles file uploads, video streaming, compression, and media processing
for the GSL learning platform.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import os
from pathlib import Path

from app.core.database import get_db
from app.core.security import get_current_user
from app.services.media_service import MediaService
from app.schemas.media import (
    MediaUploadRequest, MediaResponse, VideoResponse,
    ImageResponse, AudioResponse, MediaProcessingRequest,
    MediaProcessingResponse, MediaListResponse
)
from app.schemas.common import SuccessResponse

router = APIRouter()


@router.post("/upload", response_model=MediaResponse, status_code=status.HTTP_201_CREATED)
async def upload_media_file(
    file: UploadFile = File(..., description="Media file to upload"),
    description: Optional[str] = Query(None, max_length=500),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    is_public: bool = Query(False, description="Make file publicly accessible"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload media files (video, image, or audio).
    
    Supported formats:
    - Video: mp4, webm, avi, mov
    - Image: jpg, jpeg, png, gif, webp
    - Audio: mp3, wav, ogg, m4a
    
    Features:
    - Automatic thumbnail generation for videos/images
    - File deduplication (same file uploaded twice returns existing)
    - Compression and optimization for low-bandwidth
    - Validation for file size and type
    
    Max file size: 100MB
    """
    # Read file data
    file_data = await file.read()
    
    # Parse tags
    tag_list = [tag.strip() for tag in tags.split(",")] if tags else []
    
    # Upload file
    media_service = MediaService(db)
    
    try:
        result = await media_service.upload_media(
            file_data=file_data,
            filename=file.filename,
            content_type=file.content_type,
            user_id=current_user.id
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/video/{video_id}")
async def stream_video(
    video_id: UUID,
    quality: str = Query("original", description="Video quality: original, high, medium, low"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stream video content with progressive loading.
    
    - **video_id**: UUID of the video file
    - **quality**: Video quality level for bandwidth optimization
      - original: Full quality (default)
      - high: 1080p
      - medium: 720p (recommended for mobile)
      - low: 480p (for poor connections)
    
    Supports range requests for seeking and progressive loading.
    Optimized for low-bandwidth environments.
    """
    media_service = MediaService(db)
    
    try:
        video_info = await media_service.get_video_stream(video_id, quality)
        if not video_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        file_path = Path(video_info.file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found on disk"
            )
        
        # Return file response for streaming
        return FileResponse(
            path=str(file_path),
            media_type=video_info.content_type,
            filename=file_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming failed: {str(e)}"
        )


@router.get("/compressed/{media_id}")
async def get_compressed_media(
    media_id: UUID,
    compression_level: str = Query("medium", description="Compression level: low, medium, high"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get compressed version of media for low-bandwidth users.
    
    - **media_id**: UUID of the media file
    - **compression_level**: Compression level
      - low: Minimal compression, better quality
      - medium: Balanced (recommended)
      - high: Maximum compression, smaller file size
    
    Compressed versions are cached for faster delivery.
    Ideal for users with poor network conditions.
    """
    if compression_level not in ["low", "medium", "high"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid compression level. Use: low, medium, or high"
        )
    
    media_service = MediaService(db)
    
    try:
        compressed_data = await media_service.get_compressed_media(
            media_id,
            compression_level
        )
        
        if not compressed_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found"
            )
        
        # Get media info for content type
        media_info = await media_service.get_media_info(media_id)
        
        return StreamingResponse(
            iter([compressed_data]),
            media_type=media_info.get("content_type", "application/octet-stream")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression failed: {str(e)}"
        )


@router.post("/process", response_model=dict)
async def process_uploaded_media(
    media_id: UUID = Query(..., description="Media file ID to process"),
    processing_type: str = Query("gesture_recognition", description="Processing type"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process uploaded media for AI analysis.
    
    - **media_id**: UUID of uploaded media file
    - **processing_type**: Type of processing
      - gesture_recognition: Extract frames for sign recognition
      - audio_extraction: Extract audio from video for speech-to-text
      - thumbnail_generation: Generate thumbnail image
    
    Prepares media for AI model inference.
    Returns processed data ready for recognition/translation services.
    """
    valid_types = ["gesture_recognition", "audio_extraction", "thumbnail_generation"]
    if processing_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid processing type. Use: {', '.join(valid_types)}"
        )
    
    media_service = MediaService(db)
    
    try:
        result = await media_service.process_media_for_ai(
            media_id,
            processing_type
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@router.get("/info/{media_id}", response_model=dict)
async def get_media_info(
    media_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get media file information and metadata.
    
    - **media_id**: UUID of the media file
    
    Returns file details, URLs, size, and upload information.
    """
    media_service = MediaService(db)
    
    info = await media_service.get_media_info(media_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found"
        )
    
    return info


@router.delete("/{media_id}", response_model=SuccessResponse)
async def delete_media(
    media_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete media file.
    
    Users can only delete their own uploaded files.
    
    - **media_id**: UUID of the media file
    """
    media_service = MediaService(db)
    
    success = await media_service.delete_media(media_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or you don't have permission to delete it"
        )
    
    return SuccessResponse(message="Media deleted successfully")


@router.get("/thumbnail/{media_id}")
async def get_thumbnail(
    media_id: UUID,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get thumbnail image for video or image file.
    
    - **media_id**: UUID of the media file
    
    Returns thumbnail image (JPEG format).
    """
    media_service = MediaService(db)
    
    media_info = await media_service.get_media_info(media_id)
    if not media_info or not media_info.get("thumbnail_url"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thumbnail not found"
        )
    
    # Extract thumbnail path from URL
    # Assuming thumbnail_url format: /api/v1/media/thumbnail/{media_id}
    from app.config.settings import get_settings
    settings = get_settings()
    thumbnail_path = Path(settings.upload_dir) / f"{media_id}_thumb.jpg"
    
    if not thumbnail_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thumbnail file not found"
        )
    
    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg"
    )