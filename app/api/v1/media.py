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
from pathlib import Path
import io

from app.core.database import get_db
from app.core.auth import get_current_user
from app.db_models.user import User
from app.services.media_service import MediaService
from app.schemas.media import (
    MediaUploadResponse,
    MediaProcessingRequest,
    VideoStreamResponse,
    MediaResponse,
    MediaQuality
)

router = APIRouter()


@router.post("/upload", response_model=MediaUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_media_file(
    file: UploadFile = File(..., description="Media file to upload"),
    description: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload media files (video, image, or audio).
    
    Supports:
    - Video formats: mp4, mov, avi
    - Image formats: jpg, jpeg, png
    - Audio formats: wav, mp3
    
    Files are validated for size and format, deduplicated by hash,
    and thumbnails are automatically generated for videos and images.
    """
    try:
        # Read file data
        file_data = await file.read()
        
        # Create media service
        media_service = MediaService(db)
        
        # Upload media
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
            detail=f"Failed to upload media: {str(e)}"
        )


@router.get("/video/{video_id}")
async def stream_video(
    video_id: UUID,
    quality: str = Query("original", description="Video quality: original, high, medium, low"),
    db: Session = Depends(get_db)
):
    """
    Stream video content with progressive loading.
    
    Supports multiple quality levels for bandwidth optimization:
    - original: Full quality
    - high: High quality compressed
    - medium: Medium quality compressed
    - low: Low quality for poor connections
    """
    try:
        media_service = MediaService(db)
        
        # Get video stream info
        stream_info = await media_service.get_video_stream(video_id, quality)
        
        if not stream_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found"
            )
        
        file_path = Path(stream_info.file_path)
        
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video file not found on disk"
            )
        
        # Return file response for streaming
        return FileResponse(
            path=str(file_path),
            media_type=stream_info.content_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream video: {str(e)}"
        )


@router.get("/thumbnail/{media_id}")
async def get_thumbnail(
    media_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get thumbnail for video or image.
    
    Thumbnails are automatically generated during upload
    and cached for performance.
    """
    try:
        media_service = MediaService(db)
        
        # Get media info
        media_info = await media_service.get_media_info(media_id)
        
        if not media_info or not media_info.get("thumbnail_url"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thumbnail not found"
            )
        
        # Extract thumbnail path from URL
        thumbnail_filename = f"{media_id}_thumb.jpg"
        thumbnail_path = Path(media_service.upload_dir) / thumbnail_filename
        
        if not thumbnail_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Thumbnail file not found"
            )
        
        return FileResponse(
            path=str(thumbnail_path),
            media_type="image/jpeg",
            filename=thumbnail_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get thumbnail: {str(e)}"
        )


@router.get("/compressed/{media_id}")
async def get_compressed_media(
    media_id: UUID,
    compression_level: str = Query("medium", description="Compression level: low, medium, high"),
    db: Session = Depends(get_db)
):
    """
    Get compressed media for low-bandwidth users.
    
    Compression levels:
    - low: Minimal compression, better quality
    - medium: Balanced compression (default)
    - high: Maximum compression, smaller file size
    
    Compressed versions are cached for performance.
    """
    try:
        if compression_level not in ["low", "medium", "high"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid compression level. Use: low, medium, or high"
            )
        
        media_service = MediaService(db)
        
        # Get compressed media
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
        content_type = media_info.get("content_type", "application/octet-stream")
        
        # Return compressed data as streaming response
        return StreamingResponse(
            io.BytesIO(compressed_data),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename=compressed_{media_id}",
                "X-Compression-Level": compression_level
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get compressed media: {str(e)}"
        )


@router.post("/process")
async def process_uploaded_media(
    request: MediaProcessingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Process uploaded media for AI analysis.
    
    Processing types:
    - gesture_recognition: Extract frames for sign recognition
    - audio_extraction: Extract audio from video for speech-to-text
    
    Returns processed data ready for AI model inference.
    """
    try:
        media_service = MediaService(db)
        
        # Determine processing type from operations
        processing_type = "gesture_recognition"
        if "audio_extraction" in request.operations:
            processing_type = "audio_extraction"
        
        # Process media
        result = await media_service.process_media_for_ai(
            file_id=request.media_id,
            processing_type=processing_type
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
            detail=f"Failed to process media: {str(e)}"
        )


@router.get("/info/{media_id}", response_model=dict)
async def get_media_info(
    media_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get media file information and metadata.
    
    Returns details about the media file including size, type,
    upload date, and processing status.
    """
    try:
        media_service = MediaService(db)
        
        media_info = await media_service.get_media_info(media_id)
        
        if not media_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found"
            )
        
        return media_info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get media info: {str(e)}"
        )


@router.delete("/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_media(
    media_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete media file.
    
    Only the user who uploaded the file can delete it.
    Deletes both the database record and physical file.
    """
    try:
        media_service = MediaService(db)
        
        success = await media_service.delete_media(media_id, current_user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found or you don't have permission to delete it"
            )
        
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete media: {str(e)}"
        )



@router.post("/sync/upload")
async def sync_offline_uploads(
    files: list[UploadFile] = File(..., description="Multiple media files for offline sync"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Synchronize offline uploads when connectivity is restored.
    
    Accepts multiple files for batch upload to minimize
    network requests in poor connectivity scenarios.
    """
    try:
        media_service = MediaService(db)
        results = []
        
        for file in files:
            try:
                file_data = await file.read()
                
                result = await media_service.upload_media(
                    file_data=file_data,
                    filename=file.filename,
                    content_type=file.content_type,
                    user_id=current_user.id
                )
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "file_id": result.file_id,
                    "message": result.message
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "total_files": len(files),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync uploads: {str(e)}"
        )


@router.get("/sync/content")
async def get_offline_content(
    content_type: Optional[str] = Query(None, description="Filter by content type"),
    limit: int = Query(50, ge=1, le=100, description="Number of items to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get frequently accessed content for offline caching.
    
    Returns a list of media files that should be cached
    for offline use, prioritizing frequently accessed content.
    """
    try:
        from app.db_models.media import MediaFile
        
        query = db.query(MediaFile).filter(MediaFile.is_public == True)
        
        if content_type:
            query = query.filter(MediaFile.content_type.like(f"{content_type}%"))
        
        # Get most recent public media files
        media_files = query.order_by(MediaFile.uploaded_at.desc()).limit(limit).all()
        
        return {
            "content": [
                {
                    "file_id": str(media.id),
                    "filename": media.filename,
                    "file_url": media.file_url,
                    "thumbnail_url": media.thumbnail_url,
                    "file_size": media.file_size,
                    "content_type": media.content_type,
                    "uploaded_at": media.uploaded_at.isoformat()
                }
                for media in media_files
            ],
            "total_count": len(media_files),
            "cache_priority": "high"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get offline content: {str(e)}"
        )


@router.get("/cache/status/{media_id}")
async def get_cache_status(
    media_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Check if media is cached and get cache metadata.
    
    Useful for offline-first applications to determine
    if content needs to be downloaded or is available locally.
    """
    try:
        media_service = MediaService(db)
        
        # Check if compressed versions are cached
        cache_status = {}
        for level in ["low", "medium", "high"]:
            cache_key = f"compressed_media:{media_id}:{level}"
            is_cached = await media_service.cache.exists(cache_key)
            cache_status[level] = is_cached
        
        media_info = await media_service.get_media_info(media_id)
        
        if not media_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Media not found"
            )
        
        return {
            "media_id": str(media_id),
            "cached_versions": cache_status,
            "file_size": media_info.get("file_size"),
            "content_type": media_info.get("content_type"),
            "recommended_quality": "medium" if any(cache_status.values()) else "low"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache status: {str(e)}"
        )
