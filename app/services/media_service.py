"""
Media Service

Handles file uploads, video streaming, compression, and media processing
for the GSL learning platform with optimization for low-bandwidth environments.
"""

import os
import hashlib
from typing import Optional, BinaryIO
from uuid import UUID, uuid4
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session

from app.config.settings import get_settings
# Note: MediaFile will be handled via Supabase
from app.schemas.media import (
    MediaUploadResponse, MediaProcessingRequest,
    VideoStreamResponse
)
from app.utils.file_handler import FileHandler
from app.utils.cache import CacheManager

settings = get_settings()


class MediaService:
    """Service for media file handling and optimization."""
    
    def __init__(self, db: Session):
        self.db = db
        self.file_handler = FileHandler()
        self.cache = CacheManager()
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def upload_media(
        self,
        file_data: bytes,
        filename: str,
        content_type: str,
        user_id: Optional[UUID] = None
    ) -> MediaUploadResponse:
        """
        Upload and store media file.
        
        Args:
            file_data: File bytes
            filename: Original filename
            content_type: MIME type
            user_id: Optional user ID
            
        Returns:
            Upload response with file information
            
        Raises:
            ValueError: If file validation fails
        """
        # Validate file
        await self._validate_file(file_data, filename, content_type)
        
        # Generate unique file ID and path
        file_id = uuid4()
        file_extension = Path(filename).suffix
        file_hash = self._calculate_file_hash(file_data)
        
        # Check for duplicate files
        existing_file = self.db.query(MediaFile).filter(
            MediaFile.file_hash == file_hash
        ).first()
        
        if existing_file:
            return MediaUploadResponse(
                file_id=existing_file.id,
                filename=existing_file.filename,
                file_url=existing_file.file_url,
                thumbnail_url=existing_file.thumbnail_url,
                file_size=existing_file.file_size,
                content_type=existing_file.content_type,
                message="File already exists (deduplicated)"
            )
        
        # Save file to storage
        file_path = self.upload_dir / f"{file_id}{file_extension}"
        await self.file_handler.save_file(file_data, file_path)
        
        # Generate thumbnail for videos and images
        thumbnail_path = None
        if content_type.startswith(('video/', 'image/')):
            thumbnail_path = await self.file_handler.generate_thumbnail(
                file_path,
                self.upload_dir / f"{file_id}_thumb.jpg"
            )
        
        # Create database record
        media_file = MediaFile(
            id=file_id,
            filename=filename,
            file_path=str(file_path),
            file_url=f"/api/v1/media/video/{file_id}",
            thumbnail_url=f"/api/v1/media/thumbnail/{file_id}" if thumbnail_path else None,
            file_size=len(file_data),
            content_type=content_type,
            file_hash=file_hash,
            user_id=user_id,
            uploaded_at=datetime.utcnow()
        )
        
        self.db.add(media_file)
        self.db.commit()
        self.db.refresh(media_file)
        
        return MediaUploadResponse(
            file_id=media_file.id,
            filename=media_file.filename,
            file_url=media_file.file_url,
            thumbnail_url=media_file.thumbnail_url,
            file_size=media_file.file_size,
            content_type=media_file.content_type,
            message="File uploaded successfully"
        )
    
    async def get_video_stream(
        self,
        file_id: UUID,
        quality: str = "original"
    ) -> Optional[VideoStreamResponse]:
        """
        Get video file for streaming.
        
        Args:
            file_id: Media file ID
            quality: Video quality (original, high, medium, low)
            
        Returns:
            Video stream information
        """
        media_file = self.db.query(MediaFile).filter(
            MediaFile.id == file_id
        ).first()
        
        if not media_file:
            return None
        
        # Check if compressed version exists for requested quality
        if quality != "original":
            compressed_path = await self._get_compressed_version(
                media_file,
                quality
            )
            if compressed_path:
                file_path = compressed_path
            else:
                file_path = Path(media_file.file_path)
        else:
            file_path = Path(media_file.file_path)
        
        if not file_path.exists():
            return None
        
        return VideoStreamResponse(
            file_id=media_file.id,
            file_path=str(file_path),
            content_type=media_file.content_type,
            file_size=file_path.stat().st_size,
            quality=quality
        )
    
    async def get_compressed_media(
        self,
        file_id: UUID,
        compression_level: str = "medium"
    ) -> Optional[bytes]:
        """
        Get compressed version of media for low-bandwidth users.
        
        Args:
            file_id: Media file ID
            compression_level: Compression level (low, medium, high)
            
        Returns:
            Compressed file bytes
        """
        # Check cache first
        cache_key = f"compressed_media:{file_id}:{compression_level}"
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            return cached_data
        
        media_file = self.db.query(MediaFile).filter(
            MediaFile.id == file_id
        ).first()
        
        if not media_file:
            return None
        
        file_path = Path(media_file.file_path)
        if not file_path.exists():
            return None
        
        # Compress the file
        compressed_data = await self.file_handler.compress_media(
            file_path,
            compression_level
        )
        
        # Cache the compressed version
        await self.cache.set(cache_key, compressed_data, ttl=7200)
        
        return compressed_data
    
    async def process_media_for_ai(
        self,
        file_id: UUID,
        processing_type: str = "gesture_recognition"
    ) -> dict:
        """
        Process uploaded media for AI analysis.
        
        Args:
            file_id: Media file ID
            processing_type: Type of processing needed
            
        Returns:
            Processing result with extracted features
        """
        media_file = self.db.query(MediaFile).filter(
            MediaFile.id == file_id
        ).first()
        
        if not media_file:
            raise ValueError("Media file not found")
        
        file_path = Path(media_file.file_path)
        
        if processing_type == "gesture_recognition":
            # Extract frames from video
            frames = await self.file_handler.extract_video_frames(
                file_path,
                max_frames=30,
                target_size=(224, 224)
            )
            
            return {
                "file_id": file_id,
                "processing_type": processing_type,
                "frames_extracted": len(frames),
                "frames": frames,
                "status": "ready_for_inference"
            }
        
        elif processing_type == "audio_extraction":
            # Extract audio from video
            audio_path = await self.file_handler.extract_audio(
                file_path,
                self.upload_dir / f"{file_id}_audio.wav"
            )
            
            return {
                "file_id": file_id,
                "processing_type": processing_type,
                "audio_path": str(audio_path),
                "status": "ready_for_transcription"
            }
        
        return {
            "file_id": file_id,
            "processing_type": processing_type,
            "status": "unknown_processing_type"
        }
    
    async def delete_media(self, file_id: UUID, user_id: UUID) -> bool:
        """
        Delete media file.
        
        Args:
            file_id: Media file ID
            user_id: User requesting deletion
            
        Returns:
            True if deleted successfully
        """
        media_file = self.db.query(MediaFile).filter(
            MediaFile.id == file_id,
            MediaFile.user_id == user_id
        ).first()
        
        if not media_file:
            return False
        
        # Delete physical file
        file_path = Path(media_file.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Delete thumbnail if exists
        if media_file.thumbnail_url:
            thumbnail_path = self.upload_dir / f"{file_id}_thumb.jpg"
            if thumbnail_path.exists():
                thumbnail_path.unlink()
        
        # Delete database record
        self.db.delete(media_file)
        self.db.commit()
        
        # Clear cache
        await self.cache.delete(f"compressed_media:{file_id}:*")
        
        return True
    
    async def get_media_info(self, file_id: UUID) -> Optional[dict]:
        """Get media file information."""
        media_file = self.db.query(MediaFile).filter(
            MediaFile.id == file_id
        ).first()
        
        if not media_file:
            return None
        
        return {
            "file_id": media_file.id,
            "filename": media_file.filename,
            "file_url": media_file.file_url,
            "thumbnail_url": media_file.thumbnail_url,
            "file_size": media_file.file_size,
            "content_type": media_file.content_type,
            "uploaded_at": media_file.uploaded_at
        }
    
    async def _validate_file(
        self,
        file_data: bytes,
        filename: str,
        content_type: str
    ) -> None:
        """Validate uploaded file."""
        # Check file size
        if len(file_data) > settings.max_file_size:
            raise ValueError(
                f"File size exceeds maximum allowed size of {settings.max_file_size} bytes"
            )
        
        # Check file extension
        file_extension = Path(filename).suffix.lstrip('.')
        if file_extension.lower() not in settings.allowed_file_types:
            raise ValueError(
                f"File type '{file_extension}' not allowed. "
                f"Allowed types: {', '.join(settings.allowed_file_types)}"
            )
        
        # Validate content type
        valid_content_types = [
            'video/mp4', 'video/quicktime', 'video/x-msvideo',
            'image/jpeg', 'image/png', 'image/jpg',
            'audio/wav', 'audio/mpeg'
        ]
        if content_type not in valid_content_types:
            raise ValueError(f"Content type '{content_type}' not supported")
    
    def _calculate_file_hash(self, file_data: bytes) -> str:
        """Calculate SHA-256 hash of file for deduplication."""
        return hashlib.sha256(file_data).hexdigest()
    
    async def _get_compressed_version(
        self,
        media_file: MediaFile,
        quality: str
    ) -> Optional[Path]:
        """Get or create compressed version of media file."""
        compressed_filename = f"{media_file.id}_{quality}{Path(media_file.filename).suffix}"
        compressed_path = self.upload_dir / "compressed" / compressed_filename
        
        if compressed_path.exists():
            return compressed_path
        
        # Create compressed version
        compressed_path.parent.mkdir(parents=True, exist_ok=True)
        original_path = Path(media_file.file_path)
        
        await self.file_handler.compress_video(
            original_path,
            compressed_path,
            quality
        )
        
        return compressed_path if compressed_path.exists() else None