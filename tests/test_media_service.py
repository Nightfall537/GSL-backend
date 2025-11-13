"""
Tests for Media Service

Tests file upload, video streaming, compression, caching,
and offline synchronization functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from pathlib import Path
import tempfile
import shutil

from app.services.media_service import MediaService
from app.db_models.media import MediaFile, MediaType, ProcessingStatus
from app.schemas.media import MediaUploadResponse


@pytest.fixture
def temp_upload_dir():
    """Create temporary upload directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_file_handler():
    """Mock file handler."""
    handler = Mock()
    handler.save_file = AsyncMock(return_value=True)
    handler.generate_thumbnail = AsyncMock(return_value=Path("/fake/thumb.jpg"))
    handler.extract_video_frames = AsyncMock(return_value=[b"frame1", b"frame2"])
    handler.compress_media = AsyncMock(return_value=b"compressed-data")
    handler.compress_video = AsyncMock(return_value=True)
    handler.extract_audio = AsyncMock(return_value=Path("/fake/audio.wav"))
    return handler


@pytest.fixture
def mock_cache():
    """Mock cache manager."""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.exists = AsyncMock(return_value=False)
    return cache


@pytest.fixture
def media_service(test_db, temp_upload_dir, mock_file_handler, mock_cache):
    """Create media service with mocked dependencies."""
    service = MediaService(test_db)
    service.upload_dir = temp_upload_dir
    service.file_handler = mock_file_handler
    service.cache = mock_cache
    return service


class TestMediaUpload:
    """Test media file upload functionality."""
    
    @pytest.mark.asyncio
    async def test_upload_video_success(self, media_service, test_db, sample_video_bytes):
        """Test successful video upload."""
        result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="test_video.mp4",
            content_type="video/mp4",
            user_id=uuid4()
        )
        
        assert isinstance(result, MediaUploadResponse)
        assert result.filename == "test_video.mp4"
        assert result.content_type == "video/mp4"
        assert result.file_size == len(sample_video_bytes)
        assert result.file_url is not None
        assert "uploaded successfully" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_upload_image_success(self, media_service, sample_image_bytes):
        """Test successful image upload."""
        result = await media_service.upload_media(
            file_data=sample_image_bytes,
            filename="test_image.jpg",
            content_type="image/jpeg",
            user_id=uuid4()
        )
        
        assert isinstance(result, MediaUploadResponse)
        assert result.filename == "test_image.jpg"
        assert result.content_type == "image/jpeg"
    
    @pytest.mark.asyncio
    async def test_upload_file_too_large(self, media_service):
        """Test upload fails for oversized files."""
        large_file = b"x" * (100 * 1024 * 1024)  # 100MB
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            await media_service.upload_media(
                file_data=large_file,
                filename="large.mp4",
                content_type="video/mp4"
            )
    
    @pytest.mark.asyncio
    async def test_upload_invalid_file_type(self, media_service):
        """Test upload fails for invalid file types."""
        with pytest.raises(ValueError, match="not allowed"):
            await media_service.upload_media(
                file_data=b"test",
                filename="test.exe",
                content_type="application/x-msdownload"
            )
    
    @pytest.mark.asyncio
    async def test_upload_invalid_content_type(self, media_service):
        """Test upload fails for unsupported content types."""
        with pytest.raises(ValueError, match="not supported"):
            await media_service.upload_media(
                file_data=b"test",
                filename="test.mp4",
                content_type="application/octet-stream"
            )
    
    @pytest.mark.asyncio
    async def test_upload_deduplication(self, media_service, test_db, sample_video_bytes):
        """Test file deduplication by hash."""
        user_id = uuid4()
        
        # Upload first file
        result1 = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="video1.mp4",
            content_type="video/mp4",
            user_id=user_id
        )
        
        # Upload same file with different name
        result2 = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="video2.mp4",
            content_type="video/mp4",
            user_id=user_id
        )
        
        # Should return same file ID (deduplicated)
        assert result1.file_id == result2.file_id
        assert "already exists" in result2.message.lower()


class TestVideoStreaming:
    """Test video streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_get_video_stream_original(self, media_service, test_db, sample_video_bytes):
        """Test getting original quality video stream."""
        # Upload video first
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="stream_test.mp4",
            content_type="video/mp4"
        )
        
        # Get stream
        stream_info = await media_service.get_video_stream(
            upload_result.file_id,
            quality="original"
        )
        
        assert stream_info is not None
        assert stream_info.file_id == upload_result.file_id
        assert stream_info.quality == "original"
        assert stream_info.content_type == "video/mp4"
    
    @pytest.mark.asyncio
    async def test_get_video_stream_not_found(self, media_service):
        """Test getting stream for non-existent video."""
        result = await media_service.get_video_stream(uuid4())
        assert result is None


class TestMediaCompression:
    """Test media compression functionality."""
    
    @pytest.mark.asyncio
    async def test_compress_media_cached(self, media_service, test_db, sample_video_bytes):
        """Test compressed media is cached."""
        # Upload video
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="compress_test.mp4",
            content_type="video/mp4"
        )
        
        # Get compressed version
        compressed = await media_service.get_compressed_media(
            upload_result.file_id,
            compression_level="medium"
        )
        
        assert compressed is not None
        assert isinstance(compressed, bytes)
        
        # Verify cache was called
        media_service.cache.set.assert_called()
    
    @pytest.mark.asyncio
    async def test_compress_media_from_cache(self, media_service, mock_cache):
        """Test compressed media retrieved from cache."""
        file_id = uuid4()
        cached_data = b"cached-compressed-data"
        
        # Set up cache to return data
        mock_cache.get = AsyncMock(return_value=cached_data)
        
        result = await media_service.get_compressed_media(file_id, "medium")
        
        # Should return cached data without processing
        assert result == cached_data
        mock_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compress_different_levels(self, media_service, test_db, sample_video_bytes):
        """Test different compression levels."""
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="compress_levels.mp4",
            content_type="video/mp4"
        )
        
        for level in ["low", "medium", "high"]:
            compressed = await media_service.get_compressed_media(
                upload_result.file_id,
                compression_level=level
            )
            assert compressed is not None


class TestMediaProcessing:
    """Test media processing for AI analysis."""
    
    @pytest.mark.asyncio
    async def test_process_for_gesture_recognition(self, media_service, test_db, sample_video_bytes):
        """Test processing video for gesture recognition."""
        # Upload video
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="gesture_test.mp4",
            content_type="video/mp4"
        )
        
        # Process for gesture recognition
        result = await media_service.process_media_for_ai(
            upload_result.file_id,
            processing_type="gesture_recognition"
        )
        
        assert result["file_id"] == upload_result.file_id
        assert result["processing_type"] == "gesture_recognition"
        assert result["status"] == "ready_for_inference"
        assert "frames_extracted" in result
    
    @pytest.mark.asyncio
    async def test_process_for_audio_extraction(self, media_service, test_db, sample_video_bytes):
        """Test extracting audio from video."""
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="audio_test.mp4",
            content_type="video/mp4"
        )
        
        result = await media_service.process_media_for_ai(
            upload_result.file_id,
            processing_type="audio_extraction"
        )
        
        assert result["processing_type"] == "audio_extraction"
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, media_service):
        """Test processing non-existent file raises error."""
        with pytest.raises(ValueError, match="not found"):
            await media_service.process_media_for_ai(
                uuid4(),
                processing_type="gesture_recognition"
            )


class TestMediaDeletion:
    """Test media file deletion."""
    
    @pytest.mark.asyncio
    async def test_delete_media_success(self, media_service, test_db, sample_video_bytes):
        """Test successful media deletion."""
        user_id = uuid4()
        
        # Upload file
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="delete_test.mp4",
            content_type="video/mp4",
            user_id=user_id
        )
        
        # Delete file
        success = await media_service.delete_media(upload_result.file_id, user_id)
        
        assert success is True
        
        # Verify file is deleted from database
        media_file = test_db.query(MediaFile).filter(
            MediaFile.id == upload_result.file_id
        ).first()
        assert media_file is None
    
    @pytest.mark.asyncio
    async def test_delete_media_wrong_user(self, media_service, test_db, sample_video_bytes):
        """Test deletion fails for wrong user."""
        user_id = uuid4()
        other_user_id = uuid4()
        
        # Upload file
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="delete_test.mp4",
            content_type="video/mp4",
            user_id=user_id
        )
        
        # Try to delete with different user
        success = await media_service.delete_media(upload_result.file_id, other_user_id)
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_delete_clears_cache(self, media_service, test_db, sample_video_bytes):
        """Test deletion clears cached compressed versions."""
        user_id = uuid4()
        
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="cache_delete.mp4",
            content_type="video/mp4",
            user_id=user_id
        )
        
        await media_service.delete_media(upload_result.file_id, user_id)
        
        # Verify cache delete was called
        media_service.cache.delete.assert_called()


class TestMediaInfo:
    """Test media information retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_media_info(self, media_service, test_db, sample_video_bytes):
        """Test getting media file information."""
        upload_result = await media_service.upload_media(
            file_data=sample_video_bytes,
            filename="info_test.mp4",
            content_type="video/mp4"
        )
        
        info = await media_service.get_media_info(upload_result.file_id)
        
        assert info is not None
        assert info["file_id"] == upload_result.file_id
        assert info["filename"] == "info_test.mp4"
        assert info["content_type"] == "video/mp4"
        assert info["file_size"] == len(sample_video_bytes)
        assert "uploaded_at" in info
    
    @pytest.mark.asyncio
    async def test_get_media_info_not_found(self, media_service):
        """Test getting info for non-existent media."""
        info = await media_service.get_media_info(uuid4())
        assert info is None
