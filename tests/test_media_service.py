"""
Unit Tests for Media Service

Tests file upload, compression, streaming, and processing.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import UUID, uuid4
from pathlib import Path

from app.services.media_service import MediaService


class TestMediaService:
    """Test cases for MediaService."""
    
    @pytest.fixture
    def media_service(self, test_db):
        """Create MediaService instance."""
        return MediaService(test_db)
    
    @pytest.mark.asyncio
    async def test_upload_media_success(
        self,
        media_service,
        sample_video_bytes
    ):
        """Test successful media upload."""
        filename = "test_video.mp4"
        content_type = "video/mp4"
        user_id = uuid4()
        
        with patch.object(media_service, '_validate_file'):
            with patch.object(media_service.file_handler, 'save_file', return_value=True):
                with patch.object(media_service.file_handler, 'generate_thumbnail', return_value=Path("thumb.jpg")):
                    result = await media_service.upload_media(
                        sample_video_bytes,
                        filename,
                        content_type,
                        user_id
                    )
                    
                    assert result.filename == filename
                    assert result.content_type == content_type
                    assert result.file_size == len(sample_video_bytes)
    
    @pytest.mark.asyncio
    async def test_upload_media_duplicate(
        self,
        media_service,
        sample_video_bytes
    ):
        """Test uploading duplicate file (deduplication)."""
        filename = "test_video.mp4"
        content_type = "video/mp4"
        
        # Mock existing file
        existing_file = Mock(
            id=uuid4(),
            filename=filename,
            file_url="https://example.com/video.mp4",
            file_size=len(sample_video_bytes)
        )
        
        with patch.object(media_service, '_validate_file'):
            with patch.object(media_service, '_calculate_file_hash', return_value="hash123"):
                media_service.db.query = Mock(return_value=Mock(
                    filter=Mock(return_value=Mock(first=Mock(return_value=existing_file)))
                ))
                
                result = await media_service.upload_media(
                    sample_video_bytes,
                    filename,
                    content_type
                )
                
                assert result.file_id == existing_file.id
                assert "deduplicated" in result.message.lower()
    
    @pytest.mark.asyncio
    async def test_get_video_stream(self, media_service):
        """Test getting video for streaming."""
        file_id = uuid4()
        
        mock_file = Mock(
            id=file_id,
            file_path="uploads/video.mp4",
            content_type="video/mp4"
        )
        
        media_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_file)))
        ))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat', return_value=Mock(st_size=1024)):
                result = await media_service.get_video_stream(file_id)
                
                assert result is not None
                assert result.file_id == file_id
    
    @pytest.mark.asyncio
    async def test_get_compressed_media(
        self,
        media_service,
        sample_video_bytes
    ):
        """Test getting compressed media."""
        file_id = uuid4()
        
        mock_file = Mock(
            id=file_id,
            file_path="uploads/video.mp4"
        )
        
        media_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_file)))
        ))
        
        with patch.object(media_service.cache, 'get', return_value=None):
            with patch('pathlib.Path.exists', return_value=True):
                with patch.object(media_service.file_handler, 'compress_media', return_value=sample_video_bytes):
                    with patch.object(media_service.cache, 'set', return_value=True):
                        result = await media_service.get_compressed_media(file_id)
                        
                        assert result is not None
    
    @pytest.mark.asyncio
    async def test_process_media_for_ai_gesture_recognition(
        self,
        media_service
    ):
        """Test processing media for gesture recognition."""
        file_id = uuid4()
        
        mock_file = Mock(
            id=file_id,
            file_path="uploads/video.mp4"
        )
        
        media_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_file)))
        ))
        
        with patch.object(media_service.file_handler, 'extract_video_frames', return_value=[]):
            result = await media_service.process_media_for_ai(
                file_id,
                processing_type="gesture_recognition"
            )
            
            assert result["processing_type"] == "gesture_recognition"
            assert "frames_extracted" in result
    
    @pytest.mark.asyncio
    async def test_delete_media(self, media_service):
        """Test deleting media file."""
        file_id = uuid4()
        user_id = uuid4()
        
        mock_file = Mock(
            id=file_id,
            user_id=user_id,
            file_path="uploads/video.mp4",
            thumbnail_url=None
        )
        
        media_service.db.query = Mock(return_value=Mock(
            filter=Mock(return_value=Mock(first=Mock(return_value=mock_file)))
        ))
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.unlink'):
                with patch.object(media_service.cache, 'delete', return_value=True):
                    result = await media_service.delete_media(file_id, user_id)
                    
                    assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_file_valid(self, media_service, sample_video_bytes):
        """Test file validation with valid file."""
        filename = "test.mp4"
        content_type = "video/mp4"
        
        # Should not raise exception
        await media_service._validate_file(sample_video_bytes, filename, content_type)
    
    @pytest.mark.asyncio
    async def test_validate_file_size_exceeded(self, media_service):
        """Test file validation with oversized file."""
        large_file = b"x" * (media_service.max_file_size + 1)
        filename = "large.mp4"
        content_type = "video/mp4"
        
        with pytest.raises(ValueError, match="File size exceeds"):
            await media_service._validate_file(large_file, filename, content_type)
    
    @pytest.mark.asyncio
    async def test_validate_file_invalid_type(self, media_service, sample_video_bytes):
        """Test file validation with invalid file type."""
        filename = "test.exe"
        content_type = "application/x-msdownload"
        
        with pytest.raises(ValueError, match="File type"):
            await media_service._validate_file(sample_video_bytes, filename, content_type)
    
    def test_calculate_file_hash(self, media_service, sample_video_bytes):
        """Test file hash calculation."""
        hash1 = media_service._calculate_file_hash(sample_video_bytes)
        hash2 = media_service._calculate_file_hash(sample_video_bytes)
        
        assert hash1 == hash2  # Same data should produce same hash
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string