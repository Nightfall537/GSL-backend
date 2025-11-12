"""
Unit Tests for File Handler

Tests file operations, video processing, and compression.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import numpy as np

from app.utils.file_handler import FileHandler


class TestFileHandler:
    """Test cases for FileHandler."""
    
    @pytest.fixture
    def file_handler(self):
        """Create FileHandler instance."""
        return FileHandler()
    
    @pytest.mark.asyncio
    async def test_save_file(self, file_handler, sample_video_bytes, tmp_path):
        """Test saving file to disk."""
        file_path = tmp_path / "test_video.mp4"
        
        result = await file_handler.save_file(sample_video_bytes, file_path)
        
        assert result is True
        assert file_path.exists()
        assert file_path.read_bytes() == sample_video_bytes
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_video(self, file_handler, tmp_path):
        """Test generating thumbnail from video."""
        source_path = tmp_path / "video.mp4"
        thumbnail_path = tmp_path / "thumb.jpg"
        
        # Create dummy video file
        source_path.write_bytes(b"fake-video")
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap.return_value.release = Mock()
            
            with patch('cv2.imwrite', return_value=True):
                result = await file_handler.generate_thumbnail(
                    source_path,
                    thumbnail_path,
                    size=(320, 240)
                )
                
                assert result == thumbnail_path
    
    @pytest.mark.asyncio
    async def test_generate_thumbnail_image(self, file_handler, tmp_path):
        """Test generating thumbnail from image."""
        source_path = tmp_path / "image.jpg"
        thumbnail_path = tmp_path / "thumb.jpg"
        
        # Create dummy image file
        source_path.write_bytes(b"fake-image")
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.thumbnail = Mock()
            mock_img.save = Mock()
            mock_open.return_value = mock_img
            
            result = await file_handler.generate_thumbnail(
                source_path,
                thumbnail_path,
                size=(320, 240)
            )
            
            assert result == thumbnail_path
    
    @pytest.mark.asyncio
    async def test_extract_video_frames(self, file_handler, tmp_path):
        """Test extracting frames from video."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock video with 100 frames
            mock_cap.return_value.get.return_value = 100
            mock_cap.return_value.read.side_effect = [
                (True, np.zeros((480, 640, 3), dtype=np.uint8))
                for _ in range(30)
            ] + [(False, None)]
            mock_cap.return_value.release = Mock()
            
            with patch('cv2.resize', return_value=np.zeros((224, 224, 3), dtype=np.uint8)):
                with patch('cv2.cvtColor', return_value=np.zeros((224, 224, 3), dtype=np.uint8)):
                    frames = await file_handler.extract_video_frames(
                        video_path,
                        max_frames=30
                    )
                    
                    assert isinstance(frames, list)
                    assert len(frames) <= 30
    
    @pytest.mark.asyncio
    async def test_compress_image(self, file_handler, tmp_path):
        """Test image compression."""
        image_path = tmp_path / "image.jpg"
        image_path.write_bytes(b"fake-image")
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.mode = 'RGB'
            mock_img.save = Mock()
            mock_open.return_value = mock_img
            
            result = await file_handler.compress_media(
                image_path,
                compression_level="medium"
            )
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_compress_video(self, file_handler, tmp_path):
        """Test video compression."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        
        result = await file_handler.compress_media(
            video_path,
            compression_level="medium"
        )
        
        # Currently returns original file
        assert result is not None
    
    def test_validate_file_type_valid(self, file_handler):
        """Test file type validation with valid type."""
        assert file_handler.validate_file_type("video.mp4") is True
        assert file_handler.validate_file_type("image.jpg") is True
        assert file_handler.validate_file_type("photo.png") is True
    
    def test_validate_file_type_invalid(self, file_handler):
        """Test file type validation with invalid type."""
        assert file_handler.validate_file_type("document.pdf") is False
        assert file_handler.validate_file_type("script.exe") is False
    
    def test_validate_file_size_valid(self, file_handler):
        """Test file size validation with valid size."""
        assert file_handler.validate_file_size(1024) is True
        assert file_handler.validate_file_size(1024 * 1024) is True
    
    def test_validate_file_size_invalid(self, file_handler):
        """Test file size validation with oversized file."""
        assert file_handler.validate_file_size(file_handler.max_file_size + 1) is False
    
    @pytest.mark.asyncio
    async def test_get_video_info(self, file_handler, tmp_path):
        """Test getting video information."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake-video")
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.get.side_effect = [
                1920,  # width
                1080,  # height
                30.0,  # fps
                900    # frame count
            ]
            mock_cap.return_value.release = Mock()
            
            info = await file_handler.get_video_info(video_path)
            
            assert info["width"] == 1920
            assert info["height"] == 1080
            assert info["fps"] == 30.0
            assert info["frame_count"] == 900
            assert "duration" in info