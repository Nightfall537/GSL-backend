"""
File Handler Utilities

Provides utilities for file operations including video processing,
image manipulation, compression, and thumbnail generation.
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image
import tempfile

from app.config.settings import get_settings

settings = get_settings()


class FileHandler:
    """Handler for file operations and media processing."""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size
        self.allowed_types = settings.allowed_file_types
    
    async def save_file(self, file_data: bytes, file_path: Path) -> bool:
        """
        Save file to disk.
        
        Args:
            file_data: File bytes
            file_path: Destination path
            
        Returns:
            True if successful
        """
        try:
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            return True
            
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    
    async def generate_thumbnail(
        self,
        source_path: Path,
        thumbnail_path: Path,
        size: Tuple[int, int] = (320, 240)
    ) -> Optional[Path]:
        """
        Generate thumbnail from video or image.
        
        Args:
            source_path: Source file path
            thumbnail_path: Thumbnail destination path
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail path or None if failed
        """
        try:
            # Check if source is video or image
            if source_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
                return await self._generate_video_thumbnail(
                    source_path,
                    thumbnail_path,
                    size
                )
            else:
                return await self._generate_image_thumbnail(
                    source_path,
                    thumbnail_path,
                    size
                )
                
        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            return None
    
    async def _generate_video_thumbnail(
        self,
        video_path: Path,
        thumbnail_path: Path,
        size: Tuple[int, int]
    ) -> Optional[Path]:
        """Generate thumbnail from video first frame."""
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            
            # Read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Resize frame
            resized = cv2.resize(frame, size)
            
            # Save thumbnail
            cv2.imwrite(str(thumbnail_path), resized)
            
            return thumbnail_path
            
        except Exception as e:
            print(f"Error generating video thumbnail: {e}")
            return None
    
    async def _generate_image_thumbnail(
        self,
        image_path: Path,
        thumbnail_path: Path,
        size: Tuple[int, int]
    ) -> Optional[Path]:
        """Generate thumbnail from image."""
        try:
            # Open image
            img = Image.open(image_path)
            
            # Resize maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            img.save(thumbnail_path, quality=85)
            
            return thumbnail_path
            
        except Exception as e:
            print(f"Error generating image thumbnail: {e}")
            return None
    
    async def extract_video_frames(
        self,
        video_path: Path,
        max_frames: int = 30,
        target_size: Tuple[int, int] = (224, 224)
    ) -> List[np.ndarray]:
        """
        Extract frames from video for AI processing.
        
        Args:
            video_path: Video file path
            max_frames: Maximum number of frames to extract
            target_size: Target frame size
            
        Returns:
            List of frame arrays
        """
        try:
            frames = []
            cap = cv2.VideoCapture(str(video_path))
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            interval = max(1, total_frames // max_frames)
            
            frame_count = 0
            while len(frames) < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Resize frame
                    resized = cv2.resize(frame, target_size)
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            print(f"Error extracting video frames: {e}")
            return []
    
    async def compress_media(
        self,
        source_path: Path,
        compression_level: str = "medium"
    ) -> Optional[bytes]:
        """
        Compress media file for low-bandwidth delivery.
        
        Args:
            source_path: Source file path
            compression_level: Compression level (low, medium, high)
            
        Returns:
            Compressed file bytes or None if failed
        """
        try:
            if source_path.suffix.lower() in ['.mp4', '.mov', '.avi']:
                return await self._compress_video(source_path, compression_level)
            else:
                return await self._compress_image(source_path, compression_level)
                
        except Exception as e:
            print(f"Error compressing media: {e}")
            return None
    
    async def _compress_image(
        self,
        image_path: Path,
        compression_level: str
    ) -> Optional[bytes]:
        """Compress image file."""
        try:
            # Quality mapping
            quality_map = {
                "low": 50,
                "medium": 70,
                "high": 85
            }
            quality = quality_map.get(compression_level, 70)
            
            # Open and compress image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Save to bytes
            from io import BytesIO
            output = BytesIO()
            img.save(output, format='JPEG', quality=quality, optimize=True)
            
            return output.getvalue()
            
        except Exception as e:
            print(f"Error compressing image: {e}")
            return None
    
    async def _compress_video(
        self,
        video_path: Path,
        compression_level: str
    ) -> Optional[bytes]:
        """Compress video file."""
        # TODO: Implement video compression using ffmpeg
        # For now, return original file
        try:
            with open(video_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading video: {e}")
            return None
    
    async def compress_video(
        self,
        source_path: Path,
        output_path: Path,
        quality: str = "medium"
    ) -> bool:
        """
        Compress video to output path.
        
        Args:
            source_path: Source video path
            output_path: Output video path
            quality: Quality level
            
        Returns:
            True if successful
        """
        # TODO: Implement with ffmpeg
        # For now, just copy the file
        try:
            import shutil
            shutil.copy(source_path, output_path)
            return True
        except Exception as e:
            print(f"Error compressing video: {e}")
            return False
    
    async def extract_audio(
        self,
        video_path: Path,
        audio_path: Path
    ) -> Optional[Path]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Video file path
            audio_path: Output audio path
            
        Returns:
            Audio file path or None if failed
        """
        # TODO: Implement audio extraction using ffmpeg
        # For now, return None
        print(f"Audio extraction not yet implemented: {video_path} -> {audio_path}")
        return None
    
    def validate_file_type(self, filename: str) -> bool:
        """
        Validate file type by extension.
        
        Args:
            filename: File name
            
        Returns:
            True if valid
        """
        extension = Path(filename).suffix.lstrip('.').lower()
        return extension in self.allowed_types
    
    def validate_file_size(self, file_size: int) -> bool:
        """
        Validate file size.
        
        Args:
            file_size: File size in bytes
            
        Returns:
            True if valid
        """
        return file_size <= self.max_file_size
    
    async def get_video_info(self, video_path: Path) -> dict:
        """
        Get video file information.
        
        Args:
            video_path: Video file path
            
        Returns:
            Dictionary with video info
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }
            
            cap.release()
            return info
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}