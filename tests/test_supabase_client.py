"""
Unit Tests for Supabase Client

Tests Supabase integration for authentication, database, and storage.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from app.core.supabase_client import SupabaseManager


class TestSupabaseManager:
    """Test cases for SupabaseManager."""
    
    @pytest.fixture
    def supabase_manager(self, test_settings):
        """Create SupabaseManager instance."""
        with patch('app.core.supabase_client.settings', test_settings):
            return SupabaseManager()
    
    @pytest.mark.asyncio
    async def test_sign_up_success(self, supabase_manager):
        """Test successful user sign up."""
        email = "test@example.com"
        password = "SecurePass123!"
        metadata = {"full_name": "Test User"}
        
        mock_response = Mock(
            user=Mock(id="user-id", email=email),
            session=Mock(access_token="token")
        )
        
        with patch.object(supabase_manager.client.auth, 'sign_up', return_value=mock_response):
            result = await supabase_manager.sign_up(email, password, metadata)
            
            assert result["success"] is True
            assert result["user"].email == email
    
    @pytest.mark.asyncio
    async def test_sign_up_failure(self, supabase_manager):
        """Test failed user sign up."""
        email = "test@example.com"
        password = "weak"
        
        with patch.object(supabase_manager.client.auth, 'sign_up', side_effect=Exception("Weak password")):
            result = await supabase_manager.sign_up(email, password)
            
            assert result["success"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    async def test_sign_in_success(self, supabase_manager):
        """Test successful user sign in."""
        email = "test@example.com"
        password = "SecurePass123!"
        
        mock_response = Mock(
            user=Mock(id="user-id", email=email),
            session=Mock(access_token="token", refresh_token="refresh")
        )
        
        with patch.object(supabase_manager.client.auth, 'sign_in_with_password', return_value=mock_response):
            result = await supabase_manager.sign_in(email, password)
            
            assert result["success"] is True
            assert "access_token" in result
            assert "refresh_token" in result
    
    @pytest.mark.asyncio
    async def test_get_user(self, supabase_manager):
        """Test getting user from token."""
        access_token = "valid-token"
        
        mock_response = Mock(user=Mock(id="user-id", email="test@example.com"))
        
        with patch.object(supabase_manager.client.auth, 'get_user', return_value=mock_response):
            result = await supabase_manager.get_user(access_token)
            
            assert result is not None
            assert result.id == "user-id"
    
    @pytest.mark.asyncio
    async def test_select_data(self, supabase_manager):
        """Test selecting data from table."""
        table = "gsl_signs"
        
        mock_response = Mock(data=[
            {"id": "1", "sign_name": "hello"},
            {"id": "2", "sign_name": "thank_you"}
        ])
        
        with patch.object(supabase_manager.client, 'table') as mock_table:
            mock_table.return_value.select.return_value.execute.return_value = mock_response
            
            result = await supabase_manager.select(table)
            
            assert isinstance(result, list)
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_select_with_filters(self, supabase_manager):
        """Test selecting data with filters."""
        table = "gsl_signs"
        filters = {"difficulty_level": 1}
        
        mock_response = Mock(data=[{"id": "1", "sign_name": "hello"}])
        
        with patch.object(supabase_manager.client, 'table') as mock_table:
            mock_query = Mock()
            mock_query.eq.return_value.execute.return_value = mock_response
            mock_table.return_value.select.return_value = mock_query
            
            result = await supabase_manager.select(table, filters=filters)
            
            assert isinstance(result, list)
    
    @pytest.mark.asyncio
    async def test_insert_data(self, supabase_manager):
        """Test inserting data into table."""
        table = "gsl_signs"
        data = {"sign_name": "hello", "difficulty_level": 1}
        
        mock_response = Mock(data=[{"id": "new-id", **data}])
        
        with patch.object(supabase_manager.client, 'table') as mock_table:
            mock_table.return_value.insert.return_value.execute.return_value = mock_response
            
            result = await supabase_manager.insert(table, data)
            
            assert result is not None
            assert result["sign_name"] == "hello"
    
    @pytest.mark.asyncio
    async def test_update_data(self, supabase_manager):
        """Test updating data in table."""
        table = "gsl_signs"
        data = {"difficulty_level": 2}
        filters = {"id": "sign-id"}
        
        mock_response = Mock(data=[{"id": "sign-id", "difficulty_level": 2}])
        
        with patch.object(supabase_manager.client, 'table') as mock_table:
            mock_query = Mock()
            mock_query.eq.return_value.execute.return_value = mock_response
            mock_table.return_value.update.return_value = mock_query
            
            result = await supabase_manager.update(table, data, filters)
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_delete_data(self, supabase_manager):
        """Test deleting data from table."""
        table = "gsl_signs"
        filters = {"id": "sign-id"}
        
        with patch.object(supabase_manager.client, 'table') as mock_table:
            mock_query = Mock()
            mock_query.eq.return_value.execute.return_value = Mock()
            mock_table.return_value.delete.return_value = mock_query
            
            result = await supabase_manager.delete(table, filters)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_upload_file(self, supabase_manager, sample_video_bytes):
        """Test uploading file to storage."""
        bucket = "user-uploads"
        path = "user/video.mp4"
        
        with patch.object(supabase_manager.client.storage, 'from_') as mock_storage:
            mock_bucket = Mock()
            mock_bucket.upload.return_value = Mock()
            mock_bucket.get_public_url.return_value = "https://example.com/video.mp4"
            mock_storage.return_value = mock_bucket
            
            result = await supabase_manager.upload_file(
                bucket,
                path,
                sample_video_bytes,
                "video/mp4"
            )
            
            assert result is not None
            assert result.startswith("https://")
    
    @pytest.mark.asyncio
    async def test_download_file(self, supabase_manager):
        """Test downloading file from storage."""
        bucket = "user-uploads"
        path = "user/video.mp4"
        
        with patch.object(supabase_manager.client.storage, 'from_') as mock_storage:
            mock_bucket = Mock()
            mock_bucket.download.return_value = b"file-data"
            mock_storage.return_value = mock_bucket
            
            result = await supabase_manager.download_file(bucket, path)
            
            assert result == b"file-data"
    
    @pytest.mark.asyncio
    async def test_delete_file(self, supabase_manager):
        """Test deleting file from storage."""
        bucket = "user-uploads"
        path = "user/video.mp4"
        
        with patch.object(supabase_manager.client.storage, 'from_') as mock_storage:
            mock_bucket = Mock()
            mock_bucket.remove.return_value = Mock()
            mock_storage.return_value = mock_bucket
            
            result = await supabase_manager.delete_file(bucket, path)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_get_public_url(self, supabase_manager):
        """Test getting public URL for file."""
        bucket = "sign-videos"
        path = "hello.mp4"
        
        with patch.object(supabase_manager.client.storage, 'from_') as mock_storage:
            mock_bucket = Mock()
            mock_bucket.get_public_url.return_value = "https://example.com/hello.mp4"
            mock_storage.return_value = mock_bucket
            
            result = await supabase_manager.get_public_url(bucket, path)
            
            assert result.startswith("https://")
    
    @pytest.mark.asyncio
    async def test_call_function(self, supabase_manager):
        """Test calling RPC function."""
        function_name = "get_user_stats"
        params = {"user_id": "user-id"}
        
        mock_response = Mock(data={"total_lessons": 5})
        
        with patch.object(supabase_manager.client, 'rpc') as mock_rpc:
            mock_rpc.return_value.execute.return_value = mock_response
            
            result = await supabase_manager.call_function(function_name, params)
            
            assert result == {"total_lessons": 5}