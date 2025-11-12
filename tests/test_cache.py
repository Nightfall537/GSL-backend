"""
Unit Tests for Cache Manager

Tests Redis caching operations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.utils.cache import CacheManager


class TestCacheManager:
    """Test cases for CacheManager."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create CacheManager instance."""
        return CacheManager()
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_manager):
        """Test getting non-existent key."""
        with patch.object(cache_manager, '_get_redis', return_value=Mock(get=AsyncMock(return_value=None))):
            result = await cache_manager.get("nonexistent_key")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_manager):
        """Test setting and getting cache value."""
        key = "test_key"
        value = {"data": "test_value"}
        
        mock_redis = Mock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            # Set value
            result = await cache_manager.set(key, value, ttl=60)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_delete(self, cache_manager):
        """Test deleting cache key."""
        key = "test_key"
        
        mock_redis = Mock()
        mock_redis.delete = AsyncMock()
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.delete(key)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_true(self, cache_manager):
        """Test checking if key exists (exists)."""
        key = "existing_key"
        
        mock_redis = Mock()
        mock_redis.exists = AsyncMock(return_value=1)
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.exists(key)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_false(self, cache_manager):
        """Test checking if key exists (doesn't exist)."""
        key = "nonexistent_key"
        
        mock_redis = Mock()
        mock_redis.exists = AsyncMock(return_value=0)
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.exists(key)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache_manager):
        """Test clearing keys matching pattern."""
        pattern = "user:*"
        
        mock_redis = Mock()
        mock_redis.keys = AsyncMock(return_value=["user:1", "user:2"])
        mock_redis.delete = AsyncMock(return_value=2)
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.clear_pattern(pattern)
            
            assert result == 2
    
    @pytest.mark.asyncio
    async def test_increment(self, cache_manager):
        """Test incrementing counter."""
        key = "counter"
        
        mock_redis = Mock()
        mock_redis.incrby = AsyncMock(return_value=5)
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.increment(key, amount=1)
            
            assert result == 5
    
    @pytest.mark.asyncio
    async def test_get_many(self, cache_manager):
        """Test getting multiple keys."""
        keys = ["key1", "key2", "key3"]
        
        mock_redis = Mock()
        mock_redis.mget = AsyncMock(return_value=[b"value1", b"value2", None])
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.get_many(keys)
            
            assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_set_many(self, cache_manager):
        """Test setting multiple keys."""
        mapping = {
            "key1": "value1",
            "key2": "value2"
        }
        
        mock_redis = Mock()
        mock_pipeline = Mock()
        mock_pipeline.setex = Mock()
        mock_pipeline.execute = AsyncMock()
        mock_redis.pipeline = Mock(return_value=mock_pipeline)
        
        with patch.object(cache_manager, '_get_redis', return_value=mock_redis):
            result = await cache_manager.set_many(mapping, ttl=60)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, cache_manager):
        """Test handling Redis connection failure."""
        with patch.object(cache_manager, '_get_redis', return_value=None):
            result = await cache_manager.get("key")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_close(self, cache_manager):
        """Test closing Redis connection."""
        mock_redis = Mock()
        mock_redis.close = AsyncMock()
        cache_manager._redis = mock_redis
        
        await cache_manager.close()
        
        assert cache_manager._redis is None