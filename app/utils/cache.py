"""
Cache Manager

Provides Redis caching utilities for performance optimization
and offline data synchronization support.
"""

import json
import pickle
from typing import Any, Optional
from datetime import timedelta
import redis.asyncio as aioredis

from app.config.settings import get_settings

settings = get_settings()


class CacheManager:
    """Manager for Redis caching operations."""
    
    def __init__(self):
        self.redis_url = settings.redis_url
        self.default_ttl = settings.cache_ttl
        self._redis = None
    
    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False
                )
            except Exception as e:
                print(f"Redis connection error: {e}")
                self._redis = None
        return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return None
            
            value = await redis.get(key)
            if value is None:
                return None
            
            # Try to unpickle, fall back to JSON
            try:
                return pickle.loads(value)
            except:
                return json.loads(value)
                
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return False
            
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except:
                serialized = json.dumps(value)
            
            # Set with TTL
            expire_time = ttl if ttl is not None else self.default_ttl
            await redis.setex(key, expire_time, serialized)
            
            return True
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Supports wildcard patterns (e.g., "user:*").
        
        Args:
            key: Cache key or pattern
            
        Returns:
            True if successful
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return False
            
            # Check if key contains wildcard
            if '*' in key or '?' in key:
                return await self.clear_pattern(key) > 0
            
            await redis.delete(key)
            return True
            
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return False
            
            return await redis.exists(key) > 0
            
        except Exception as e:
            print(f"Cache exists error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "user:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return 0
            
            keys = await redis.keys(pattern)
            if keys:
                return await redis.delete(*keys)
            return 0
            
        except Exception as e:
            print(f"Cache clear pattern error: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment counter in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None if failed
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return None
            
            return await redis.incrby(key, amount)
            
        except Exception as e:
            print(f"Cache increment error: {e}")
            return None
    
    async def get_many(self, keys: list) -> dict:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return {}
            
            values = await redis.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = pickle.loads(value)
                    except:
                        try:
                            result[key] = json.loads(value)
                        except:
                            result[key] = value
            
            return result
            
        except Exception as e:
            print(f"Cache get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: dict, ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            redis = await self._get_redis()
            if redis is None:
                return False
            
            pipe = redis.pipeline()
            expire_time = ttl if ttl is not None else self.default_ttl
            
            for key, value in mapping.items():
                try:
                    serialized = pickle.dumps(value)
                except:
                    serialized = json.dumps(value)
                
                pipe.setex(key, expire_time, serialized)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            print(f"Cache set_many error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Global cache instance
_cache_instance = None


def get_cache() -> CacheManager:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance