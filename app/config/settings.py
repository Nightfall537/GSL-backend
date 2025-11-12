"""
Application Settings and Configuration

Manages environment variables and application configuration using Pydantic settings.
Supports different environments (development, testing, production) with appropriate defaults.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "GSL Backend"
    environment: str = "development"
    debug: bool = True
    secret_key: str = "your-secret-key-change-in-production"
    
    # Database - Supabase
    supabase_url: str = "https://your-project.supabase.co"
    supabase_key: str = "your-supabase-anon-key"
    supabase_service_key: str = "your-supabase-service-role-key"
    database_url: str = "postgresql://postgres:password@db.your-project.supabase.co:5432/postgres"
    database_echo: bool = False
    use_supabase_auth: bool = True  # Use Supabase Auth instead of custom JWT
    
    # Redis Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 3600  # 1 hour default TTL
    
    # Authentication
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    
    # CORS
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # File Storage
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: List[str] = ["mp4", "mov", "avi", "jpg", "jpeg", "png"]
    
    # AI Models
    ai_models_dir: str = "models"
    cv_model_path: str = "models/gsl_recognition.tflite"
    stt_model_name: str = "openai/whisper-base"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    ai_rate_limit_requests: int = 20  # Lower limit for AI endpoints
    
    # Performance
    max_workers: int = 4
    request_timeout: int = 30
    ai_processing_timeout: int = 10
    
    @validator("allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        valid_envs = ["development", "testing", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()