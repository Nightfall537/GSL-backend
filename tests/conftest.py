"""
Pytest Configuration and Fixtures

Provides shared fixtures and configuration for all tests.
"""

import pytest
import asyncio
from typing import Generator
from unittest.mock import Mock, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.core.database import Base
from app.core.supabase_client import SupabaseManager
from app.config.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        environment="testing",
        database_url="sqlite:///:memory:",
        supabase_url="https://test.supabase.co",
        supabase_key="test-key",
        supabase_service_key="test-service-key",
        secret_key="test-secret-key",
        debug=True
    )


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """Create test database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def mock_supabase() -> Mock:
    """Create mock Supabase client."""
    mock = Mock(spec=SupabaseManager)
    
    # Mock authentication methods
    mock.sign_up = AsyncMock(return_value={
        "user": Mock(id="test-user-id", email="test@example.com"),
        "session": Mock(access_token="test-token"),
        "success": True
    })
    
    mock.sign_in = AsyncMock(return_value={
        "user": Mock(id="test-user-id", email="test@example.com"),
        "session": Mock(access_token="test-token", refresh_token="refresh-token"),
        "access_token": "test-token",
        "success": True
    })
    
    mock.get_user = AsyncMock(return_value=Mock(
        id="test-user-id",
        email="test@example.com"
    ))
    
    # Mock database methods
    mock.select = AsyncMock(return_value=[])
    mock.insert = AsyncMock(return_value={"id": "test-id"})
    mock.update = AsyncMock(return_value={"id": "test-id"})
    mock.delete = AsyncMock(return_value=True)
    
    # Mock storage methods
    mock.upload_file = AsyncMock(return_value="https://test.supabase.co/storage/test.jpg")
    mock.download_file = AsyncMock(return_value=b"test-data")
    mock.delete_file = AsyncMock(return_value=True)
    
    return mock


@pytest.fixture(scope="function")
def sample_user_data() -> dict:
    """Sample user data for testing."""
    return {
        "id": "test-user-id",
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "SecurePass123!"
    }


@pytest.fixture(scope="function")
def sample_gsl_sign() -> dict:
    """Sample GSL sign data for testing."""
    return {
        "id": "sign-id-1",
        "sign_name": "hello",
        "description": "Greeting sign",
        "category_id": "category-1",
        "difficulty_level": 1,
        "video_url": "https://example.com/hello.mp4",
        "thumbnail_url": "https://example.com/hello_thumb.jpg",
        "related_signs": [],
        "usage_examples": ["Hello, how are you?"]
    }


@pytest.fixture(scope="function")
def sample_lesson() -> dict:
    """Sample lesson data for testing."""
    return {
        "id": "lesson-id-1",
        "title": "Basic Greetings",
        "description": "Learn basic greeting signs",
        "level": 1,
        "category": "greetings",
        "sequence_order": 1,
        "signs_covered": ["sign-id-1", "sign-id-2"],
        "estimated_duration": 15
    }


@pytest.fixture(scope="function")
def sample_video_bytes() -> bytes:
    """Sample video data for testing."""
    return b"fake-video-data-for-testing"


@pytest.fixture(scope="function")
def sample_audio_bytes() -> bytes:
    """Sample audio data for testing."""
    return b"fake-audio-data-for-testing"


@pytest.fixture(scope="function")
def sample_image_bytes() -> bytes:
    """Sample image data for testing."""
    return b"fake-image-data-for-testing"