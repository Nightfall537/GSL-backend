"""
GSL Backend - Schemas Usage Demo

This script demonstrates how to use the Pydantic schemas for validation
and data handling in the GSL learning platform.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from uuid import uuid4

# Import schemas
from app.schemas import (
    # User schemas
    UserCreate, UserLogin, UserResponse, AgeGroup, LearningLevel,
    # GSL schemas
    TextToSignRequest, SignRecognitionRequest, GSLSignResponse,
    SignCategory, DifficultyLevel, TranslationResponse,
    # Learning schemas
    LessonSearchRequest, PracticeSessionCreate, LessonResponse,
    LessonLevel, LessonCategory, PracticeSessionType,
    # Common schemas
    PaginationParams, ErrorResponse, ErrorCode, SuccessResponse,
    # Media schemas
    MediaUploadRequest, MediaType, VideoUploadRequest,
    # Analytics schemas
    AnalyticsRequest, TimeRange, MetricType, AggregationType
)


def demo_user_schemas():
    """Demonstrate user schema validation."""
    print("=" * 60)
    print("USER SCHEMAS DEMO")
    print("=" * 60)
    
    # Create user request
    print("\n1. User Registration:")
    try:
        user_create = UserCreate(
            username="john_doe",
            email="john@example.com",
            password="SecurePass123",
            full_name="John Doe",
            age_group=AgeGroup.adult,
            learning_level=LearningLevel.beginner,
            preferred_language="english"
        )
        print(f"✓ Valid user creation: {user_create.username}")
        print(f"  Email: {user_create.email}")
        print(f"  Age Group: {user_create.age_group.value}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    
    # Invalid password
    print("\n2. Invalid Password (too short):")
    try:
        invalid_user = UserCreate(
            username="test",
            email="test@example.com",
            password="short"  # Too short
        )
    except ValueError as e:
        print(f"✓ Caught validation error: {e}")
    
    # Login request
    print("\n3. User Login:")
    login = UserLogin(
        email="john@example.com",
        password="SecurePass123"
    )
    print(f"✓ Login request created for: {login.email}")


def demo_gsl_schemas():
    """Demonstrate GSL schema validation."""
    print("\n" + "=" * 60)
    print("GSL SCHEMAS DEMO")
    print("=" * 60)
    
    # Text to sign translation
    print("\n1. Text-to-Sign Translation Request:")
    text_to_sign = TextToSignRequest(
        text="Hello, how are you?",
        include_fingerspelling=False,
        grammar_rules=True,
        simplify_text=True
    )
    print(f"✓ Translation request: '{text_to_sign.text}'")
    print(f"  Grammar rules: {text_to_sign.grammar_rules}")
    
    # Sign recognition request
    print("\n2. Sign Recognition Request:")
    recognition = SignRecognitionRequest(
        media_type="video",
        confidence_threshold=0.75,
        include_alternatives=True,
        max_alternatives=3
    )
    print(f"✓ Recognition request for {recognition.media_type}")
    print(f"  Confidence threshold: {recognition.confidence_threshold}")
    
    # GSL Sign response
    print("\n3. GSL Sign Response:")
    sign = GSLSignResponse(
        id=uuid4(),
        sign_name="HELLO",
        description="Greeting sign for hello",
        category=SignCategory.greetings,
        difficulty_level=DifficultyLevel.beginner,
        video_url="https://example.com/signs/hello.mp4",
        thumbnail_url="https://example.com/signs/hello_thumb.jpg",
        usage_examples=["Hello!", "Hi there!"],
        related_signs=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    print(f"✓ Sign: {sign.sign_name}")
    print(f"  Category: {sign.category.value}")
    print(f"  Difficulty: {sign.difficulty_level.value}")


def demo_learning_schemas():
    """Demonstrate learning schema validation."""
    print("\n" + "=" * 60)
    print("LEARNING SCHEMAS DEMO")
    print("=" * 60)
    
    # Lesson search
    print("\n1. Lesson Search Request:")
    search = LessonSearchRequest(
        level=LessonLevel.beginner,
        category=LessonCategory.greetings,
        completed=False,
        limit=10,
        offset=0
    )
    print(f"✓ Searching for {search.level.name} lessons")
    print(f"  Category: {search.category.value}")
    print(f"  Limit: {search.limit}")
    
    # Practice session
    print("\n2. Practice Session Creation:")
    practice = PracticeSessionCreate(
        session_type=PracticeSessionType.free_practice,
        signs_practiced=[uuid4(), uuid4(), uuid4()],
        duration_seconds=600,
        accuracy_score=85.5,
        notes="Good practice session"
    )
    print(f"✓ Practice session: {practice.session_type.value}")
    print(f"  Duration: {practice.duration_seconds}s")
    print(f"  Accuracy: {practice.accuracy_score}%")
    print(f"  Signs practiced: {len(practice.signs_practiced)}")


def demo_common_schemas():
    """Demonstrate common schema usage."""
    print("\n" + "=" * 60)
    print("COMMON SCHEMAS DEMO")
    print("=" * 60)
    
    # Pagination
    print("\n1. Pagination Parameters:")
    pagination = PaginationParams(page=2, page_size=20)
    print(f"✓ Page: {pagination.page}")
    print(f"  Page size: {pagination.page_size}")
    print(f"  Offset: {pagination.offset}")
    print(f"  Limit: {pagination.limit}")
    
    # Success response
    print("\n2. Success Response:")
    success = SuccessResponse(
        message="Operation completed successfully",
        data={"user_id": str(uuid4()), "status": "active"}
    )
    print(f"✓ {success.message}")
    print(f"  Status: {success.status.value}")
    
    # Error response
    print("\n3. Error Response:")
    error = ErrorResponse(
        error_code=ErrorCode.VALIDATION_ERROR,
        message="Invalid input data",
        details=[{
            "field": "email",
            "message": "Invalid email format",
            "code": "invalid_format"
        }]
    )
    print(f"✓ Error: {error.message}")
    print(f"  Code: {error.error_code.value}")


def demo_media_schemas():
    """Demonstrate media schema validation."""
    print("\n" + "=" * 60)
    print("MEDIA SCHEMAS DEMO")
    print("=" * 60)
    
    # Video upload
    print("\n1. Video Upload Request:")
    video_upload = VideoUploadRequest(
        filename="sign_demo.mp4",
        description="Demonstration of greeting signs",
        tags=["greetings", "beginner", "tutorial"],
        is_public=True,
        duration=120,
        resolution="1920x1080",
        fps=30
    )
    print(f"✓ Video upload: {video_upload.filename}")
    print(f"  Duration: {video_upload.duration}s")
    print(f"  Resolution: {video_upload.resolution}")
    print(f"  Tags: {', '.join(video_upload.tags)}")


def demo_analytics_schemas():
    """Demonstrate analytics schema validation."""
    print("\n" + "=" * 60)
    print("ANALYTICS SCHEMAS DEMO")
    print("=" * 60)
    
    # Analytics request
    print("\n1. Analytics Request:")
    analytics = AnalyticsRequest(
        metric_type=MetricType.user_engagement,
        time_range=TimeRange.last_30_days,
        aggregation=AggregationType.average,
        group_by="day"
    )
    print(f"✓ Metric: {analytics.metric_type.value}")
    print(f"  Time range: {analytics.time_range.value}")
    print(f"  Aggregation: {analytics.aggregation.value}")


def demo_validation_errors():
    """Demonstrate validation error handling."""
    print("\n" + "=" * 60)
    print("VALIDATION ERRORS DEMO")
    print("=" * 60)
    
    # Invalid email
    print("\n1. Invalid Email Format:")
    try:
        UserCreate(
            username="test",
            email="not-an-email",  # Invalid
            password="SecurePass123"
        )
    except ValueError as e:
        print(f"✓ Caught: {e}")
    
    # Invalid confidence threshold
    print("\n2. Invalid Confidence Threshold:")
    try:
        SignRecognitionRequest(
            media_type="video",
            confidence_threshold=1.5  # Must be 0.0-1.0
        )
    except ValueError as e:
        print(f"✓ Caught: {e}")
    
    # Invalid page number
    print("\n3. Invalid Page Number:")
    try:
        PaginationParams(page=0, page_size=20)  # Page must be >= 1
    except ValueError as e:
        print(f"✓ Caught: {e}")


def demo_json_serialization():
    """Demonstrate JSON serialization."""
    print("\n" + "=" * 60)
    print("JSON SERIALIZATION DEMO")
    print("=" * 60)
    
    # Create a sign
    sign = GSLSignResponse(
        id=uuid4(),
        sign_name="THANK YOU",
        description="Expression of gratitude",
        category=SignCategory.greetings,
        difficulty_level=DifficultyLevel.beginner,
        video_url="https://example.com/signs/thank_you.mp4",
        thumbnail_url="https://example.com/signs/thank_you_thumb.jpg",
        usage_examples=["Thank you", "Thanks"],
        related_signs=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Convert to JSON
    print("\n1. Model to JSON:")
    json_data = sign.model_dump_json(indent=2)
    print(json_data[:200] + "...")
    
    # Convert to dict
    print("\n2. Model to Dict:")
    dict_data = sign.model_dump()
    print(f"✓ Converted to dict with {len(dict_data)} fields")
    print(f"  Keys: {', '.join(list(dict_data.keys())[:5])}...")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("GSL BACKEND - PYDANTIC SCHEMAS USAGE DEMO")
    print("=" * 60)
    
    try:
        demo_user_schemas()
        demo_gsl_schemas()
        demo_learning_schemas()
        demo_common_schemas()
        demo_media_schemas()
        demo_analytics_schemas()
        demo_validation_errors()
        demo_json_serialization()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Schemas provide automatic validation")
        print("2. Type hints ensure data consistency")
        print("3. Enums enforce valid values")
        print("4. Custom validators add business logic")
        print("5. Easy JSON serialization/deserialization")
        print("6. Clear error messages for invalid data")
        
    except Exception as e:
        print(f"\n✗ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
