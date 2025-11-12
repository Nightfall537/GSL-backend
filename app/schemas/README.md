# GSL Backend - Pydantic Schemas

This directory contains all Pydantic models for API request/response validation in the GSL (Ghanaian Sign Language) learning platform backend.

## Structure

```
app/schemas/
├── __init__.py          # Package exports
├── user.py              # User authentication and profile schemas
├── gsl.py               # GSL sign recognition and translation schemas
├── learning.py          # Learning, lessons, and progress schemas
├── common.py            # Common/shared schemas (pagination, errors, etc.)
├── media.py             # Media upload and processing schemas
├── analytics.py         # Analytics and reporting schemas
└── README.md            # This file
```

## Schema Files

### user.py
User-related schemas for authentication and profile management:
- **Request**: `UserCreate`, `UserLogin`, `UserUpdate`, `PasswordChange`
- **Response**: `UserResponse`, `UserProfile`, `UserLoginResponse`, `UserStatistics`
- **Enums**: `AgeGroup`, `LearningLevel`, `UserRole`

### gsl.py
GSL sign language operations:
- **Request**: `SignRecognitionRequest`, `TextToSignRequest`, `SpeechToSignRequest`, `SignCreateRequest`
- **Response**: `GSLSignResponse`, `SignRecognitionResponse`, `TranslationResponse`, `DictionarySearchResponse`
- **Enums**: `SignCategory`, `DifficultyLevel`, `RecognitionStatus`, `TranslationType`

### learning.py
Learning and educational content:
- **Request**: `LessonSearchRequest`, `LessonProgressUpdate`, `PracticeSessionCreate`, `QuizSubmission`
- **Response**: `LessonResponse`, `AchievementResponse`, `QuizResultResponse`, `LearningAnalytics`
- **Enums**: `LessonLevel`, `LessonCategory`, `ExerciseType`, `AchievementType`

### common.py
Shared schemas used across the API:
- **Generic**: `PaginatedResponse[T]`, `SearchResponse[T]`
- **Responses**: `SuccessResponse`, `ErrorResponse`, `HealthCheckResponse`
- **Utilities**: `PaginationParams`, `PaginationMeta`, `ErrorDetail`
- **Enums**: `StatusEnum`, `ErrorCode`

### media.py
Media file handling:
- **Request**: `MediaUploadRequest`, `VideoUploadRequest`, `ImageUploadRequest`, `MediaProcessingRequest`
- **Response**: `MediaResponse`, `VideoResponse`, `ImageResponse`, `MediaProcessingResponse`
- **Enums**: `MediaType`, `ProcessingStatus`, `MediaQuality`

### analytics.py
Analytics and reporting:
- **Request**: `AnalyticsRequest`, `UserAnalyticsRequest`, `LessonAnalyticsRequest`
- **Response**: `AnalyticsResponse`, `DashboardMetrics`, `TrendAnalysis`, `InsightResponse`
- **Metrics**: `UserEngagementMetrics`, `LearningProgressMetrics`, `SignRecognitionMetrics`
- **Enums**: `TimeRange`, `MetricType`, `AggregationType`

## Usage Examples

### Basic Request Validation

```python
from app.schemas import UserCreate, UserLogin

# Registration
user_data = UserCreate(
    username="john_doe",
    email="john@example.com",
    password="SecurePass123",
    age_group="adult",
    learning_level="beginner"
)

# Login
login_data = UserLogin(
    email="john@example.com",
    password="SecurePass123"
)
```

### Response Models

```python
from app.schemas import UserResponse, GSLSignResponse
from fastapi import APIRouter

router = APIRouter()

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    # Your logic here
    return user_data

@router.get("/signs/{sign_id}", response_model=GSLSignResponse)
async def get_sign(sign_id: str):
    # Your logic here
    return sign_data
```

### Pagination

```python
from app.schemas import PaginatedResponse, LessonResponse, PaginationParams

@router.get("/lessons", response_model=PaginatedResponse[LessonResponse])
async def list_lessons(pagination: PaginationParams):
    # Use pagination.offset and pagination.limit
    lessons = get_lessons(offset=pagination.offset, limit=pagination.limit)
    
    return PaginatedResponse(
        items=lessons,
        pagination=PaginationMeta(
            page=pagination.page,
            page_size=pagination.page_size,
            total_items=total_count,
            total_pages=(total_count + pagination.page_size - 1) // pagination.page_size,
            has_next=pagination.page * pagination.page_size < total_count,
            has_previous=pagination.page > 1
        )
    )
```

### Error Handling

```python
from app.schemas import ErrorResponse, ErrorCode, ErrorDetail
from fastapi import HTTPException

def raise_validation_error(field: str, message: str):
    raise HTTPException(
        status_code=422,
        detail=ErrorResponse(
            error_code=ErrorCode.VALIDATION_ERROR,
            message="Validation failed",
            details=[
                ErrorDetail(field=field, message=message, code="invalid_value")
            ]
        ).dict()
    )
```

### Translation Request

```python
from app.schemas import TextToSignRequest, TranslationResponse

@router.post("/translate/text-to-sign", response_model=TranslationResponse)
async def translate_text_to_sign(request: TextToSignRequest):
    # Process translation
    signs = translate_service.text_to_sign(
        text=request.text,
        include_fingerspelling=request.include_fingerspelling,
        grammar_rules=request.grammar_rules
    )
    
    return TranslationResponse(
        source_type="text_to_sign",
        target_type="gsl_signs",
        source_content=request.text,
        translated_signs=signs,
        confidence_score=0.95,
        message="Translation successful"
    )
```

### Analytics Request

```python
from app.schemas import AnalyticsRequest, AnalyticsResponse, TimeRange, MetricType

@router.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: AnalyticsRequest):
    data = analytics_service.get_metrics(
        metric_type=request.metric_type,
        time_range=request.time_range,
        aggregation=request.aggregation
    )
    
    return data
```

## Validation Features

All schemas include:
- **Type validation**: Automatic type checking
- **Field constraints**: min/max length, value ranges, patterns
- **Custom validators**: Business logic validation
- **Default values**: Sensible defaults for optional fields
- **Documentation**: Field descriptions for API docs

## Best Practices

1. **Use appropriate schemas**: Choose request schemas for inputs, response schemas for outputs
2. **Leverage enums**: Use provided enums for consistent values
3. **Handle errors**: Use `ErrorResponse` for consistent error formatting
4. **Paginate large results**: Use `PaginatedResponse` for list endpoints
5. **Document fields**: Field descriptions auto-generate API documentation
6. **Validate early**: Let Pydantic catch validation errors before processing

## Integration with FastAPI

These schemas integrate seamlessly with FastAPI:

```python
from fastapi import FastAPI, Depends
from app.schemas import UserCreate, UserResponse, ErrorResponse

app = FastAPI()

@app.post(
    "/users",
    response_model=UserResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        409: {"model": ErrorResponse, "description": "User Already Exists"}
    }
)
async def create_user(user: UserCreate):
    # Pydantic automatically validates the request body
    # Returns 422 if validation fails
    return await user_service.create(user)
```

## Testing

Example test using schemas:

```python
from app.schemas import UserCreate

def test_user_creation_validation():
    # Valid user
    user = UserCreate(
        username="test_user",
        email="test@example.com",
        password="SecurePass123"
    )
    assert user.username == "test_user"
    
    # Invalid password (too short)
    with pytest.raises(ValueError):
        UserCreate(
            username="test",
            email="test@example.com",
            password="short"
        )
```

## Contributing

When adding new schemas:
1. Place them in the appropriate file based on domain
2. Add exports to `__init__.py`
3. Include field descriptions and validation
4. Add usage examples to this README
5. Write tests for custom validators
