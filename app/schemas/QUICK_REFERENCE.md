# Pydantic Schemas - Quick Reference

## Import Patterns

```python
# Import specific schemas
from app.schemas import UserCreate, UserResponse, GSLSignResponse

# Import enums
from app.schemas import SignCategory, LearningLevel, MediaType

# Import common utilities
from app.schemas import PaginatedResponse, ErrorResponse, PaginationParams
```

## Common Patterns

### 1. User Registration & Authentication

```python
from app.schemas import UserCreate, UserLogin, UserResponse

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

### 2. GSL Translation

```python
from app.schemas import TextToSignRequest, TranslationResponse

# Text to sign
request = TextToSignRequest(
    text="Hello, how are you?",
    include_fingerspelling=False,
    grammar_rules=True
)
```

### 3. Sign Recognition

```python
from app.schemas import SignRecognitionRequest, SignRecognitionResponse

request = SignRecognitionRequest(
    media_type="video",
    confidence_threshold=0.75,
    include_alternatives=True
)
```

### 4. Lesson Management

```python
from app.schemas import LessonSearchRequest, LessonResponse

search = LessonSearchRequest(
    level=1,  # beginner
    category="greetings",
    limit=20
)
```

### 5. Practice Sessions

```python
from app.schemas import PracticeSessionCreate

session = PracticeSessionCreate(
    session_type="free_practice",
    signs_practiced=[sign_id1, sign_id2],
    duration_seconds=600,
    accuracy_score=85.5
)
```

### 6. Pagination

```python
from app.schemas import PaginationParams, PaginatedResponse

# Request
pagination = PaginationParams(page=1, page_size=20)

# Response
response = PaginatedResponse(
    items=[...],
    pagination=PaginationMeta(...)
)
```

### 7. Error Handling

```python
from app.schemas import ErrorResponse, ErrorCode, ErrorDetail

error = ErrorResponse(
    error_code=ErrorCode.VALIDATION_ERROR,
    message="Invalid input",
    details=[
        ErrorDetail(
            field="email",
            message="Invalid format",
            code="invalid_format"
        )
    ]
)
```

### 8. Media Upload

```python
from app.schemas import VideoUploadRequest

upload = VideoUploadRequest(
    filename="demo.mp4",
    description="Sign demonstration",
    tags=["tutorial", "beginner"],
    duration=120,
    resolution="1920x1080"
)
```

### 9. Analytics

```python
from app.schemas import AnalyticsRequest, TimeRange, MetricType

analytics = AnalyticsRequest(
    metric_type=MetricType.user_engagement,
    time_range=TimeRange.last_30_days,
    aggregation="average"
)
```

## FastAPI Integration

### Basic Endpoint

```python
from fastapi import APIRouter
from app.schemas import UserResponse

router = APIRouter()

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    return user_data
```

### With Request Body

```python
from app.schemas import UserCreate, UserResponse

@router.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    # user is automatically validated
    return created_user
```

### With Pagination

```python
from app.schemas import PaginatedResponse, LessonResponse, PaginationParams

@router.get("/lessons", response_model=PaginatedResponse[LessonResponse])
async def list_lessons(pagination: PaginationParams = Depends()):
    return paginated_results
```

### With Error Responses

```python
from app.schemas import UserResponse, ErrorResponse

@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    responses={
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse}
    }
)
async def get_user(user_id: str):
    return user_data
```

## Validation Examples

### Valid Data

```python
# ✓ Valid user
user = UserCreate(
    username="john_doe",
    email="john@example.com",
    password="SecurePass123"
)

# ✓ Valid confidence
recognition = SignRecognitionRequest(
    media_type="video",
    confidence_threshold=0.75
)

# ✓ Valid pagination
pagination = PaginationParams(page=1, page_size=20)
```

### Invalid Data (Raises ValidationError)

```python
# ✗ Invalid email
user = UserCreate(
    username="test",
    email="not-an-email",  # Invalid
    password="SecurePass123"
)

# ✗ Invalid confidence (must be 0.0-1.0)
recognition = SignRecognitionRequest(
    media_type="video",
    confidence_threshold=1.5  # Invalid
)

# ✗ Invalid page (must be >= 1)
pagination = PaginationParams(page=0, page_size=20)
```

## Enum Values

### User Enums
```python
AgeGroup: "child", "teen", "adult", "senior"
LearningLevel: "beginner", "intermediate", "advanced"
UserRole: "learner", "teacher", "admin"
```

### GSL Enums
```python
SignCategory: "greetings", "family", "colors", "animals", "food", 
              "numbers", "emotions", "actions", "objects", "places",
              "time", "weather", "clothing", "body_parts", "grammar"

DifficultyLevel: 1 (beginner), 2 (intermediate), 3 (advanced)

RecognitionStatus: "success", "low_confidence", "failed", "processing"

TranslationType: "speech_to_sign", "text_to_sign", "sign_to_text"
```

### Learning Enums
```python
LessonLevel: 1 (beginner), 2 (intermediate), 3 (advanced)

LessonCategory: "greetings", "family", "daily_life", "emotions",
                "numbers", "colors", "food", "animals", "places",
                "actions", "grammar", "conversation"

ExerciseType: "recognition", "translation", "practice", 
              "quiz", "matching", "sequence"

PracticeSessionType: "free_practice", "lesson_practice",
                     "review_session", "challenge"
```

### Media Enums
```python
MediaType: "video", "image", "audio"
ProcessingStatus: "pending", "processing", "completed", "failed"
MediaQuality: "low", "medium", "high", "original"
```

### Analytics Enums
```python
TimeRange: "today", "yesterday", "last_7_days", "last_30_days",
           "last_90_days", "this_month", "last_month", 
           "this_year", "custom"

MetricType: "user_engagement", "learning_progress",
            "sign_recognition", "lesson_completion",
            "practice_sessions", "system_performance"

AggregationType: "sum", "average", "count", "min", "max", "median"
```

## JSON Serialization

### Model to JSON
```python
sign = GSLSignResponse(...)
json_str = sign.model_dump_json()
```

### Model to Dict
```python
sign = GSLSignResponse(...)
dict_data = sign.model_dump()
```

### JSON to Model
```python
json_data = '{"username": "john", ...}'
user = UserCreate.model_validate_json(json_data)
```

### Dict to Model
```python
dict_data = {"username": "john", ...}
user = UserCreate.model_validate(dict_data)
```

## Field Constraints

### Common Constraints
- `min_length` / `max_length`: String length
- `ge` / `le`: Greater/less than or equal (numbers)
- `gt` / `lt`: Greater/less than (numbers)
- `pattern`: Regex pattern matching
- `min_items` / `max_items`: List length

### Examples
```python
username: str = Field(..., min_length=3, max_length=50)
password: str = Field(..., min_length=8)
confidence: float = Field(..., ge=0.0, le=1.0)
page: int = Field(1, ge=1)
tags: List[str] = Field(default_factory=list, max_items=10)
```

## Custom Validators

### Using @validator
```python
from pydantic import validator

class UserCreate(BaseModel):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password too short')
        return v
```

## Tips & Tricks

1. **Use Enums**: Enforce valid values and get autocomplete
2. **Default Factories**: Use `Field(default_factory=list)` for mutable defaults
3. **Optional Fields**: Use `Optional[Type]` for nullable fields
4. **Generic Types**: Use `PaginatedResponse[T]` for reusable patterns
5. **Field Descriptions**: Add descriptions for auto-generated docs
6. **Custom Validators**: Add business logic validation
7. **from_attributes**: Set `Config.from_attributes = True` for ORM models

## Common Mistakes

❌ **Don't**: Use mutable defaults directly
```python
tags: List[str] = []  # Wrong!
```

✓ **Do**: Use default_factory
```python
tags: List[str] = Field(default_factory=list)  # Correct!
```

❌ **Don't**: Forget to validate enums
```python
category: str  # Wrong - any string accepted
```

✓ **Do**: Use enum types
```python
category: SignCategory  # Correct - only valid categories
```

❌ **Don't**: Mix up request and response schemas
```python
@router.post("/users", response_model=UserCreate)  # Wrong!
```

✓ **Do**: Use appropriate schemas
```python
@router.post("/users", response_model=UserResponse)  # Correct!
```

## Testing Schemas

```python
import pytest
from app.schemas import UserCreate

def test_valid_user():
    user = UserCreate(
        username="test",
        email="test@example.com",
        password="SecurePass123"
    )
    assert user.username == "test"

def test_invalid_password():
    with pytest.raises(ValueError):
        UserCreate(
            username="test",
            email="test@example.com",
            password="short"  # Too short
        )
```

## Performance Tips

1. **Reuse schemas**: Don't create new schema classes unnecessarily
2. **Use model_dump()**: Faster than dict() for serialization
3. **Validate once**: Don't re-validate already validated data
4. **Use exclude/include**: Only serialize needed fields
5. **Lazy imports**: Import schemas only when needed

## Resources

- Full documentation: `app/schemas/README.md`
- Demo script: `examples/schemas_usage_demo.py`
- Pydantic docs: https://docs.pydantic.dev/
