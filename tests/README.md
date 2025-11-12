# GSL Backend Test Suite

## Overview
Comprehensive test suite for the Ghanaian Sign Language Backend API.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_user_service.py          # User management tests
├── test_recognition_service.py   # Gesture recognition tests
├── test_translation_service.py   # Translation tests
├── test_learning_service.py      # Learning content tests
├── test_media_service.py         # Media handling tests
├── test_computer_vision.py       # AI vision model tests
├── test_speech_to_text.py        # Speech recognition tests
├── test_nlp_processor.py         # NLP processing tests
├── test_cache.py                 # Redis caching tests
├── test_file_handler.py          # File operations tests
└── test_supabase_client.py       # Supabase integration tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_user_service.py
```

### Run Specific Test Class
```bash
pytest tests/test_user_service.py::TestUserService
```

### Run Specific Test Method
```bash
pytest tests/test_user_service.py::TestUserService::test_hash_password
```

### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only async tests
pytest -m asyncio
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=app --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

### Run with Verbose Output
```bash
pytest -v
```

### Run and Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

## Test Categories

### Unit Tests
Test individual components in isolation with mocked dependencies.

**Files:**
- `test_user_service.py`
- `test_recognition_service.py`
- `test_translation_service.py`
- `test_learning_service.py`
- `test_media_service.py`
- `test_computer_vision.py`
- `test_speech_to_text.py`
- `test_nlp_processor.py`
- `test_cache.py`
- `test_file_handler.py`
- `test_supabase_client.py`

### Integration Tests
Test multiple components working together (to be added).

### End-to-End Tests
Test complete user workflows (to be added).

## Test Coverage Goals

| Component | Target Coverage | Current Status |
|-----------|----------------|----------------|
| Services | 80%+ | ✅ Tests written |
| AI Modules | 70%+ | ✅ Tests written |
| Utilities | 80%+ | ✅ Tests written |
| Core | 80%+ | ✅ Tests written |
| API Endpoints | 80%+ | ⏳ To be added |

## Writing New Tests

### Test File Template
```python
"""
Unit Tests for [Component Name]

Brief description of what is being tested.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.[module].[component] import [ClassName]


class Test[ClassName]:
    """Test cases for [ClassName]."""
    
    @pytest.fixture
    def component_instance(self):
        """Create component instance for testing."""
        return [ClassName]()
    
    def test_something(self, component_instance):
        """Test description."""
        # Arrange
        input_data = "test"
        
        # Act
        result = component_instance.method(input_data)
        
        # Assert
        assert result == expected_value
```

### Async Test Template
```python
@pytest.mark.asyncio
async def test_async_method(self, component_instance):
    """Test async method."""
    result = await component_instance.async_method()
    assert result is not None
```

### Mock Template
```python
def test_with_mock(self, component_instance):
    """Test with mocked dependency."""
    with patch.object(component_instance, 'dependency') as mock_dep:
        mock_dep.return_value = "mocked_value"
        
        result = component_instance.method()
        
        assert result == "expected"
        mock_dep.assert_called_once()
```

## Fixtures

### Available Fixtures (from conftest.py)

- `test_settings` - Test configuration settings
- `test_db` - Test database session
- `mock_supabase` - Mocked Supabase client
- `sample_user_data` - Sample user data
- `sample_gsl_sign` - Sample GSL sign data
- `sample_lesson` - Sample lesson data
- `sample_video_bytes` - Sample video data
- `sample_audio_bytes` - Sample audio data
- `sample_image_bytes` - Sample image data

### Using Fixtures
```python
def test_with_fixture(self, sample_user_data):
    """Test using fixture."""
    assert sample_user_data["username"] == "testuser"
```

## Best Practices

### 1. Test Naming
- Use descriptive names: `test_user_registration_with_valid_data`
- Follow pattern: `test_[what]_[condition]_[expected_result]`

### 2. Test Structure (AAA Pattern)
```python
def test_example(self):
    # Arrange - Set up test data
    input_data = "test"
    
    # Act - Execute the code being tested
    result = function(input_data)
    
    # Assert - Verify the result
    assert result == expected
```

### 3. Test Independence
- Each test should be independent
- Don't rely on test execution order
- Clean up after tests

### 4. Mock External Dependencies
- Mock database calls
- Mock API calls
- Mock file system operations
- Mock AI model inference

### 5. Test Edge Cases
- Empty inputs
- None values
- Invalid data
- Boundary conditions
- Error conditions

## Continuous Integration

Tests are automatically run on:
- Every commit
- Every pull request
- Before deployment

## Troubleshooting

### Tests Failing Locally

1. **Check dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check environment variables**
   ```bash
   cp .env.example .env
   # Update .env with test values
   ```

3. **Clear pytest cache**
   ```bash
   pytest --cache-clear
   ```

### Slow Tests

1. **Run specific tests**
   ```bash
   pytest tests/test_user_service.py
   ```

2. **Skip slow tests**
   ```bash
   pytest -m "not slow"
   ```

### Import Errors

1. **Check PYTHONPATH**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:${PWD}"
   ```

2. **Install package in development mode**
   ```bash
   pip install -e .
   ```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Python Mock](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Note**: These tests are written but not executed. Run `pytest` to execute them after setting up your environment.