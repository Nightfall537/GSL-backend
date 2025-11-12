# GSL Backend - Ghanaian Sign Language Learning Platform

A FastAPI-based backend system that powers an AI-driven mobile application for learning and translating Ghanaian Sign Language. The system enables real-time bidirectional translation between sign language, speech, and text, transforming the static Harmonized GSL dictionary into an interactive learning platform.

## Features

- **AI-Powered Sign Recognition**: Computer vision models for real-time GSL gesture recognition
- **Speech-to-Sign Translation**: Convert Ghanaian English speech to GSL signs
- **Text-to-Sign Translation**: Transform text input into GSL sign demonstrations
- **Interactive Learning**: Structured lessons, tutorials, and gamified progress tracking
- **Offline-First Design**: Optimized for low-bandwidth environments in rural Ghana
- **Cultural Localization**: Support for Ghanaian English accents and local phrases

## Technology Stack

- **Backend Framework**: FastAPI 0.104.1
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis for performance optimization
- **AI/ML**: TensorFlow Lite, Whisper, OpenCV
- **Authentication**: JWT tokens with bcrypt hashing
- **Containerization**: Docker and Docker Compose

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gsl-backend
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose (Recommended)**
   ```bash
   docker-compose up -d
   ```

4. **Or run locally**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Start PostgreSQL and Redis (using Docker)
   docker-compose up -d postgres redis
   
   # Run database migrations
   alembic upgrade head
   
   # Start the application
   uvicorn app.main:app --reload
   ```

### API Documentation

Once the application is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Project Structure

```
gsl-backend/
├── app/
│   ├── api/v1/           # API route handlers
│   ├── core/             # Core utilities (database, security)
│   ├── models/           # SQLAlchemy database models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Business logic services
│   ├── ai/               # AI/ML model integration
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── alembic/              # Database migrations
├── docker-compose.yml    # Development environment
└── requirements.txt      # Python dependencies
```

## API Endpoints

### User Management (`/api/v1/users`)
- `POST /register` - Create new learner account
- `POST /login` - Authenticate and get token
- `GET /profile` - Get user profile and progress
- `PUT /profile` - Update user information
- `GET /progress` - Retrieve learning analytics

### Sign Recognition (`/api/v1/recognition`)
- `POST /recognize` - Upload video/image for gesture recognition
- `GET /confidence/{id}` - Get recognition confidence scores
- `POST /validate` - Validate user's gesture attempt
- `GET /similar/{gesture}` - Get similar gestures

### Translation (`/api/v1/translate`)
- `POST /speech-to-sign` - Convert audio to GSL signs
- `POST /text-to-sign` - Convert text to GSL signs
- `POST /sign-to-text` - Convert gesture to text
- `GET /sign-video/{id}` - Retrieve sign demonstration

### Learning (`/api/v1/learning`)
- `GET /lessons` - Get structured lesson content
- `GET /lessons/{id}` - Get specific lesson details
- `POST /progress` - Update lesson completion
- `GET /achievements` - Get user badges
- `GET /dictionary` - Search GSL dictionary

### Media (`/api/v1/media`)
- `POST /upload` - Upload media files
- `GET /video/{id}` - Stream video content
- `GET /compressed/{id}` - Get compressed media
- `POST /process` - Process media for AI analysis

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_users.py
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://gsl_user:gsl_password@localhost:5432/gsl_db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SECRET_KEY` | JWT signing key | `your-secret-key-change-in-production` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:3000,http://localhost:8080` |
| `AI_MODELS_DIR` | Directory for AI models | `models` |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Harmonized Ghanaian Sign Language Dictionary
- Ghana Federation of the Disabled
- AI for Inclusive Learning Hackathon participants