"""
GSL Backend - Main FastAPI Application

This is the main entry point for the Ghanaian Sign Language learning platform backend.
It provides RESTful API endpoints for AI-powered sign language recognition, translation,
and learning services.
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.config.settings import get_settings
from app.core.database import init_db
from app.core.middleware import setup_middleware
from app.api.v1 import users, recognition, translate, learning, media


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    settings = get_settings()
    await init_db()
    print(f"GSL Backend starting up - Environment: {settings.environment}")
    
    yield
    
    # Shutdown
    print("GSL Backend shutting down")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="GSL Backend API",
        description="""
        ## Ghanaian Sign Language Learning Platform Backend
        
        A FastAPI-based system that powers an AI-driven mobile application for learning 
        and translating Ghanaian Sign Language. The system enables real-time bidirectional 
        translation between sign language, speech, and text.
        
        ### Key Features
        - **AI-Powered Sign Recognition**: Real-time GSL gesture recognition from video/images
        - **Speech-to-Sign Translation**: Convert Ghanaian English speech to GSL signs
        - **Text-to-Sign Translation**: Transform text input into GSL sign demonstrations
        - **Interactive Learning**: Structured lessons with gamified progress tracking
        - **Offline-First Design**: Optimized for low-bandwidth environments
        - **Cultural Localization**: Support for Ghanaian English accents and local phrases
        
        ### Authentication
        Most endpoints require JWT authentication. Use the `/api/v1/users/login` endpoint 
        to obtain an access token, then include it in the Authorization header:
        ```
        Authorization: Bearer <your-token>
        ```
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        contact={
            "name": "GSL Backend Team",
            "email": "support@gsl-backend.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        tags_metadata=[
            {
                "name": "users",
                "description": "User management, authentication, and learning progress tracking",
            },
            {
                "name": "recognition",
                "description": "AI-powered sign recognition from video and image uploads",
            },
            {
                "name": "translation",
                "description": "Bidirectional translation between speech, text, and GSL signs",
            },
            {
                "name": "learning",
                "description": "Structured lessons, tutorials, and GSL dictionary access",
            },
            {
                "name": "media",
                "description": "File upload, video streaming, and media processing services",
            },
        ]
    )
    
    # Setup all middleware (CORS, rate limiting, logging, error handling)
    setup_middleware(app)
    
    # Include API routers
    app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
    app.include_router(recognition.router, prefix="/api/v1/recognition", tags=["recognition"])
    app.include_router(translate.router, prefix="/api/v1/translate", tags=["translation"])
    app.include_router(learning.router, prefix="/api/v1/learning", tags=["learning"])
    app.include_router(media.router, prefix="/api/v1/media", tags=["media"])
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "GSL Backend API",
            "version": "1.0.0",
            "description": "Ghanaian Sign Language Learning Platform Backend",
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {"status": "healthy", "service": "gsl-backend"}
    
    return app


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )