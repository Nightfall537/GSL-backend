"""
Database Configuration and Session Management

Provides SQLAlchemy database engine, session management, and initialization utilities
for the GSL Backend application with Supabase integration.
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator

from app.config.settings import get_settings

settings = get_settings()

# Create database engine
if settings.environment == "testing":
    # Use in-memory SQLite for testing
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=settings.database_echo
    )
else:
    # Use Supabase PostgreSQL for development and production
    # Supabase provides a PostgreSQL connection string
    engine = create_engine(
        settings.database_url,
        echo=settings.database_echo,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=10,
        max_overflow=20
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()


async def init_db() -> None:
    """
    Initialize database tables.
    
    Note: With Supabase, tables are typically created via the Supabase Dashboard
    or migrations. This function is kept for local development and testing.
    """
    # Import all models to ensure they are registered
    # from app.models import user, gsl, learning  # noqa
    
    # Create all tables (only for local development)
    # In production with Supabase, use Supabase migrations
    if settings.environment == "development":
        Base.metadata.create_all(bind=engine)
    
    print("Database initialization complete")


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """Database management utilities."""
    
    @staticmethod
    def create_tables():
        """Create all database tables."""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def drop_tables():
        """Drop all database tables."""
        Base.metadata.drop_all(bind=engine)
    
    @staticmethod
    def reset_database():
        """Reset database by dropping and recreating all tables."""
        DatabaseManager.drop_tables()
        DatabaseManager.create_tables()