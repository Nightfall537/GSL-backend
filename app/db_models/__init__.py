"""
Database Models Package

SQLAlchemy ORM models for database tables.
Note: With Supabase, these are primarily used for type hints and testing.
Actual database schema is managed through Supabase migrations.
"""

from app.db_models.user import User, LearningProgress
from app.db_models.gsl import GSLSign, SignCategory
from app.db_models.learning import Lesson, Achievement, PracticeSession
from app.db_models.media import MediaFile, MediaType, ProcessingStatus

__all__ = [
    "User",
    "LearningProgress",
    "GSLSign",
    "SignCategory",
    "Lesson",
    "Achievement",
    "PracticeSession",
    "MediaFile",
    "MediaType",
    "ProcessingStatus",
]
