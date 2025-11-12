"""
Learning Service

Handles lesson content management, tutorial progression, achievements,
and GSL dictionary search for the learning platform.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.config.settings import get_settings
from app.db_models.learning import Lesson, Achievement, PracticeSession
from app.db_models.gsl import GSLSign, SignCategory
# Note: TutorialStep and LearningProgress will be handled via Supabase
from app.schemas.learning import (
    LessonResponse, LessonDetailResponse,
    AchievementResponse, DictionarySearchRequest
)
from app.utils.cache import CacheManager

settings = get_settings()


class LearningService:
    """Service for learning content and progress management."""
    
    def __init__(self, db: Session):
        self.db = db
        self.cache = CacheManager()
    
    async def get_lessons(
        self,
        user_id: Optional[UUID] = None,
        level: Optional[int] = None,
        category: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[LessonResponse]:
        """
        Get structured lesson content with optional filtering.
        
        Args:
            user_id: Optional user ID to include progress
            level: Filter by difficulty level
            category: Filter by category
            limit: Maximum number of lessons to return
            offset: Pagination offset
            
        Returns:
            List of lessons with metadata
        """
        query = self.db.query(Lesson)
        
        # Apply filters
        if level is not None:
            query = query.filter(Lesson.level == level)
        if category:
            query = query.filter(Lesson.category == category)
        
        # Order by level and sequence
        query = query.order_by(Lesson.level, Lesson.sequence_order)
        
        # Pagination
        lessons = query.limit(limit).offset(offset).all()
        
        # Get user progress if user_id provided
        user_progress = None
        if user_id:
            user_progress = self.db.query(LearningProgress).filter(
                LearningProgress.user_id == user_id
            ).first()
        
        # Build response
        results = []
        for lesson in lessons:
            is_completed = False
            if user_progress and lesson.id in (user_progress.completed_lessons or []):
                is_completed = True
            
            results.append(LessonResponse(
                id=lesson.id,
                title=lesson.title,
                description=lesson.description,
                level=lesson.level,
                category=lesson.category,
                estimated_duration=lesson.estimated_duration,
                signs_count=len(lesson.signs_covered) if lesson.signs_covered else 0,
                is_completed=is_completed,
                is_locked=self._is_lesson_locked(lesson, user_progress)
            ))
        
        return results
    
    async def get_lesson_details(
        self,
        lesson_id: UUID,
        user_id: Optional[UUID] = None
    ) -> Optional[LessonDetailResponse]:
        """
        Get detailed lesson information including tutorial steps.
        
        Args:
            lesson_id: Lesson ID
            user_id: Optional user ID for progress tracking
            
        Returns:
            Detailed lesson information
        """
        lesson = self.db.query(Lesson).filter(Lesson.id == lesson_id).first()
        
        if not lesson:
            return None
        
        # Get tutorial steps
        steps = self.db.query(TutorialStep).filter(
            TutorialStep.lesson_id == lesson_id
        ).order_by(TutorialStep.step_number).all()
        
        # Get signs covered in this lesson
        signs = []
        if lesson.signs_covered:
            signs = self.db.query(GSLSign).filter(
                GSLSign.id.in_(lesson.signs_covered)
            ).all()
        
        # Check completion status
        is_completed = False
        if user_id:
            progress = self.db.query(LearningProgress).filter(
                LearningProgress.user_id == user_id
            ).first()
            if progress and lesson.id in (progress.completed_lessons or []):
                is_completed = True
        
        return LessonDetailResponse(
            id=lesson.id,
            title=lesson.title,
            description=lesson.description,
            level=lesson.level,
            category=lesson.category,
            estimated_duration=lesson.estimated_duration,
            tutorial_steps=steps,
            signs_covered=signs,
            is_completed=is_completed
        )
    
    async def update_lesson_progress(
        self,
        user_id: UUID,
        lesson_id: UUID,
        completed: bool = True
    ) -> dict:
        """
        Update user's lesson completion status.
        
        Args:
            user_id: User ID
            lesson_id: Lesson ID
            completed: Whether lesson is completed
            
        Returns:
            Updated progress information
        """
        progress = self.db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id
        ).first()
        
        if not progress:
            # Create new progress record
            progress = LearningProgress(
                user_id=user_id,
                total_lessons_completed=0,
                current_level=1,
                completed_lessons=[],
                last_activity=datetime.utcnow()
            )
            self.db.add(progress)
        
        # Update completed lessons
        if completed:
            if not progress.completed_lessons:
                progress.completed_lessons = []
            
            if lesson_id not in progress.completed_lessons:
                progress.completed_lessons.append(lesson_id)
                progress.total_lessons_completed += 1
                progress.last_activity = datetime.utcnow()
                
                # Check for achievements
                await self._check_and_award_achievements(user_id, progress)
        
        self.db.commit()
        self.db.refresh(progress)
        
        return {
            "lesson_id": lesson_id,
            "completed": completed,
            "total_lessons_completed": progress.total_lessons_completed,
            "current_level": progress.current_level
        }
    
    async def get_user_achievements(self, user_id: UUID) -> List[AchievementResponse]:
        """
        Get user's earned achievements and badges.
        
        Args:
            user_id: User ID
            
        Returns:
            List of achievements
        """
        progress = self.db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id
        ).first()
        
        if not progress or not progress.achievements:
            return []
        
        # Get achievement details
        achievements = self.db.query(Achievement).filter(
            Achievement.id.in_(progress.achievements)
        ).all()
        
        return [
            AchievementResponse(
                id=achievement.id,
                name=achievement.name,
                description=achievement.description,
                icon_url=achievement.icon_url,
                earned_at=achievement.earned_at
            )
            for achievement in achievements
        ]
    
    async def search_dictionary(
        self,
        query: str,
        category_id: Optional[UUID] = None,
        difficulty_level: Optional[int] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[GSLSign]:
        """
        Search GSL dictionary with filters.
        
        Args:
            query: Search query string
            category_id: Filter by category
            difficulty_level: Filter by difficulty
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching GSL signs
        """
        # Check cache
        cache_key = f"dictionary_search:{query}:{category_id}:{difficulty_level}:{limit}:{offset}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Build query
        db_query = self.db.query(GSLSign)
        
        # Search in name, description, and usage examples
        if query:
            search_filter = or_(
                GSLSign.sign_name.ilike(f"%{query}%"),
                GSLSign.description.ilike(f"%{query}%")
            )
            db_query = db_query.filter(search_filter)
        
        # Apply filters
        if category_id:
            db_query = db_query.filter(GSLSign.category_id == category_id)
        if difficulty_level is not None:
            db_query = db_query.filter(GSLSign.difficulty_level == difficulty_level)
        
        # Order by relevance (simplified)
        db_query = db_query.order_by(GSLSign.sign_name)
        
        # Pagination
        results = db_query.limit(limit).offset(offset).all()
        
        # Cache results
        await self.cache.set(cache_key, results, ttl=3600)
        
        return results
    
    async def get_sign_categories(self) -> List[SignCategory]:
        """Get all sign categories."""
        return self.db.query(SignCategory).order_by(SignCategory.name).all()
    
    async def get_related_signs(self, sign_id: UUID) -> List[GSLSign]:
        """
        Get signs related to a specific sign.
        
        Args:
            sign_id: GSL sign ID
            
        Returns:
            List of related signs
        """
        sign = self.db.query(GSLSign).filter(GSLSign.id == sign_id).first()
        
        if not sign or not sign.related_signs:
            return []
        
        return self.db.query(GSLSign).filter(
            GSLSign.id.in_(sign.related_signs)
        ).all()
    
    async def record_practice_session(
        self,
        user_id: UUID,
        lesson_id: Optional[UUID] = None,
        signs_practiced: Optional[List[UUID]] = None,
        duration_seconds: int = 0
    ) -> PracticeSession:
        """
        Record a practice session for analytics.
        
        Args:
            user_id: User ID
            lesson_id: Optional lesson ID
            signs_practiced: List of sign IDs practiced
            duration_seconds: Session duration
            
        Returns:
            Created practice session
        """
        session = PracticeSession(
            user_id=user_id,
            lesson_id=lesson_id,
            signs_practiced=signs_practiced or [],
            duration_seconds=duration_seconds,
            completed_at=datetime.utcnow()
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        # Update user's last activity
        progress = self.db.query(LearningProgress).filter(
            LearningProgress.user_id == user_id
        ).first()
        if progress:
            progress.last_activity = datetime.utcnow()
            self.db.commit()
        
        return session
    
    def _is_lesson_locked(
        self,
        lesson: Lesson,
        user_progress: Optional[LearningProgress]
    ) -> bool:
        """Check if lesson is locked based on user progress."""
        if not user_progress:
            # First lesson is always unlocked
            return lesson.level > 1 or lesson.sequence_order > 1
        
        # Lesson is unlocked if user's level is high enough
        if user_progress.current_level >= lesson.level:
            return False
        
        # Check if prerequisites are met
        if lesson.prerequisites:
            completed = user_progress.completed_lessons or []
            for prereq_id in lesson.prerequisites:
                if prereq_id not in completed:
                    return True
        
        return False
    
    async def _check_and_award_achievements(
        self,
        user_id: UUID,
        progress: LearningProgress
    ) -> None:
        """Check and award achievements based on progress."""
        # Define achievement criteria
        achievement_criteria = {
            "first_lesson": lambda p: p.total_lessons_completed >= 1,
            "five_lessons": lambda p: p.total_lessons_completed >= 5,
            "ten_lessons": lambda p: p.total_lessons_completed >= 10,
            "level_up": lambda p: p.current_level >= 2,
            "dedicated_learner": lambda p: len(p.practice_sessions or []) >= 10
        }
        
        # Check each criterion
        for achievement_key, criterion in achievement_criteria.items():
            if criterion(progress):
                # Check if achievement already awarded
                achievement = self.db.query(Achievement).filter(
                    Achievement.key == achievement_key
                ).first()
                
                if achievement and achievement.id not in (progress.achievements or []):
                    if not progress.achievements:
                        progress.achievements = []
                    progress.achievements.append(achievement.id)
                    achievement.earned_at = datetime.utcnow()