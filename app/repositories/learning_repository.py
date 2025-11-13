"""
Learning Repository

Repository classes for learning-related database operations.
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.core.repository import BaseRepository
from app.db_models.learning import Lesson, Achievement, PracticeSession


class LessonRepository(BaseRepository[Lesson]):
    """Repository for Lesson model operations."""
    
    def __init__(self, db: Session):
        super().__init__(Lesson, db)
    
    def get_by_level(
        self,
        level: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Lesson]:
        """
        Get lessons by difficulty level.
        
        Args:
            level: Difficulty level (1-3)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of lessons at the specified level
        """
        return self.db.query(Lesson).filter(
            Lesson.level == level
        ).order_by(Lesson.sequence_order).offset(skip).limit(limit).all()
    
    def get_by_category(
        self,
        category: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Lesson]:
        """
        Get lessons by category.
        
        Args:
            category: Lesson category
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of lessons in the category
        """
        return self.db.query(Lesson).filter(
            Lesson.category == category
        ).order_by(Lesson.sequence_order).offset(skip).limit(limit).all()
    
    def search_lessons(
        self,
        query: Optional[str] = None,
        level: Optional[int] = None,
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 20
    ) -> List[Lesson]:
        """
        Search lessons with filters.
        
        Args:
            query: Search query for title or description
            level: Filter by difficulty level
            category: Filter by category
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching lessons
        """
        db_query = self.db.query(Lesson)
        
        # Text search
        if query:
            search_term = f"%{query.lower()}%"
            db_query = db_query.filter(
                or_(
                    Lesson.title.ilike(search_term),
                    Lesson.description.ilike(search_term)
                )
            )
        
        # Level filter
        if level:
            db_query = db_query.filter(Lesson.level == level)
        
        # Category filter
        if category:
            db_query = db_query.filter(Lesson.category == category)
        
        return db_query.order_by(
            Lesson.level, Lesson.sequence_order
        ).offset(skip).limit(limit).all()
    
    def get_prerequisites(self, lesson_id: UUID) -> List[Lesson]:
        """
        Get prerequisite lessons for a lesson.
        
        Args:
            lesson_id: Lesson ID
            
        Returns:
            List of prerequisite lessons
        """
        lesson = self.get(lesson_id)
        if not lesson or not lesson.prerequisites:
            return []
        
        return self.db.query(Lesson).filter(
            Lesson.id.in_(lesson.prerequisites)
        ).all()
    
    def get_lessons_covering_sign(self, sign_id: UUID) -> List[Lesson]:
        """
        Get lessons that cover a specific sign.
        
        Args:
            sign_id: GSL sign ID
            
        Returns:
            List of lessons covering the sign
        """
        return self.db.query(Lesson).filter(
            Lesson.signs_covered.contains([sign_id])
        ).all()
    
    def get_next_lesson(self, current_lesson_id: UUID) -> Optional[Lesson]:
        """
        Get the next lesson in sequence.
        
        Args:
            current_lesson_id: Current lesson ID
            
        Returns:
            Next lesson or None
        """
        current_lesson = self.get(current_lesson_id)
        if not current_lesson:
            return None
        
        return self.db.query(Lesson).filter(
            and_(
                Lesson.level == current_lesson.level,
                Lesson.category == current_lesson.category,
                Lesson.sequence_order > current_lesson.sequence_order
            )
        ).order_by(Lesson.sequence_order).first()
    
    def get_beginner_lessons(self, limit: int = 10) -> List[Lesson]:
        """
        Get beginner-friendly lessons.
        
        Args:
            limit: Maximum number of lessons to return
            
        Returns:
            List of beginner lessons
        """
        return self.db.query(Lesson).filter(
            Lesson.level == 1
        ).order_by(Lesson.sequence_order).limit(limit).all()


class AchievementRepository(BaseRepository[Achievement]):
    """Repository for Achievement model operations."""
    
    def __init__(self, db: Session):
        super().__init__(Achievement, db)
    
    def get_by_type(self, achievement_type: str) -> List[Achievement]:
        """
        Get achievements by type.
        
        Args:
            achievement_type: Type of achievement
            
        Returns:
            List of achievements of the specified type
        """
        return self.db.query(Achievement).filter(
            Achievement.type == achievement_type
        ).all()
    
    def get_available_achievements(self, user_progress: Dict[str, Any]) -> List[Achievement]:
        """
        Get achievements that a user can potentially earn.
        
        Args:
            user_progress: User's current progress data
            
        Returns:
            List of available achievements
        """
        # This is a simplified implementation
        # In a real system, you'd evaluate criteria against user progress
        return self.db.query(Achievement).all()


class PracticeSessionRepository(BaseRepository[PracticeSession]):
    """Repository for PracticeSession model operations."""
    
    def __init__(self, db: Session):
        super().__init__(PracticeSession, db)
    
    def get_by_user(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[PracticeSession]:
        """
        Get practice sessions for a user.
        
        Args:
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of user's practice sessions
        """
        return self.db.query(PracticeSession).filter(
            PracticeSession.user_id == user_id
        ).order_by(PracticeSession.completed_at.desc()).offset(skip).limit(limit).all()
    
    def get_by_lesson(
        self,
        lesson_id: UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[PracticeSession]:
        """
        Get practice sessions for a specific lesson.
        
        Args:
            lesson_id: Lesson ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of practice sessions for the lesson
        """
        return self.db.query(PracticeSession).filter(
            PracticeSession.lesson_id == lesson_id
        ).order_by(PracticeSession.completed_at.desc()).offset(skip).limit(limit).all()
    
    def get_user_session_stats(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get practice session statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with session statistics
        """
        sessions = self.db.query(PracticeSession).filter(
            PracticeSession.user_id == user_id
        ).all()
        
        if not sessions:
            return {
                "total_sessions": 0,
                "total_time": 0,
                "average_accuracy": 0.0,
                "signs_practiced": 0
            }
        
        total_time = sum(session.duration_seconds for session in sessions)
        accuracies = [s.accuracy_score for s in sessions if s.accuracy_score is not None]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Count unique signs practiced
        all_signs = []
        for session in sessions:
            if session.signs_practiced:
                all_signs.extend(session.signs_practiced)
        unique_signs = len(set(all_signs))
        
        return {
            "total_sessions": len(sessions),
            "total_time": total_time,
            "average_accuracy": avg_accuracy,
            "signs_practiced": unique_signs
        }
    
    def get_recent_sessions(
        self,
        user_id: UUID,
        days: int = 7,
        limit: int = 10
    ) -> List[PracticeSession]:
        """
        Get recent practice sessions for a user.
        
        Args:
            user_id: User ID
            days: Number of days to look back
            limit: Maximum number of sessions to return
            
        Returns:
            List of recent practice sessions
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        return self.db.query(PracticeSession).filter(
            and_(
                PracticeSession.user_id == user_id,
                PracticeSession.completed_at >= cutoff_date
            )
        ).order_by(PracticeSession.completed_at.desc()).limit(limit).all()