"""
Achievement Service

Handles achievement tracking, badge system, and gamification logic.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.repositories.learning_repository import AchievementRepository, PracticeSessionRepository
from app.repositories.user_repository import LearningProgressRepository
from app.db_models.learning import Achievement


class AchievementService:
    """Service for managing achievements and gamification."""
    
    def __init__(self, db: Session):
        self.db = db
        self.achievement_repo = AchievementRepository(db)
        self.progress_repo = LearningProgressRepository(db)
        self.session_repo = PracticeSessionRepository(db)
    
    def check_and_award_achievements(self, user_id: UUID) -> List[Achievement]:
        """
        Check user progress and award any newly earned achievements.
        
        Args:
            user_id: User's ID
            
        Returns:
            List of newly awarded achievements
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress:
            return []
        
        current_achievements = set(progress.achievements or [])
        all_achievements = self.achievement_repo.get_all()
        newly_awarded = []
        
        for achievement in all_achievements:
            if achievement.id not in current_achievements:
                if self._check_achievement_criteria(achievement, progress, user_id):
                    self.progress_repo.add_achievement(user_id, achievement.id)
                    newly_awarded.append(achievement)
        
        return newly_awarded
    
    def _check_achievement_criteria(
        self,
        achievement: Achievement,
        progress: Any,
        user_id: UUID
    ) -> bool:
        """
        Check if user meets achievement criteria.
        
        Args:
            achievement: Achievement to check
            progress: User's learning progress
            user_id: User's ID
            
        Returns:
            True if criteria met
        """
        criteria = achievement.criteria or {}
        achievement_type = achievement.type
        
        if achievement_type == "lesson_completion":
            required_lessons = criteria.get("lessons_required", 0)
            return progress.total_lessons_completed >= required_lessons
        
        elif achievement_type == "streak":
            required_streak = criteria.get("days_required", 0)
            return progress.current_streak >= required_streak
        
        elif achievement_type == "accuracy":
            required_accuracy = criteria.get("accuracy_required", 0.0)
            return progress.accuracy_rate >= required_accuracy
        
        elif achievement_type == "practice_time":
            required_minutes = criteria.get("minutes_required", 0)
            return progress.practice_time_minutes >= required_minutes
        
        elif achievement_type == "signs_learned":
            required_signs = criteria.get("signs_required", 0)
            return progress.signs_learned >= required_signs
        
        elif achievement_type == "level_up":
            required_level = criteria.get("level_required", 0)
            return progress.current_level >= required_level
        
        return False
    
    def get_user_achievements(self, user_id: UUID) -> List[Achievement]:
        """
        Get all achievements earned by a user.
        
        Args:
            user_id: User's ID
            
        Returns:
            List of earned achievements
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress or not progress.achievements:
            return []
        
        return self.achievement_repo.get_multi_by_ids(progress.achievements)
    
    def get_available_achievements(self, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get achievements available for user to earn with progress.
        
        Args:
            user_id: User's ID
            
        Returns:
            List of achievements with progress information
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress:
            return []
        
        current_achievements = set(progress.achievements or [])
        all_achievements = self.achievement_repo.get_all()
        
        available = []
        for achievement in all_achievements:
            if achievement.id not in current_achievements:
                progress_pct = self._calculate_achievement_progress(
                    achievement, progress, user_id
                )
                available.append({
                    "achievement": achievement,
                    "progress_percentage": progress_pct,
                    "earned": False
                })
        
        return available
    
    def _calculate_achievement_progress(
        self,
        achievement: Achievement,
        progress: Any,
        user_id: UUID
    ) -> float:
        """
        Calculate progress towards an achievement.
        
        Args:
            achievement: Achievement to check
            progress: User's learning progress
            user_id: User's ID
            
        Returns:
            Progress percentage (0-100)
        """
        criteria = achievement.criteria or {}
        achievement_type = achievement.type
        
        if achievement_type == "lesson_completion":
            required = criteria.get("lessons_required", 1)
            current = progress.total_lessons_completed
            return min(100.0, (current / required) * 100)
        
        elif achievement_type == "streak":
            required = criteria.get("days_required", 1)
            current = progress.current_streak
            return min(100.0, (current / required) * 100)
        
        elif achievement_type == "accuracy":
            required = criteria.get("accuracy_required", 1.0)
            current = progress.accuracy_rate
            return min(100.0, (current / required) * 100)
        
        elif achievement_type == "practice_time":
            required = criteria.get("minutes_required", 1)
            current = progress.practice_time_minutes
            return min(100.0, (current / required) * 100)
        
        elif achievement_type == "signs_learned":
            required = criteria.get("signs_required", 1)
            current = progress.signs_learned
            return min(100.0, (current / required) * 100)
        
        elif achievement_type == "level_up":
            required = criteria.get("level_required", 1)
            current = progress.current_level
            return min(100.0, (current / required) * 100)
        
        return 0.0
    
    def create_achievement(self, achievement_data: Dict[str, Any]) -> Achievement:
        """
        Create a new achievement.
        
        Args:
            achievement_data: Achievement data
            
        Returns:
            Created achievement
        """
        return self.achievement_repo.create(achievement_data)
    
    def get_achievement_leaderboard(
        self,
        achievement_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of users by achievement count.
        
        Args:
            achievement_type: Optional filter by achievement type
            limit: Number of top users to return
            
        Returns:
            List of users with achievement counts
        """
        # This would require a more complex query in production
        # For now, return empty list as placeholder
        return []
