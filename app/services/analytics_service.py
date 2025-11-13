"""
Analytics Service

Provides learning analytics, performance tracking, and insights.
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.repositories.learning_repository import PracticeSessionRepository, LessonRepository
from app.repositories.user_repository import LearningProgressRepository
from app.db_models.learning import PracticeSession


class AnalyticsService:
    """Service for learning analytics and insights."""
    
    def __init__(self, db: Session):
        self.db = db
        self.session_repo = PracticeSessionRepository(db)
        self.progress_repo = LearningProgressRepository(db)
        self.lesson_repo = LessonRepository(db)
    
    def get_user_analytics(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a user.
        
        Args:
            user_id: User's ID
            
        Returns:
            Dictionary with analytics data
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress:
            return self._empty_analytics()
        
        session_stats = self.session_repo.get_user_session_stats(user_id)
        recent_sessions = self.session_repo.get_recent_sessions(user_id, days=30)
        
        return {
            "overview": {
                "total_lessons_completed": progress.total_lessons_completed,
                "current_level": progress.current_level,
                "experience_points": progress.experience_points,
                "signs_learned": progress.signs_learned,
                "accuracy_rate": progress.accuracy_rate,
                "practice_time_minutes": progress.practice_time_minutes,
            },
            "streaks": {
                "current_streak": progress.current_streak,
                "longest_streak": progress.longest_streak,
                "days_active": progress.days_active,
            },
            "practice_stats": session_stats,
            "recent_activity": self._format_recent_activity(recent_sessions),
            "learning_velocity": self._calculate_learning_velocity(user_id),
            "strengths_weaknesses": self._analyze_strengths_weaknesses(user_id),
        }
    
    def get_learning_patterns(self, user_id: UUID) -> Dict[str, Any]:
        """
        Analyze user's learning patterns and habits.
        
        Args:
            user_id: User's ID
            
        Returns:
            Dictionary with learning pattern insights
        """
        sessions = self.session_repo.get_by_user(user_id, limit=100)
        
        if not sessions:
            return {
                "preferred_time": None,
                "average_session_duration": 0,
                "consistency_score": 0.0,
                "most_practiced_signs": [],
            }
        
        return {
            "preferred_time": self._find_preferred_practice_time(sessions),
            "average_session_duration": self._calculate_avg_duration(sessions),
            "consistency_score": self._calculate_consistency_score(user_id),
            "most_practiced_signs": self._get_most_practiced_signs(sessions),
            "weekly_activity": self._get_weekly_activity(user_id),
        }
    
    def get_performance_trends(
        self,
        user_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get performance trends over time.
        
        Args:
            user_id: User's ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend data
        """
        sessions = self.session_repo.get_recent_sessions(user_id, days=days)
        
        if not sessions:
            return {
                "accuracy_trend": [],
                "practice_time_trend": [],
                "signs_learned_trend": [],
            }
        
        # Group sessions by day
        daily_data = self._group_sessions_by_day(sessions)
        
        return {
            "accuracy_trend": self._calculate_accuracy_trend(daily_data),
            "practice_time_trend": self._calculate_time_trend(daily_data),
            "improvement_rate": self._calculate_improvement_rate(sessions),
        }
    
    def get_recommendations(self, user_id: UUID) -> List[Dict[str, Any]]:
        """
        Get personalized learning recommendations.
        
        Args:
            user_id: User's ID
            
        Returns:
            List of recommendations
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress:
            return []
        
        recommendations = []
        
        # Check practice consistency
        if progress.current_streak < 3:
            recommendations.append({
                "type": "consistency",
                "priority": "high",
                "message": "Practice daily to build a learning streak",
                "action": "Start a practice session"
            })
        
        # Check accuracy
        if progress.accuracy_rate < 0.7:
            recommendations.append({
                "type": "accuracy",
                "priority": "medium",
                "message": "Focus on accuracy - review challenging signs",
                "action": "Review difficult signs"
            })
        
        # Suggest level progression
        if progress.total_lessons_completed >= (progress.current_level * 5):
            recommendations.append({
                "type": "progression",
                "priority": "high",
                "message": "You're ready for the next level!",
                "action": "Start next level lessons"
            })
        
        return recommendations
    
    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            "overview": {
                "total_lessons_completed": 0,
                "current_level": 1,
                "experience_points": 0,
                "signs_learned": 0,
                "accuracy_rate": 0.0,
                "practice_time_minutes": 0,
            },
            "streaks": {
                "current_streak": 0,
                "longest_streak": 0,
                "days_active": 0,
            },
            "practice_stats": {
                "total_sessions": 0,
                "total_time": 0,
                "average_accuracy": 0.0,
                "signs_practiced": 0,
            },
            "recent_activity": [],
            "learning_velocity": 0.0,
            "strengths_weaknesses": {"strengths": [], "weaknesses": []},
        }
    
    def _format_recent_activity(
        self,
        sessions: List[PracticeSession]
    ) -> List[Dict[str, Any]]:
        """Format recent practice sessions for display."""
        return [
            {
                "date": session.completed_at,
                "duration": session.duration_seconds,
                "accuracy": session.accuracy_score,
                "signs_count": len(session.signs_practiced or []),
            }
            for session in sessions[:10]
        ]
    
    def _calculate_learning_velocity(self, user_id: UUID) -> float:
        """
        Calculate learning velocity (signs learned per week).
        
        Args:
            user_id: User's ID
            
        Returns:
            Signs learned per week
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress or progress.days_active == 0:
            return 0.0
        
        weeks_active = max(1, progress.days_active / 7)
        return progress.signs_learned / weeks_active
    
    def _analyze_strengths_weaknesses(
        self,
        user_id: UUID
    ) -> Dict[str, List[str]]:
        """
        Analyze user's strengths and weaknesses.
        
        Args:
            user_id: User's ID
            
        Returns:
            Dictionary with strengths and weaknesses
        """
        sessions = self.session_repo.get_recent_sessions(user_id, days=30)
        
        if not sessions:
            return {"strengths": [], "weaknesses": []}
        
        # Analyze accuracy patterns
        high_accuracy_sessions = [s for s in sessions if s.accuracy_score and s.accuracy_score >= 0.8]
        low_accuracy_sessions = [s for s in sessions if s.accuracy_score and s.accuracy_score < 0.6]
        
        strengths = []
        weaknesses = []
        
        if len(high_accuracy_sessions) > len(sessions) * 0.5:
            strengths.append("Consistent high accuracy")
        
        if len(low_accuracy_sessions) > len(sessions) * 0.3:
            weaknesses.append("Accuracy needs improvement")
        
        return {"strengths": strengths, "weaknesses": weaknesses}
    
    def _find_preferred_practice_time(
        self,
        sessions: List[PracticeSession]
    ) -> Optional[str]:
        """Find user's preferred practice time of day."""
        if not sessions:
            return None
        
        hour_counts = {}
        for session in sessions:
            hour = session.completed_at.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        if not hour_counts:
            return None
        
        preferred_hour = max(hour_counts, key=hour_counts.get)
        
        if preferred_hour < 12:
            return "morning"
        elif preferred_hour < 17:
            return "afternoon"
        else:
            return "evening"
    
    def _calculate_avg_duration(self, sessions: List[PracticeSession]) -> int:
        """Calculate average session duration in seconds."""
        if not sessions:
            return 0
        
        total = sum(s.duration_seconds for s in sessions)
        return total // len(sessions)
    
    def _calculate_consistency_score(self, user_id: UUID) -> float:
        """
        Calculate consistency score (0-1) based on practice regularity.
        
        Args:
            user_id: User's ID
            
        Returns:
            Consistency score
        """
        progress = self.progress_repo.get_by_user_id(user_id)
        if not progress:
            return 0.0
        
        # Simple consistency based on streak
        max_possible_streak = 30  # Consider last 30 days
        return min(1.0, progress.current_streak / max_possible_streak)
    
    def _get_most_practiced_signs(
        self,
        sessions: List[PracticeSession]
    ) -> List[UUID]:
        """Get most frequently practiced signs."""
        sign_counts = {}
        
        for session in sessions:
            if session.signs_practiced:
                for sign_id in session.signs_practiced:
                    sign_counts[sign_id] = sign_counts.get(sign_id, 0) + 1
        
        # Sort by count and return top 10
        sorted_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)
        return [sign_id for sign_id, _ in sorted_signs[:10]]
    
    def _get_weekly_activity(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get activity for the past week."""
        sessions = self.session_repo.get_recent_sessions(user_id, days=7)
        
        # Group by day
        daily_activity = {}
        for session in sessions:
            date_key = session.completed_at.date()
            if date_key not in daily_activity:
                daily_activity[date_key] = {
                    "date": date_key,
                    "sessions": 0,
                    "duration": 0,
                }
            daily_activity[date_key]["sessions"] += 1
            daily_activity[date_key]["duration"] += session.duration_seconds
        
        return list(daily_activity.values())
    
    def _group_sessions_by_day(
        self,
        sessions: List[PracticeSession]
    ) -> Dict[datetime, List[PracticeSession]]:
        """Group sessions by day."""
        daily_sessions = {}
        
        for session in sessions:
            date_key = session.completed_at.date()
            if date_key not in daily_sessions:
                daily_sessions[date_key] = []
            daily_sessions[date_key].append(session)
        
        return daily_sessions
    
    def _calculate_accuracy_trend(
        self,
        daily_data: Dict[datetime, List[PracticeSession]]
    ) -> List[Dict[str, Any]]:
        """Calculate daily accuracy trend."""
        trend = []
        
        for date, sessions in sorted(daily_data.items()):
            accuracies = [s.accuracy_score for s in sessions if s.accuracy_score is not None]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                trend.append({
                    "date": date,
                    "accuracy": avg_accuracy
                })
        
        return trend
    
    def _calculate_time_trend(
        self,
        daily_data: Dict[datetime, List[PracticeSession]]
    ) -> List[Dict[str, Any]]:
        """Calculate daily practice time trend."""
        trend = []
        
        for date, sessions in sorted(daily_data.items()):
            total_time = sum(s.duration_seconds for s in sessions)
            trend.append({
                "date": date,
                "minutes": total_time // 60
            })
        
        return trend
    
    def _calculate_improvement_rate(
        self,
        sessions: List[PracticeSession]
    ) -> float:
        """
        Calculate improvement rate based on accuracy over time.
        
        Args:
            sessions: List of practice sessions
            
        Returns:
            Improvement rate percentage
        """
        if len(sessions) < 2:
            return 0.0
        
        # Compare first 5 and last 5 sessions
        early_sessions = sessions[-5:] if len(sessions) >= 5 else sessions
        recent_sessions = sessions[:5] if len(sessions) >= 5 else sessions
        
        early_accuracy = [s.accuracy_score for s in early_sessions if s.accuracy_score is not None]
        recent_accuracy = [s.accuracy_score for s in recent_sessions if s.accuracy_score is not None]
        
        if not early_accuracy or not recent_accuracy:
            return 0.0
        
        early_avg = sum(early_accuracy) / len(early_accuracy)
        recent_avg = sum(recent_accuracy) / len(recent_accuracy)
        
        if early_avg == 0:
            return 0.0
        
        return ((recent_avg - early_avg) / early_avg) * 100
