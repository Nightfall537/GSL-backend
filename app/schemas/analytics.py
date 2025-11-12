"""
Analytics Schemas

Pydantic models for analytics and reporting API requests and responses.
Handles validation for usage statistics, performance metrics, and insights.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID
from enum import Enum


class TimeRange(str, Enum):
    """Time range options for analytics."""
    today = "today"
    yesterday = "yesterday"
    last_7_days = "last_7_days"
    last_30_days = "last_30_days"
    last_90_days = "last_90_days"
    this_month = "this_month"
    last_month = "last_month"
    this_year = "this_year"
    custom = "custom"


class MetricType(str, Enum):
    """Types of metrics."""
    user_engagement = "user_engagement"
    learning_progress = "learning_progress"
    sign_recognition = "sign_recognition"
    lesson_completion = "lesson_completion"
    practice_sessions = "practice_sessions"
    system_performance = "system_performance"


class AggregationType(str, Enum):
    """Aggregation types for metrics."""
    sum = "sum"
    average = "average"
    count = "count"
    min = "min"
    max = "max"
    median = "median"


class ChartType(str, Enum):
    """Chart types for visualization."""
    line = "line"
    bar = "bar"
    pie = "pie"
    area = "area"
    scatter = "scatter"


# Request Schemas

class AnalyticsRequest(BaseModel):
    """Schema for analytics data request."""
    metric_type: MetricType = Field(..., description="Type of metric to retrieve")
    time_range: TimeRange = Field(..., description="Time range for data")
    start_date: Optional[date] = Field(None, description="Start date for custom range")
    end_date: Optional[date] = Field(None, description="End date for custom range")
    aggregation: AggregationType = Field(AggregationType.sum, description="Aggregation method")
    group_by: Optional[str] = Field(None, description="Field to group by")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate end date is after start date."""
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError('End date must be after start date')
        return v


class UserAnalyticsRequest(BaseModel):
    """Schema for user-specific analytics request."""
    user_id: UUID = Field(..., description="User ID")
    metrics: List[MetricType] = Field(..., min_length=1, description="Metrics to retrieve")
    time_range: TimeRange = Field(TimeRange.last_30_days, description="Time range")
    include_comparisons: bool = Field(False, description="Include period comparisons")


class LessonAnalyticsRequest(BaseModel):
    """Schema for lesson analytics request."""
    lesson_id: Optional[UUID] = Field(None, description="Specific lesson ID")
    category: Optional[str] = Field(None, description="Lesson category")
    level: Optional[int] = Field(None, ge=1, le=3, description="Lesson level")
    time_range: TimeRange = Field(TimeRange.last_30_days)


class SignRecognitionAnalyticsRequest(BaseModel):
    """Schema for sign recognition analytics."""
    sign_id: Optional[UUID] = Field(None, description="Specific sign ID")
    category: Optional[str] = Field(None, description="Sign category")
    time_range: TimeRange = Field(TimeRange.last_30_days)
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence threshold")


# Response Schemas

class DataPoint(BaseModel):
    """Schema for a single data point."""
    timestamp: datetime
    value: float
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MetricSummary(BaseModel):
    """Schema for metric summary."""
    metric_name: str
    current_value: float
    previous_value: Optional[float] = None
    change_percentage: Optional[float] = None
    trend: Optional[str] = Field(None, description="up, down, or stable")
    unit: Optional[str] = None


class AnalyticsResponse(BaseModel):
    """Schema for analytics data response."""
    metric_type: MetricType
    time_range: TimeRange
    start_date: date
    end_date: date
    data_points: List[DataPoint]
    summary: MetricSummary
    aggregation: AggregationType
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class UserEngagementMetrics(BaseModel):
    """Schema for user engagement metrics."""
    total_users: int
    active_users: int
    new_users: int
    returning_users: int
    average_session_duration: float = Field(..., description="Average session duration in minutes")
    total_sessions: int
    sessions_per_user: float
    retention_rate: float = Field(..., ge=0.0, le=100.0)
    churn_rate: float = Field(..., ge=0.0, le=100.0)


class LearningProgressMetrics(BaseModel):
    """Schema for learning progress metrics."""
    total_lessons_completed: int
    average_completion_rate: float = Field(..., ge=0.0, le=100.0)
    average_score: float = Field(..., ge=0.0, le=100.0)
    total_practice_time: int = Field(..., description="Total practice time in minutes")
    signs_learned: int
    achievements_earned: int
    average_streak: float


class SignRecognitionMetrics(BaseModel):
    """Schema for sign recognition metrics."""
    total_recognitions: int
    successful_recognitions: int
    success_rate: float = Field(..., ge=0.0, le=100.0)
    average_confidence: float = Field(..., ge=0.0, le=1.0)
    most_recognized_signs: List[Dict[str, Any]]
    least_recognized_signs: List[Dict[str, Any]]
    recognition_by_category: Dict[str, int]


class LessonCompletionMetrics(BaseModel):
    """Schema for lesson completion metrics."""
    total_completions: int
    unique_users: int
    average_completion_time: float = Field(..., description="Average time in minutes")
    completion_rate_by_level: Dict[str, float]
    most_popular_lessons: List[Dict[str, Any]]
    dropout_points: List[Dict[str, Any]]


class SystemPerformanceMetrics(BaseModel):
    """Schema for system performance metrics."""
    average_response_time: float = Field(..., description="Average response time in ms")
    total_requests: int
    error_rate: float = Field(..., ge=0.0, le=100.0)
    uptime_percentage: float = Field(..., ge=0.0, le=100.0)
    peak_concurrent_users: int
    database_query_time: float
    ai_model_latency: float


class DashboardMetrics(BaseModel):
    """Schema for dashboard overview metrics."""
    user_engagement: UserEngagementMetrics
    learning_progress: LearningProgressMetrics
    sign_recognition: SignRecognitionMetrics
    lesson_completion: LessonCompletionMetrics
    system_performance: SystemPerformanceMetrics
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TrendAnalysis(BaseModel):
    """Schema for trend analysis."""
    metric_name: str
    trend_direction: str = Field(..., description="increasing, decreasing, or stable")
    trend_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of trend")
    forecast: List[DataPoint] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)


class ComparisonReport(BaseModel):
    """Schema for comparison report."""
    current_period: Dict[str, Any]
    previous_period: Dict[str, Any]
    changes: Dict[str, float]
    significant_changes: List[Dict[str, Any]]
    period_start: date
    period_end: date


class ExportRequest(BaseModel):
    """Schema for analytics export request."""
    metric_types: List[MetricType] = Field(..., min_length=1)
    time_range: TimeRange
    format: str = Field("csv", pattern="^(csv|json|xlsx)$", description="Export format")
    include_charts: bool = Field(False, description="Include chart images")


class ExportResponse(BaseModel):
    """Schema for analytics export response."""
    export_id: UUID
    download_url: str
    format: str
    file_size: int
    expires_at: datetime
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class InsightResponse(BaseModel):
    """Schema for AI-generated insights."""
    insight_type: str
    title: str
    description: str
    severity: str = Field(..., description="info, warning, or critical")
    recommendations: List[str]
    affected_metrics: List[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
