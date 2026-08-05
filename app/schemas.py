from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FeedbackBase(BaseModel):
    raw_text: str
    original_sentiment: str | None = None
    source: str | None = None
    customer_id: str | None = None
    location: str | None = None
    confidence_score: float | None = None
    feedback_date: datetime | None = None


class FeedbackResponse(FeedbackBase):
    id: int
    created_at: datetime
    llm_sentiment: str | None = None
    topic: str | None = None
    summary: str | None = None
    recommendation: str | None = None
    model_config = ConfigDict(from_attributes=True)


class PaginatedFeedbackResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[FeedbackResponse]


class TopicCount(BaseModel):
    topic: str
    count: int


class ExecutiveSummaryResponse(BaseModel):
    total_feedback_analyzed: int
    sentiment_breakdown: dict[str, int]
    top_topics: list[TopicCount]
    executive_summary: str
    global_recommendations: list[str]
