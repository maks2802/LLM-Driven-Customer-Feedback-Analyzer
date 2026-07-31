from datetime import datetime

from pydantic import BaseModel


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

    class Config:
        from_attributes = True
