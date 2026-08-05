from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from .database import Base


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    raw_text = Column(Text, nullable=False)
    original_sentiment = Column(String, nullable=True)
    source = Column(String, nullable=True)
    customer_id = Column(String, index=True, nullable=True)
    location = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    feedback_date = Column(DateTime, nullable=True)

    batch_id = Column(String, index=True, nullable=True)
    llm_sentiment = Column(String, nullable=True)
    topic = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
