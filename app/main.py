import logging
from typing import Annotated

import models
import schemas
from database import engine, get_db
from fastapi import Depends, FastAPI, HTTPException
from llm_service import analyze_feedback
from pydantic import BaseModel
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)


class TextFeedbackRequest(BaseModel):
    text: str
    customer_id: str | None = None
    source: str | None = "Manual Input"


app = FastAPI(
    title="Customer Feedback Analyzer API",
    description="API for analyzing customer feedback using LLM",
    version="1.0.0",
)


@app.get("/")
def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "Feedback Analyzer is running."}


@app.post("/analyze/text/", response_model=schemas.FeedbackResponse)
def analyze_single_text(request: TextFeedbackRequest, db: Annotated[Session, Depends(get_db)]):
    """Receives raw feedback, analyzes it via OpenAI, and saves the result to the database."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Feedback text cannot be empty.")

    llm_analysis = analyze_feedback(request.text)

    if llm_analysis.get("llm_sentiment") == "Error":
        raise HTTPException(status_code=500, detail=llm_analysis.get("summary"))

    db_feedback = models.Feedback(
        raw_text=request.text,
        customer_id=request.customer_id,
        source=request.source,
        llm_sentiment=llm_analysis.get("llm_sentiment"),
        topic=llm_analysis.get("topic"),
        summary=llm_analysis.get("summary"),
        recommendation=llm_analysis.get("recommendation"),
    )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)

    return db_feedback
