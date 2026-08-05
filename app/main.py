import io
import logging
import uuid
from collections import Counter
from typing import Annotated, Optional

import models
import pandas as pd
import schemas
from database import engine, get_db
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from llm_service import analyze_feedback, generate_executive_summary
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tasks import process_csv_in_background

logging.basicConfig(level=logging.INFO)
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
    try:
        db.commit()
        db.refresh(db_feedback)
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to save feedback to the database."
        ) from e

    return db_feedback


@app.post("/upload/csv/")
async def upload_csv(background_tasks: BackgroundTasks, file: Annotated[UploadFile, File(...)]):
    """
    Accepts a CSV file, reads the feedback column, analyzes each row
    via LLM, and saves to the database.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()

        decoded_content = contents.decode("utf-8")

        cleaned_lines = []
        for line in decoded_content.splitlines():
            line = line.strip()

            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1].replace('""', '"')

            cleaned_lines.append(line)

        cleaned_csv = "\n".join(cleaned_lines)

        df = pd.read_csv(io.StringIO(cleaned_csv), skipinitialspace=True)
        df.columns = df.columns.str.strip()
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")

        batch_id = str(uuid.uuid4())

        background_tasks.add_task(process_csv_in_background, records=records, batch_id=batch_id)

        return {
            "message": "File successfully uploaded.",
            "batch_id": batch_id,
            "details": f"Accepted {len(records)} rows for processing.",
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.") from None
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while reading the file."
        ) from e


@app.get("/analyze/summary/", response_model=schemas.ExecutiveSummaryResponse)
def get_global_executive_summary(
    db: Annotated[Session, Depends(get_db)],
    batch_id: Optional[str] = Query(
        None, description="Filter summary by specific CSV upload batch"
    ),
):
    """Generates a global executive summary and actionable recommendations
    based on the whole dataset stored in the database."""
    query = db.query(models.Feedback)

    if batch_id:
        query = query.filter(models.Feedback.batch_id == batch_id)

    feedbacks = query.all()

    if not feedbacks:
        raise HTTPException(
            status_code=404,
            detail="No feedback records found for the given criteria.",
        )

    total_count = len(feedbacks)
    sentiments = Counter([f.llm_sentiment for f in feedbacks if f.llm_sentiment])
    topics = Counter([f.topic for f in feedbacks if f.topic])

    sample_summaries = [
        {"topic": f.topic, "sentiment": f.llm_sentiment, "summary": f.summary}
        for f in feedbacks
        if f.summary
    ]

    aggregated_context = {
        "total_feedbacks": total_count,
        "sentiment_breakdown": dict(sentiments),
        "topic_breakdown": dict(topics),
        "individual_summaries": sample_summaries[:150],
    }

    llm_report = generate_executive_summary(aggregated_context)

    top_topic_list = [{"topic": topic, "count": count} for topic, count in topics.most_common()]

    return {
        "total_feedback_analyzed": total_count,
        "sentiment_breakdown": dict(sentiments),
        "top_topics": top_topic_list,
        "executive_summary": llm_report.get("executive_summary", ""),
        "global_recommendations": llm_report.get("global_recommendations", []),
    }


@app.get("/feedbacks/", response_model=schemas.PaginatedFeedbackResponse)
def get_feedbacks(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1, description="Page Number"),
    size: int = Query(10, ge=1, le=100, description="Number of items per page"),
    sentiment: Optional[str] = Query(
        None, description="Filter by sentiment (Positive, Negative, Neutral)"
    ),
    topic: Optional[str] = Query(None, description="Filter by topic (e.g. Support, UX, Bugs)"),
):
    """Retrieve a list of analyzed feedbacks from the database
    with pagination and optional filtering support."""
    query = db.query(models.Feedback)

    if sentiment:
        query = query.filter(models.Feedback.llm_sentiment == sentiment)
    if topic:
        query = query.filter(models.Feedback.topic == topic)

    total = query.count()
    offset = (page - 1) * size
    feedbacks = query.order_by(models.Feedback.created_at.desc()).offset(offset).limit(size).all()

    return {"total": total, "page": page, "size": size, "items": feedbacks}


@app.get("/feedbacks/{feedback_id}", response_model=schemas.FeedbackResponse)
def get_feedback_by_id(feedback_id: int, db: Annotated[Session, Depends(get_db)]):
    """Retrieve detailed information for a specific feedback entry by its ID."""
    feedback = db.query(models.Feedback).filter(models.Feedback.id == feedback_id).first()

    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found.")

    return feedback
