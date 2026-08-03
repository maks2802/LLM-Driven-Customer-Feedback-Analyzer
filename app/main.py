import io
import logging
from typing import Annotated

import models
import pandas as pd
import schemas
from database import engine, get_db
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from llm_service import analyze_feedback
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from tasks import process_csv_in_background

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
        logger.error("Database error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to save feedback to the database."
        ) from e

    return db_feedback


@app.post("/analyze/csv/")
async def analyze_csv_upload(
    background_tasks: BackgroundTasks, file: Annotated[UploadFile, File(...)]
):
    """
    Accepts a CSV file, reads the feedback column, analyzes each row
    via LLM, and saves to the database.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")

        background_tasks.add_task(process_csv_in_background, records=records)

        return {
            "message": "File successfully uploaded.",
            "details": f"Accepted {len(records)} rows for processing.",
        }

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.") from None
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while reading the file."
        ) from e
