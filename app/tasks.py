import asyncio
import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from . import models
from .database import SessionLocal
from .llm_service import analyze_feedback_async

logger = logging.getLogger(__name__)


async def process_csv_in_background(records: list[dict], batch_id: Optional[str] = None):
    """Asynchronous background function for CSV processing."""
    semaphore = asyncio.Semaphore(20)

    async def fetch_analysis(row: dict, text: str):
        """Helper function to process a single line via a semaphore."""
        async with semaphore:
            llm_analysis = await analyze_feedback_async(text)
            return row, text, llm_analysis

    tasks = []
    for row in records:
        raw_text = None
        for key, val in row.items():
            if key and key.strip().lower() in ["text", "feedback", "review", "comment"]:
                raw_text = val
                break

        if not raw_text or not str(raw_text).strip():
            continue

        tasks.append(fetch_analysis(row=row, text=str(raw_text)))

    if not tasks:
        logger.warning("No valid text records found to process.")
        return

    logger.info(f"Starting async processing of {len(tasks)} records via OpenAI...")

    results = await asyncio.gather(*tasks)

    logger.info("OpenAI processing completed. Saving to database...")

    db: Session = SessionLocal()
    try:
        processed_count = 0
        for row, text, llm_analysis in results:
            raw_date = row.get("Date/Time")
            feedback_date = None
            if raw_date:
                try:
                    feedback_date = pd.to_datetime(raw_date).to_pydatetime()
                except (ValueError, TypeError):
                    pass

            raw_confidence = row.get("Confidence Score")
            confidence_score = None
            if raw_confidence is not None:
                try:
                    confidence_score = float(raw_confidence)
                except (ValueError, TypeError):
                    pass

            db_feedback = models.Feedback(
                raw_text=text,
                original_sentiment=row.get("Sentiment"),
                source=row.get("Source"),
                feedback_date=feedback_date,
                customer_id=str(row.get("User ID")) if row.get("User ID") else None,
                location=row.get("Location"),
                confidence_score=confidence_score,
                batch_id=batch_id,
                llm_sentiment=llm_analysis.get("llm_sentiment"),
                topic=llm_analysis.get("topic"),
                summary=llm_analysis.get("summary"),
                recommendation=llm_analysis.get("recommendation"),
            )

            db.add(db_feedback)
            processed_count += 1

        db.commit()
        logger.info(
            f"Background processing successfully completed. Total records saved: {processed_count}."
        )

    except Exception as e:
        logger.error(f"Critical error in database saving: {e}")
        db.rollback()
    finally:
        db.close()
