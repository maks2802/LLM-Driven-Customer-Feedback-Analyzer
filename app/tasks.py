import logging

import models
import pandas as pd
from database import SessionLocal
from llm_service import analyze_feedback
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def process_csv_in_background(records: list[dict]):
    """Background function for CSV row processing."""
    db: Session = SessionLocal()
    BATCH_SIZE = 50

    try:
        processed_count = 0

        for row in records:
            raw_text = row.get("Text")

            if not raw_text or not str(raw_text).strip():
                logger.warning(f"Skipping empty feedback row: {row}")
                continue

            llm_analysis = analyze_feedback(str(raw_text))

            raw_date = row.get("Date/Time")
            feedback_date = None
            if row.get("Date/Time"):
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
                raw_text=str(raw_text),
                original_sentiment=row.get("Sentiment"),
                source=row.get("Source"),
                feedback_date=feedback_date,
                customer_id=str(row.get("User ID")) if row.get("User ID") else None,
                location=row.get("Location"),
                confidence_score=confidence_score,
                llm_sentiment=llm_analysis.get("llm_sentiment"),
                topic=llm_analysis.get("topic"),
                summary=llm_analysis.get("summary"),
                recommendation=llm_analysis.get("recommendation"),
            )

            db.add(db_feedback)
            processed_count += 1

            if processed_count % BATCH_SIZE == 0:
                db.commit()
                logger.info(f"Batch saved: {processed_count} records...")

        if processed_count % BATCH_SIZE != 0:
            db.commit()

        logger.info(
            f"Background processing successfully completed. Total records saved: {processed_count}."
        )

    except Exception as e:
        logger.error(f"Critical error in background CSV processing: {e}")
        db.rollback()
    finally:
        db.close()
