import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from main import app, get_db
from models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///:memory:")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Creates a new clean database session for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Overrides the database dependency in FastAPI with a testing one."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


def test_read_root(client):
    """Checks the API availability and root endpoint response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Feedback Analyzer is running."}


@patch("main.analyze_feedback")
def test_analyze_single_text_success(mock_analyze, client):
    """Verifies successful text analysis."""
    mock_analyze.return_value = {
        "llm_sentiment": "Positive",
        "topic": "UX",
        "summary": "User praised the intuitive interface",
        "recommendation": "Maintain current design language.",
    }

    payload = {
        "text": "The new dashboard is incredibly easy to use and looks great!",
        "customer_id": "cust_001",
    }

    response = client.post("/analyze/text/", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["raw_text"] == payload["text"]
    assert data["customer_id"] == "cust_001"
    assert data["llm_sentiment"] == "Positive"
    assert data["topic"] == "UX"
    assert data["summary"] == "User praised the intuitive interface"
    assert data["recommendation"] == "Maintain current design language."
    assert "id" in data


def test_analyze_single_text_empty(client):
    """Verifies validation behavior when an empty text is provided."""
    payload = {"text": "   "}
    response = client.post("/analyze/text/", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Feedback text cannot be empty."


@patch("main.analyze_feedback")
def test_get_feedbacks_pagination_and_filtering(mock_analyze, client):
    """Verifies the feedback retrieval endpoint, including pagination and filtering support."""
    mock_analyze.return_value = {
        "llm_sentiment": "Negative",
        "topic": "Bugs",
        "summary": "App crashes on startup.",
        "recommendation": "Investigate startup sequence.",
    }

    payload = {"text": "It crashes every time."}

    create_response = client.post("/analyze/text/", json=payload)
    assert create_response.status_code == 200

    response = client.get("/feedbacks/?page=1&size=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1

    response_filtered = client.get("/feedbacks/?sentiment=Negative&topic=Bugs")
    assert response_filtered.json()["total"] == 1

    response_empty = client.get("/feedbacks/?sentiment=Positive")
    assert response_empty.json()["total"] == 0
