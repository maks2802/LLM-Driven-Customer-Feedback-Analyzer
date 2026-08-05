# LLM-Driven Customer Feedback Analyzer

## Description

The LLM-Driven Customer Feedback Analyzer is a robust backend service built with **FastAPI** that processes raw customer feedback (reviews, support tickets, NPS comments) and extracts actionable insights using **OpenAI's LLM (gpt-4o-mini)**.

The application supports both single-text inputs and bulk CSV uploads. It automatically classifies sentiment, identifies core topics (e.g., UX, Pricing, Bugs), generates concise summaries, and provides actionable recommendations. It also features an endpoint to generate a high-level executive summary based on the aggregated dataset.

**Tech Stack:** FastAPI, PostgreSQL, SQLAlchemy, Pandas, OpenAI API, Pytest, Ruff.

---

## Structure of the Project

The project follows a modular architecture for better maintainability and scalability:

```
.
├── .github/workflows/      # CI/CD pipelines (GitHub Actions for linting & testing)
├── app/                    # Main application directory
│   ├── __init__.py
│   ├── main.py             # FastAPI application instance and routing setup
│   ├── database.py         # Database connection and session management (SQLAlchemy)
│   ├── models.py           # Database ORM models (PostgreSQL tables)
│   ├── schemas.py          # Pydantic models for request/response validation
│   ├── llm_service.py      # OpenAI API integration, prompt engineering, and error handling
│   ├── tasks.py            # Async background tasks (for processing large CSV files)
│   └── tests.py            # Unit and integration tests (Pytest)
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── pytest.ini              # Pytest configuration and coverage settings
├── requirements.txt        # Python dependencies
├── ruff.toml               # Code linter and formatter configuration (Ruff)
└── README.md               # Project documentation
```

---

## How to Run It

### Prerequisites

- Python 3.12+
- PostgreSQL database (or you can fall back to SQLite for local testing)
- An active OpenAI API Key

### Step-by-Step Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Set up environment variables:**

   Create a `.env` file in the root directory and add your credentials:

   ```env
   OPENAI_API_KEY="your-openai-api-key-here"
   DATABASE_URL="postgresql://user:password@localhost:5432/feedback_db"

   # Note: For quick local testing without PostgreSQL, you can use SQLite:
   # DATABASE_URL="sqlite:///./feedback.db"
   ```

3. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the API Documentation:**

   Open your browser and navigate to the auto-generated Swagger UI at:
   `http://127.0.0.1:8000/docs`

---

## Details About Commands and Parameters

### Core API Endpoints

The API is divided into logical tags for easy navigation in Swagger UI:

#### System

- `GET /` — Checks API health.

#### Analysis

- `POST /analyze/text/` — Analyzes a single raw text feedback.
  - **Payload:** `{"text": "Your review here", "customer_id": "optional_id"}`

- `POST /upload/csv/` — Uploads a CSV file for asynchronous batch processing. The CSV must contain a column named "Text", "Feedback", "Review", or "Comment".

- `GET /analyze/summary/` — Generates a global executive summary and actionable recommendations based on the entire stored dataset.
  - **Query Parameters:** `batch_id` (optional, to filter the report by a specific CSV upload batch).

- `GET /feedbacks/` — Retrieves a list of processed feedbacks.
  - **Query Parameters:** `page` (default: 1), `size` (default: 10), `sentiment` (filter), `topic` (filter).
- `GET /feedbacks/{feedback_id}` — Retrieves detailed information for a specific feedback entry by its ID.

### Development Commands

- **Run Tests:**

  ```bash
  pytest
  ```

  _(Generates a coverage report in the terminal)._

- **Code Linting & Formatting (Ruff):**

  ```bash
  ruff check .        # Check for style violations
  ruff check --fix .  # Automatically fix issues
  ruff format .        # Format code
  ```
