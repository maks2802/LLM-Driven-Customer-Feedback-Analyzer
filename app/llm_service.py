import json
import logging
import os

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)


def analyze_feedback(raw_text: str) -> dict:
    """Sends review text to OpenAI and returns structured analysis as a dictionary."""

    system_prompt = """
    You are an expert customer feedback analyzer.
    Analyze the provided customer feedback and extract the following information
    strictly in JSON format:
    {
        "llm_sentiment": "Classify as 'Positive', 'Neutral', or 'Negative'",
        "topic": "Identify the main topic (e.g., 'Pricing', 'UX', 'Bugs', 'Support',
        'Features', 'Other')",
        "summary": "A brief 1-2 sentence summary of the core issue or praise.",
        "recommendation": "A short, actionable recommendation for the product or
        support team based on this feedback."
    }

    Respond ONLY with a valid JSON object.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Customer Feedback: {raw_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw_json_response = response.choices[0].message.content
        result = json.loads(raw_json_response)

        return result

    except AuthenticationError as e:
        logger.error(f"Authentication error (check API key): {e}")
        return _get_error_response("Authentication failed.")
    except RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded: {e}")
        return _get_error_response("Rate limit exceeded.")
    except APIConnectionError as e:
        logger.error(f"OpenAI connection error: {e}")
        return _get_error_response("Connection error.")
    except OpenAIError as e:
        logger.error(f"Internal OpenAI API error: {e}")
        return _get_error_response("OpenAI API error.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model: {e}")
        return _get_error_response("Failed to parse model response.")


def _get_error_response(message: str) -> dict:
    """Helper function to return a standard error response."""
    return {
        "llm_sentiment": "Error",
        "topic": "Error",
        "summary": message,
        "recommendation": "None",
    }
