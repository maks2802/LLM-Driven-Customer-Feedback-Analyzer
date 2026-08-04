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


def generate_executive_summary(feedbacks_summary_data: dict) -> dict:
    """
    Generates an executive summary and global actionable recommendations
    based on the aggregated dataset provided.
    """
    system_prompt = """
    You are a Strategic Customer Success Executive and Product Analyst.
    You will be provided with aggregated data and summaries of customer feedback.

    Your task is to generate a high-level executive report strictly in JSON format:
    {
        "executive_summary": "A concise 3-4 sentence high-level executive summary
        detailing overall customer satisfaction, recurring themes, and critical problem areas.",
        "global_recommendations": [
            "Actionable recommendation 1 for overall product/service improvement",
            "Actionable recommendation 2...",
            "Actionable recommendation 3..."
        ]
    }

    Respond ONLY with a valid JSON object.
    """

    user_content = f"Customer Feedback Aggregated Data:\n{json.dumps(
        feedbacks_summary_data,
        ensure_ascii=False,
        indent=2
    )}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        raw_json_response = response.choices[0].message.content
        result = json.loads(raw_json_response)

        return result

    except AuthenticationError as e:
        logger.error(f"Authentication error in summary generation: {e}")
        return _get_summary_error_response("Authentication failed.")
    except RateLimitError as e:
        logger.error(f"OpenAI rate limit exceeded in summary generation: {e}")
        return _get_summary_error_response("Rate limit exceeded.")
    except APIConnectionError as e:
        logger.error(f"OpenAI connection error in summary generation: {e}")
        return _get_summary_error_response("Connection error.")
    except OpenAIError as e:
        logger.error(f"Internal OpenAI API error in summary generation: {e}")
        return _get_summary_error_response("OpenAI API error.")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from summary model: {e}")
        return _get_summary_error_response("Failed to parse model response.")


def _get_summary_error_response(message: str) -> dict:
    """Helper function to return a standard error response for global summary."""
    return {
        "executive_summary": f"Failed to generate summary: {message}",
        "global_recommendations": ["Check system logs and verify OpenAI service status."],
    }
