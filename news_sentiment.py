"""Fetch recent headlines from NewsAPI and score their sentiment.

The Hugging Face model is only imported when ``load_classifier`` is called,
because importing ``transformers`` is slow and not needed unless news is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import requests

NEWSAPI_URL = "https://newsapi.org/v2/everything"
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class NewsError(Exception):
    """Headlines could not be fetched."""


@dataclass(frozen=True)
class ScoredHeadline:
    title: str
    label: str
    confidence: float


def fetch_headlines(api_key: str, query: str = "gold", limit: int = 5) -> list[str]:
    """Return up to ``limit`` recent English headlines matching ``query``.

    The API key is sent as a header, not in the URL, so it cannot end up in
    logs or error messages. Raises ``NewsError`` on any failure.
    """
    try:
        response = requests.get(
            NEWSAPI_URL,
            params={"q": query, "searchIn": "title", "sortBy": "publishedAt", "pageSize": limit, "language": "en"},
            headers={"X-Api-Key": api_key},
            timeout=10,
        )
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise NewsError(f"Could not reach NewsAPI ({type(exc).__name__}).") from exc

    if response.status_code != 200 or data.get("status") == "error":
        reason = data.get("message") or f"HTTP {response.status_code}"
        raise NewsError(f"NewsAPI error: {reason}")

    titles = [(article.get("title") or "").strip() for article in data.get("articles", [])]
    return [t for t in titles if t and t != "[Removed]"]


def load_classifier(model_name: str = DEFAULT_MODEL) -> Callable:
    """Load a Hugging Face sentiment-analysis pipeline (downloads it on first use)."""
    from transformers import pipeline

    return pipeline("sentiment-analysis", model=model_name)


def score_headlines(titles: list[str], classifier: Callable) -> list[ScoredHeadline]:
    """Score all headlines in one batch call to the classifier."""
    if not titles:
        return []
    results = classifier(titles, truncation=True)
    return [
        ScoredHeadline(title=title, label=str(result["label"]), confidence=float(result["score"]))
        for title, result in zip(titles, results)
    ]
