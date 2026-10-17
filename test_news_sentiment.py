import sys
import types

import pytest
import requests

import news_sentiment
from news_sentiment import (
    NewsError,
    ScoredHeadline,
    fetch_headlines,
    load_classifier,
    score_headlines,
)

SECRET = "super-secret-key"


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=False):
        self._payload = payload
        self.status_code = status_code
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Replace requests.get and record how it was called."""
    calls = []
    state = {"response": FakeResponse({"status": "ok", "articles": []}), "raises": None}

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if state["raises"]:
            raise state["raises"]
        return state["response"]

    monkeypatch.setattr(news_sentiment.requests, "get", get)
    return types.SimpleNamespace(calls=calls, state=state)


# ----------------------------------------------------------------- fetch_headlines


def test_returns_titles_from_the_response(fake_get):
    fake_get.state["response"] = FakeResponse(
        {"status": "ok", "articles": [{"title": "Gold hits record"}, {"title": "Gold slips"}]}
    )

    assert fetch_headlines(SECRET, "gold", 5) == ["Gold hits record", "Gold slips"]


def test_api_key_goes_in_a_header_not_the_url_or_params(fake_get):
    fetch_headlines(SECRET, "gold", 5)

    (url, kwargs), = fake_get.calls
    assert kwargs["headers"] == {"X-Api-Key": SECRET}
    assert SECRET not in url
    assert SECRET not in str(kwargs["params"])


def test_request_has_a_timeout_and_the_expected_params(fake_get):
    fetch_headlines(SECRET, "gold", 7)

    (_, kwargs), = fake_get.calls
    assert kwargs["timeout"] == 10
    assert kwargs["params"] == {"q": "gold", "searchIn": "title", "sortBy": "publishedAt", "pageSize": 7, "language": "en"}


def test_blank_missing_and_removed_titles_are_dropped(fake_get):
    fake_get.state["response"] = FakeResponse(
        {
            "status": "ok",
            "articles": [
                {"title": "Real headline"},
                {"title": ""},
                {"title": None},
                {},
                {"title": "[Removed]"},
                {"title": "  padded  "},
            ],
        }
    )

    assert fetch_headlines(SECRET) == ["Real headline", "padded"]


def test_api_error_message_is_reported(fake_get):
    fake_get.state["response"] = FakeResponse(
        {"status": "error", "message": "You have made too many requests."}, status_code=429
    )

    with pytest.raises(NewsError, match="too many requests"):
        fetch_headlines(SECRET)


def test_http_error_without_a_message_reports_the_status_code(fake_get):
    fake_get.state["response"] = FakeResponse({}, status_code=500)

    with pytest.raises(NewsError, match="HTTP 500"):
        fetch_headlines(SECRET)


def test_network_failure_becomes_a_news_error_without_leaking_the_key(fake_get):
    fake_get.state["raises"] = requests.ConnectionError(f"failed for key {SECRET}")

    with pytest.raises(NewsError) as excinfo:
        fetch_headlines(SECRET)
    assert SECRET not in str(excinfo.value)


def test_non_json_response_becomes_a_news_error(fake_get):
    fake_get.state["response"] = FakeResponse(json_error=True)

    with pytest.raises(NewsError):
        fetch_headlines(SECRET)


# --------------------------------------------------------------- score_headlines


def test_scores_every_headline_in_one_batch_call():
    calls = []

    def classifier(texts, **kwargs):
        calls.append((list(texts), kwargs))
        return [{"label": "POSITIVE", "score": 0.91}, {"label": "NEGATIVE", "score": 0.77}]

    scored = score_headlines(["Gold rallies", "Gold sinks"], classifier)

    assert scored == [
        ScoredHeadline("Gold rallies", "POSITIVE", 0.91),
        ScoredHeadline("Gold sinks", "NEGATIVE", 0.77),
    ]
    assert calls == [(["Gold rallies", "Gold sinks"], {"truncation": True})]


def test_no_headlines_means_the_classifier_is_never_called():
    def classifier(*args, **kwargs):
        raise AssertionError("should not be called")

    assert score_headlines([], classifier) == []


# --------------------------------------------------------------- load_classifier


def test_load_classifier_builds_a_sentiment_pipeline_for_the_named_model(monkeypatch):
    captured = {}

    def fake_pipeline(task, **kwargs):
        captured["task"] = task
        captured.update(kwargs)
        return "classifier"

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))

    assert load_classifier("some/model") == "classifier"
    assert captured == {"task": "sentiment-analysis", "model": "some/model"}
