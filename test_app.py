"""Smoke tests that run the real Streamlit script with Streamlit's test harness.

No network access is used: the WebSocket client and the news functions are
replaced with stand-ins.
"""

import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import news_sentiment
from finnhub_stream import STATUS, TICK

APP = str(Path(__file__).with_name("app.py"))


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    # Never read a developer's real .env or keys during tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.delenv("FINNHUB_KEY", raising=False)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    st.cache_data.clear()
    st.cache_resource.clear()


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def metrics(at):
    return {m.label: m for m in at.metric}


def tick(seconds, price):
    return (TICK, (datetime.fromtimestamp(seconds, tz=timezone.utc), price))


# ------------------------------------------------------------------ configuration


def test_missing_finnhub_key_shows_a_setup_message_and_stops():
    at = run_app()

    assert not at.exception
    assert len(at.error) == 1 and "FINNHUB_KEY" in at.error[0].value
    assert not at.button  # nothing else was drawn


def test_idle_state_with_a_key(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    assert not at.exception
    assert button(at, "Start stream").disabled is False
    assert button(at, "Stop stream").disabled is True
    assert metrics(at)["Last price (USD/oz)"].value == "—"
    assert any("Start stream" in i.value for i in at.info)


# --------------------------------------------------------------------- price view


def test_queued_ticks_are_drained_and_displayed(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    queue = at.session_state["queue"]
    queue.put(tick(1_700_000_000, 2000.00))
    queue.put(tick(1_700_000_001, 2001.50))
    at.run()

    assert not at.exception
    shown = metrics(at)
    assert shown["Last price (USD/oz)"].value == "2,001.50"
    assert shown["Last price (USD/oz)"].delta == "+1.50"
    assert shown["High / low (buffered ticks)"].value == "2,001.50 / 2,000.00"
    assert shown["Ticks buffered"].value == "2"


def test_a_single_tick_has_no_change_value(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()
    at.session_state["queue"].put(tick(1_700_000_000, 2000.00))
    at.run()

    assert metrics(at)["Last price (USD/oz)"].value == "2,000.00"
    assert metrics(at)["Last price (USD/oz)"].delta in (None, "")


def test_status_and_error_messages_reach_the_page(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    at.session_state["queue"].put((STATUS, "Connected to OANDA:XAU_USD"))
    at.run()
    assert any("Connected to OANDA:XAU_USD" in m.value for m in at.markdown)

    at.session_state["queue"].put(("error", "Finnhub: Invalid API key"))
    at.run()
    assert any("Error: Finnhub: Invalid API key" in m.value for m in at.markdown)


def test_a_dead_stream_thread_resets_the_running_state(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    at.session_state["running"] = True  # claims to run, but there is no thread
    at.run()

    assert button(at, "Start stream").disabled is False
    assert any("Stream ended" in m.value for m in at.markdown)


# ------------------------------------------------------------------- start / stop


class FakeStreamer:
    instances = []

    def __init__(self, token, symbol, out_queue):
        self.token, self.symbol, self.out_queue = token, symbol, out_queue
        self.release = threading.Event()
        FakeStreamer.instances.append(self)

    def run(self):
        self.out_queue.put((STATUS, f"Connected to {self.symbol}"))
        self.release.wait(timeout=10)

    def stop(self):
        self.release.set()


@pytest.fixture
def fake_streamer(monkeypatch):
    FakeStreamer.instances = []
    monkeypatch.setattr("finnhub_stream.FinnhubStreamer", FakeStreamer)
    yield FakeStreamer
    for instance in FakeStreamer.instances:
        instance.release.set()


def test_start_then_stop_runs_the_full_state_machine(monkeypatch, fake_streamer):
    monkeypatch.setenv("FINNHUB_KEY", "secret-token")
    at = run_app()

    button(at, "Start stream").click().run()

    assert not at.exception
    (streamer,) = fake_streamer.instances
    assert (streamer.token, streamer.symbol) == ("secret-token", "OANDA:XAU_USD")
    assert button(at, "Start stream").disabled is True
    assert button(at, "Stop stream").disabled is False
    assert next(t for t in at.sidebar.text_input if t.label == "Finnhub symbol").disabled is True

    button(at, "Stop stream").click().run()

    assert not at.exception
    assert button(at, "Start stream").disabled is False
    assert button(at, "Stop stream").disabled is True
    assert streamer.release.is_set()


def test_start_uses_the_symbol_typed_in_the_sidebar(monkeypatch, fake_streamer):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    next(t for t in at.sidebar.text_input if t.label == "Finnhub symbol").set_value("OANDA:EUR_USD").run()
    button(at, "Start stream").click().run()

    assert fake_streamer.instances[0].symbol == "OANDA:EUR_USD"


def test_blank_symbol_is_rejected(monkeypatch, fake_streamer):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    next(t for t in at.sidebar.text_input if t.label == "Finnhub symbol").set_value("   ").run()
    button(at, "Start stream").click().run()

    assert fake_streamer.instances == []
    assert any("enter a symbol" in m.value for m in at.markdown)


def test_restarting_clears_the_old_ticks(monkeypatch, fake_streamer):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()
    at.session_state["queue"].put(tick(1_700_000_000, 2000.00))
    at.run()
    assert metrics(at)["Ticks buffered"].value == "1"

    button(at, "Start stream").click().run()

    assert metrics(at)["Ticks buffered"].value == "0"


# ------------------------------------------------------------------------- news


class StubClassifier:
    def __call__(self, texts, **kwargs):
        return [{"label": "POSITIVE", "score": 0.9} for _ in texts]


@pytest.fixture
def news_stubs(monkeypatch):
    calls = {"fetch": 0, "load": 0}

    def fetch(api_key, query="gold", limit=5):
        calls["fetch"] += 1
        return ["Gold climbs", "Gold steady"]

    def load(model_name=None):
        calls["load"] += 1
        return StubClassifier()

    monkeypatch.setattr(news_sentiment, "fetch_headlines", fetch)
    monkeypatch.setattr(news_sentiment, "load_classifier", load)
    return calls


def test_news_without_a_key_explains_how_to_enable_it(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    at = run_app()

    assert any(s.value == "Gold news sentiment" for s in at.subheader)
    assert any("NEWS_API_KEY" in c.value for c in at.caption)
    assert not at.dataframe


def test_news_headlines_are_scored_and_shown(monkeypatch, news_stubs):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    monkeypatch.setenv("NEWS_API_KEY", "n")
    at = run_app()

    assert not at.exception
    table = at.dataframe[0].value
    assert table["Headline"].tolist() == ["Gold climbs", "Gold steady"]
    assert table["Sentiment"].tolist() == ["POSITIVE", "POSITIVE"]


def test_news_is_cached_across_refreshes(monkeypatch, news_stubs):
    # Regression: the original script called NewsAPI and reloaded the model on every refresh.
    monkeypatch.setenv("FINNHUB_KEY", "k")
    monkeypatch.setenv("NEWS_API_KEY", "n")
    at = run_app()
    at.run()
    at.run()

    assert news_stubs == {"fetch": 1, "load": 1}


def test_news_can_be_hidden(monkeypatch, news_stubs):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    monkeypatch.setenv("NEWS_API_KEY", "n")
    at = run_app()

    next(c for c in at.sidebar.checkbox if c.label == "Show news sentiment").uncheck().run()

    assert not at.dataframe
    assert not any(s.value == "Gold news sentiment" for s in at.subheader)


def test_news_failure_is_shown_as_a_warning(monkeypatch):
    monkeypatch.setenv("FINNHUB_KEY", "k")
    monkeypatch.setenv("NEWS_API_KEY", "n")

    def failing_fetch(api_key, query="gold", limit=5):
        raise news_sentiment.NewsError("NewsAPI error: rate limited")

    monkeypatch.setattr(news_sentiment, "fetch_headlines", failing_fetch)
    at = run_app()

    assert not at.exception
    assert any("rate limited" in w.value for w in at.warning)
