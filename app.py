"""XAU/USD real-time dashboard: live Finnhub ticks plus news headline sentiment.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import os
import queue
import threading

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

from finnhub_stream import ERROR, STATUS, TICK, FinnhubStreamer
from news_sentiment import (
    DEFAULT_MODEL,
    NewsError,
    fetch_headlines,
    load_classifier,
    score_headlines,
)

# ---------------------------------------------------------------- configuration

load_dotenv()
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL)
DEFAULT_SYMBOL = "OANDA:XAU_USD"  # Finnhub forex/CFD symbol for gold in USD
NEWS_QUERY = '"gold price" OR XAUUSD OR bullion OR "spot gold"'  # matched against headlines only
NEWS_CACHE_SECONDS = 900  # keeps the free NewsAPI quota safe while the page refreshes

st.set_page_config(page_title="XAU/USD Real-Time Dashboard", layout="wide")
st.title("XAU/USD real-time dashboard")
st.caption("Live gold price ticks from Finnhub, with sentiment scores for recent gold headlines.")

if not FINNHUB_KEY:
    st.error("FINNHUB_KEY is missing. Copy `.env.example` to `.env` and add your Finnhub API key.")
    st.stop()

# ---------------------------------------------------------------------- state


def empty_ticks() -> pd.DataFrame:
    return pd.DataFrame({"time": pd.Series(dtype="datetime64[ns, UTC]"), "price": pd.Series(dtype="float64")})


if "ticks" not in st.session_state:
    st.session_state.ticks = empty_ticks()
if "queue" not in st.session_state:
    st.session_state.queue = queue.Queue()
if "streamer" not in st.session_state:
    st.session_state.streamer = None
if "thread" not in st.session_state:
    st.session_state.thread = None
if "running" not in st.session_state:
    st.session_state.running = False
if "status" not in st.session_state:
    st.session_state.status = "Disconnected"
if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = DEFAULT_SYMBOL

# If the background thread died (dropped connection, bad key), reflect that.
thread = st.session_state.thread
if st.session_state.running and (thread is None or not thread.is_alive()):
    st.session_state.running = False
    if not st.session_state.status.startswith(("Error", "Closed")):
        st.session_state.status = "Stream ended"
running: bool = st.session_state.running

# -------------------------------------------------------------------- sidebar

st.sidebar.header("Connection")
symbol_input = st.sidebar.text_input(
    "Finnhub symbol", value=DEFAULT_SYMBOL, key="symbol_input", disabled=running
)
max_ticks = st.sidebar.slider("Ticks to keep", 200, 5000, 1500, step=100)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh every (s)", 1, 10, 2, disabled=not auto_refresh)

st.sidebar.header("News")
show_news = st.sidebar.checkbox("Show news sentiment", value=True)

# -------------------------------------------------------------------- stream


def start_stream() -> None:
    symbol = symbol_input.strip()
    if not symbol:
        st.session_state.status = "Error: enter a symbol first"
        return
    st.session_state.ticks = empty_ticks()
    st.session_state.queue = queue.Queue()  # fresh queue so old messages cannot leak in
    streamer = FinnhubStreamer(FINNHUB_KEY, symbol, st.session_state.queue)
    st.session_state.streamer = streamer
    st.session_state.thread = threading.Thread(target=streamer.run, daemon=True, name="finnhub-stream")
    st.session_state.active_symbol = symbol
    st.session_state.status = f"Connecting to {symbol}..."
    st.session_state.running = True
    st.session_state.thread.start()


def stop_stream() -> None:
    if st.session_state.streamer:
        st.session_state.streamer.stop()
    st.session_state.running = False


def drain_queue(max_items: int = 5000) -> None:
    """Move everything the background thread has queued into session state."""
    new_ticks = []
    for _ in range(max_items):
        try:
            kind, payload = st.session_state.queue.get_nowait()
        except queue.Empty:
            break
        if kind == TICK:
            new_ticks.append(payload)
        elif kind == STATUS:
            st.session_state.status = str(payload)
        elif kind == ERROR:
            st.session_state.status = f"Error: {payload}"
    if new_ticks:
        fresh = pd.DataFrame(new_ticks, columns=["time", "price"])
        old = st.session_state.ticks
        combined = fresh if old.empty else pd.concat([old, fresh], ignore_index=True)
        st.session_state.ticks = combined.tail(max_ticks).reset_index(drop=True)


start_col, stop_col, status_col = st.columns([1, 1, 4])
if start_col.button("Start stream", type="primary", disabled=running):
    start_stream()
    st.rerun()
if stop_col.button("Stop stream", disabled=not running):
    stop_stream()
    st.rerun()

drain_queue()  # always drain, so a final "Closed" status still shows after stopping
status_col.markdown(f"**Symbol:** {st.session_state.active_symbol} &nbsp;•&nbsp; **Status:** {st.session_state.status}")

# ---------------------------------------------------------------- price view

ticks: pd.DataFrame = st.session_state.ticks
metric_price, metric_range, metric_count = st.columns(3)

if len(ticks):
    last = ticks["price"].iloc[-1]
    change = f"{last - ticks['price'].iloc[-2]:+.2f}" if len(ticks) > 1 else None
    metric_price.metric("Last price (USD/oz)", f"{last:,.2f}", delta=change)
    metric_range.metric("High / low (buffered ticks)", f"{ticks['price'].max():,.2f} / {ticks['price'].min():,.2f}")
    metric_count.metric("Ticks buffered", f"{len(ticks):,}")
    # Gold moves a few dollars around ~4,000, so the y-axis must not start at zero.
    chart = (
        alt.Chart(ticks)
        .mark_line()
        .encode(
            x=alt.X("time:T", title=None),
            y=alt.Y("price:Q", title="USD/oz", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("time:T", format="%H:%M:%S"), alt.Tooltip("price:Q", format=",.2f")],
        )
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(
        ticks.tail(20).iloc[::-1],
        hide_index=True,
        column_config={
            "time": st.column_config.DatetimeColumn("Time (UTC)", format="HH:mm:ss"),
            "price": st.column_config.NumberColumn("Price", format="%.2f"),
        },
    )
else:
    metric_price.metric("Last price (USD/oz)", "—")
    metric_range.metric("High / low (buffered ticks)", "—")
    metric_count.metric("Ticks buffered", "0")
    if running:
        st.info("Waiting for ticks. Forex and gold markets are closed at weekends, so none may arrive then.")
    else:
        st.info("Press **Start stream** to begin receiving live prices.")

# ----------------------------------------------------------------------- news


@st.cache_resource(show_spinner="Loading sentiment model (the first run downloads it)...")
def get_classifier(model_name: str):
    return load_classifier(model_name)


@st.cache_data(ttl=NEWS_CACHE_SECONDS, show_spinner="Fetching headlines...")
def get_news_sentiment(query: str, model_name: str) -> dict:
    """Cached so the page's frequent refreshes do not hit NewsAPI every time."""
    try:
        titles = fetch_headlines(NEWS_API_KEY, query, limit=5)
        scored = score_headlines(titles, get_classifier(model_name))
    except NewsError as exc:
        return {"items": [], "error": str(exc)}
    except ImportError:
        return {"items": [], "error": "transformers is not installed. Run `pip install -r requirements.txt`."}
    except OSError as exc:
        return {"items": [], "error": f"Could not load the sentiment model: {exc}"}
    return {"items": [dataclasses.asdict(s) for s in scored], "error": None}


if show_news:
    st.subheader("Gold news sentiment")
    if not NEWS_API_KEY:
        st.caption("Add NEWS_API_KEY to your `.env` file to enable headline sentiment.")
    else:
        result = get_news_sentiment(NEWS_QUERY, SENTIMENT_MODEL)
        if result["error"]:
            st.warning(result["error"])
        elif result["items"]:
            news = pd.DataFrame(result["items"]).rename(
                columns={"title": "Headline", "label": "Sentiment", "confidence": "Confidence"}
            )
            st.dataframe(
                news,
                hide_index=True,
                column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")},
            )
        else:
            st.caption("No headlines were returned.")
        st.caption(
            f"Scored by `{SENTIMENT_MODEL}`, a general-purpose model: treat it as a rough guide, not a trading signal. "
            f"Headlines refresh at most every {NEWS_CACHE_SECONDS // 60} minutes."
        )
        if st.button("Refresh headlines now"):
            get_news_sentiment.clear()
            st.rerun()

# ---------------------------------------------------------------- auto refresh

if running and auto_refresh:
    st_autorefresh(interval=refresh_seconds * 1000, limit=None, key="tick_refresh")
