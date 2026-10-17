# XAU/USD Real-Time Dashboard

A Streamlit dashboard that streams live gold (XAU/USD) price ticks from the Finnhub WebSocket API and scores recent gold news headlines with a Hugging Face sentiment model.

<!-- Add your own screenshot: save it as docs/screenshot.png and uncomment the next line -->
<!-- ![Dashboard screenshot](docs/screenshot.png) -->

## Features

- **Live prices:** WebSocket stream from Finnhub with last price (and change from the previous tick), high/low of the buffered ticks, a line chart and the 20 most recent ticks.
- **News sentiment:** the latest gold headlines from NewsAPI, classified in a single batch by a Hugging Face model (DistilBERT SST-2 by default).
- **Start/Stop control:** the symbol can be changed only while the stream is stopped; restarting clears the old ticks.
- **Auto-refresh:** configurable 1-10 s page refresh while streaming.
- **Resilient:** a dropped connection, bad API key, malformed message or NewsAPI failure is shown on the page instead of crashing it.

## Architecture

```
Finnhub WebSocket ──> FinnhubStreamer (background thread) ──> queue.Queue ──> app.py (drains on each rerun) ──> charts/metrics
NewsAPI ──> fetch_headlines ──> score_headlines (Hugging Face) ──> cached 15 min ──> table
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI, session state, queue draining, news display |
| `finnhub_stream.py` | WebSocket client and message parsing (`parse_trade_message`) |
| `news_sentiment.py` | NewsAPI fetching and sentiment scoring |
| `test_*.py` | Unit tests and Streamlit `AppTest` smoke tests (no network needed) |

Design choices:

- The background thread never touches Streamlit state. It only puts messages on a queue, which the app drains on each rerun.
- The sentiment model is loaded once (`st.cache_resource`) and headlines are cached for 15 minutes (`st.cache_data`), so the frequent price refreshes do not reload the model or exhaust the free NewsAPI quota.
- The NewsAPI key is sent in an `X-Api-Key` header, not the URL, so it cannot leak into logs or error messages.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/abukar916/xauusd-dashboard.git
cd xauusd-dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # then add your keys
streamlit run app.py
```

`torch` and `transformers` are large downloads, and the sentiment model is downloaded on first use. The price stream works without NEWS_API_KEY.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `FINNHUB_KEY` | Yes | Finnhub API key ([finnhub.io](https://finnhub.io), free tier works) |
| `NEWS_API_KEY` | No | NewsAPI key ([newsapi.org](https://newsapi.org)); enables the news panel |
| `SENTIMENT_MODEL` | No | Hugging Face sentiment model; defaults to `distilbert-base-uncased-finetuned-sst-2-english`. For finance-specific scoring try `ProsusAI/finbert` |

## Usage

1. Run the app and press **Start stream**.
2. Adjust how many ticks to keep and the refresh rate in the sidebar.
3. Press **Stop stream** to disconnect. Use **Refresh headlines now** to bypass the 15-minute news cache.

Gold and forex markets are closed at weekends, so no ticks arrive then.

## Tests

```bash
pip install pytest
pytest
```

51 tests cover message parsing, the WebSocket client callbacks, NewsAPI handling (including that the key is never in the URL), sentiment batching, and the full Start/Stop UI flow using Streamlit's headless test harness. The network and the Hugging Face model are stubbed.

## Limitations

- Closing the browser tab without pressing **Stop** leaves the WebSocket thread running until the server process exits.
- The default sentiment model is general-purpose (movie-review trained), so treat scores as a rough guide, not a trading signal.
- Ticks are held in memory only; there is no persistence or historical data.
- The real model and live APIs are not exercised by the tests.


