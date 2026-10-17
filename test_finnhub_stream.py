import json
from datetime import datetime, timezone
from queue import Queue

import pytest

from finnhub_stream import (
    ERROR,
    STATUS,
    TICK,
    FinnhubError,
    FinnhubStreamer,
    parse_trade_message,
)


def trade_message(*trades):
    return json.dumps({"type": "trade", "data": list(trades)})


# ------------------------------------------------------------ parse_trade_message


def test_parses_a_single_trade_into_a_utc_tick():
    msg = trade_message({"p": 2345.67, "s": "OANDA:XAU_USD", "t": 1_700_000_000_000, "v": 1})

    assert parse_trade_message(msg) == [
        (datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc), 2345.67)
    ]


def test_parses_several_trades_in_one_message_in_order():
    msg = trade_message({"p": 1.0, "t": 1_000}, {"p": 2.0, "t": 2_000}, {"p": 3.0, "t": 3_000})

    assert [price for _, price in parse_trade_message(msg)] == [1.0, 2.0, 3.0]


def test_millisecond_timestamps_keep_their_fraction():
    (ts, _), = parse_trade_message(trade_message({"p": 1.0, "t": 1_700_000_000_250}))
    assert ts.microsecond == 250_000


@pytest.mark.parametrize(
    "message",
    [
        json.dumps({"type": "ping"}),
        json.dumps({"type": "trade", "data": None}),
        json.dumps({"type": "trade", "data": []}),
        json.dumps([1, 2, 3]),
        json.dumps("hello"),
    ],
)
def test_messages_without_trades_give_no_ticks(message):
    assert parse_trade_message(message) == []


def test_error_message_raises_finnhub_error():
    with pytest.raises(FinnhubError, match="Invalid API key"):
        parse_trade_message(json.dumps({"type": "error", "msg": "Invalid API key"}))


def test_error_message_without_text_still_raises():
    with pytest.raises(FinnhubError, match="unknown error"):
        parse_trade_message(json.dumps({"type": "error"}))


def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError):
        parse_trade_message("{not json")


@pytest.mark.parametrize(
    "trade",
    [{"t": 1_000}, {"p": 1.0}, {"p": "abc", "t": 1_000}, {"p": 1.0, "t": None}, None],
)
def test_trade_missing_or_bad_fields_raises_value_error(trade):
    with pytest.raises(ValueError):
        parse_trade_message(trade_message(trade))


# ----------------------------------------------------------------- FinnhubStreamer


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def make_streamer():
    queue = Queue()
    return FinnhubStreamer("token", "OANDA:XAU_USD", queue), queue


def drain(queue):
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def test_on_open_subscribes_to_the_symbol_and_reports_status():
    streamer, queue = make_streamer()
    socket = FakeSocket()

    streamer._on_open(socket)

    assert [json.loads(m) for m in socket.sent] == [{"type": "subscribe", "symbol": "OANDA:XAU_USD"}]
    assert drain(queue) == [(STATUS, "Connected to OANDA:XAU_USD")]


def test_on_message_queues_each_tick():
    streamer, queue = make_streamer()

    streamer._on_message(None, trade_message({"p": 1.5, "t": 1_000}, {"p": 2.5, "t": 2_000}))

    items = drain(queue)
    assert [kind for kind, _ in items] == [TICK, TICK]
    assert [payload[1] for _, payload in items] == [1.5, 2.5]


def test_on_message_queues_finnhub_errors():
    streamer, queue = make_streamer()

    streamer._on_message(None, json.dumps({"type": "error", "msg": "Invalid API key"}))

    assert drain(queue) == [(ERROR, "Finnhub: Invalid API key")]


def test_on_message_queues_parse_errors_instead_of_raising():
    streamer, queue = make_streamer()

    streamer._on_message(None, "{oops")

    (kind, text), = drain(queue)
    assert kind == ERROR and text.startswith("Could not parse message")


def test_a_bad_message_does_not_stop_later_good_ones():
    streamer, queue = make_streamer()

    streamer._on_message(None, "{oops")
    streamer._on_message(None, trade_message({"p": 9.0, "t": 1_000}))

    assert [kind for kind, _ in drain(queue)] == [ERROR, TICK]


def test_on_error_and_on_close_report_to_the_queue():
    streamer, queue = make_streamer()

    streamer._on_error(None, "boom")
    streamer._on_close(None, 1000, "bye")
    streamer._on_close(None, None, None)

    assert drain(queue) == [
        (ERROR, "WebSocket error: boom"),
        (STATUS, "Closed (1000) bye"),
        (STATUS, "Closed (None)"),
    ]


def test_stop_closes_the_socket():
    streamer, _ = make_streamer()
    streamer.ws = FakeSocket()

    streamer.stop()

    assert streamer.ws.closed


def test_stop_before_connecting_is_safe():
    streamer, _ = make_streamer()
    streamer.stop()  # no socket yet: must not raise


def test_stop_swallows_errors_from_an_already_broken_socket():
    streamer, _ = make_streamer()

    class Broken:
        def close(self):
            raise RuntimeError("already closed")

    streamer.ws = Broken()
    streamer.stop()  # must not raise
