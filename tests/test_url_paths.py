from urllib.parse import ParseResult, urlencode, urlparse

import pandas as pd

from upoly.utility import str_to_path, trades_path


def test_toolz_map() -> None:
    got = list(map(lambda x: x + 1, range(10)))
    want = list(range(1, 11, 1))
    assert got == want


def test_qs() -> None:
    got = urlencode({"reverse": True, "limit": 50000, "apiKey": "smokie"})
    want = "reverse=True&limit=50000&apiKey=smokie"
    assert got == want


def test_path_parsing() -> None:
    u = urlparse("//www.cwi.nl:80/%7Eguido/Python.html")
    assert u.scheme == ""
    assert u.netloc == "www.cwi.nl:80"


# def test_breaking_up_polygon_trades_url() -> None:
#     got = trades_path(
#         "AAPL",
#         pd.Timestamp(datetime(2020, 10, 14)),
#         pd.Timestamp(datetime(2020, 10, 15)),
#     )
#     want = "https://api.polygon.io/v2/ticks/stocks/trades/AAPL/2020-10-14?reverse=True&limit=50000&apiKey=smokie"
#     assert got == want


def test_string_to_path() -> None:
    got = ParseResult(
        "https",
        "api.polygon.io",
        "/v2/ticks/stocks/trades/AAPL/2020-10-14",
        "",
        "reverse=true&limit=50000&apiKey=smokie",
        "",
    )
    want = str_to_path()
    assert got == want
