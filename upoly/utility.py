import os
from urllib.parse import ParseResult, urlencode, urlparse, urlunparse

import pandas as pd

from .constants import POLYGON_BASE_URL, TRADES_PATH


def trades_path(
    ticker: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    max_records: int = 50000,
    is_ascending: bool = True,
) -> str:

    date_path = ts_start.date()

    path = f"{TRADES_PATH}{ticker}/{date_path}"

    query = urlencode(
        {
            "reverse": str(is_ascending).lower(),
            "limit": max_records,
            "timestamp": ts_start.value,
            "timestampLimit": ts_end.value,
            "apiKey": os.environ.get("POLYGON_KEY_ID"),
        }
    )

    x = ParseResult("https", "api.polygon.io", path, "", query, "")

    return urlunparse(x)


def str_to_path() -> ParseResult:

    placeholder_key = "smokie"

    input = f"https://api.polygon.io/v2/ticks/stocks/trades/AAPL/2020-10-14?reverse=true&limit=50000&apiKey={placeholder_key}"
    return urlparse(input)
