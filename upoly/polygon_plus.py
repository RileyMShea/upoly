"""This module provides functionality to interact
with the polygon api asynchonously.
"""
import asyncio
import os
from datetime import datetime
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import httpx
import nest_asyncio
import orjson
import pandas as pd
import pandas_market_calendars as mcal
import pytz
import uvloop
from joblib import Memory
from pandas_market_calendars.exchange_calendar_nyse import NYSEExchangeCalendar

from .settings import unwrap

cachedir = "./.joblib_cache"
memory = Memory(cachedir, verbose=0)

NY = pytz.timezone("America/New_York")


# Create a calendar
nyse: NYSEExchangeCalendar = mcal.get_calendar("NYSE")


# prevent errors in Jupyter/Ipython; otherwise use enhanced event loop
try:
    get_ipython()  # type: ignore
    nest_asyncio.apply()
except NameError:
    uvloop.install()


async def _produce_polygon_aggs(
    queue: asyncio.Queue,
    symbol: str,
    timespan: str,
    interval: int,
    _from: datetime,
    to: datetime,
    unadjusted: bool = False,
    error_threshold: int = 1_000_000,
) -> None:
    """Produce a chunk of polygon results and put in asyncio que.

    Args:
        :param:queue (asyncio.Queue): special collection to put results asynchonously
        :param:symbol (str): the stock symbol
        :param:timespan (str): unit of time, "minute", "hour", "day"
        :param:interval (int): how many units of timespan to agg bars by
        :param:_from (datetime): start of time interval
        :param:to (datetime): end of time interval
        :param:unadjusted (bool, optional): Whether results should be adjusted
        :param:for splits and dividends. Defaults to False.
        :param:error_threshold (int): how many nanoseconds out of request bounds
        before considering invalid.  defaults to 1_000_000 (0.001 seconds)
    """

    # print(f"input:{_from=} {to}")
    start = int(_from.timestamp() * 1_000)  # timestamp in micro seconds
    end = int(to.timestamp() * 1_000)  # timestamp in micro seconds

    assert start < end

    POLYGON_KEY_ID = unwrap(os.getenv("POLYGON_KEY_ID"))
    async with httpx.AsyncClient(http2=True) as client:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/{timespan}/{start}/{end}?unadjusted={unadjusted}&sort=asc&limit=50000&apiKey={POLYGON_KEY_ID}"
        res = await client.get(url)
        if res.status_code != 200:
            ValueError(f"Bad statuscode {res.status_code=};expected 200")

        data: Dict[str, Any] = orjson.loads(res.text)
        results = data.get("results", None)
        if results is None:
            raise ValueError(f"Missing results for range {start=} {end=}")
        df_chunk = pd.DataFrame(results)
        if 10_000 >= (num_results := df_chunk.shape[0]) >= 45_000:
            raise ValueError(f"{num_results=} results, response possibly truncated")

        if (min_bar_time := df_chunk["t"].min()) < (start - 75_000):  # type: ignore
            raise ValueError(
                f"Result contains bar:{min_bar_time=} that pre-dates start time:{start}\n Difference: {start-min_bar_time} ns"  # type:ignore
            )

        if (max_bar_time := df_chunk["t"].max()) > end + 75_000:  # type: ignore
            raise ValueError(
                f"Result contains bar:{max_bar_time=} that post-dates end time:{end} Difference: {max_bar_time-end} ns"  # type: ignore
            )

        await queue.put(df_chunk)


async def _dispatch_consume_polygon(
    intervals: Tuple[pd.Timestamp, pd.Timestamp],
    symbol: str,
    timespan: str,
    interval: int,
    unadjusted: bool = False,
) -> List[pd.DataFrame]:

    queue = asyncio.Queue()

    await asyncio.gather(
        *(
            _produce_polygon_aggs(
                queue, symbol, timespan, interval, _from, to, unadjusted
            )
            for _from, to in intervals
        )
    )
    results: List[pd.DataFrame] = []
    while not queue.empty():
        # print(queue.qsize())
        results.append(await queue.get())
        queue.task_done()
    return results


@memory.cache
def async_polygon_aggs(
    symbol: str,
    timespan: str,
    interval: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    unadjusted: bool = False,
    max_chunk_days: int = 100,
) -> pd.DataFrame:
    """Produce calendar formatted pandas dataframe for a stock.

    Args:
        :param: symbol (str): Stock symbol name; i.e. "AAPL"
        :param: timespan (str): "minute" | "hour" | "day" | "month"
        :param: interval (int): aggregate by this amount; i.e. 15 minutes
        :param: start (pd.Timestamp): requires pytz `NY` tz
        :param: end (pd.Timestamp): requires  pytz `NY` tz
        :param: unadjusted (bool, optional): whether bars should be adjusted for splits. Defaults to False.
        :param: max_chunk_days (int, optional): Limit amount of days to retrieve per request to prevent truncated results. Defaults to 100.


    Returns:
        pd.DataFrame: All valid trading minutes for requested params. All times
        are for "America/New_York" so they account for transitions between
        EST and EDT.  The times are tz-naive for easy usage with numpy, since
        numpy does not support timezones at the time of this writing.

    Example:
    >>> start = pd.Timestamp("2020-10-01", tz=NY)
    >>> end = pd.Timestamp("2020-11-01", tz=NY)

    >>> df = async_polygon_aggs("AAPL", "minute", 1, start, end)
    """

    results: Optional[List[pd.DataFrame]] = None

    if start.tz.zone != "America/New_York" or end.tz.zone != "America/New_York":
        raise ValueError(
            "start and end time must be a NYS, timezone-aware, pandas Timestamp"
        )

    periods = ceil((end - start).days / max_chunk_days)

    intervals: Tuple[pd.Timestamp, pd.Timestamp] = pd.interval_range(
        start=start, end=end, periods=periods
    ).to_tuples()

    results = unwrap(
        asyncio.run(
            _dispatch_consume_polygon(intervals, symbol, timespan, interval, unadjusted)
        )
    )

    if results is None:
        raise ValueError

    df = pd.concat(results, ignore_index=True)
    df.t = pd.to_datetime(df.t.astype(int), unit="ms", utc=True)
    df.set_index("t", inplace=True)

    schedule = nyse.schedule(start, end)
    valid_minutes = mcal.date_range(schedule, "1min")

    expected_sessions = schedule.shape[0]

    actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore

    print(
        f"{expected_sessions=}\t{actual_sessions=}\tpct_diff: {(actual_sessions/expected_sessions)-1.:.2%}"
    )

    expected_minutes = valid_minutes.shape[0]

    actual_minutes = df.shape[0]

    print(
        f"{expected_minutes=}\t{actual_minutes=}\tpct_diff: {(actual_minutes/expected_minutes)-1.:.2%}"
    )

    if (duplicated_indice_count := df.index.duplicated().sum()) > 0:
        print("\n")
        print(f"Found the following duplicated indexes:")
        print(df[df.index.duplicated(keep=False)].sort_index()["vw"])  # type: ignore
        print(f"Dropping {duplicated_indice_count} row(s) w/ duplicate Datetimeindex ")
        df = df[~df.index.duplicated()]

        df = df.reindex(valid_minutes)
        print("Reindexing...")

    print("After Reindexing by trading calender:")

    actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore
    actual_minutes = df.shape[0]

    print(f"{expected_sessions=} {actual_sessions=}")
    print(f"{expected_minutes=} {actual_minutes=}")

    # Rename polygon json keys to canonical pandas headers
    df = df.rename(
        columns={  # type: ignore
            "v": "volume",
            "vw": "volwavg",
            "o": "open",
            "c": "close",
            "h": "high",
            "l": "low",
            "t": "date",
            "n": "trades",
        }
    )  # type:ignore

    # Converting UTC-aware timestamps to NY-naive
    df.index = df.index.tz_convert(NY).tz_localize(None)  # type:ignore

    # Let pandas infer optimal column dtypes; converts trades,volume float->Int64
    df = df.convert_dtypes()

    # Reorder columns so that ohlcv comes first
    df = df[["open", "high", "low", "close", "volume", "volwavg", "trades"]]

    return df


if __name__ == "__main__":
    start = pd.Timestamp("2019-01-01", tz=NY)
    end = pd.Timestamp("2020-01-01", tz=NY)

    data = async_polygon_aggs("AAPL", "minute", 1, start, end)
    print(data.head())
    print(data.info())
