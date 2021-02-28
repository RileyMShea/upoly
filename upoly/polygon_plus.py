"""This module provides functionality to interact
with the polygon api asynchonously.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from asyncio.proactor_events import _ProactorBasePipeTransport
from datetime import datetime, timedelta
from functools import wraps
from math import ceil
from time import perf_counter
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, TypeVar, cast

import aiohttp
import nest_asyncio
import orjson
import pandas as pd
import pandas_market_calendars as mcal
import pytz
from dotenv import find_dotenv, load_dotenv

# import uvloop
from joblib import Memory
from pandas_market_calendars.exchange_calendar_nyse import NYSEExchangeCalendar

from .constants import STAMP_TO_MICRO_FACTOR
from .models import PolyAggResponse
from .settings import unwrap

dirname = os.path.dirname(__file__)
cachedir = os.path.join(dirname, ".joblib_cache")

memory = Memory(cachedir, verbose=0)

NY = pytz.timezone("America/New_York")

try:
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
except IOError:
    try:
        load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))
    except IOError:
        logging.warning("missing .env file")


# prevent errors in Jupyter/Ipython; otherwise attempt to use enhanced event loop
try:
    get_ipython()  # type: ignore
    nest_asyncio.apply()
except NameError:
    try:
        import uvloop

        uvloop.install()
    except:
        pass

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()  # type:ignore
    )

F = TypeVar("F", bound=Callable[..., Any])


def typed_cache(
    fn: F,
) -> F:
    def wrapper(*args, **kwargs):
        is_debug = os.environ.get("UPOLY_DEBUG", None)
        if is_debug:
            return fn(*args, **kwargs)
        else:
            return memory.cache(fn)(*args, **kwargs)

    return cast(F, wrapper)


def silence_event_loop_closed(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if str(e) != "Event loop is closed":
                raise

    return wrapper


_ProactorBasePipeTransport.__del__ = silence_event_loop_closed(
    _ProactorBasePipeTransport.__del__
)


async def _produce_polygon_aggs(
    polygon_id: str,
    client: aiohttp.ClientSession,
    queue: asyncio.Queue,  # type:ignore
    symbol: str,
    timespan: Literal["minute", "hour", "day"],
    interval: int,
    _from: datetime,
    to: datetime,
    unadjusted: bool = False,
    error_threshold: int = 1_000_000,
    response_limit: int = 50_000,
    debug_mode: bool = False,
) -> None:
    """Produce a chunk of polygon results and put in asyncio que.

    Args:
        :param: polygon_id (str): Polygon API key
        :param: client (aiohttp.ClientSession): An open http2 client session
        :param: queue (asyncio.Queue): special collection to put results asynchonously
        :param: symbol (str): the stock symbol
        :param: timespan (str): unit of time, "minute", "hour", "day"
        :param: interval (int): how many units of timespan to agg bars by
        :param: _from (datetime): start of time interval
        :param: to (datetime): end of time interval
        :param: unadjusted (bool, optional): Whether results should be adjusted
        :param: for splits and dividends. Defaults to False.
        :param: error_threshold (int): max number of agg bars in each request
        :param: error_threshold (int): how many nanoseconds requests timestamps
        can be out-of-bounds before considering invalid.  defaults to 1_000_000
        (0.001 seconds)
    """
    # timestamp in micro seconds
    start = int(_from.timestamp() * STAMP_TO_MICRO_FACTOR)
    end = int(to.timestamp() * STAMP_TO_MICRO_FACTOR)

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/{timespan}/{start}/{end}?unadjusted={unadjusted}&sort=asc&limit={response_limit}&apiKey={polygon_id}"
    res = await client.get(url)
    if res.status != 200:
        raise ValueError(f"Bad statuscode {res.status=};expected 200")

    await queue.put(await res.json(loads=orjson.loads))


async def _dispatch_consume_polygon(
    intervals: Tuple[pd.Timestamp, pd.Timestamp],
    symbol: str,
    timespan: Literal["minute", "hour", "day"],
    interval: int,
    unadjusted: bool = False,
) -> List[bytes]:

    queue: asyncio.Queue[bytes] = asyncio.Queue()

    try:
        POLYGON_KEY_ID = unwrap(os.getenv("POLYGON_KEY_ID"))
    except ValueError:
        raise ValueError(f"Missing POLYGON_KEY_ID, please load with from python-dotenv")

    client = aiohttp.ClientSession()
    async with client:

        await asyncio.gather(
            *(
                _produce_polygon_aggs(
                    POLYGON_KEY_ID,
                    client,
                    queue,
                    symbol,
                    timespan,
                    interval,
                    _from,
                    to,
                    unadjusted,
                )
                for _from, to in intervals
            )
        )
        results: List[bytes] = []
        while not queue.empty():
            results.append(await queue.get())
            queue.task_done()

        return results


def combine_chunks(chunks: List[bytes]) -> Iterator[PolyAggResponse]:
    for chunk in chunks:
        yield orjson.loads(chunk)


@typed_cache
@silence_event_loop_closed
def async_polygon_aggs(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    /,
    timespan: Literal["minute", "hour", "day"] = "minute",
    interval: int = 1,
    unadjusted: bool = False,
    max_chunk_days: int = 100,
    paucity_threshold: float = 0.70,
) -> Optional[pd.DataFrame]:
    """Produce calendar formatted pandas dataframe for a stock.

    Args:
        :param: symbol (str): Stock symbol name; i.e. "AAPL"
        :param: start (pd.Timestamp): requires pytz `NY` tz
        :param: end (pd.Timestamp): requires  pytz `NY` tz
        :param: timespan (str): "minute" | "hour" | "day" | "month"
        :param: interval (int): aggregate by this amount; i.e. 15 minutes
        :param: unadjusted (bool, optional): whether bars should be adjusted for splits. Defaults to False.
        :param: max_chunk_days (int, optional): Limit amount of days to retrieve per request to prevent truncated results. Defaults to 100.
        :param: paucity_threshold (float): Minimum ratio of response_records-to-expected_records
        before triggering an exception.  Defaults to 0.7


    Returns:
        pd.DataFrame: All valid trading minutes for requested params. All times
        are for "America/New_York" so they account for transitions between
        EST and EDT.  The times are tz-naive for easy usage with numpy, since
        numpy does not support timezones at the time of this writing.

    Example:
    >>> start = pd.Timestamp("2020-10-01", tz=NY)
    >>> end = pd.Timestamp("2020-11-01", tz=NY)

    >>> df = async_polygon_aggs("AAPL", start, end)
    """

    if start.tz.zone != "America/New_York" or end.tz.zone != "America/New_York":
        raise ValueError(
            "start and end time must be a NYS, timezone-aware, pandas Timestamp"
        )

    periods = ceil((end - start).days / max_chunk_days)

    intervals: Tuple[pd.Timestamp, pd.Timestamp] = pd.interval_range(
        start=start, end=end + timedelta(days=1), periods=periods
    ).to_tuples()

    print(f"Retrieving {periods} mini-batches for {symbol}...")

    network_io_start = perf_counter()

    raw_results = unwrap(
        asyncio.run(
            _dispatch_consume_polygon(intervals, symbol, timespan, interval, unadjusted)
        )
    )
    network_io_stop = perf_counter()
    print(f"Data retrieved in {network_io_stop-network_io_start:.2f} seconds.")
    print("Performing Transforms...")

    df: pd.DataFrame = pd.concat(
        (pd.DataFrame(result_chunk.get("results", None)) for result_chunk in raw_results),
        ignore_index=True,
    )

    nyse: NYSEExchangeCalendar = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start, end)
    valid_minutes: pd.DatetimeIndex = mcal.date_range(schedule, "1min") - pd.Timedelta(
        value=1, unit="T"
    )

    if df is None or df.empty:
        print(f"No results for {symbol}.")
        return None
    df.t = pd.to_datetime(pd.to_numeric(df.t), unit="ms", utc=True)
    df.set_index("t", inplace=True)

    expected_sessions = schedule.shape[0]

    actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore

    print(
        f"{expected_sessions=}\t{actual_sessions=}\tpct_diff: {(actual_sessions/expected_sessions)-1.:+.2%}"
    )

    expected_minutes = valid_minutes.shape[0]

    actual_minutes = df.loc[~df.isnull().all(axis="columns")].shape[0]

    print(
        f"{expected_minutes=}\t{actual_minutes=}\tpct_diff: {(actual_minutes/expected_minutes)-1.:+.2%}"
    )

    if (duplicated_indice_count := df.index.duplicated().sum()) > 0:
        # print("\n")
        # print(f"Found the following duplicated indexes:")
        # print(df[df.index.duplicated(keep=False)].sort_index()["vw"])  # type: ignore
        print(f"Dropping {duplicated_indice_count} row(s) w/ duplicate Datetimeindex ")
        df = df[~df.index.duplicated()]  # type: ignore

    # print("Reindexing...")
    df = df.reindex(valid_minutes)  # type: ignore

    print("After Reindexing by trading calender:")

    actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore
    actual_minutes = df.loc[~df.isnull().all(axis="columns")].shape[0]

    print(f"{expected_sessions = }\t{actual_sessions = }")
    print(f"{expected_minutes = }\t{actual_minutes = }")

    pct_minutes_not_null = actual_minutes / expected_minutes

    print(f"{pct_minutes_not_null = :.3%}")

    if pct_minutes_not_null < paucity_threshold:
        print(f"{symbol} below threshold: {paucity_threshold}")
        return None

    if pct_minutes_not_null > 1.01:
        raise ValueError(f"{pct_minutes_not_null=} Riley messed up, yell at him")

    # Rename polygon json keys to canonical pandas headers
    df = df.rename(
        columns={
            "v": "volume",
            "vw": "volwavg",
            "o": "open",
            "c": "close",
            "h": "high",
            "l": "low",
            "t": "date",
            "n": "trades",
        }
    )
    # Converting UTC-aware timestamps to NY-naive
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_convert(NY).tz_localize(None)  # type:ignore
    else:
        raise TypeError

    # Reorder columns so that ohlcv comes first
    df = df[["open", "high", "low", "close", "volume", "volwavg", "trades"]]
    df.index.name = "time"

    first_index = df.index.min()
    last_index = df.index.max()

    print(f"{first_index = } {last_index = }")

    if isinstance(df, pd.DataFrame):
        return df
    elif df is None:
        return None
    else:
        raise ValueError(f"Expected Dataframe return; Got: {type(df)}")
