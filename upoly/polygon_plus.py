"""This module provides functionality to interact
with the polygon api asynchonously.
"""
import asyncio
import os
from datetime import datetime, time
from math import ceil
from time import perf_counter
from typing import List, Tuple

import httpx
import nest_asyncio
import numpy as np
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
nyse: NYSEExchangeCalendar = mcal.get_calendar("NYSE", open_time=time(9, 29))


# prevent errors in Jupyter/Ipython; otherwise use enhanced event loop
try:
    get_ipython()  # type: ignore
    nest_asyncio.apply()
except NameError:
    uvloop.install()


async def _produce_polygon_aggs(
    polygon_id: str,
    client: httpx.AsyncClient,
    queue: asyncio.Queue,
    symbol: str,
    timespan: str,
    interval: int,
    _from: datetime,
    to: datetime,
    unadjusted: bool = False,
    error_threshold: int = 1_000_000,
    response_limit: int = 50_000,
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
        :param:error_threshold (int): max number of agg bars in each request
        :param:error_threshold (int): how many nanoseconds requests timestamps
        can be out-of-bounds before considering invalid.  defaults to 1_000_000
        (0.001 seconds)
    """

    # print(f"input:{_from=} {to}")
    start = int(_from.timestamp() * 1_000)  # timestamp in micro seconds
    end = int(to.timestamp() * 1_000)  # timestamp in micro seconds

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{interval}/{timespan}/{start}/{end}?unadjusted={unadjusted}&sort=asc&limit={response_limit}&apiKey={polygon_id}"
    res = await client.get(url)
    if res.status_code != 200:
        ValueError(f"Bad statuscode {res.status_code=};expected 200")

    await queue.put(res.content)
    return
    # data: Dict[str, Any] = orjson.loads(res.content)

    # results = data.get("results", None)
    # if results is None:
    #     raise ValueError(f"Missing results for range {start=} {end=}")
    # df_chunk = pd.DataFrame(results)
    # if 10_000 >= (num_results := df_chunk.shape[0]) >= 45_000:
    #     raise ValueError(f"{num_results=} results, response possibly truncated")

    # if (min_bar_time := df_chunk["t"].min()) < (start - error_threshold):  # type: ignore
    #     raise ValueError(
    #         f"Result contains bar:{min_bar_time=} that pre-dates start time:{start}\n Difference: {start-min_bar_time} ns"  # type:ignore
    #     )

    # if (max_bar_time := df_chunk["t"].max()) > end + error_threshold:  # type: ignore
    #     raise ValueError(
    #         f"Result contains bar:{max_bar_time=} that post-dates end time:{end} Difference: {max_bar_time-end} ns"  # type: ignore
    #     )


async def _dispatch_consume_polygon(
    intervals: Tuple[pd.Timestamp, pd.Timestamp],
    symbol: str,
    timespan: str,
    interval: int,
    unadjusted: bool = False,
) -> List[bytes]:

    queue: asyncio.Queue[bytes] = asyncio.Queue()

    POLYGON_KEY_ID = unwrap(os.getenv("POLYGON_KEY_ID"))
    client = httpx.AsyncClient(http2=True)  # use httpx
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
    await client.aclose()
    results: List[bytes] = []
    while not queue.empty():
        results.append(await queue.get())
        queue.task_done()

    return results


# @memory.cache
def async_polygon_aggs(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timespan: str = "minute",
    interval: int = 1,
    unadjusted: bool = False,
    max_chunk_days: int = 100,
    stats: bool = False,
    debug_mode: bool = False,
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

    def wrapper(
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        timespan: str = "minute",
        interval: int = 1,
        unadjusted: bool = False,
        max_chunk_days: int = 100,
        stats: bool = False,
        debug_mode: bool = False,
    ) -> pd.DataFrame:
        if start.tz.zone != "America/New_York" or end.tz.zone != "America/New_York":
            raise ValueError(
                "start and end time must be a NYS, timezone-aware, pandas Timestamp"
            )

        periods = ceil((end - start).days / max_chunk_days)

        intervals: Tuple[pd.Timestamp, pd.Timestamp] = pd.interval_range(
            start=start, end=end, periods=periods
        ).to_tuples()

        print(f"Retrieving {periods} mini-batches...")
        network_io_start = perf_counter()
        raw_results = unwrap(
            asyncio.run(
                _dispatch_consume_polygon(
                    intervals, symbol, timespan, interval, unadjusted
                )
            )
        )
        network_io_stop = perf_counter()
        print(f"Data retrieved in {network_io_stop-network_io_start:.2f} seconds.")
        print("Performing Transforms...")
        # results = pd.read_json(raw_results,ignore_index=True)

        df: pd.DataFrame = pd.concat(
            (
                pd.DataFrame(orjson.loads(result_chunk)["results"])
                for result_chunk in raw_results
            ),
            ignore_index=True,
        )
        df.t = pd.to_datetime(df.t.astype(int), unit="ms", utc=True)
        df.set_index("t", inplace=True)

        schedule = nyse.schedule(start, end)
        valid_minutes = mcal.date_range(schedule, "1min")

        expected_sessions = schedule.shape[0]

        actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore

        print(
            f"{expected_sessions=}\t{actual_sessions=}\tpct_diff: {(actual_sessions/expected_sessions)-1.:+.2%}"
        )

        expected_minutes = valid_minutes.shape[0]

        actual_minutes = df.shape[0]

        print(
            f"{expected_minutes=}\t{actual_minutes=}\tpct_diff: {(actual_minutes/expected_minutes)-1.:+.2%}"
        )

        if (duplicated_indice_count := df.index.duplicated().sum()) > 0:
            print("\n")
            print(f"Found the following duplicated indexes:")
            print(df[df.index.duplicated(keep=False)].sort_index()["vw"])  # type: ignore
            print(
                f"Dropping {duplicated_indice_count} row(s) w/ duplicate Datetimeindex "
            )
            df = df[~df.index.duplicated()]

            df = df.reindex(valid_minutes)
            print("Reindexing...")

        print("After Reindexing by trading calender:")

        actual_sessions = df.groupby(df.index.date).count().shape[0]  # type:ignore
        actual_minutes = df.shape[0]

        print(f"{expected_sessions = }\t{actual_sessions = }")
        print(f"{expected_minutes = }\t{actual_minutes = }")

        pct_minutes_not_null = df.dropna().shape[0] / actual_minutes
        print(f"{pct_minutes_not_null = :.3%}")

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
        df = df.astype(np.float64)

        # Reorder columns so that ohlcv comes first
        df = df[["open", "high", "low", "close", "volume", "volwavg", "trades"]]
        df.index.name = "time"

        first_index = df.index.min()
        last_index = df.index.max()

        print(f"{first_index = } {last_index = }")

        return df

    if debug_mode == True:
        return wrapper(
            symbol,
            start,
            end,
            timespan,
            interval,
            unadjusted,
            max_chunk_days,
            stats,
            debug_mode,
        )
    wrapped = memory.cache(wrapper)(
        symbol,
        start,
        end,
        timespan,
        interval,
        unadjusted,
        max_chunk_days,
        stats,
        debug_mode,
    )
    if isinstance(wrapped, pd.DataFrame):
        return wrapped
    else:
        raise ValueError(f"Expected Dataframe return; Got: {type(wrapped)}")


if __name__ == "__main__":
    start = pd.Timestamp("2019-01-01", tz=NY)
    end = pd.Timestamp("2020-01-01", tz=NY)

    data = async_polygon_aggs("AAPL", start, end)

    print(data.head())
    print(data.info())
