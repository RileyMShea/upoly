#%%
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.dataframe.core import DataFrame as DaskFrame
from numpy.core.fromnumeric import partition

from upoly import NY, async_polygon_aggs

#%%
# short time period(1day)
start = pd.Timestamp("2020-01-02", tz=NY)
end = pd.Timestamp("2020-01-03", tz=NY)
coup = async_polygon_aggs("COUP", start, end)
coup
#%%
# medium time period
start = pd.Timestamp("2019-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
aapl = async_polygon_aggs("AAPL", start, end)
aapl.shape[0] % 390
#%%
#%%
# long time period
start = pd.Timestamp("2015-06-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
goog = async_polygon_aggs("GOOG", start, end)
msft = async_polygon_aggs("MSFT", start, end)
goog

#%%
goog: DaskFrame = dd.from_pandas(goog, npartitions=1)  # type:ignore
goog
#%%

#%%
goog["ticker"] = "GOOG"
goog.ticker = goog.ticker.astype("category")
#%%
goog["month"] = goog.index.month.astype(str).astype("category")
goog
#%%
#%%


def get_dt_partitions(idx: pd.DatetimeIndex):
    for el in ["month", "day", "hour", "minute"]:
        yield idx.month, idx.day, idx.hour, id.minute


#%%
goog.set_index([goog.ticker, goog.month], append=True, drop=True, inplace=True)

#%%
goog = goog.stack()
goog.name = "goog"
goog

#%%

msft = msft.stack()
msft.name = "msft"
msft
#%%

msft = msft.to_frame()
msft.index.set_names(["time", "ohlvc"], inplace=True)
#%%
msft.rename_axis("ticker", axis="columns", inplace=True)
msft
#%%
msft = msft.reset_index()
msft
#%%
msft.time.dt.year
#%%

msft["year"] = msft.time.dt.year
msft["month"] = msft.time.dt.month
msft["day"] = msft.time.dt.day
msft["hour"] = msft.time.dt.hour
msft["minute"] = msft.time.dt.minute
msft
#%%
msft.groupby("ohlvc").get_group("open")


#%%
# period exceeding stock history
start = pd.Timestamp("2012-06-01", tz=NY)
end = pd.Timestamp("2018-01-01", tz=NY)
shop = async_polygon_aggs("SHOP", start, end)
shop
#%%  Check holiday close has a 12:59 bar
aapl.loc["2019-07-03 12"].tail()
#%%
#%%  Check holiday close doesn't have a 13:00 bar
aapl.loc["2019-07-03 13"].head()
#%%
aapl["volume"] = pd.to_numeric(
    aapl.volume.fillna(0, downcast="infer"), downcast="unsigned"
)

aapl["trades"] = pd.to_numeric(
    aapl.trades.fillna(0, downcast="infer"), downcast="unsigned"
)
aapl
#%%
aapl.iloc[:, :4].info()

#%%
aapl["symbol"] = "AAPL"
aapl["symbol"] = aapl.symbol.astype("category")
aapl
#%%
aapl.to_parquet("./data", "pyarrow", "brotli", True, ["symbol"])
#%%
aapl.open.sort_values()
#%%
pd.read_parquet("./data").info()

#%%
aapl.agg(["min", "max"])

#%%
b = pd.to_numeric(aapl.trades.dropna(), downcast="unsigned")


#%%
c = b.to_numpy()
c
#%%
np.append(c, np.NaN)
#%%
pd.Series([0, 1, np.NaN, 33333.0], dtype=np.float32)

#%%
# tsla = async_polygon_aggs("TSLA", start, end)

# #%%
# print(os.getcwd())
# a = {"a": 1}

# x = brotli.compress(orjson.dumps(a))

# with open("a.json.zip", "wb") as f:
#     f.write(x)

# with open("a.json.zip", "rb") as f:
#     res = orjson.loads(brotli.decompress(f.read()))
# print(res)

#%%

from typing import Any, Callable, TypeVar, cast

from joblib import Memory

cachedir = "./.joblib_cache"
memory = Memory(cachedir, verbose=0)
from time import sleep

F = TypeVar("F", bound=Callable[..., Any])


def typed_cache(
    fn: F,
) -> F:
    "do I see this"

    def wrapper(*args, **kwargs):
        return memory.cache(fn)(*args, **kwargs)

    return cast(F, wrapper)


@typed_cache
def stuff(a: int) -> int:
    """Apple pie"""
    sleep(1)
    return a


@typed_cache
def mstuff(a: float) -> float:
    """Apple pie"""
    return a


#%%
stuff(5)
mstuff(5)
# %%
