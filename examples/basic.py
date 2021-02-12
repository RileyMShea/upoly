#%%
import numpy as np
import pandas as pd

from upoly import NY, async_polygon_aggs

#%%
# short time period(1day)
start = pd.Timestamp("2020-01-02", tz=NY)
end = pd.Timestamp("2020-01-03", tz=NY)
coup = async_polygon_aggs("COUP", start, end, debug_mode=True)
coup
#%%
# medium time period
start = pd.Timestamp("2019-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
aapl = async_polygon_aggs("AAPL", start, end, debug_mode=True)
aapl.shape[0] % 390
#%%
#%%
# long time period
start = pd.Timestamp("2015-06-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
aapl = async_polygon_aggs("GOOG", start, end, debug_mode=True)
aapl
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
