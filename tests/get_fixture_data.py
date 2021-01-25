#%%
import os

import brotli
import orjson
import pandas as pd
import pytz

from upoly import NY, async_polygon_aggs

# print(os.getcwd())

# #%%
# %%time
# with open("./fixtures/SHOP-1559361600000-1565526000000.json", "rb") as f:
#     res = orjson.loads(f.read())

# with open("./fixtures/SHOP-1559361600000-1565526000000.json.zip", "wb") as f:
#     x = brotli.compress(orjson.dumps(res))
#     f.write(x)
# #%%
# %%time
# with open("./fixtures/SHOP-1559361600000-1565526000000.json.zip", "rb") as f:
#     res = orjson.loads(brotli.decompress(f.read()))


#%%


def no_missing() -> None:
    print(os.getcwd())
    try:
        os.chdir("./fixtures")
    except:
        pass
    start = pd.Timestamp("2019-06-01", tz=NY)
    end = pd.Timestamp("2020-01-01", tz=NY)
    async_polygon_aggs("SHOP", start, end, debug_mode=True)


def some_missing():
    ...


def all_missing():
    ...


if __name__ == "__main__":
    no_missing()
