#%%
import pandas as pd
import pytz
from dotenv import load_dotenv

from upoly import NY, async_polygon_aggs

load_dotenv()

NY = pytz.timezone("America/New_York")

start = pd.Timestamp("2018-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
#%%
# asset_names = ["AAPL", "TSLA", "MSFT"]
aapl = async_polygon_aggs("SHOP", start, end, debug_mode=True)
aapl
#%%
tsla = async_polygon_aggs("TSLA", start, end)

#%%
aapl

#%%
pd.concat([aapl, aapl], ignore_index=False)
#%%
pd.concat([None, None], ignore_index=True)
#%%
# orjson bytes (res.content)
# 1.75 s ± 68.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


# orjson str (res.text)
# 1.71 s ± 37.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# httpx json
# 2.23 s ± 285 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
