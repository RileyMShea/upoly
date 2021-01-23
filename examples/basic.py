#%%
import pandas as pd
import pytz

from upoly import NY, async_polygon_aggs

NY = pytz.timezone("America/New_York")

start = pd.Timestamp("2018-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)
#%%
# asset_names = ["AAPL", "TSLA", "MSFT"]
aapl = async_polygon_aggs("SHOP", start, end, debug_mode=True)
aapl
#%%
tsla = async_polygon_aggs("TSLA", start, end)
