#%%
import os
from datetime import datetime

import httpx
import numba
import numpy as np
import pandas as pd
from cytoolz import map

from upoly import NY
from upoly.models import PolyTradeResponse
from upoly.utility import trades_path

pd.options.display.float_format = "{:.3f}".format

#%%

ts_start = pd.Timestamp(datetime(2020, 10, 14, 9, 30), tz=NY)
ts_end = ts_start + pd.Timedelta(5, "minute")

url = trades_path("SHOP", ts_start, ts_end, is_ascending=False)
url


#%%

res = httpx.get(url).json()["results"]
print(len(res))
#%%
res
#%%


res_times = list(map(lambda x: x["t"], res))
min(res_times), max(res_times)
#%%
data: PolyTradeResponse = res.json()

data
#%%
column_renamer = {k: v["name"] for k, v in data["map"].items()}
column_retyper = {v["name"]: v["type"] for _, v in data["map"].items()}
column_retyper
#%%
df = pd.DataFrame(data["results"])
df.rename(columns=column_renamer, inplace=True)

#%%
df["sip_timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
df["participant_timestamp"] = pd.to_datetime(
    df["participant_timestamp"], unit="ns", utc=True
)
df["trf_timestamp"] = pd.to_datetime(df["participant_timestamp"], unit="ns", utc=True)
#%%
df.sip_timestamp.dt.tz_convert(NY)


#%%
exchanges = httpx.get(
    f"https://api.polygon.io/v1/meta/exchanges?apiKey={os.environ.get('POLYGON_KEY_ID')}"
).json()

exchanges

#%%
exchange_df = pd.DataFrame(exchanges).sort_values("id")
exchange_df
#%%
# TODO: convert with dummies
df.pop("conditions")
#%%
df.id = df.id.astype("category")
exchange_df = exchange_df.astype("category")
#%%
exchange_df = exchange_df.rename(columns={"id": "exchange_id"})
df = df.rename(columns={"exchange": "exchange_id"})


#%%
combined = df.merge(exchange_df, on="exchange_id")
combined.pop("exchange_id")
combined.pop("sequence_number")
combined.pop("id")
combined.pop("trf_id")
#%%
combined.head()
#%%
#%%
#%%
# combined.sequence_number = combined.sequence_number.astype("category")
# combined.exchange_id = combined.exchange_id.astype("category")

#%%
combined.groupby("mic").describe().loc[["XNAS", "XNYS", "XNGS"], :].T

#%%
combined["fill_lag"] = combined.iloc[:, 0] - combined.iloc[:, 1]
#%%
combined = df.rename(columns={"size": "volume"})
combined

# %%

combined.fill_lag.agg(["min", "max", "median", "mean"])
#%%

combined.plot(x="sip_timestamp", y="volume")
#%%
combined.sip_timestamp
