#%%
from datetime import datetime

import numpy as np
import pandas as pd

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

#%%
#%%
