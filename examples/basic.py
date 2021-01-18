import pandas as pd
import pytz
from dotenv import load_dotenv

from upoly import NY, async_polygon_aggs

load_dotenv()


NY = pytz.timezone("America/New_York")

start = pd.Timestamp("2019-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)

df = async_polygon_aggs("AAPL", "minute", 1, start, end)
