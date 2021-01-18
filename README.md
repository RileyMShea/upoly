# upoly

An Asyncio based, high performance, REST client libary for interacting
with the polygon REST api.

Requires Python >=3.8,<=3.9

## Installation

This library makes use of some high performance packages written in `C`/`Rust`
(uvloop, orjson) so it may require `python-dev` on ubuntu or similar on
other OS's.

## Usage

Reccomend to create a copy of `./env.sample` as `./env`. Make sure `.env` is listed
in `.gitignore`.

```env
POLYGON_KEY_ID=REPACEWITHPOLYGONORALPACAKEYHERE
```

Many alternatives to `.env` exist. One such alternative is exporting
like so:

```bash
#!/bin/env bash
export POLYGON_KEY_ID=REPACEWITHPOLYGONORALPACAKEYHERE
```

```python
# yourscript.py
import pytz
from dotenv import load_dotenv
import pandas as pd

# load Polygon key from .env file
load_dotenv()
# alternatively run from cli with:
# POLYGON_KEY_ID=@#*$sdfasd python yourscript.py

# Not recommend but can be set with os.environ["POLYGON_KEY_ID"] as well

from upoly import async_polygon_aggs


NY = pytz.timezone("America/New_York")

# Must be a NY, pandas Timestamp
start = pd.Timestamp("2015-01-01", tz=NY)
end = pd.Timestamp("2020-01-01", tz=NY)

df = async_polygon_aggs("AAPL", "minute", 1, start, end)
```

## TODO

- [ ] unit tests
- [ ] regression tests
- [ ] integration tests
- [ ] `/trade` endpoint functionality for tick data
