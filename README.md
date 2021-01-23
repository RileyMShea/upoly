# Upoly

<p align="center">
    <img src="upoly.png" alt="upoly">
</p>
<!-- ![upoly logo](upoly.png) -->

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Dependency Status](https://img.shields.io/librariesio/github/RileyMShea/upoly)]("")
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/RileyMShea/upoly/Tests)
![Lines of code](https://img.shields.io/tokei/lines/github/RileyMShea/upoly)
![GitHub issues](https://img.shields.io/github/issues-raw/RileyMShea/upoly)
![GitHub](https://img.shields.io/github/license/RileyMShea/upoly)
![PyPI](https://img.shields.io/pypi/v/upoly)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/upoly)
<a href="https://codecov.io/gh/RileyMShea/upoly" target="_blank">
<img src="https://img.shields.io/codecov/c/github/RileyMShea/upoly?color=%2334D058" alt="Coverage">
</a>

An Asyncio based, high performance, REST client libary for interacting
with the polygon REST api.

## Abstract

The two main python rest-client libs for polygon.io(alpaca-trade-api,
polygonio) do not provide an effective means to gather more than 50,000 trade
bars at once. This library aims to address that by providing an easy and
performant solution to getting results from timespans where the resultset
exceeds 50,000 trade bars.

## Installation

This library makes use of some high performance packages written in `C`/`Rust`
(uvloop, orjson) so it may require `sudo apt install python3-dev` on Ubuntu or similar on
other OS's. It is currently only compatible with Python 3.8.x but aims to be
compatible with 3.9 once it's dependendencies support 3.9
and all future CPython versions moving forward.

pip/poetry w/ venv

```bash
#!/bin/env bash
python3.8 -m venv .venv && source .venv/bin/activate

poetry add upoly
# or
pip install upoly
```

## Usage

Reccomend to create a copy of `./env.sample` as `./env`. Make sure `.env` is listed
in `.gitignore`.

```env
# ./.env
POLYGON_KEY_ID=REPACEWITHPOLYGONORALPACAKEYHERE
```

Many alternatives to `.env` exist. One such alternative is exporting
like so:

```bash
#!/bin/env bash
export POLYGON_KEY_ID=REPACEWITHPOLYGONORALPACAKEYHERE
```

or adding to your shell startup script, either `.zshrc` or `.bashrc` to have
it be globally available to all projects.

```bash
#/home/youruseraccount/.bashrc
...
export POLYGON_KEY_ID=REPACEWITHPOLYGONORALPACAKEYHERE
...
```

```python
# ./yourscript.py
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

df = async_polygon_aggs("AAPL", start, end)
```

## TODO

- [ ] unit tests
- [ ] regression tests
- [ ] integration tests
- [ ] `/trade` endpoint functionality for tick data
