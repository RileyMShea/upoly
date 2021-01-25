# conftest.py
from typing import Iterator

import brotli
import orjson
import pytest
from httpx import Response

from upoly.models import PolyAggResponse


@pytest.fixture
def first_file() -> Iterator[PolyAggResponse]:
    with open("./tests/fixtures/SHOP-1559361600000-1565526000000.json.zip", "rb") as f:
        yield orjson.loads(brotli.decompress(f.read()))
