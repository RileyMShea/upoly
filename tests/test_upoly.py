# mypy: allow-untyped-decorators
import asyncio
from datetime import datetime
from typing import Any, Dict, Iterator

import httpx
import orjson
import pandas as pd
import pytest
import respx
from httpx import Response
from respx.patterns import M
from respx.router import MockRouter

from upoly import NY, __version__, async_polygon_aggs
from upoly.constants import MINUTE_PATH, POLYGON_BASE_URL
from upoly.models import PolyAggResponse
from upoly.polygon_plus import _dispatch_consume_polygon, _produce_polygon_aggs


@pytest.fixture
def polymock(first_file: PolyAggResponse) -> Iterator[MockRouter]:
    """

    Args:
        first_file (PolyAggResponse): [description]

    Raises:
        TypeError: [description]
        TypeError: [description]

    Yields:
        Iterator[MockRouter]: [description]
    """

    polygon_minute_mock = respx.mock(assert_all_called=False)

    if isinstance(polygon_minute_mock, MockRouter):
        pass
    else:
        raise TypeError

    poly_min_pattern = M(
        scheme="https",
        host="api.polygon.io",
        method="GET",
        path__startswith=MINUTE_PATH,
    )
    polygon_minute_mock.route(poly_min_pattern, name="polymin").mock(
        return_value=Response(200, json=first_file),
    )
    if isinstance(polygon_minute_mock, MockRouter):
        pass
    else:
        raise TypeError
    yield polygon_minute_mock


@pytest.fixture
def produce_interface() -> Iterator[Dict[str, Any]]:
    yield dict(
        polygon_id="asdfasdf",
        client=httpx.AsyncClient(),
        queue=asyncio.Queue(),
        symbol="AAPL",
        timespan="minute",
        interval=1,
        _from=datetime(2020, 1, 1),
        to=datetime(2020, 1, 15),
        debug_mode=True,
    )


@pytest.mark.asyncio
async def test_file_read() -> None:
    ...


@pytest.mark.asyncio
async def test_min_route(polymock: MockRouter, produce_interface: Dict[str, Any]) -> None:
    with polymock:
        x = await _produce_polygon_aggs(**produce_interface)
        assert polymock["polymin"].called == True
        # assert x is not None


def test_version() -> None:
    assert __version__ == "0.1.21"
