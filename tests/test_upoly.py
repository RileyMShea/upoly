import os

import httpx
import orjson
import pandas as pd
import pytest
import respx
from httpx import Response

from upoly import NY, __version__, async_polygon_aggs


def test_version() -> None:
    assert __version__ == "0.1.4"


@respx.mock
def test_example() -> None:
    my_route = respx.get("https://foo.bar/").mock(return_value=Response(204))
    response = httpx.get("https://foo.bar/")
    assert my_route.called
    assert response.status_code == 204
    assert response.text == ""
    assert len(response.headers) == 0


@respx.mock(base_url="https://api.polygon.io/")
def test_example(respx_mock) -> None:

    start = pd.Timestamp("2019-01-01", tz=NY)
    end = pd.Timestamp("2020-01-01", tz=NY)

    os.environ["POLYGON_KEY_ID"] = "apples"

    my_route = respx_mock.get(
        "v2/aggs/ticker/AAPL/range/1/minute/1546318800000/1554202800000?unadjusted=False&sort=asc&limit=50000&apiKey=apples"
    ).mock(return_value=Response(204))

    with pytest.raises(orjson.JSONDecodeError):
        data = async_polygon_aggs("AAPL", "minute", 1, start, end)

    # assert my_route.called
    # assert response.status_code == 204
    # assert response.text == ""
    # assert len(response.headers) == 0
