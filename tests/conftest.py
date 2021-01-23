# conftest.py
import orjson
import pytest
from httpx import Response


@pytest.fixture
def first_file():
    with open("./tests/fixtures/SHOP-1559361600000-1565526000000.json", "rb") as f:
        data = f.read()
        if isinstance(data, bytes):
            yield orjson.loads(data)
        else:
            raise TypeError
