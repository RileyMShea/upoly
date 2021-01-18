# conftest.py
import pytest
import respx
from httpx import Response


@pytest.fixture
def mocked_api():
    with respx.mock(base_url="https://foo.bar", assert_all_called=False) as respx_mock:
        users_route = respx_mock.get("/users/", name="list_users")
        users_route.return_value = Response(200, json=[])
        ...
        yield respx_mock
