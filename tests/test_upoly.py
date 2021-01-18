import httpx
import respx
from httpx import Response

from upoly import __version__


def test_version():
    assert __version__ == "0.1.4"


@respx.mock
def test_example():
    my_route = respx.get("https://foo.bar/").mock(return_value=Response(204))
    response = httpx.get("https://foo.bar/")
    assert my_route.called
    assert response.status_code == 204
