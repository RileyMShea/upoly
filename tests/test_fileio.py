from unittest import mock

import aiofiles
import pytest
from aiofiles import threadpool

from upoly.models import PolyAggResponse

# Real file IO can be mocked by patching aiofiles.threadpool.sync_open as
# desired. The return type also needs to be registered with the
# aiofiles.threadpool.wrap dispatcher
aiofiles.threadpool.wrap.register(mock.MagicMock)(  # type: ignore
    lambda *args, **kwargs: threadpool.AsyncBufferedIOBase(*args, **kwargs)  # type: ignore
)


@pytest.mark.asyncio
async def test_single_file(first_file: PolyAggResponse) -> None:
    assert isinstance(first_file, dict)
    assert (
        diff_keys := set(first_file.keys()) ^ set(PolyAggResponse.__annotations__.keys())
    ) == set(), diff_keys


@pytest.mark.asyncio
async def test_stuff() -> None:
    data = "data"
    mock_file = mock.MagicMock()

    with mock.patch("aiofiles.threadpool.sync_open", return_value=mock_file) as mock_open:
        async with aiofiles.open("filename", "w") as f:
            await f.write(data)

        mock_file.write.assert_called_once_with(data)

        async with aiofiles.open("filename", "r") as f:
            file_data = await f.read()
