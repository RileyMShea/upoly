import os
from typing import cast
from unittest.mock import MagicMock

from pytest import MonkeyPatch
from pytest_mock.plugin import MockerFixture


class UnixFS:
    @staticmethod
    def rm(filename: str) -> None:
        os.remove(filename)


def test_unix_fs(mocker: MockerFixture) -> None:
    mocker.patch("os.remove")
    UnixFS.rm("file")
    os.remove = cast(MagicMock, os.remove)
    os.remove.assert_called_once_with("file")


def test_stuff(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("apple", "TestingUser")

    got = os.environ.get("apple")
    want = "TestingUser"

    assert got == want
