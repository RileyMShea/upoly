from typing import Optional, TypeVar

T = TypeVar("T")


def unwrap(element: Optional[T]) -> T:
    """Check an optional variable for `None` to get
    the type checker to recognize it.

    Args:
        element (Optional[T]): Any type `T` that is also
        optional

    Raises:
        ValueError: If variable is None

    Returns:
        T: The unwrapped unwrapped optional variable

    Examples:
    >>> # Type checking will complain if passing `Optional[str]`
    >>> # to functions expecting a `str`
    >>> a = os.getenv("MY_PASSWORD"))  # getenv returns `Optional[str]
    >>> a = unwrap(a)  # unwrap verifies a is not None, Type inference removes `Optional`.
    """
    if element is None:
        raise ValueError("Got None, Expected a value")
    else:
        return element
