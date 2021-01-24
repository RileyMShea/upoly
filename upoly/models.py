from typing import List, TypedDict


class PolyApiBar(TypedDict):
    v: int
    vw: float
    o: float
    c: float
    h: float
    l: float
    t: int
    n: int


class PolyAggResponse(TypedDict):
    ticker: str
    status: str
    count: int
    adjusted: bool
    queryCount: int
    resultsCount: int
    request_id: str
    results: List[PolyApiBar]
