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


class PolyTrade(TypedDict):
    t: int
    y: int
    q: int
    i: int
    x: int
    s: int
    c: List[int]
    p: float
    z: int


class TradeMeta(TypedDict):
    name: str
    type: str


class PolyTradeMap(TypedDict):
    r: TradeMeta
    f: TradeMeta
    q: TradeMeta
    I: TradeMeta
    e: TradeMeta
    x: TradeMeta
    p: TradeMeta
    z: TradeMeta
    t: TradeMeta
    y: TradeMeta
    c: TradeMeta
    i: TradeMeta
    s: TradeMeta


class PolyTradeResponse(TypedDict):
    results: List[PolyTrade]
    success: bool
    map: PolyTradeMap
    ticker: str
    results_count: int
    db_latency: int
