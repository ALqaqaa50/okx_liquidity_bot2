import pytest
from main import LiquidityStrategy


def test_calc_trend_score_flat():
    s = LiquidityStrategy(None)
    candles = []
    assert s._calc_trend_score(candles) == 0.0


def test_calc_trend_score_up():
    s = LiquidityStrategy(None)
    # generate 10 candles with rising closes
    candles = []
    price = 100
    for i in range(10):
        candles.append({"close": price + i})
    score = s._calc_trend_score(candles)
    assert score > 0
