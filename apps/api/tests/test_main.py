import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the path so we can import the main module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from main import (  # noqa: E402
    BacktestRequest,
    apply_fees_and_slippage,
    calculate_bollinger,
    calculate_ema,
    calculate_metrics,
    calculate_rsi,
    calculate_sma,
    generate_signal,
    monte_carlo,
    parameter_sweep,
    rolling_metrics,
    run_backtest,
    _validate_ticker,
)
from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

STRATEGIES = [
    "ema_crossover",
    "rsi_mean_reversion",
    "sma_crossover",
    "bollinger_breakout",
    "momentum",
]


# ---------------------------------------------------------------------------
# Synthetic market data so tests never touch the network.
# ---------------------------------------------------------------------------
def make_bars(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Trending random walk with oscillation so every strategy can trade.
    drift = np.linspace(0, 0.6, n)
    noise = np.cumsum(rng.normal(0, 0.01, n))
    wave = 0.1 * np.sin(np.linspace(0, 12 * np.pi, n))
    close = 100 * np.exp(drift + noise + wave)
    return pd.DataFrame({"Close": close}, index=dates)


@pytest.fixture
def patched_bars(monkeypatch):
    bars = make_bars()
    monkeypatch.setattr(main, "get_bars", lambda ticker, start, end: bars.copy())
    return bars


@pytest.fixture
def client(patched_bars):
    return TestClient(main.app)


def base_request(**overrides):
    params = dict(ticker="SPY", start="2020-01-01", end="2021-06-01")
    params.update(overrides)
    return BacktestRequest(**params)


# ---------------------------------------------------------------------------
# Original helper tests (kept).
# ---------------------------------------------------------------------------
def test_apply_fees_and_slippage():
    price = 100.0
    long_price = apply_fees_and_slippage(price, 1.0, 2.0, "long")
    assert long_price > price
    assert abs(long_price - 100.03) < 0.001

    flat_price = apply_fees_and_slippage(price, 1.0, 2.0, "flat")
    assert flat_price < price
    assert abs(flat_price - 99.97) < 0.001


def test_apply_fees_and_slippage_validation():
    with pytest.raises(ValueError):
        apply_fees_and_slippage(-1, 1, 1, "long")
    with pytest.raises(ValueError):
        apply_fees_and_slippage(100, -1, 1, "long")
    with pytest.raises(ValueError):
        apply_fees_and_slippage(100, 1, 1, "sideways")


def test_calculate_metrics():
    equity_curve = pd.Series([10000, 11000, 10500, 12000, 11500])
    trades = [
        {"return": 0.1},
        {"return": -0.05},
        {"return": 0.15},
        {"return": -0.04},
    ]
    metrics = calculate_metrics(equity_curve, trades)
    assert metrics.cagr > 0
    assert metrics.total_trades == 4
    assert metrics.win_rate == 0.5
    assert metrics.max_drawdown < 0


def test_calculate_metrics_validation():
    with pytest.raises(ValueError):
        calculate_metrics(pd.Series([10000]), [])
    with pytest.raises(ValueError):
        calculate_metrics(pd.Series([0, 100]), [])


def test_sharpe_calculation():
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    equity_curve = pd.Series(10000 * np.cumprod(1 + returns))
    trades = [{"return": r} for r in returns[1:]]
    metrics = calculate_metrics(equity_curve, trades)
    assert -2 < metrics.sharpe < 2
    assert metrics.volatility > 0


# ---------------------------------------------------------------------------
# Indicators.
# ---------------------------------------------------------------------------
def test_indicators_basic():
    prices = pd.Series(np.linspace(100, 200, 60))
    assert len(calculate_ema(prices, 10)) == len(prices)
    assert len(calculate_sma(prices, 10)) == len(prices)

    rsi = calculate_rsi(prices, 14).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()

    upper, middle, lower = calculate_bollinger(prices, 20, 2.0)
    valid = upper.dropna().index
    assert (upper.loc[valid] >= middle.loc[valid]).all()
    assert (lower.loc[valid] <= middle.loc[valid]).all()


def test_indicator_validation():
    prices = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        calculate_ema(prices, 0)
    with pytest.raises(ValueError):
        calculate_sma(prices, 0)
    with pytest.raises(ValueError):
        calculate_rsi(prices, 0)
    with pytest.raises(ValueError):
        calculate_bollinger(prices, 0)


def test_validate_ticker():
    assert _validate_ticker("spy") == "SPY"
    assert _validate_ticker("BRK.B") == "BRK.B"
    with pytest.raises(ValueError):
        _validate_ticker("AAPL; DROP TABLE")
    with pytest.raises(ValueError):
        _validate_ticker("")


# ---------------------------------------------------------------------------
# Signal generation.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_generate_signal_is_binary(strategy):
    bars = make_bars()
    params = base_request(strategy=strategy)
    sig = generate_signal(bars, params)
    assert len(sig) == len(bars)
    assert set(np.unique(sig.to_numpy())).issubset({0, 1})


def test_generate_signal_unknown_strategy():
    bars = make_bars()
    params = base_request()
    object.__setattr__(params, "strategy", "does_not_exist")
    with pytest.raises(ValueError):
        generate_signal(bars, params)


# ---------------------------------------------------------------------------
# run_backtest across all strategies.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", STRATEGIES)
def test_run_backtest_all_strategies(patched_bars, strategy):
    result = run_backtest(base_request(strategy=strategy))
    assert len(result.equity_curve) > 0
    assert len(result.benchmark_curve) > 0
    assert result.metrics.total_trades >= 0
    assert result.metrics.volatility >= 0
    assert result.params_used.strategy == strategy


def test_run_backtest_without_vol_target(patched_bars):
    result = run_backtest(base_request(vol_target=False))
    assert len(result.equity_curve) > 0


def test_run_backtest_bad_date_format(patched_bars):
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request(start="01-01-2020"))
    assert exc.value.status_code == 400


def test_run_backtest_start_after_end(patched_bars):
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request(start="2021-01-01", end="2020-01-01"))
    assert exc.value.status_code == 400


def test_run_backtest_range_too_long(patched_bars):
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request(start="2000-01-01", end="2020-01-01"))
    assert exc.value.status_code == 400


def test_run_backtest_data_service_error(monkeypatch):
    def boom(ticker, start, end):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(main, "get_bars", boom)
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request())
    assert exc.value.status_code == 503


def test_run_backtest_flattens_multiindex_columns(monkeypatch):
    """Recent yfinance returns MultiIndex columns (price, ticker); the engine
    must flatten them so 'Close' is reachable."""
    base = make_bars()
    multi = base.copy()
    multi.columns = pd.MultiIndex.from_product([["Close"], ["SPY"]])
    monkeypatch.setattr(main, "get_bars", lambda t, s, e: multi.copy())

    result = run_backtest(base_request())
    assert len(result.equity_curve) > 0


def test_run_backtest_empty_data(monkeypatch):
    monkeypatch.setattr(main, "get_bars", lambda t, s, e: pd.DataFrame())
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request())
    assert exc.value.status_code in (400, 503)


def test_run_backtest_missing_close_column(monkeypatch):
    frame = make_bars().rename(columns={"Close": "Adj Close"})
    monkeypatch.setattr(main, "get_bars", lambda t, s, e: frame.copy())
    with pytest.raises(HTTPException) as exc:
        run_backtest(base_request())
    assert exc.value.status_code in (400, 503)


# ---------------------------------------------------------------------------
# Analytics functions.
# ---------------------------------------------------------------------------
def test_monte_carlo():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))
    result = monte_carlo(returns, n_sims=200)
    assert result["n_sims"] == 200
    p = result["percentiles"]
    assert p["p5"] <= p["p50"] <= p["p95"]
    assert 0.0 <= result["prob_loss"] <= 1.0


def test_monte_carlo_insufficient_data():
    with pytest.raises(ValueError):
        monte_carlo(pd.Series([0.01]))


def test_rolling_metrics(patched_bars):
    df, _ = main._simulate(base_request())
    result = rolling_metrics(df, window=21)
    assert len(result["dates"]) == len(result["rolling_sharpe"])
    assert result["window"] >= 2


def test_parameter_sweep(patched_bars):
    result = parameter_sweep(base_request())
    assert len(result["sharpe"]) == 4
    assert len(result["sharpe"][0]) == 4
    # Crossover keeps fast < slow, so the [0][0] cell (fast==slow band) is None.
    assert result["x_param"] == "ema_fast"
    assert result["y_param"] == "ema_slow"
    assert "fast_grid" in result and "slow_grid" in result  # legacy aliases


@pytest.mark.parametrize(
    "strategy,x_param,y_param",
    [
        ("sma_crossover", "sma_fast", "sma_slow"),
        ("rsi_mean_reversion", "rsi_period", "oversold"),
        ("bollinger_breakout", "bb_window", "bb_std"),
        ("momentum", "lookback", "threshold"),
    ],
)
def test_parameter_sweep_is_strategy_aware(patched_bars, strategy, x_param, y_param):
    result = parameter_sweep(base_request(strategy=strategy))
    assert result["strategy"] == strategy
    assert result["x_param"] == x_param
    assert result["y_param"] == y_param
    assert len(result["sharpe"]) == 4 and len(result["sharpe"][0]) == 4


# ---------------------------------------------------------------------------
# HTTP endpoints.
# ---------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_list_strategies(client):
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    assert set(resp.json()["strategies"].keys()) == set(STRATEGIES)


def test_backtest_endpoints(client):
    payload = base_request().model_dump()
    for route in ("/backtest", "/api/backtest"):
        resp = client.post(route, json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "metrics" in body
        assert "equity_curve" in body
        assert "benchmark_curve" in body


def test_backtest_validation_error(client):
    payload = base_request().model_dump()
    payload["ema_fast"] = 1  # below the minimum of 2
    resp = client.post("/backtest", json=payload)
    assert resp.status_code == 422


def test_analytics_endpoints(client):
    payload = base_request().model_dump()
    for route in ("/api/monte-carlo", "/api/rolling", "/api/benchmarks", "/api/parameter-sweep"):
        resp = client.post(route, json=payload)
        assert resp.status_code == 200, f"{route}: {resp.text}"


def test_export_csv(client):
    payload = base_request().model_dump()
    resp = client.post("/api/export.csv", json=payload)
    assert resp.status_code == 200
    assert "date,equity,benchmark" in resp.text


# ---------------------------------------------------------------------------
# Market data cache + graceful fallback.
# ---------------------------------------------------------------------------
def test_get_bars_uses_fresh_cache(monkeypatch, tmp_path):
    bars = make_bars()
    monkeypatch.setattr(main, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CACHE_TTL", 3600)
    main._write_cache(main._cache_path("SPY", "2020-01-01", "2021-01-01"), bars)

    def fail(*args, **kwargs):
        raise AssertionError("network must not be hit on a fresh cache hit")

    monkeypatch.setattr(main.yf, "download", fail)
    out = main.get_bars("SPY", "2020-01-01", "2021-01-01")
    assert len(out) == len(bars)


def test_get_bars_writes_cache_on_fetch(monkeypatch, tmp_path):
    bars = make_bars()
    monkeypatch.setattr(main, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CACHE_TTL", 3600)
    monkeypatch.setattr(main.yf, "download", lambda *a, **k: bars.copy())
    main.get_bars("AAPL", "2020-01-01", "2021-01-01")
    assert os.path.exists(main._cache_path("AAPL", "2020-01-01", "2021-01-01"))


def test_get_bars_serves_stale_on_failure(monkeypatch, tmp_path):
    bars = make_bars()
    monkeypatch.setattr(main, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CACHE_TTL", 0)  # disable freshness so live path runs
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    main._write_cache(main._cache_path("SPY", "2020-01-01", "2021-01-01"), bars)

    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(main.yf, "download", boom)
    out = main.get_bars("SPY", "2020-01-01", "2021-01-01")
    assert len(out) == len(bars)


def test_get_bars_raises_without_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "CACHE_TTL", 0)
    monkeypatch.setattr(main.time, "sleep", lambda *_: None)
    monkeypatch.setattr(main.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError):
        main.get_bars("ZZZZ", "2020-01-01", "2021-01-01")


# ---------------------------------------------------------------------------
# Optional API-key auth.
# ---------------------------------------------------------------------------
def test_api_key_blocks_without_header(monkeypatch, patched_bars):
    monkeypatch.setattr(main, "API_KEY", "secret")
    c = TestClient(main.app)
    resp = c.post("/backtest", json=base_request().model_dump())
    assert resp.status_code == 401


def test_api_key_allows_with_header(monkeypatch, patched_bars):
    monkeypatch.setattr(main, "API_KEY", "secret")
    c = TestClient(main.app)
    resp = c.post(
        "/backtest",
        json=base_request().model_dump(),
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 200, resp.text


def test_api_key_health_is_public(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "secret")
    c = TestClient(main.app)
    assert c.get("/health").status_code == 200
