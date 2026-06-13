"""EMA + Sharpe Dashboard API.

A multi-strategy quantitative backtesting service. Supports five strategies
(EMA crossover, RSI mean reversion, SMA crossover, Bollinger breakout, and
momentum) plus analytics endpoints (Monte Carlo, rolling metrics, benchmark
comparison, parameter sweep, and CSV export).

Network access (yfinance) is isolated in ``get_bars`` so the engine and
endpoints can be unit-tested deterministically by patching that one function.
"""

from __future__ import annotations

import functools
import hashlib
import io
import csv
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

app = FastAPI(title="EMA Sharpe Dashboard API", version="2.0.0")

# ---------------------------------------------------------------------------
# Configuration (environment driven so production can lock things down).
# ---------------------------------------------------------------------------
# CORS: comma separated origins, or "*" to allow any (local default).
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
if _origins_env == "*":
    ALLOWED_ORIGINS: List[str] = ["*"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

# Optional shared secret. When set, every endpoint except the public paths
# requires a matching X-API-Key header.
API_KEY = os.environ.get("API_KEY", "").strip()
PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}

# Market data cache (keeps the app responsive and provides a stale fallback
# when the upstream data provider is unavailable).
CACHE_DIR = os.environ.get(
    "DATA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "qt_data_cache")
)
CACHE_TTL = int(os.environ.get("DATA_CACHE_TTL", "3600"))  # seconds; 0 disables

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Enforce an API key when one is configured; otherwise pass through."""
    if (
        API_KEY
        and request.method != "OPTIONS"
        and request.url.path not in PUBLIC_PATHS
    ):
        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing API key"}
            )
    return await call_next(request)

# ---------------------------------------------------------------------------
# Rate limiting (naive in-memory, per-IP, fixed window).
# Buckets for stale windows are evicted so memory stays bounded.
# ---------------------------------------------------------------------------
WINDOW, LIMIT = 60, 60  # 60 requests per minute per IP
buckets: Dict[tuple, int] = {}


@app.middleware("http")
async def throttle(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = int(time.time())
    current_window = now // WINDOW
    key = (ip, current_window)

    # Evict stale windows to keep memory bounded.
    stale = [k for k in buckets if k[1] < current_window]
    for k in stale:
        del buckets[k]

    buckets[key] = buckets.get(key, 0) + 1
    if buckets[key] > LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")
    return await call_next(request)


STRATEGIES = {
    "ema_crossover": {
        "name": "EMA Crossover",
        "description": "Go long when the fast EMA is above the slow EMA.",
        "parameters": ["ema_fast", "ema_slow"],
    },
    "rsi_mean_reversion": {
        "name": "RSI Mean Reversion",
        "description": "Buy oversold (RSI < oversold), exit overbought (RSI > overbought).",
        "parameters": ["rsi_period", "oversold", "overbought"],
    },
    "sma_crossover": {
        "name": "SMA Crossover",
        "description": "Go long when the fast SMA is above the slow SMA.",
        "parameters": ["sma_fast", "sma_slow"],
    },
    "bollinger_breakout": {
        "name": "Bollinger Breakout",
        "description": "Go long on a breakout above the upper band, exit below the lower band.",
        "parameters": ["bb_window", "bb_std"],
    },
    "momentum": {
        "name": "Momentum",
        "description": "Go long when trailing momentum exceeds the threshold.",
        "parameters": ["lookback", "threshold"],
    },
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class BacktestRequest(BaseModel):
    ticker: str = Field("SPY", min_length=1, max_length=10)
    start: str
    end: str
    strategy: str = Field(
        "ema_crossover",
        pattern="^(ema_crossover|rsi_mean_reversion|sma_crossover|bollinger_breakout|momentum)$",
    )

    # EMA parameters
    ema_fast: int = Field(20, ge=2, le=250)
    ema_slow: int = Field(100, ge=3, le=500)

    # RSI parameters
    rsi_period: int = Field(14, ge=2, le=50)
    oversold: float = Field(30, ge=5, le=45)
    overbought: float = Field(70, ge=55, le=95)

    # SMA parameters
    sma_fast: int = Field(20, ge=2, le=250)
    sma_slow: int = Field(50, ge=3, le=500)

    # Bollinger Bands parameters
    bb_window: int = Field(20, ge=5, le=100)
    bb_std: float = Field(2.0, ge=1.0, le=3.0)

    # Momentum parameters
    lookback: int = Field(20, ge=2, le=100)
    threshold: float = Field(0.02, ge=0.0, le=0.5)

    # Trading costs
    fees_bps: float = Field(1, ge=0, le=100)
    slip_bps: float = Field(2, ge=0, le=100)

    # Volatility targeting
    vol_target: bool = True
    target_vol: float = Field(0.15, gt=0, le=1)


class BacktestMetrics(BaseModel):
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float


class EquityPoint(BaseModel):
    date: str
    equity: float


class Trade(BaseModel):
    date: str
    side: str
    price: float
    quantity: float


class BacktestResponse(BaseModel):
    metrics: BacktestMetrics
    equity_curve: List[EquityPoint]
    benchmark_curve: List[EquityPoint]
    trades: List[Trade]
    params_used: BacktestRequest


# ---------------------------------------------------------------------------
# Data fetching (the only place that touches the network)
# ---------------------------------------------------------------------------
def _validate_ticker(ticker: str) -> str:
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    if not all(c.isalnum() or c == "." for c in ticker.upper()):
        raise ValueError("Invalid ticker format. Use alphanumeric characters and dots only.")
    return ticker.upper().strip()


@functools.lru_cache(maxsize=256)
def _cache_path(ticker: str, start: str, end: str) -> str:
    """Return a stable on-disk path for a (ticker, start, end) query."""
    digest = hashlib.sha256(f"{ticker}|{start}|{end}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.pkl")


def _read_cache(path: str, max_age: Optional[int]) -> Optional[pd.DataFrame]:
    """Read a cached frame if present and (optionally) younger than max_age."""
    if not os.path.exists(path):
        return None
    if max_age is not None and (time.time() - os.path.getmtime(path)) > max_age:
        return None
    try:
        return pd.read_pickle(path)
    except Exception as e:  # noqa: BLE001 - a corrupt cache should never be fatal
        print(f"Cache read failed for {path}: {e}")
        return None


def _write_cache(path: str, df: pd.DataFrame) -> None:
    """Persist a frame to the cache, ignoring any write errors."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(path)
    except Exception as e:  # noqa: BLE001 - caching is best effort
        print(f"Cache write failed for {path}: {e}")


def get_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV market data, with an on-disk cache and stale fallback.

    A fresh cache entry (younger than ``CACHE_TTL``) is served immediately. On a
    cache miss the data provider is queried with retries and the result cached.
    If every live attempt fails but a stale cache entry exists, that stale copy
    is returned so the service degrades gracefully instead of erroring.

    Raises:
        ValueError: If the ticker is invalid.
        RuntimeError: If no live data and no cached data are available.
    """
    ticker = _validate_ticker(ticker)
    path = _cache_path(ticker, start, end)

    if CACHE_TTL > 0:
        fresh = _read_cache(path, CACHE_TTL)
        if fresh is not None:
            return fresh

    last_exc: Optional[Exception] = None
    for i in range(3):
        try:
            df = yf.download(
                ticker, start=start, end=end, auto_adjust=True, progress=False
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                _write_cache(path, df)
                return df
        except Exception as e:  # noqa: BLE001 - yfinance raises a variety of errors
            last_exc = e
            print(f"Attempt {i + 1} failed: {e}")
        time.sleep(1 + i)

    stale = _read_cache(path, None)
    if stale is not None:
        print(f"Serving stale cached data for {ticker} after live fetch failed")
        return stale

    raise RuntimeError(
        f"Data unavailable for {ticker}. Try a broader date range or a different ticker."
    ) from last_exc


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    if period < 1:
        raise ValueError("EMA period must be at least 1")
    return prices.ewm(span=period).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period < 1:
        raise ValueError("SMA period must be at least 1")
    return prices.rolling(window=period).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder-style simple rolling mean)."""
    if period < 1:
        raise ValueError("RSI period must be at least 1")
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bollinger(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (upper, middle, lower) Bollinger bands."""
    if window < 1:
        raise ValueError("Bollinger window must be at least 1")
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return middle + num_std * std, middle, middle - num_std * std


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualized rolling volatility (252 trading days)."""
    if window < 1:
        raise ValueError("Volatility window must be at least 1")
    return returns.rolling(window=window).std() * np.sqrt(252)


def apply_fees_and_slippage(price: float, fees_bps: float, slip_bps: float, side: str) -> float:
    """Apply realistic trading costs (fees and slippage).

    A long entry pays up; a flat (exit) receives less. Costs are in basis
    points (1 bp = 0.01%).
    """
    if price <= 0:
        raise ValueError("Price must be positive")
    if fees_bps < 0 or slip_bps < 0:
        raise ValueError("Fees and slippage must be non-negative")

    fee_cost = price * (fees_bps / 10000)
    slip_cost = price * (slip_bps / 10000)

    if side == "long":
        return price + fee_cost + slip_cost
    if side == "flat":
        return price - fee_cost - slip_cost
    raise ValueError(f"Invalid side: {side}. Must be 'long' or 'flat'")


# ---------------------------------------------------------------------------
# Signal generation (long/flat, value in {0, 1})
# ---------------------------------------------------------------------------
def generate_signal(df: pd.DataFrame, params: BacktestRequest) -> pd.Series:
    """Generate a long/flat signal series (0 or 1) for the chosen strategy."""
    close = df["Close"]
    strategy = params.strategy

    if strategy == "ema_crossover":
        fast = calculate_ema(close, params.ema_fast)
        slow = calculate_ema(close, params.ema_slow)
        valid = fast.notna() & slow.notna()
        return (valid & (fast > slow)).astype(int)

    if strategy == "sma_crossover":
        fast = calculate_sma(close, params.sma_fast)
        slow = calculate_sma(close, params.sma_slow)
        valid = fast.notna() & slow.notna()
        return (valid & (fast > slow)).astype(int)

    if strategy == "momentum":
        mom = close.pct_change(params.lookback)
        return (mom.notna() & (mom > params.threshold)).astype(int)

    if strategy == "rsi_mean_reversion":
        rsi = calculate_rsi(close, params.rsi_period).to_numpy()
        pos = np.zeros(len(df), dtype=int)
        state = 0
        for i in range(len(rsi)):
            if not np.isnan(rsi[i]):
                if state == 0 and rsi[i] < params.oversold:
                    state = 1
                elif state == 1 and rsi[i] > params.overbought:
                    state = 0
            pos[i] = state
        return pd.Series(pos, index=df.index)

    if strategy == "bollinger_breakout":
        upper, _, lower = calculate_bollinger(close, params.bb_window, params.bb_std)
        c = close.to_numpy()
        u = upper.to_numpy()
        low = lower.to_numpy()
        pos = np.zeros(len(df), dtype=int)
        state = 0
        for i in range(len(c)):
            if not np.isnan(u[i]) and not np.isnan(low[i]):
                if state == 0 and c[i] > u[i]:
                    state = 1
                elif state == 1 and c[i] < low[i]:
                    state = 0
            pos[i] = state
        return pd.Series(pos, index=df.index)

    raise ValueError(f"Unknown strategy: {strategy}")


def calculate_metrics(equity_curve: pd.Series, trades: List[Dict]) -> BacktestMetrics:
    """Calculate performance metrics from an equity curve and trade list."""
    if equity_curve.empty or len(equity_curve) < 2:
        raise ValueError("Equity curve must have at least 2 data points")
    if equity_curve.iloc[0] <= 0:
        raise ValueError("Initial equity must be positive")

    returns = equity_curve.pct_change().dropna()

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = len(equity_curve) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    trade_returns = [t.get("return", 0) for t in trades if isinstance(t, dict)]
    win_rate = (
        len([r for r in trade_returns if r > 0]) / len(trade_returns)
        if trade_returns
        else 0
    )
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.0

    return BacktestMetrics(
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=float(win_rate),
        total_trades=len(trades),
        avg_trade_return=avg_trade_return,
        volatility=float(volatility),
    )


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def _prepare_frame(params: BacktestRequest) -> pd.DataFrame:
    """Validate inputs, fetch data, and return a clean single-index frame."""
    try:
        start_date = datetime.strptime(params.start, "%Y-%m-%d")
        end_date = datetime.strptime(params.end, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError("Invalid date format. Use YYYY-MM-DD") from e
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
    if (end_date - start_date).days > 3650:  # ~10 years max
        raise ValueError("Date range cannot exceed 10 years")

    df = get_bars(params.ticker, params.start, params.end)
    if df.empty:
        raise ValueError("No data available for the given parameters")
    # Flatten yfinance MultiIndex columns if present.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError("Data missing 'Close' price column")
    return df


def _simulate(params: BacktestRequest):
    """Run the position-level backtest.

    Returns a tuple of ``(df, trades)`` where ``df`` is indexed by date and
    contains ``Close`` and ``equity`` columns, and ``trades`` is a list of
    dicts with a ``return`` key (used by :func:`calculate_metrics`).
    """
    df = _prepare_frame(params).copy()

    df["signal"] = generate_signal(df, params)
    df["returns"] = df["Close"].pct_change()

    if params.vol_target:
        df["vol"] = calculate_volatility(df["returns"])
        df["vol_scalar"] = (params.target_vol / df["vol"]).replace([np.inf, -np.inf], np.nan)
        df["vol_scalar"] = df["vol_scalar"].fillna(1).clip(0, 2)
    else:
        df["vol_scalar"] = 1.0

    df["cash"] = 10000.0
    df["equity"] = 10000.0
    df["shares"] = 0.0
    df["entry_price"] = 0.0

    trades: List[Dict] = []
    cash_loc = df.columns.get_loc("cash")
    shares_loc = df.columns.get_loc("shares")
    entry_loc = df.columns.get_loc("entry_price")
    equity_loc = df.columns.get_loc("equity")

    for i in range(1, len(df)):
        prev_signal = int(df.iloc[i - 1]["signal"])
        curr_signal = int(df.iloc[i]["signal"])
        price = float(df.iloc[i]["Close"])
        vol_scalar = float(df.iloc[i]["vol_scalar"])
        target_position = curr_signal * vol_scalar

        if prev_signal != curr_signal:
            # Close any open position.
            prev_shares = float(df.iloc[i - 1]["shares"])
            if prev_shares != 0:
                exit_price = apply_fees_and_slippage(price, params.fees_bps, params.slip_bps, "flat")
                df.iloc[i, cash_loc] = float(df.iloc[i - 1]["cash"]) + prev_shares * exit_price
                entry_price = float(df.iloc[i - 1]["entry_price"]) or price
                trade_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                trades.append(
                    {
                        "date": df.index[i].strftime("%Y-%m-%d"),
                        "side": "flat",
                        "price": exit_price,
                        "quantity": prev_shares,
                        "return": trade_return,
                    }
                )
            else:
                df.iloc[i, cash_loc] = float(df.iloc[i - 1]["cash"])

            # Open a new position.
            if target_position > 0:
                entry_price = apply_fees_and_slippage(price, params.fees_bps, params.slip_bps, "long")
                available = float(df.iloc[i, cash_loc])
                shares_to_buy = (available * min(target_position, 1.0)) / entry_price
                df.iloc[i, cash_loc] = available - shares_to_buy * entry_price
                df.iloc[i, shares_loc] = shares_to_buy
                df.iloc[i, entry_loc] = entry_price
                trades.append(
                    {
                        "date": df.index[i].strftime("%Y-%m-%d"),
                        "side": "long",
                        "price": entry_price,
                        "quantity": shares_to_buy,
                    }
                )
            else:
                df.iloc[i, shares_loc] = 0.0
                df.iloc[i, entry_loc] = 0.0
        else:
            df.iloc[i, cash_loc] = float(df.iloc[i - 1]["cash"])
            df.iloc[i, shares_loc] = float(df.iloc[i - 1]["shares"])
            df.iloc[i, entry_loc] = float(df.iloc[i - 1]["entry_price"])

        df.iloc[i, equity_loc] = float(df.iloc[i, cash_loc]) + float(df.iloc[i, shares_loc]) * price

    return df, trades


def _equity_points(index, values) -> List[EquityPoint]:
    points: List[EquityPoint] = []
    for date, value in zip(index, values):
        v = float(value)
        if np.isfinite(v):
            points.append(EquityPoint(date=date.strftime("%Y-%m-%d"), equity=v))
    return points


def run_backtest(params: BacktestRequest) -> BacktestResponse:
    """Run a backtest and assemble the response (metrics, curves, trades)."""
    try:
        df, trades = _simulate(params)

        equity_curve = _equity_points(df.index, df["equity"])
        if not equity_curve:
            raise ValueError("Failed to generate equity curve. Check data quality.")

        # Buy-and-hold benchmark, normalized to the same starting capital.
        first_close = float(df["Close"].iloc[0])
        bh_equity = 10000.0 * (df["Close"] / first_close)
        benchmark_curve = _equity_points(df.index, bh_equity)

        formatted_trades = [
            Trade(date=t["date"], side=t["side"], price=float(t["price"]), quantity=float(t["quantity"]))
            for t in trades
        ]

        metrics = calculate_metrics(df["equity"], trades)

        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            trades=formatted_trades,
            params_used=params,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Data service error: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def monte_carlo(daily_returns: pd.Series, n_sims: int = 500, horizon: Optional[int] = None) -> Dict:
    """Bootstrap daily returns to estimate the distribution of terminal equity."""
    rets = daily_returns.dropna().to_numpy()
    if len(rets) < 2:
        raise ValueError("Not enough return data for Monte Carlo simulation")
    horizon = horizon or len(rets)
    rng = np.random.default_rng(42)
    finals = np.empty(n_sims)
    for s in range(n_sims):
        sample = rng.choice(rets, size=horizon, replace=True)
        finals[s] = float(np.prod(1.0 + sample))
    return {
        "n_sims": n_sims,
        "horizon": horizon,
        "percentiles": {
            "p5": float(np.percentile(finals, 5)),
            "p25": float(np.percentile(finals, 25)),
            "p50": float(np.percentile(finals, 50)),
            "p75": float(np.percentile(finals, 75)),
            "p95": float(np.percentile(finals, 95)),
        },
        "mean": float(np.mean(finals)),
        "std": float(np.std(finals)),
        "prob_loss": float(np.mean(finals < 1.0)),
    }


def rolling_metrics(df: pd.DataFrame, window: int = 63) -> Dict:
    """Compute rolling Sharpe, return, and drawdown over the equity curve."""
    returns = df["equity"].pct_change()
    if len(returns.dropna()) < window:
        window = max(2, len(returns.dropna()) // 2)

    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    rolling_sharpe = (roll_mean / roll_std * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
    rolling_return = df["equity"].pct_change(window)

    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    def clean(series: pd.Series) -> List[Optional[float]]:
        return [None if pd.isna(v) else float(v) for v in series]

    return {
        "window": window,
        "dates": dates,
        "rolling_sharpe": clean(rolling_sharpe),
        "rolling_return": clean(rolling_return),
    }


# Two-axis sweep grids per strategy. Each entry is (param_name, [values]).
SWEEP_AXES: Dict[str, tuple] = {
    "ema_crossover": (("ema_fast", [10, 20, 30, 50]), ("ema_slow", [50, 100, 150, 200])),
    "sma_crossover": (("sma_fast", [10, 20, 30, 50]), ("sma_slow", [50, 100, 150, 200])),
    "rsi_mean_reversion": (("rsi_period", [7, 14, 21, 28]), ("oversold", [20, 25, 30, 35])),
    "bollinger_breakout": (("bb_window", [10, 20, 30, 40]), ("bb_std", [1.5, 2.0, 2.5, 3.0])),
    "momentum": (("lookback", [20, 40, 60, 90]), ("threshold", [0.01, 0.02, 0.05, 0.1])),
}


def _sweep_sharpe(df: pd.DataFrame, params: BacktestRequest) -> float:
    """Annualized Sharpe for one parameter set, reusing the live signal logic."""
    base_returns = df["Close"].pct_change().fillna(0.0)
    cost = (params.fees_bps + params.slip_bps) / 10000.0
    sig = generate_signal(df, params).astype(float)
    pos = sig.shift().fillna(0.0)
    trade = (pos - pos.shift().fillna(0.0)).abs()
    strat_ret = pos * base_returns - trade * cost
    std = strat_ret.std()
    return float(strat_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0


def parameter_sweep(params: BacktestRequest) -> Dict:
    """Sweep the two primary parameters of the selected strategy.

    Returns a Sharpe heatmap over the strategy's two-axis grid. ``x``/``y`` keys
    describe the axes generically; ``fast_grid``/``slow_grid`` are kept as
    aliases for backward compatibility.
    """
    (x_param, x_grid), (y_param, y_grid) = SWEEP_AXES.get(
        params.strategy, SWEEP_AXES["ema_crossover"]
    )
    df = _prepare_frame(params)
    is_crossover = params.strategy in ("ema_crossover", "sma_crossover")

    heatmap: List[List[Optional[float]]] = []
    for xv in x_grid:
        row: List[Optional[float]] = []
        for yv in y_grid:
            # For crossover strategies, fast must stay below slow.
            if is_crossover and xv >= yv:
                row.append(None)
                continue
            candidate = params.model_copy(update={x_param: xv, y_param: yv})
            row.append(_sweep_sharpe(df, candidate))
        heatmap.append(row)

    return {
        "strategy": params.strategy,
        "x_param": x_param,
        "y_param": y_param,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "sharpe": heatmap,
        # Legacy aliases.
        "fast_grid": x_grid,
        "slow_grid": y_grid,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/strategies")
async def list_strategies():
    """List the available trading strategies and their parameters."""
    return {"strategies": STRATEGIES}


@app.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    """Run a backtest with the specified strategy and parameters."""
    return run_backtest(request)


# Alias kept for compatibility with the documented `/api/backtest` route.
@app.post("/api/backtest", response_model=BacktestResponse)
async def backtest_api(request: BacktestRequest):
    return run_backtest(request)


@app.post("/api/monte-carlo")
async def monte_carlo_endpoint(request: BacktestRequest):
    """Bootstrap Monte Carlo simulation of terminal equity."""
    try:
        df, _ = _simulate(request)
        return monte_carlo(df["equity"].pct_change())
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Data service error: {e}")


@app.post("/api/rolling")
async def rolling_endpoint(request: BacktestRequest):
    """Rolling Sharpe and return metrics over the equity curve."""
    try:
        df, _ = _simulate(request)
        return rolling_metrics(df)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Data service error: {e}")


@app.post("/api/benchmarks")
async def benchmarks_endpoint(request: BacktestRequest):
    """Compare strategy performance against buy-and-hold."""
    result = run_backtest(request)
    bh_series = pd.Series([p.equity for p in result.benchmark_curve])
    strat_series = pd.Series([p.equity for p in result.equity_curve])
    bh_return = float(bh_series.iloc[-1] / bh_series.iloc[0] - 1) if len(bh_series) > 1 else 0.0
    strat_return = float(strat_series.iloc[-1] / strat_series.iloc[0] - 1) if len(strat_series) > 1 else 0.0
    return {
        "strategy_metrics": result.metrics,
        "strategy_total_return": strat_return,
        "buy_hold_total_return": bh_return,
        "equity_curve": result.equity_curve,
        "benchmark_curve": result.benchmark_curve,
    }


@app.post("/api/parameter-sweep")
async def parameter_sweep_endpoint(request: BacktestRequest):
    """Strategy-aware parameter sweep producing a Sharpe heatmap."""
    try:
        return parameter_sweep(request)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Data service error: {e}")


@app.post("/api/export.csv", response_class=PlainTextResponse)
async def export_csv(request: BacktestRequest):
    """Export the equity and benchmark curves as CSV."""
    result = run_backtest(request)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["date", "equity", "benchmark"])
    bh_by_date = {p.date: p.equity for p in result.benchmark_curve}
    for point in result.equity_curve:
        writer.writerow([point.date, point.equity, bh_by_date.get(point.date, "")])
    return PlainTextResponse(content=buffer.getvalue(), media_type="text/csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
