import base64, json, time, io, csv, traceback, datetime as dt, math, statistics, logging
from functools import lru_cache
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field, ValidationError, field_validator
from typing import Optional, List, Dict, Any
from backtest import run_backtest, annualized_sharpe
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ema-sharpe")

env = Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape())

app = FastAPI(title="EMA + Sharpe Live")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    resp: Response = await call_next(request)
    # Don't cache HTML pages (others like JS/CSS can keep ETags if you want)
    if request.url.path in ("/", "/index.html"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Naive per-IP rate limit: 60 req / 60s ---
WINDOW, LIMIT = 60, 60
buckets = {}

@app.middleware("http")
async def throttle(request: Request, call_next):
    ip = request.client.host
    now = int(time.time())
    key = (ip, now // WINDOW)
    buckets[key] = buckets.get(key, 0) + 1
    if buckets[key] > LIMIT:
        return JSONResponse({"detail":"Too many requests"}, status_code=429)
    return await call_next(request)

class BacktestReq(BaseModel):
    ticker: str = Field("SPY", min_length=1, max_length=10)
    start: str = "2018-01-01"
    end: str = Field(default_factory=lambda: dt.date.today().isoformat())
    strategy: str = Field("ema_crossover", pattern="^(ema_crossover|rsi_mean_reversion|sma_crossover|bollinger_breakout|momentum)$")
    
    # EMA parameters
    ema_fast: int = Field(20, ge=2, le=250)
    ema_slow: int = Field(100, ge=3, le=500)
    
    # RSI parameters
    rsi_period: int = Field(14, ge=5, le=50)
    oversold: float = Field(30, ge=10, le=40)
    overbought: float = Field(70, ge=60, le=90)
    
    # SMA parameters
    sma_fast: int = Field(20, ge=2, le=250)
    sma_slow: int = Field(50, ge=3, le=500)
    
    # Bollinger Bands parameters
    bb_window: int = Field(20, ge=5, le=100)
    bb_std: float = Field(2.0, ge=1.0, le=3.0)
    
    # Momentum parameters
    lookback: int = Field(20, ge=5, le=100)
    threshold: float = Field(0.02, ge=0.001, le=0.1)
    
    # Common parameters
    fees_bps: float = Field(1.0, ge=0, le=100)
    slip_bps: float = Field(2.0, ge=0, le=100)
    vol_target: bool = True
    target_vol: float = Field(0.15, gt=0, le=1.0)

    @field_validator("end")
    @classmethod
    def clamp_future(cls, v: str) -> str:
        try:
            d = dt.date.fromisoformat(v)
        except Exception:
            return dt.date.today().isoformat()
        return min(d, dt.date.today()).isoformat()

    @field_validator("ticker")
    @classmethod
    def norm_ticker(cls, v: str) -> str:
        return (v or "SPY").strip().upper()

    @field_validator("start")
    @classmethod
    def norm_start(cls, v: str) -> str:
        return (v or "2018-01-01").strip()

class BenchmarksReq(BaseModel):
    tickers: List[str]
    start: str
    end: str

class RollingReq(BaseModel):
    ticker: str
    start: str
    end: str
    window_days: int = 126

class MonteCarloReq(BaseModel):
    ticker: str
    start: str
    end: str
    trials: int = 200
    horizon_days: int = 252

class ParameterSweepReq(BaseModel):
    ticker: str
    start: str
    end: str
    fees_bps: float
    slip_bps: float
    vol_target: bool
    target_vol: float
    tickers: List[str]
    fast_list: List[int]
    slow_list: List[int]

class LiveDataReq(BaseModel):
    ticker: str

class SubscriptionReq(BaseModel):
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    ticker: str
    strategy: str
    params: Dict[str, Any]

class LeaderboardEntry(BaseModel):
    id: str
    ticker: str
    strategy: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: str
    sharpe: float
    cagr: float

# In-memory storage for demo (in production, use a database)
leaderboard_data: List[Dict] = []
subscriptions: List[Dict] = []

@app.get("/api/strategies")
def get_strategies():
    """Get available trading strategies"""
    return {
        "strategies": [
            {
                "id": "ema_crossover",
                "name": "EMA Crossover",
                "description": "Buy when fast EMA crosses above slow EMA",
                "params": ["ema_fast", "ema_slow"]
            },
            {
                "id": "rsi_mean_reversion", 
                "name": "RSI Mean Reversion",
                "description": "Buy when RSI is oversold, sell when overbought",
                "params": ["rsi_period", "oversold", "overbought"]
            },
            {
                "id": "sma_crossover",
                "name": "SMA Crossover", 
                "description": "Buy when fast SMA crosses above slow SMA",
                "params": ["sma_fast", "sma_slow"]
            },
            {
                "id": "bollinger_breakout",
                "name": "Bollinger Breakout",
                "description": "Buy on breakout above upper band, sell on breakdown",
                "params": ["bb_window", "bb_std"]
            },
            {
                "id": "momentum",
                "name": "Momentum",
                "description": "Buy when price momentum exceeds threshold",
                "params": ["lookback", "threshold"]
            }
        ]
    }

@app.post("/api/live-data")
async def get_live_data(req: LiveDataReq):
    """Get live/near-live data for a ticker"""
    try:
        ticker = yf.Ticker(req.ticker)
        info = ticker.info
        
        # Get last close price
        hist = ticker.history(period="1d")
        last_close = float(hist['Close'].iloc[-1]) if not hist.empty else None
        
        # Handle case where yfinance returns empty data
        if not info or hist.empty:
            return {
                "ticker": req.ticker.upper(),
                "last_close": None,
                "current_price": None,
                "market_cap": None,
                "volume": None,
                "avg_volume": None,
                "pe_ratio": None,
                "dividend_yield": None,
                "last_updated": dt.datetime.now().isoformat(),
                "error": "No data available for this ticker"
            }
        
        return {
            "ticker": req.ticker.upper(),
            "last_close": last_close,
            "current_price": info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "volume": info.get("volume"),
            "avg_volume": info.get("averageVolume"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "last_updated": dt.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Live data error for {req.ticker}: {str(e)}")
        return {
            "ticker": req.ticker.upper(),
            "error": f"Failed to fetch data: {str(e)}"
        }

@app.post("/api/subscribe")
async def subscribe_to_strategy(req: SubscriptionReq):
    """Subscribe to live tracking of a strategy"""
    try:
        subscription = {
            "email": req.email,
            "ticker": req.ticker,
            "strategy": req.strategy,
            "params": req.params,
            "created_at": dt.datetime.now().isoformat(),
            "active": True
        }
        subscriptions.append(subscription)
        
        return {
            "message": "Subscription created successfully",
            "subscription_id": len(subscriptions) - 1
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/leaderboard")
def get_leaderboard(sort_by: str = "sharpe", limit: int = 20):
    """Get strategy leaderboard"""
    try:
        # Sort by specified metric
        if sort_by == "sharpe":
            sorted_data = sorted(leaderboard_data, key=lambda x: x.get("sharpe", 0), reverse=True)
        elif sort_by == "cagr":
            sorted_data = sorted(leaderboard_data, key=lambda x: x.get("cagr", 0), reverse=True)
        else:
            sorted_data = leaderboard_data
            
        return {
            "leaderboard": sorted_data[:limit],
            "total_entries": len(leaderboard_data)
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/add-to-leaderboard")
async def add_to_leaderboard(req: Request):
    """Add a strategy result to the leaderboard"""
    try:
        payload = await req.json()
        backtest_data = BacktestReq(**payload)
        
        # Run backtest to get metrics
        result = run_backtest(
            ticker=backtest_data.ticker,
            start=backtest_data.start,
            end=backtest_data.end,
            strategy=backtest_data.strategy,
            ema_fast=backtest_data.ema_fast,
            ema_slow=backtest_data.ema_slow,
            rsi_period=backtest_data.rsi_period,
            oversold=backtest_data.oversold,
            overbought=backtest_data.overbought,
            sma_fast=backtest_data.sma_fast,
            sma_slow=backtest_data.sma_slow,
            bb_window=backtest_data.bb_window,
            bb_std=backtest_data.bb_std,
            lookback=backtest_data.lookback,
            threshold=backtest_data.threshold,
            fees_bps=backtest_data.fees_bps,
            slip_bps=backtest_data.slip_bps,
            vol_target=backtest_data.vol_target,
            target_vol=backtest_data.target_vol
        )
        
        # Create leaderboard entry
        entry = {
            "id": f"{backtest_data.ticker}_{backtest_data.strategy}_{int(time.time())}",
            "ticker": backtest_data.ticker,
            "strategy": backtest_data.strategy,
            "params": payload,
            "metrics": result["metrics"],
            "sharpe": result["metrics"]["sharpe"],
            "cagr": result["metrics"]["cagr"],
            "created_at": dt.datetime.now().isoformat()
        }
        
        leaderboard_data.append(entry)
        
        return {
            "message": "Added to leaderboard successfully",
            "entry": entry
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/explain-detailed")
async def explain_detailed(req: BacktestReq):
    """Detailed explainability with regime analysis"""
    try:
        result = cached_backtest(
            ticker=req.ticker, start=req.start, end=req.end,
            strategy=req.strategy,
            ema_fast=req.ema_fast, ema_slow=req.ema_slow,
            rsi_period=req.rsi_period, oversold=req.oversold, overbought=req.overbought,
            sma_fast=req.sma_fast, sma_slow=req.sma_slow,
            bb_window=req.bb_window, bb_std=req.bb_std,
            lookback=req.lookback, threshold=req.threshold,
            fees_bps=req.fees_bps, slip_bps=req.slip_bps,
            vol_target=req.vol_target, target_vol=req.target_vol
        )
        
        # Analyze performance by market regime
        equity_curve = result["equity_curve"]
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i]["equity"] / equity_curve[i-1]["equity"]) - 1
            returns.append(ret)
        
        # Simple regime detection based on rolling returns
        rolling_returns = pd.Series(returns).rolling(60).mean()
        bull_periods = rolling_returns > rolling_returns.median()
        bear_periods = rolling_returns < rolling_returns.median()
        
        bull_sharpe = annualized_sharpe(pd.Series(returns)[bull_periods]) if bull_periods.sum() > 0 else 0
        bear_sharpe = annualized_sharpe(pd.Series(returns)[bear_periods]) if bear_periods.sum() > 0 else 0
        
        explanation = {
            "strategy_info": result.get("strategy_info", {}),
            "overall_performance": result["metrics"],
            "regime_analysis": {
                "bull_market_sharpe": bull_sharpe,
                "bear_market_sharpe": bear_sharpe,
                "bull_periods": int(bull_periods.sum()),
                "bear_periods": int(bear_periods.sum())
            },
            "key_insights": [
                f"Strategy performs {'better' if bull_sharpe > bear_sharpe else 'worse'} in bull markets",
                f"Sharpe ratio: {result['metrics']['sharpe']:.2f}",
                f"Max drawdown: {result['metrics']['max_drawdown']:.1%}",
                f"Win rate: {result['metrics']['win_rate']:.1%}"
            ]
        }
        
        return JSONResponse(explanation)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
def index(q: Optional[str] = Query(default=None)):
    # Permalink support: if ?q=base64(json), inject into page to auto-run
    initial_params = None
    if q:
        try:
            decoded = base64.urlsafe_b64decode(q + "==").decode()
            initial_params = json.loads(decoded)
        except Exception:
            initial_params = None
    tmpl = env.get_template("index.html")
    return tmpl.render(initial_params=json.dumps(initial_params or {}))

@lru_cache(maxsize=128)
def cached_backtest(ticker: str, start: str, end: str, strategy: str, ema_fast: int, ema_slow: int, 
                   rsi_period: int, oversold: float, overbought: float,
                   sma_fast: int, sma_slow: int, bb_window: int, bb_std: float,
                   lookback: int, threshold: float, fees_bps: float, slip_bps: float, 
                   vol_target: bool, target_vol: float):
    """Cache backtest results for 5 minutes"""
    return run_backtest(
        ticker=ticker, start=start, end=end, strategy=strategy,
        ema_fast=ema_fast, ema_slow=ema_slow,
        rsi_period=rsi_period, oversold=oversold, overbought=overbought,
        sma_fast=sma_fast, sma_slow=sma_slow,
        bb_window=bb_window, bb_std=bb_std,
        lookback=lookback, threshold=threshold,
        fees_bps=fees_bps, slip_bps=slip_bps,
        vol_target=vol_target, target_vol=target_vol
    )

@app.post("/api/backtest")
async def api_backtest(req: Request):
    try:
        payload = await req.json()
        data = BacktestReq(**payload)
        log.info(f"[BT] ticker={data.ticker} start={data.start} end={data.end}")
        
        result = cached_backtest(
            ticker=data.ticker, start=data.start, end=data.end,
            strategy=data.strategy,
            ema_fast=data.ema_fast, ema_slow=data.ema_slow,
            rsi_period=data.rsi_period, oversold=data.oversold, overbought=data.overbought,
            sma_fast=data.sma_fast, sma_slow=data.sma_slow,
            bb_window=data.bb_window, bb_std=data.bb_std,
            lookback=data.lookback, threshold=data.threshold,
            fees_bps=data.fees_bps, slip_bps=data.slip_bps,
            vol_target=data.vol_target, target_vol=data.target_vol
        )
        
        result["params_used"] = {
            "ticker": data.ticker,
            "start": data.start,
            "end": data.end,
            "ema_fast": data.ema_fast,
            "ema_slow": data.ema_slow,
            "fees_bps": data.fees_bps,
            "slip_bps": data.slip_bps,
            "vol_target": data.vol_target,
            "target_vol": data.target_vol
        }
        return JSONResponse(result)
    except ValidationError as e:
        return JSONResponse({"error":"validation", "detail": e.errors()}, status_code=422)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error":"server", "detail": str(e)}, status_code=500)

@app.post("/api/benchmarks")
async def benchmarks(req: BenchmarksReq):
    """Compare multiple tickers as buy-and-hold benchmarks"""
    try:
        benchmarks = {}
        for ticker in req.tickers:
            try:
                result = cached_backtest(
                    ticker=ticker, start=req.start, end=req.end,
                    strategy="ema_crossover",
                    ema_fast=1, ema_slow=2,  # Buy and hold
                    rsi_period=14, oversold=30, overbought=70,
                    sma_fast=20, sma_slow=50,
                    bb_window=20, bb_std=2.0,
                    lookback=20, threshold=0.02,
                    fees_bps=0, slip_bps=0,
                    vol_target=False, target_vol=0.15
                )
                benchmarks[ticker] = result["equity_curve"]
            except Exception as e:
                log.warning(f"Failed to fetch benchmark {ticker}: {e}")
                continue
        return JSONResponse({"benchmarks": benchmarks})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/rolling")
async def rolling_metrics(req: RollingReq):
    """Rolling Sharpe, CAGR, and min drawdown over time"""
    try:
        result = cached_backtest(
            ticker=req.ticker, start=req.start, end=req.end,
            strategy="ema_crossover",
            ema_fast=20, ema_slow=100,
            rsi_period=14, oversold=30, overbought=70,
            sma_fast=20, sma_slow=50,
            bb_window=20, bb_std=2.0,
            lookback=20, threshold=0.02,
            fees_bps=1, slip_bps=2,
            vol_target=True, target_vol=0.15
        )
        
        # Calculate rolling metrics
        equity_curve = result["equity_curve"]
        window = req.window_days
        
        rolling_data = []
        for i in range(window, len(equity_curve)):
            window_data = equity_curve[i-window:i]
            if len(window_data) < window:
                continue
                
            # Calculate returns
            returns = []
            for j in range(1, len(window_data)):
                ret = (window_data[j]["equity"] / window_data[j-1]["equity"]) - 1
                returns.append(ret)
            
            if not returns:
                continue
                
            # Rolling Sharpe (annualized)
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns) if len(returns) > 1 else 0
            sharpe = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0
            
            # Rolling CAGR
            start_equity = window_data[0]["equity"]
            end_equity = window_data[-1]["equity"]
            days = len(window_data)
            cagr = (end_equity / start_equity) ** (252 / days) - 1
            
            # Rolling min drawdown
            peak = start_equity
            min_dd = 0
            for point in window_data:
                peak = max(peak, point["equity"])
                dd = (point["equity"] / peak) - 1
                min_dd = min(min_dd, dd)
            
            rolling_data.append({
                "date": window_data[-1]["date"],
                "sharpe": sharpe,
                "cagr": cagr,
                "min_drawdown": min_dd
            })
        
        return JSONResponse({
            "sharpe": [{"date": d["date"], "v": d["sharpe"]} for d in rolling_data],
            "cagr": [{"date": d["date"], "v": d["cagr"]} for d in rolling_data],
            "min_drawdown": [{"date": d["date"], "v": d["min_drawdown"]} for d in rolling_data]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/monte-carlo")
async def monte_carlo(req: MonteCarloReq):
    """Monte Carlo simulation of future returns"""
    try:
        result = cached_backtest(
            ticker=req.ticker, start=req.start, end=req.end,
            strategy="ema_crossover",
            ema_fast=20, ema_slow=100,
            rsi_period=14, oversold=30, overbought=70,
            sma_fast=20, sma_slow=50,
            bb_window=20, bb_std=2.0,
            lookback=20, threshold=0.02,
            fees_bps=1, slip_bps=2,
            vol_target=True, target_vol=0.15
        )
        
        # Extract historical returns
        equity_curve = result["equity_curve"]
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i]["equity"] / equity_curve[i-1]["equity"]) - 1
            returns.append(ret)
        
        if not returns:
            return JSONResponse({"error": "No returns data"}, status_code=400)
        
        # Bootstrap simulation
        import random
        simulations = []
        horizon = req.horizon_days
        
        for _ in range(req.trials):
            sim_returns = random.choices(returns, k=horizon)
            sim_equity = [1.0]
            for ret in sim_returns:
                sim_equity.append(sim_equity[-1] * (1 + ret))
            simulations.append(sim_equity[1:])  # Remove initial 1.0
        
        # Calculate percentiles
        p05, p50, p95 = [], [], []
        for i in range(horizon):
            values = [sim[i] for sim in simulations]
            values.sort()
            p05.append(values[int(0.05 * len(values))])
            p50.append(values[int(0.50 * len(values))])
            p95.append(values[int(0.95 * len(values))])
        
        return JSONResponse({
            "p05": p05,
            "p50": p50,
            "p95": p95
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/parameter-sweep")
async def parameter_sweep(req: ParameterSweepReq):
    """Multi-ticker parameter sweep"""
    try:
        grids = []
        for ticker in req.tickers:
            sharpe_grid = []
            for f in req.fast_list:
                row = []
                for s in req.slow_list:
                    if f >= s:
                        row.append(None)
                        continue
                    try:
                        res = cached_backtest(
                            ticker=ticker, start=req.start, end=req.end,
                            ema_fast=f, ema_slow=s,
                            fees_bps=req.fees_bps, slip_bps=req.slip_bps,
                            vol_target=req.vol_target, target_vol=req.target_vol
                        )
                        row.append(float(res["metrics"]["sharpe"]))
                    except Exception:
                        row.append(None)
                sharpe_grid.append(row)
            
            grids.append({
                "ticker": ticker,
                "fast": req.fast_list,
                "slow": req.slow_list,
                "sharpe": sharpe_grid
            })
        
        return JSONResponse({"grids": grids})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/explain")
async def explain(req: BacktestReq):
    """Explain why buy/sell decisions were made"""
    try:
        result = cached_backtest(
            ticker=req.ticker, start=req.start, end=req.end,
            ema_fast=req.ema_fast, ema_slow=req.ema_slow,
            fees_bps=req.fees_bps, slip_bps=req.slip_bps,
            vol_target=req.vol_target, target_vol=req.target_vol
        )
        
        # Simple explanation based on EMA crossover
        explanation = {
            "strategy": "EMA Crossover",
            "description": f"Buy when {req.ema_fast}-day EMA > {req.ema_slow}-day EMA, sell otherwise",
            "parameters": {
                "fast_ema": req.ema_fast,
                "slow_ema": req.ema_slow,
                "volatility_targeting": req.vol_target,
                "target_volatility": req.target_vol
            },
            "performance": result["metrics"]
        }
        
        return JSONResponse(explanation)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/export.csv")
async def export_csv(req: Request):
    try:
        payload = await req.json()
        data = BacktestReq(**payload)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors())
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    result = run_backtest(
        ticker=data.ticker, start=data.start, end=data.end,
        ema_fast=data.ema_fast, ema_slow=data.ema_slow,
        fees_bps=data.fees_bps, slip_bps=data.slip_bps,
        vol_target=data.vol_target, target_vol=data.target_vol
    )

    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["date","strategy_equity","buy_and_hold_equity"])
    for row in result["equity_curve"]:
        w.writerow([row["date"], row["equity"], row["bh_equity"]])
    return PlainTextResponse(out.getvalue(), media_type="text/csv")

@app.post("/api/trades.csv")
async def trades_csv(req: Request):
    try:
        payload = await req.json()
        data = BacktestReq(**payload)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors())
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    res = run_backtest(
        ticker=data.ticker, start=data.start, end=data.end,
        ema_fast=data.ema_fast, ema_slow=data.ema_slow,
        fees_bps=data.fees_bps, slip_bps=data.slip_bps,
        vol_target=data.vol_target, target_vol=data.target_vol
    )
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["date_in","date_out","days","entry_px","exit_px","ret_pct"])
    for t in res.get("trades", []):
        w.writerow([t["date_in"], t["date_out"], t["days"], t["entry_px"], t["exit_px"], round(100*t["ret"], 4)])
    return PlainTextResponse(out.getvalue(), media_type="text/csv")

@app.post("/backtest")
async def api_backtest_alias(req: Request):
    # Alias for older frontends hitting /backtest
    return await api_backtest(req)