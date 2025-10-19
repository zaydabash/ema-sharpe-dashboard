from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import functools
import time
from typing import List, Dict, Any

app = FastAPI(title="EMA Sharpe Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
WINDOW, LIMIT = 60, 60  # 60 requests per minute per IP
buckets = {}

@app.middleware("http")
async def throttle(request: Request, call_next):
    ip = request.client.host
    now = int(time.time())
    key = (ip, now // WINDOW)
    buckets[key] = buckets.get(key, 0) + 1
    if buckets[key] > LIMIT:
        raise HTTPException(429, "Too many requests. Slow down.")
    return await call_next(request)

# Pydantic models
class BacktestRequest(BaseModel):
    ticker: str = Field("SPY", min_length=1, max_length=10)
    start: str
    end: str
    ema_fast: int = Field(20, ge=2, le=250)
    ema_slow: int = Field(100, ge=20, le=500)
    fees_bps: float = Field(1, ge=0, le=100)
    slip_bps: float = Field(2, ge=0, le=100)
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
    trades: List[Trade]
    params_used: BacktestRequest

# Cached data fetching
@functools.lru_cache(maxsize=256)
def get_bars(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch and cache market data with retries"""
    for i in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
        time.sleep(1 + i)
    raise RuntimeError("Data unavailable. Try a broader date range or a different ticker.")

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period).mean()

def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)

def apply_fees_and_slippage(price: float, fees_bps: float, slip_bps: float, side: str) -> float:
    """Apply realistic trading costs"""
    fee_cost = price * (fees_bps / 10000)
    slip_cost = price * (slip_bps / 10000)
    
    if side == 'long':
        return price + fee_cost + slip_cost
    else:  # flat/exit
        return price - fee_cost - slip_cost

def calculate_metrics(equity_curve: pd.Series, trades: List[Dict]) -> BacktestMetrics:
    """Calculate performance metrics"""
    returns = equity_curve.pct_change().dropna()
    
    # CAGR
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    years = len(equity_curve) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Sharpe ratio (risk-free rate = 0)
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Max drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Win rate and trade metrics
    trade_returns = [t['return'] for t in trades if 'return' in t]
    win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
    avg_trade_return = np.mean(trade_returns) if trade_returns else 0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    return BacktestMetrics(
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_trades=len(trades),
        avg_trade_return=avg_trade_return,
        volatility=volatility
    )

def run_backtest(params: BacktestRequest) -> BacktestResponse:
    """Run the EMA crossover backtest"""
    try:
        # Fetch data
        df = get_bars(params.ticker, params.start, params.end)
        if df.empty:
            raise ValueError("No data available for the given parameters")
        
        # Fix MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Flatten to single level
        
        # Calculate EMAs
        df['ema_fast'] = calculate_ema(df['Close'], params.ema_fast)
        df['ema_slow'] = calculate_ema(df['Close'], params.ema_slow)
        
        # Generate signals - simple EMA crossover strategy
        df['signal'] = 0
        
        # Calculate signal using numpy where to avoid pandas boolean ambiguity
        import numpy as np
        valid_mask = df['ema_fast'].notna() & df['ema_slow'].notna()
        df['signal'] = np.where(
            valid_mask & (df['ema_fast'] > df['ema_slow']), 
            1, 
            0
        ).astype(int)  # Ensure integer type
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Volatility targeting
        if params.vol_target:
            df['vol'] = calculate_volatility(df['returns'])
            df['vol_scalar'] = params.target_vol / df['vol']
            df['vol_scalar'] = df['vol_scalar'].fillna(1).clip(0, 2)  # Cap at 2x leverage
        else:
            df['vol_scalar'] = 1
        
        # Initialize portfolio
        df['position'] = 0
        df['equity'] = 10000  # Starting capital
        df['cash'] = 10000
        df['shares'] = 0
        df['entry_price'] = 0.0  # Initialize entry price column
        
        trades = []
        
        for i in range(1, len(df)):
            prev_signal = int(df.iloc[i-1]['signal'])  # Already scalar
            curr_signal = int(df.iloc[i]['signal'])    # Already scalar
            price = df.iloc[i]['Close']
            vol_scalar = df.iloc[i]['vol_scalar']
            
            # Position sizing with volatility targeting
            target_position = curr_signal * vol_scalar
            
            if prev_signal != curr_signal:  # Signal change
                # Close existing position
                if df.iloc[i-1]['shares'] != 0:
                    exit_price = apply_fees_and_slippage(price, params.fees_bps, params.slip_bps, 'flat')
                    cash_change = df.iloc[i-1]['shares'] * exit_price
                    df.iloc[i, df.columns.get_loc('cash')] = df.iloc[i-1]['cash'] + cash_change
                    
                    # Record trade
                    trade_return = (exit_price - df.iloc[i-1]['entry_price']) / df.iloc[i-1]['entry_price'] if 'entry_price' in df.columns else 0
                    trades.append({
                        'date': df.index[i].strftime('%Y-%m-%d'),
                        'side': 'flat',
                        'price': exit_price,
                        'quantity': df.iloc[i-1]['shares'],
                        'return': trade_return
                    })
                
                # Open new position
                if target_position > 0:
                    entry_price = apply_fees_and_slippage(price, params.fees_bps, params.slip_bps, 'long')
                    shares_to_buy = (df.iloc[i]['cash'] * target_position) / entry_price
                    cost = shares_to_buy * entry_price
                    
                    df.iloc[i, df.columns.get_loc('cash')] = df.iloc[i]['cash'] - cost
                    df.iloc[i, df.columns.get_loc('shares')] = shares_to_buy
                    df.iloc[i, df.columns.get_loc('entry_price')] = entry_price
                    
                    trades.append({
                        'date': df.index[i].strftime('%Y-%m-%d'),
                        'side': 'long',
                        'price': entry_price,
                        'quantity': shares_to_buy
                    })
                else:
                    df.iloc[i, df.columns.get_loc('shares')] = 0
                    df.iloc[i, df.columns.get_loc('entry_price')] = 0.0
            else:
                # No signal change, maintain position
                df.iloc[i, df.columns.get_loc('cash')] = df.iloc[i-1]['cash']
                df.iloc[i, df.columns.get_loc('shares')] = df.iloc[i-1]['shares']
                df.iloc[i, df.columns.get_loc('entry_price')] = df.iloc[i-1]['entry_price']
            
            # Calculate equity
            df.iloc[i, df.columns.get_loc('equity')] = df.iloc[i]['cash'] + df.iloc[i]['shares'] * price
        
        # Create equity curve - simplified approach
        equity_curve = []
        for i, (date, equity) in enumerate(zip(df.index, df['equity'])):
            try:
                equity_curve.append(EquityPoint(
                    date=date.strftime('%Y-%m-%d'), 
                    equity=float(equity)
                ))
            except Exception as e:
                print(f"Error creating equity point {i}: {e}")
                continue
        
        # Format trades
        formatted_trades = [
            Trade(
                date=trade['date'],
                side=trade['side'],
                price=trade['price'],
                quantity=trade['quantity']
            )
            for trade in trades
        ]
        
        # Calculate metrics
        metrics = calculate_metrics(df['equity'], trades)
        
        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=formatted_trades,
            params_used=params
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    """Run EMA crossover backtest"""
    return run_backtest(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
