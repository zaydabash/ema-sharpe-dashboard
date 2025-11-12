from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import functools
import time
from typing import List, Dict

app = FastAPI(title="EMA Sharpe Dashboard API", version="1.0.0")

# CORS middleware
# SECURITY NOTE: Currently allows all origins. For production, restrict to specific domains.
# Example: allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict to specific domains in production
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
    """
    Fetch and cache market data with retries.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        RuntimeError: If data fetch fails after 3 attempts
        ValueError: If ticker is invalid or date range is invalid
    """
    # Validate ticker format (basic sanitization)
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    # Sanitize ticker to prevent injection (alphanumeric and dots only)
    if not all(c.isalnum() or c == '.' for c in ticker.upper()):
        raise ValueError("Invalid ticker format. Use alphanumeric characters and dots only.")
    
    ticker = ticker.upper().strip()
    
    for i in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except ValueError as e:
            raise ValueError(f"Invalid date range or ticker: {e}")
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i == 2:  # Last attempt
                raise RuntimeError(
                    f"Data unavailable for {ticker}. "
                    "Try a broader date range or a different ticker."
                ) from e
        time.sleep(1 + i)
    
    raise RuntimeError("Data unavailable. Try a broader date range or a different ticker.")

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        prices: Series of price data
        period: EMA period (number of periods)
    
    Returns:
        Series with EMA values
    """
    if period < 1:
        raise ValueError("EMA period must be at least 1")
    return prices.ewm(span=period).mean()

def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate annualized rolling volatility.
    
    Args:
        returns: Series of daily returns
        window: Rolling window size in days (default: 20)
    
    Returns:
        Series with annualized volatility (252 trading days)
    """
    if window < 1:
        raise ValueError("Volatility window must be at least 1")
    return returns.rolling(window=window).std() * np.sqrt(252)

def apply_fees_and_slippage(price: float, fees_bps: float, slip_bps: float, side: str) -> float:
    """
    Apply realistic trading costs (fees and slippage).
    
    Args:
        price: Base price
        fees_bps: Fees in basis points (1 bp = 0.01%)
        slip_bps: Slippage in basis points
        side: Trade side ('long' or 'flat')
    
    Returns:
        Adjusted price after fees and slippage
        
    Raises:
        ValueError: If side is invalid or price is negative
    """
    if price <= 0:
        raise ValueError("Price must be positive")
    if fees_bps < 0 or slip_bps < 0:
        raise ValueError("Fees and slippage must be non-negative")
    
    fee_cost = price * (fees_bps / 10000)
    slip_cost = price * (slip_bps / 10000)
    
    if side == 'long':
        return price + fee_cost + slip_cost
    elif side == 'flat':
        return price - fee_cost - slip_cost
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'long' or 'flat'")

def calculate_metrics(equity_curve: pd.Series, trades: List[Dict]) -> BacktestMetrics:
    """
    Calculate comprehensive performance metrics from equity curve and trades.
    
    Args:
        equity_curve: Series of equity values over time
        trades: List of trade dictionaries with 'return' key
    
    Returns:
        BacktestMetrics object with calculated metrics
        
    Raises:
        ValueError: If equity_curve is empty or invalid
    """
    if equity_curve.empty or len(equity_curve) < 2:
        raise ValueError("Equity curve must have at least 2 data points")
    
    if equity_curve.iloc[0] <= 0:
        raise ValueError("Initial equity must be positive")
    
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
    trade_returns = [t.get('return', 0) for t in trades if isinstance(t, dict)]
    win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns) if trade_returns else 0
    avg_trade_return = float(np.mean(trade_returns)) if trade_returns else 0.0
    
    # Volatility
    volatility = returns.std() * np.sqrt(252) if not returns.empty else 0.0
    
    return BacktestMetrics(
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=float(max_drawdown),
        win_rate=float(win_rate),
        total_trades=len(trades),
        avg_trade_return=avg_trade_return,
        volatility=volatility
    )

def run_backtest(params: BacktestRequest) -> BacktestResponse:
    """
    Run the EMA crossover backtest strategy.
    
    Args:
        params: BacktestRequest with strategy parameters
    
    Returns:
        BacktestResponse with metrics, equity curve, and trades
        
    Raises:
        HTTPException: If backtest fails due to invalid parameters or data issues
    """
    try:
        # Validate date range
        try:
            start_date = datetime.strptime(params.start, '%Y-%m-%d')
            end_date = datetime.strptime(params.end, '%Y-%m-%d')
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            if (end_date - start_date).days > 3650:  # ~10 years max
                raise ValueError("Date range cannot exceed 10 years")
        except ValueError as e:
            if "does not match format" in str(e):
                raise ValueError("Invalid date format. Use YYYY-MM-DD") from e
            raise
        
        # Fetch data
        df = get_bars(params.ticker, params.start, params.end)
        if df.empty:
            raise ValueError("No data available for the given parameters")
        
        # Validate required columns
        if 'Close' not in df.columns:
            raise ValueError("Data missing 'Close' price column")
        
        # Fix MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Flatten to single level
        
        # Calculate EMAs
        df['ema_fast'] = calculate_ema(df['Close'], params.ema_fast)
        df['ema_slow'] = calculate_ema(df['Close'], params.ema_slow)
        
        # Generate signals - simple EMA crossover strategy
        df['signal'] = 0
        
        # Calculate signal using numpy where to avoid pandas boolean ambiguity
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
            price = float(df.iloc[i]['Close'])
            vol_scalar = float(df.iloc[i]['vol_scalar'])
            
            # Position sizing with volatility targeting
            target_position = curr_signal * vol_scalar
            
            if prev_signal != curr_signal:  # Signal change
                # Close existing position
                if df.iloc[i-1]['shares'] != 0:
                    exit_price = apply_fees_and_slippage(price, params.fees_bps, params.slip_bps, 'flat')
                    cash_change = df.iloc[i-1]['shares'] * exit_price
                    df.iloc[i, df.columns.get_loc('cash')] = df.iloc[i-1]['cash'] + cash_change
                    
                    # Record trade
                    entry_price = float(df.iloc[i-1]['entry_price']) if df.iloc[i-1]['entry_price'] > 0 else price
                    trade_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0.0
                    trades.append({
                        'date': df.index[i].strftime('%Y-%m-%d'),
                        'side': 'flat',
                        'price': exit_price,
                        'quantity': float(df.iloc[i-1]['shares']),
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
                equity_value = float(equity)
                if not np.isnan(equity_value) and np.isfinite(equity_value):
                    equity_curve.append(EquityPoint(
                        date=date.strftime('%Y-%m-%d'), 
                        equity=equity_value
                    ))
            except (ValueError, TypeError) as e:
                print(f"Error creating equity point {i}: {e}")
                continue
        
        if not equity_curve:
            raise ValueError("Failed to generate equity curve. Check data quality.")
        
        # Format trades
        formatted_trades = []
        for trade in trades:
            try:
                formatted_trades.append(Trade(
                    date=trade['date'],
                    side=trade['side'],
                    price=float(trade['price']),
                    quantity=float(trade['quantity'])
                ))
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error formatting trade: {e}")
                continue
        
        # Calculate metrics
        metrics = calculate_metrics(df['equity'], trades)
        
        return BacktestResponse(
            metrics=metrics,
            equity_curve=equity_curve,
            trades=formatted_trades,
            params_used=params
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Data service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Status and timestamp
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    """
    Run EMA crossover backtest with specified parameters.
    
    Args:
        request: BacktestRequest with strategy parameters
    
    Returns:
        BacktestResponse with metrics, equity curve, and trades
        
    Raises:
        HTTPException: 400 for invalid parameters, 503 for data service errors, 500 for internal errors
    """
    return run_backtest(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
