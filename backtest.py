import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import datetime

def download_bars(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    import time
    import datetime
    
    # retry a few times; yfinance can hiccup
    last_exc = None
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df is not None and not df.empty:
                # Handle multi-level columns if they exist
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df = df.rename_axis("Date").reset_index()
                
                # Ensure we have the required columns
                if "Close" not in df.columns:
                    raise ValueError(f"No Close price data for {ticker}")
                
                if df.empty:
                    raise ValueError(f"No data in date range {start} to {end} for {ticker}")
                    
                return df[["Date", "Close"]].dropna(), "yfinance"
        except Exception as ex:
            last_exc = ex
            print(f"Attempt {attempt + 1} failed for {ticker}: {str(ex)}")
            if attempt < 2:  # Not the last attempt
                time.sleep(1)  # Wait before retry
    
    # If all attempts failed, create mock data for demonstration
    print(f"Warning: Using mock data for {ticker} due to yfinance issues")
    mock_data = create_mock_data(ticker, start, end)
    print(f"Created mock data with shape: {mock_data.shape}")
    return mock_data, "mock"

def create_mock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Create mock data for demonstration when yfinance fails"""
    import datetime
    import pandas as pd
    import numpy as np
    
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d")
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = [d for d in dates if d.weekday() < 5]  # Only weekdays
    
    # Create mock price data (random walk with trend)
    # Make the mock series deterministic per ticker so different tickers look different
    # but remain stable between runs
    seed = (abs(hash((ticker or "SPY").upper())) % (2**32 - 1)) or 42
    rng = np.random.default_rng(seed)
    n = len(dates)
    # Vary drift/vol slightly by ticker to make results differ
    base_drift = 0.0005
    base_vol = 0.02
    drift_adj = ((seed % 1000) - 500) / 1_000_000  # ~[-0.0005, +0.0005]
    vol_adj = ((seed // 1000) % 1000) / 100_000    # ~[0, 0.01]
    mu = base_drift + drift_adj
    sigma = base_vol + vol_adj
    returns = rng.normal(mu, sigma, n)  # ticker-specific random walk
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    return df

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands"""
    sma_val = sma(series, window)
    std = series.rolling(window=window).std()
    upper = sma_val + (std * num_std)
    lower = sma_val - (std * num_std)
    return upper, sma_val, lower

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak - 1.0).min()
    return float(dd)

def annualized_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    excess = returns - rf/periods_per_year
    return float(np.sqrt(periods_per_year) * (excess.mean() / excess.std(ddof=0)))

def calculate_strategy_signals(df: pd.DataFrame, strategy: str, **params) -> Tuple[pd.Series, Dict]:
    """Calculate trading signals based on strategy type"""
    signals = pd.Series(0.0, index=df.index)
    explanations = {}
    
    if strategy == "ema_crossover":
        ema_fast = params.get("ema_fast", 20)
        ema_slow = params.get("ema_slow", 100)
        
        df["ema_fast"] = ema(df["Close"], ema_fast)
        df["ema_slow"] = ema(df["Close"], ema_slow)
        signals = (df["ema_fast"] > df["ema_slow"]).astype(float)
        
        explanations = {
            "strategy": "EMA Crossover",
            "description": f"Buy when {ema_fast}-day EMA > {ema_slow}-day EMA",
            "parameters": {"fast_ema": ema_fast, "slow_ema": ema_slow}
        }
        
    elif strategy == "rsi_mean_reversion":
        rsi_period = params.get("rsi_period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)
        
        df["rsi"] = rsi(df["Close"], rsi_period)
        signals = ((df["rsi"] < oversold).astype(float) - (df["rsi"] > overbought).astype(float))
        
        explanations = {
            "strategy": "RSI Mean Reversion",
            "description": f"Buy when RSI < {oversold}, sell when RSI > {overbought}",
            "parameters": {"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought}
        }
        
    elif strategy == "sma_crossover":
        sma_fast = params.get("sma_fast", 20)
        sma_slow = params.get("sma_slow", 50)
        
        df["sma_fast"] = sma(df["Close"], sma_fast)
        df["sma_slow"] = sma(df["Close"], sma_slow)
        signals = (df["sma_fast"] > df["sma_slow"]).astype(float)
        
        explanations = {
            "strategy": "SMA Crossover",
            "description": f"Buy when {sma_fast}-day SMA > {sma_slow}-day SMA",
            "parameters": {"fast_sma": sma_fast, "slow_sma": sma_slow}
        }
        
    elif strategy == "bollinger_breakout":
        bb_window = params.get("bb_window", 20)
        bb_std = params.get("bb_std", 2)
        
        upper, middle, lower = bollinger_bands(df["Close"], bb_window, bb_std)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        
        # Breakout: buy when price breaks above upper band, sell when below lower
        signals = ((df["Close"] > df["bb_upper"]).astype(float) - 
                  (df["Close"] < df["bb_lower"]).astype(float))
        
        explanations = {
            "strategy": "Bollinger Breakout",
            "description": f"Buy on breakout above upper band, sell on breakdown below lower band",
            "parameters": {"bb_window": bb_window, "bb_std": bb_std}
        }
        
    elif strategy == "momentum":
        lookback = params.get("lookback", 20)
        threshold = params.get("threshold", 0.02)
        
        df["momentum"] = df["Close"].pct_change(lookback)
        signals = (df["momentum"] > threshold).astype(float)
        
        explanations = {
            "strategy": "Momentum",
            "description": f"Buy when {lookback}-day momentum > {threshold:.1%}",
            "parameters": {"lookback": lookback, "threshold": threshold}
        }
    
    return signals, explanations

def run_backtest(
    ticker="SPY", start="2015-01-01", end="2025-01-01",
    strategy="ema_crossover",
    ema_fast=20, ema_slow=100,
    rsi_period=14, oversold=30, overbought=70,
    sma_fast=20, sma_slow=50,
    bb_window=20, bb_std=2,
    lookback=20, threshold=0.02,
    fees_bps=1.0, slip_bps=2.0,
    vol_target=True, target_vol=0.15, vol_window=20
):
    try:
        df, data_source = download_bars(ticker, start, end)
        
        # Ensure we have enough data
        min_periods = max(ema_fast, ema_slow, rsi_period, sma_fast, sma_slow, bb_window, lookback) + 10
        if len(df) < min_periods:
            raise ValueError(f"Not enough data for strategy calculation. Need at least {min_periods} days, got {len(df)}")
        
        df["ret"] = df["Close"].pct_change().fillna(0.0)
        
        # Calculate strategy signals
        strategy_params = {
            "ema_fast": ema_fast, "ema_slow": ema_slow,
            "rsi_period": rsi_period, "oversold": oversold, "overbought": overbought,
            "sma_fast": sma_fast, "sma_slow": sma_slow,
            "bb_window": bb_window, "bb_std": bb_std,
            "lookback": lookback, "threshold": threshold
        }
        
        df["signal_raw"], strategy_explanation = calculate_strategy_signals(df, strategy, **strategy_params)

        # exposure with optional vol targeting (cap 3x)
        exposure = df["signal_raw"].copy()
        if vol_target:
            rolling_vol = df["ret"].rolling(vol_window).std(ddof=0) * np.sqrt(252)
            scaling = (target_vol / rolling_vol).clip(0.0, 3.0)
            exposure = (exposure * scaling).fillna(0.0)

        df["pos"] = exposure
        df["pos_prev"] = df["pos"].shift().fillna(0.0)
        df["trade_size"] = (df["pos"] - df["pos_prev"]).abs()
        cost = (fees_bps + slip_bps) / 10000.0
        df["costs"] = -df["trade_size"] * cost
        df["strat_ret"] = df["pos_prev"] * df["ret"] + df["costs"]
        df["equity"] = (1.0 + df["strat_ret"]).cumprod()
        df["bh_equity"] = (1.0 + df["ret"]).cumprod()

        # ---- detailed trades (entry when pos 0→>0, exit >0→0); PnL on strategy returns ----
        trades = []
        in_idx = None
        for i in range(1, len(df)):
            was = df.at[i-1, "pos"]; now = df.at[i, "pos"]
            if was <= 1e-9 and now > 1e-9 and in_idx is None:
                in_idx = i
            elif was > 1e-9 and now <= 1e-9 and in_idx is not None:
                # include both endpoints (entry day through exit day)
                rr = (1.0 + df["strat_ret"].iloc[in_idx:i+1]).prod() - 1.0
                trades.append({
                    "date_in": df.at[in_idx, "Date"].strftime("%Y-%m-%d"),
                    "date_out": df.at[i, "Date"].strftime("%Y-%m-%d"),
                    "ret": float(rr),
                    "days": int(i - in_idx + 1),
                    "entry_px": float(df.at[in_idx, "Close"]),
                    "exit_px": float(df.at[i, "Close"])
                })
                in_idx = None
        # if still open at the end, close at last bar
        if in_idx is not None and len(df) - 1 > in_idx:
            j = len(df) - 1
            rr = (1.0 + df["strat_ret"].iloc[in_idx:j+1]).prod() - 1.0
            trades.append({
                "date_in": df.at[in_idx, "Date"].strftime("%Y-%m-%d"),
                "date_out": df.at[j, "Date"].strftime("%Y-%m-%d"),
                "ret": float(rr),
                "days": int(j - in_idx + 1),
                "entry_px": float(df.at[in_idx, "Close"]),
                "exit_px": float(df.at[j, "Close"])
            })

        # ---- monthly returns (strategy & buy-and-hold), with row/column totals ----
        dd = df[["Date", "strat_ret", "ret"]].copy()
        dd["year"] = dd["Date"].dt.year
        dd["month"] = dd["Date"].dt.month

        def monthify(series):
            return (1.0 + series).prod() - 1.0

        strat_m = dd.groupby(["year", "month"])["strat_ret"].apply(monthify).reset_index()
        bh_m    = dd.groupby(["year", "month"])["ret"].apply(monthify).reset_index()

        def to_grid(tbl):
            grid = {}
            for yr, sub in tbl.groupby("year"):
                row = [None]*12
                for _, r in sub.iterrows():
                    row[int(r["month"])-1] = float(r.iloc[-1])
                grid[int(yr)] = row
            return grid

        strat_grid = to_grid(strat_m)
        bh_grid    = to_grid(bh_m)

        # totals: yearly (row) and monthly (col), plus grand total
        def totals(grid):
            years = sorted(grid.keys())
            col_tot = [0.0]*12
            row_tot = {}
            for y in years:
                vals = [v for v in (grid[y] or []) if v is not None]
                row_tot[y] = float(np.prod([1.0+v for v in vals]) - 1.0) if vals else 0.0
                for m in range(12):
                    if grid[y][m] is not None:
                        col_tot[m] = (1.0+col_tot[m])*(1.0+grid[y][m]) - 1.0
            grand_vals = [v for v in col_tot if v is not None]
            grand = float(np.prod([1.0+v for v in grand_vals]) - 1.0) if grand_vals else 0.0
            return {"row": row_tot, "col": col_tot, "grand": grand}

        strat_tot = totals(strat_grid)
        bh_tot    = totals(bh_grid)

        # metrics
        cagr = float((df["equity"].iloc[-1]) ** (252 / max(1, len(df))) - 1.0)
        sharpe = annualized_sharpe(df["strat_ret"])
        mdd = max_drawdown(df["equity"])
        wins = int((df["strat_ret"] > 0).sum())
        trades_cnt = int((df["trade_size"] > 1e-9).sum())

        equity_curve = [
            {"date": d.strftime("%Y-%m-%d"), "equity": float(e), "bh_equity": float(b)}
            for d, e, b in zip(df["Date"], df["equity"], df["bh_equity"])
        ]

        # drawdown & exposure
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = df["equity"] / df["peak"] - 1.0
        exposure_series = df["pos"].fillna(0.0).clip(0, 3.0).astype(float).tolist()

        # --- volatility band (visual overlay), 20d rolling std of strategy daily returns
        roll_std = df["strat_ret"].rolling(20).std(ddof=0).fillna(0.0)
        k = 2.0  # ±2σ band (visual, not a prediction interval)
        band_up = (df["equity"] * (1.0 + k * roll_std)).clip(lower=0).astype(float).tolist()
        band_dn = (df["equity"] * (1.0 - k * roll_std)).clip(lower=0).astype(float).tolist()

        return {
            "metrics": {"cagr": cagr, "sharpe": sharpe, "max_drawdown": mdd,
                        "win_rate": wins / max(1, len(df)), "total_trades": trades_cnt},
            "equity_curve": equity_curve,
            "drawdown": [float(x) for x in df["drawdown"].fillna(0.0)],
            "exposure": exposure_series,
            "trades": trades,                        # detailed list
            "monthly_returns": strat_grid,           # strategy grid
            "monthly_returns_bh": bh_grid,           # buy & hold grid
            "monthly_totals": {"strategy": strat_tot, "buyhold": bh_tot},
            "band_up": band_up,
            "band_down": band_dn,
            "data_source": data_source,
            "strategy_info": strategy_explanation
        }
    except Exception as e:
        raise ValueError(f"Backtest failed: {str(e)}")