import pytest
import sys
import os

# Add the parent directory to the path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import apply_fees_and_slippage, calculate_metrics
import pandas as pd
import numpy as np

def test_apply_fees_and_slippage():
    """Test fee and slippage calculation"""
    price = 100.0
    fees_bps = 1.0  # 1 basis point
    slip_bps = 2.0  # 2 basis points
    
    # Long position should add costs
    long_price = apply_fees_and_slippage(price, fees_bps, slip_bps, 'long')
    assert long_price > price
    assert abs(long_price - 100.03) < 0.001  # 100 + 0.01 + 0.02
    
    # Flat position should subtract costs
    flat_price = apply_fees_and_slippage(price, fees_bps, slip_bps, 'flat')
    assert flat_price < price
    assert abs(flat_price - 99.97) < 0.001  # 100 - 0.01 - 0.02

def test_calculate_metrics():
    """Test metrics calculation"""
    # Create a simple equity curve
    equity_curve = pd.Series([10000, 11000, 10500, 12000, 11500])
    
    # Create some sample trades
    trades = [
        {'return': 0.1},
        {'return': -0.05},
        {'return': 0.15},
        {'return': -0.04},
    ]
    
    metrics = calculate_metrics(equity_curve, trades)
    
    # Basic checks
    assert metrics.cagr > 0  # Should be positive
    assert metrics.total_trades == 4
    assert metrics.win_rate == 0.5  # 2 wins out of 4 trades
    assert metrics.max_drawdown < 0  # Should be negative

def test_sharpe_calculation():
    """Test Sharpe ratio calculation"""
    # Create equity curve with known volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    equity_curve = pd.Series(10000 * np.cumprod(1 + returns))
    
    trades = [{'return': r} for r in returns[1:]]  # Skip first return
    
    metrics = calculate_metrics(equity_curve, trades)
    
    # Sharpe should be reasonable for this data
    assert -2 < metrics.sharpe < 2
    assert metrics.volatility > 0
