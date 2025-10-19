export interface BacktestRequest {
  ticker: string;
  start: string;
  end: string;
  ema_fast: number;
  ema_slow: number;
  fees_bps: number;
  slip_bps: number;
  vol_target: boolean;
  target_vol: number;
}

export interface BacktestMetrics {
  cagr: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  avg_trade_return: number;
  volatility: number;
  // Support for PascalCase field names from deployed API
  CAGR?: number;
  Sharpe?: number;
  MaxDrawdown?: number;
  WinRate?: number;
  Trades?: number;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface Trade {
  date: string;
  side: string;
  price: number;
  quantity: number;
}

export interface BacktestResponse {
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: Trade[];
  params_used: BacktestRequest;
}
