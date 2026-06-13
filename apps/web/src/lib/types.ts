export type StrategyId =
  | 'ema_crossover'
  | 'rsi_mean_reversion'
  | 'sma_crossover'
  | 'bollinger_breakout'
  | 'momentum';

export interface BacktestRequest {
  ticker: string;
  start: string;
  end: string;
  strategy: StrategyId;

  // EMA
  ema_fast: number;
  ema_slow: number;

  // RSI
  rsi_period: number;
  oversold: number;
  overbought: number;

  // SMA
  sma_fast: number;
  sma_slow: number;

  // Bollinger
  bb_window: number;
  bb_std: number;

  // Momentum
  lookback: number;
  threshold: number;

  // Costs
  fees_bps: number;
  slip_bps: number;

  // Volatility targeting
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
  benchmark_curve: EquityPoint[];
  trades: Trade[];
  params_used: BacktestRequest;
}
