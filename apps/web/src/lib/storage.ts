import { BacktestRequest } from '@/lib/types';

const STORAGE_KEY = 'ema-sharpe-params';

/** Default backtest parameters, matching the API's defaults. */
export function getDefaultParams(): BacktestRequest {
  const today = new Date().toISOString().slice(0, 10);
  return {
    ticker: 'SPY',
    start: '2018-01-01',
    end: today,
    strategy: 'ema_crossover',

    ema_fast: 20,
    ema_slow: 100,

    rsi_period: 14,
    oversold: 30,
    overbought: 70,

    sma_fast: 20,
    sma_slow: 50,

    bb_window: 20,
    bb_std: 2.0,

    lookback: 20,
    threshold: 0.02,

    fees_bps: 1,
    slip_bps: 2,

    vol_target: true,
    target_vol: 0.15,
  };
}

/** Persist parameters to localStorage (no-op during SSR). */
export function saveParams(params: BacktestRequest): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(params));
  } catch {
    // Ignore quota or serialization errors.
  }
}

/** Load saved parameters, merged over defaults for forward compatibility. */
export function loadParams(): BacktestRequest | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<BacktestRequest>;
    return { ...getDefaultParams(), ...parsed };
  } catch {
    return null;
  }
}
