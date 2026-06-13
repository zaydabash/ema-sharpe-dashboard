import { describe, it, expect, beforeEach } from 'vitest';
import { getDefaultParams, saveParams, loadParams } from '@/lib/storage';

describe('storage', () => {
  beforeEach(() => window.localStorage.clear());

  it('returns null when nothing is saved', () => {
    expect(loadParams()).toBeNull();
  });

  it('round-trips saved params', () => {
    const params = getDefaultParams();
    params.ticker = 'QQQ';
    saveParams(params);
    expect(loadParams()?.ticker).toBe('QQQ');
  });

  it('merges saved params over defaults for forward compatibility', () => {
    window.localStorage.setItem('ema-sharpe-params', JSON.stringify({ ticker: 'AAPL' }));
    const loaded = loadParams();
    expect(loaded?.ticker).toBe('AAPL');
    expect(loaded?.strategy).toBe('ema_crossover');
    expect(loaded?.ema_slow).toBe(100);
  });

  it('returns null on corrupt json', () => {
    window.localStorage.setItem('ema-sharpe-params', '{not valid json');
    expect(loadParams()).toBeNull();
  });

  it('defaults line up with the API defaults', () => {
    const defaults = getDefaultParams();
    expect(defaults.ticker).toBe('SPY');
    expect(defaults.ema_fast).toBe(20);
    expect(defaults.vol_target).toBe(true);
  });
});
