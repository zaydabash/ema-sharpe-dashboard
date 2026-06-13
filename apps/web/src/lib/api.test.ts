import { describe, it, expect, vi, afterEach } from 'vitest';
import { runBacktest } from '@/lib/api';
import { getDefaultParams } from '@/lib/storage';

function jsonResponse(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('runBacktest', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('returns parsed json on success', async () => {
    const payload = { metrics: { sharpe: 1 } };
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse(payload, 200)));
    const result = await runBacktest(getDefaultParams());
    expect(result).toEqual(payload);
  });

  it('throws the API detail message on error', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ detail: 'bad ticker' }, 400)));
    await expect(runBacktest(getDefaultParams())).rejects.toThrow('bad ticker');
  });

  it('falls back to a status message when the body is not json', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('oops', { status: 503 })));
    await expect(runBacktest(getDefaultParams())).rejects.toThrow('503');
  });
});
