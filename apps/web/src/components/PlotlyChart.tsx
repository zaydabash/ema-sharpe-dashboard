'use client';

import { EquityPoint, Trade } from '@/lib/types';
import dynamic from 'next/dynamic';
import type { Data, Layout, Config } from 'plotly.js';

// Plotly only runs in the browser; load it client-side to avoid SSR errors.
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PlotlyChartProps {
  equityCurve: EquityPoint[];
  benchmarkCurve?: EquityPoint[];
  trades: Trade[];
  ticker: string;
  title?: string;
}

function toXY(points: EquityPoint[]) {
  return (points || [])
    .map((p) => ({ x: new Date(p.date), y: Number(p.equity) }))
    .filter((p) => !Number.isNaN(p.y) && Number.isFinite(p.y))
    .sort((a, b) => a.x.getTime() - b.x.getTime());
}

export function PlotlyChart({ equityCurve, benchmarkCurve, trades, title }: PlotlyChartProps) {
  const ec = toXY(equityCurve);
  const bench = toXY(benchmarkCurve || []);

  const yAt = (date: string) => {
    const t = new Date(date).getTime();
    const point = ec.find((p) => p.x.getTime() === t);
    return point ? point.y : null;
  };

  const longTrades = trades.filter((trade) => trade.side === 'long');
  const flatTrades = trades.filter((trade) => trade.side === 'flat');

  const ACCENT = '#f0a830'; // amber signal
  const MUTED = '#6b7280';
  const POSITIVE = '#36b37e';
  const NEGATIVE = '#ef5350';
  const AXIS = '#8a909c';
  const GRID = 'rgba(255,255,255,0.06)';
  const monoFamily = 'var(--font-mono), ui-monospace, monospace';

  const data: Data[] = [
    {
      x: ec.map((p) => p.x),
      y: ec.map((p) => p.y),
      type: 'scatter',
      mode: 'lines',
      name: 'Strategy',
      line: { color: ACCENT, width: 2 },
      connectgaps: true,
    },
    {
      x: bench.map((p) => p.x),
      y: bench.map((p) => p.y),
      type: 'scatter',
      mode: 'lines',
      name: 'Buy & Hold',
      line: { color: MUTED, width: 1.5, dash: 'dot' },
      connectgaps: true,
    },
    {
      x: longTrades.map((trade) => new Date(trade.date)),
      y: longTrades.map((trade) => yAt(trade.date)),
      type: 'scatter',
      mode: 'markers',
      name: 'Long Entries',
      marker: { color: POSITIVE, size: 8, symbol: 'triangle-up' },
    },
    {
      x: flatTrades.map((trade) => new Date(trade.date)),
      y: flatTrades.map((trade) => yAt(trade.date)),
      type: 'scatter',
      mode: 'markers',
      name: 'Exits',
      marker: { color: NEGATIVE, size: 8, symbol: 'triangle-down' },
    },
  ];

  const layout: Partial<Layout> = {
    title: title ? { text: title, font: { color: AXIS, family: monoFamily, size: 12 } } : undefined,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: AXIS, family: monoFamily, size: 11 },
    xaxis: {
      type: 'date',
      autorange: true,
      gridcolor: GRID,
      zerolinecolor: GRID,
      linecolor: GRID,
      tickfont: { color: AXIS, family: monoFamily, size: 10 },
    },
    yaxis: {
      type: 'linear',
      title: { text: 'Portfolio Value ($)', font: { color: AXIS, family: monoFamily, size: 11 } },
      tickformat: ',.0f',
      autorange: true,
      fixedrange: false,
      gridcolor: GRID,
      zerolinecolor: GRID,
      linecolor: GRID,
      tickfont: { color: AXIS, family: monoFamily, size: 10 },
    },
    hovermode: 'closest',
    hoverlabel: { font: { family: monoFamily, size: 11 }, bordercolor: GRID },
    showlegend: true,
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.08,
      font: { color: AXIS, family: monoFamily, size: 10 },
      bgcolor: 'rgba(0,0,0,0)',
    },
    margin: { t: 30, r: 16, b: 40, l: 60 },
  };

  const config: Partial<Config> = {
    responsive: true,
    displayModeBar: 'hover',
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  };

  return (
    <div className="w-full h-96">
      <Plot data={data} layout={layout} config={config} style={{ width: '100%', height: '100%' }} useResizeHandler />
    </div>
  );
}
