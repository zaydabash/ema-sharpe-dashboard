'use client';

import { EquityPoint, Trade } from '@/lib/types';
import Plot from 'react-plotly.js';

interface PlotlyChartProps {
  equityCurve: EquityPoint[];
  trades: Trade[];
  ticker: string;
}

export function PlotlyChart({ equityCurve, trades, ticker }: PlotlyChartProps) {
  // Coerce types and ensure proper data format
  const ec = (equityCurve || [])
    .map(p => ({ x: new Date(p.date), y: Number(p.equity) }))
    .filter(p => !Number.isNaN(p.y) && Number.isFinite(p.y))  // Remove NaN/Inf values
    .sort((a, b) => a.x.getTime() - b.x.getTime());  // Sort by date

  // Separate long and flat trades for different markers
  const longTrades = trades.filter(trade => trade.side === 'long');
  const flatTrades = trades.filter(trade => trade.side === 'flat');

  const data = [
    {
      x: ec.map(p => p.x),
      y: ec.map(p => p.y),
      type: 'scatter',
      mode: 'lines',
      name: 'Equity Curve',
      line: { color: '#3b82f6', width: 2 },
      connectgaps: true,  // Smooth over tiny gaps
    },
    {
      x: longTrades.map(trade => new Date(trade.date)),
      y: longTrades.map(trade => {
        const point = ec.find(p => p.x.getTime() === new Date(trade.date).getTime());
        return point ? point.y : 0;
      }),
      type: 'scatter',
      mode: 'markers',
      name: 'Long Entries',
      marker: { 
        color: '#10b981', 
        size: 8,
        symbol: 'triangle-up'
      },
    },
    {
      x: flatTrades.map(trade => new Date(trade.date)),
      y: flatTrades.map(trade => {
        const point = ec.find(p => p.x.getTime() === new Date(trade.date).getTime());
        return point ? point.y : 0;
      }),
      type: 'scatter',
      mode: 'markers',
      name: 'Exits',
      marker: { 
        color: '#ef4444', 
        size: 8,
        symbol: 'triangle-down'
      },
    },
  ];

  const layout = {
    title: `${ticker} EMA Crossover Strategy`,
    xaxis: { 
      type: 'date',
      title: 'Date',
      autorange: true
    },
    yaxis: { 
      type: 'linear',
      title: 'Portfolio Value ($)',
      tickformat: ',.2f',
      autorange: true,
      fixedrange: false
    },
    hovermode: 'closest' as const,
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: 'rgba(255,255,255,0.8)',
    },
    margin: { t: 50, r: 50, b: 50, l: 50 },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
  };

  return (
    <div className="w-full h-96">
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
