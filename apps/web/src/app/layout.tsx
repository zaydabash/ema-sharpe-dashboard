import type { Metadata } from 'next'
import { IBM_Plex_Sans, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'

const plexSans = IBM_Plex_Sans({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-sans',
  display: 'swap',
})

const plexMono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  variable: '--font-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Quant Terminal: Multi-Strategy Backtester',
  description:
    'Institutional-grade backtesting for five quantitative strategies with Sharpe, drawdown, and benchmark analytics.',
  keywords: ['trading', 'backtest', 'quantitative finance', 'Sharpe ratio', 'systematic'],
  authors: [{ name: 'Quant Terminal' }],
  openGraph: {
    title: 'Quant Terminal: Multi-Strategy Backtester',
    description:
      'Institutional-grade backtesting for five quantitative strategies with Sharpe, drawdown, and benchmark analytics.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>{children}</body>
    </html>
  )
}
