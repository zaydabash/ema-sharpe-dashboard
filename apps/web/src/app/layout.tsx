import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'EMA + Sharpe Dashboard',
  description: 'Interactive EMA crossover backtest with Sharpe ratio and drawdown metrics',
  keywords: ['trading', 'backtest', 'EMA', 'Sharpe ratio', 'quantitative finance'],
  authors: [{ name: 'EMA Sharpe Dashboard' }],
  openGraph: {
    title: 'EMA + Sharpe Dashboard',
    description: 'Interactive EMA crossover backtest with Sharpe ratio and drawdown metrics',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  )
}
