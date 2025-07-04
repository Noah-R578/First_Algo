# Cryptocurrency Trading Algorithm

A comprehensive cryptocurrency trading algorithm that analyzes multiple technical indicators and statistical measures to generate trading signals.

## Features

- **Multi-timeframe Analysis**: Combines hourly and daily data
- **Technical Indicators**: SMA, RSI, MACD, Bollinger Bands
- **Statistical Analysis**: Hurst Exponent, Stationarity Tests (ADF, KPSS)
- **Pattern Recognition**: All 61 TALib candlestick patterns
- **Outlier Detection**: Hampel filter and statistical outlier detection
- **Multi-coin Analysis**: Analyzes top 10 cryptocurrencies by market cap

## Setup

1. **Configure API Keys**:
   - Copy `config.py` and add your API keys:
   ```python
   ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key"
   CRYPTOCOMPARE_API_KEY = "your_cryptocompare_key"
   ```

2. **Run the Algorithm**:
   ```bash
   python algo1.py
   ```

## API Keys Required

- **CryptoCompare**: Free tier with 1,000 requests/day (100,000 with API key)

## How It Works

1. **Data Collection**: Fetches hourly and daily OHLC data
2. **Data Cleaning**: Removes outliers and fills missing values
3. **Technical Analysis**: Calculates various indicators
4. **Statistical Analysis**: Determines market regime (trending vs mean-reverting)
5. **Signal Generation**: Combines all signals for final decision
6. **Multi-coin Analysis**: Runs analysis on top 10 cryptocurrencies

## Trading Signals

- **Buy Signal**: When combined signals exceed thresholds
- **Sell Signal**: When combined signals fall below thresholds
- **No Trade**: When signals are neutral

## Files

- `algo1.py`: Main algorithm
- `config.py`: API keys (not committed to git)
- `.gitignore`: Excludes sensitive files

## Disclaimer

This is for educational purposes only. Always do your own research and never invest more than you can afford to lose. 
