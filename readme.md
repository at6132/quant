# Quants Trading System

A comprehensive trading system that processes market data and generates trading signals using various technical analysis strategies.

## Quick Start

1. Get the data:
   ```bash
   python download_btc_15s.py
   ```
   This will download the last 7 days of BTCUSDT data from Binance and create:
   - `BTCUSDT_1s_last7days.csv` (raw 1-second data)
   - `BTCUSDT_15s_last7days.csv` (aggregated 15-second data)

2. Process the data with indicators:
   ```bash
   python process_indicators.py
   ```
   This will create a `processed_data` directory containing parquet files for multiple timeframes:
   - 15Second.parquet
   - 1minute.parquet
   - 15minute.parquet
   - 1hour.parquet
   - 4hours.parquet

Each parquet file contains the original price data plus all the indicators described below.

## Available Indicators

### 1. Bollinger Bands Order Block Engine
- Detects potential reversal zones using Bollinger Bands
- Identifies order blocks and breakout points
- Generates signals based on price action within bands

### 2. Breaker Signals
- Identifies breaker blocks (failed support/resistance levels)
- Generates signals for potential trend reversals
- Uses multiple timeframe analysis

### 3. ICT Smart Money Concepts
- Implements Institutional Smart Money Concepts
- Detects institutional order flow
- Identifies potential institutional trading zones

### 4. Liquidity Swings
- Analyzes market liquidity patterns
- Detects swing highs and lows
- Identifies potential liquidity grabs

### 5. PVSRA (Price Volume Supply Demand Analysis)
- Analyzes price and volume relationships
- Identifies supply and demand zones
- Generates signals based on volume profile

### 6. Trading Sessions
- Detects major trading sessions (London, New York, Tokyo)
- Analyzes session-specific price action
- Generates session-based trading signals

### 7. SMC Core
- Core Smart Money Concepts analysis
- Identifies institutional trading patterns
- Generates signals based on market structure

### 8. Trend Reality Core
- Analyzes trend strength and direction
- Identifies trend continuation and reversal patterns
- Generates signals based on trend analysis

## Data Processing Details

The system processes data in chunks to manage memory efficiently and includes:
- Automatic timezone handling (all timestamps in UTC)
- Memory optimization (float32 data types)
- Error handling and logging
- Progress tracking for long operations

## Dependencies

- Python 3.8+
- pandas
- numpy
- pyarrow
- requests (for data download)
- logging

## Notes

- All timestamps are in UTC
- Data is processed in chunks for memory efficiency
- Each indicator's output is prefixed with its name (e.g., 'bb_' for Bollinger Bands)
- The system automatically handles data type conversions for parquet storage
