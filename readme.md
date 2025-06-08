# Quants Trading System

A comprehensive trading system that processes market data and generates trading signals using various technical analysis strategies.

## Project Structure

### Core Components

- `process_indicators.py`: Main script for processing market data and generating indicators
- `Core/indicators/`: Directory containing all indicator implementations
  - `bb_ob_engine.py`: Bollinger Bands Order Block Engine for detecting potential reversal zones
  - `breaker_signals.py`: Breaker block detection and signal generation
  - `ict_sm_trades.py`: ICT Smart Money Concepts trading signals
  - `liquidity_swings.py`: Liquidity swing detection and analysis
  - `pvsra.py`: Price Volume Supply Demand Analysis
  - `sessions.py`: Trading session detection and analysis
  - `smc_core.py`: Smart Money Concepts core analysis
  - `tr_reality_core.py`: Trend Reality Core analysis

### Data Processing

- `processed_data/`: Directory containing processed data files
  - `15Second.parquet`: 15-second timeframe processed data
  - `1Minute.parquet`: 1-minute timeframe processed data
  - `5Minute.parquet`: 5-minute timeframe processed data
  - `15Minute.parquet`: 15-minute timeframe processed data
  - `1Hour.parquet`: 1-hour timeframe processed data
  - `4Hour.parquet`: 4-hour timeframe processed data
  - `1Day.parquet`: Daily timeframe processed data

## Data Requirements

### Input Data Format
The system expects a CSV file (`btcusdt_15s.csv`) with the following columns:
- Timestamp (index)
- Open
- High
- Low
- Close
- Volume

### Data Sources
1. Binance API (recommended)
2. Custom data providers
3. Historical data downloads

## Strategies

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

## Usage

1. Prepare your data:
   ```bash
   # Place your btcusdt_15s.csv file in the project root
   ```

2. Run the processing script:
   ```bash
   python process_indicators.py
   ```

3. Access processed data:
   - Processed data will be saved in the `processed_data/` directory
   - Each timeframe will have its own parquet file
   - Use pandas to read the parquet files for analysis

## Output Data

The processed data includes:
- Original price data
- Technical indicators
- Trading signals
- Session information
- Order block detection
- Liquidity analysis
- Trend analysis

## Dependencies

- Python 3.8+
- pandas
- numpy
- pyarrow
- logging

## Notes

- The system processes data in chunks to manage memory efficiently
- All timestamps are converted to UTC
- Data is downcast to float32 to reduce memory footprint
- Session names are converted to strings for parquet storage
