# Quants Trading System

A comprehensive trading system that processes market data and generates trading signals using various technical analysis strategies.

## Quick Start: Full Workflow

### 1. **Download Data**
```bash
python download_btc_15s.py
```
- Downloads the last 7 days of BTCUSDT data from Binance.
- Creates:
  - `BTCUSDT_1s_last7days.csv` (raw 1-second data)
  - `BTCUSDT_15s_last7days.csv` (aggregated 15-second data)

### 2. **Process Data with Indicators**
```bash
python process_indicators.py
```
- Creates a `processed_data` directory with parquet files for multiple timeframes:
  - 15Second.parquet
  - 1minute.parquet
  - 15minute.parquet
  - 1hour.parquet
  - 4hours.parquet
- Each file contains price data plus all indicators.

### 3. **Train Models**
```bash
python main.py -c config.yaml
```
- Loads and preprocesses the data
- Mines trading rules
- Trains LightGBM and TFT models
- Generates predictions
- Saves all results and artifacts

**Artifacts and Results:**
- `artefacts/feature_matrix.parquet`: All engineered features/labels
- `artefacts/rule_miner_report.json`: Rule miner report
- `artefacts/lgbm_model.pkl`, `artefacts/tft_model.ckpt`: Trained models
- `artefacts/pnl_curves.png`, `artefacts/pnl_stats.json`: Walk-forward PnL curves and stats

### 4. **Test Models with Paper Trading**
```bash
python "Paper Trading/start_all.py"
```
- Launches the live paper trading engine and web dashboard
- Connects to Kraken for live BTC data
- Runs the trained model(s) in real time
- Simulates trades, risk, and account management
- Web dashboard at [http://localhost:5000](http://localhost:5000):
  - Live BTC price
  - Account balance
  - Open positions
  - Trade log
  - Equity curve, PnL per trade, analytics
  - Downloadable trade log

---

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

---

## Prerequisites
- Python 3.10
- Git
- pip (Python package manager)

### Installation

1. Create and activate a virtual environment (Must have Python 3.10 installed):
```bash
cd Algorithm1
py -3.10 -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Unix/MacOS:
source venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Processing Details
- Processes data in chunks for memory efficiency
- Handles timezones (all UTC)
- Memory optimization (float32)
- Error handling and logging
- Progress tracking for long operations

## Customization
- **Configuration:** Edit `config.yaml` for model, rule mining, and backtesting settings
- **Feature Engineering:** Add/modify indicators in `src/features/`
- **Model Training:** Adjust hyperparameters, add models, change metrics

## Troubleshooting
- Memory errors: Reduce batch size in config
- Training time: Adjust number of trials
- Data issues: Check data preprocessing
- Logs: See `logs/` for details

## Dependencies
- Python 3.10+
- pandas, numpy, pyarrow, requests, logging, lightgbm, pytorch, pytorch-forecasting, optuna, scikit-learn

## License
This project is licensed under JT Capital LLC. All rights reserved.
