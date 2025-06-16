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
python "paper_trading/start_all.py"
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

### 5. **Start Everything Together (Web Dashboard + Paper Trading Engine)**

To launch both the web dashboard and the paper trading engine in one step, use the provided script:

```bash
python paper_trading/start_all.py
```

- This will start the Flask web dashboard at [http://localhost:5000](http://localhost:5000)
- It will also start the paper trading engine with $1,000,000 simulated capital, using live market data and your trained model.
- Both modules run in parallel and communicate for real-time updates.
- Stop both with `Ctrl+C` in the terminal.

This is the recommended way to run a full end-to-end paper trading simulation with live data, adaptive risk management, and real-time monitoring.

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
- Python 3.10
- pandas, numpy, pyarrow, requests, logging, lightgbm, pytorch, pytorch-forecasting, optuna, scikit-learn

## License
This project is licensed under JT Capital LLC. All rights reserved.

## Risk Model

The system uses an advanced adaptive risk model, as described in [adaptive_risk_model.md](adaptive_risk_model.md). This model combines Bayesian Kelly sizing, volatility-weighted exposure scaling, conditional Martingale reinforcement, and a stochastic drawdown barrier to dynamically adjust risk levels based on statistical confidence and market turbulence.

### Key Components:
- **Bayesian Kelly Fraction:** Uses a Beta posterior to shrink the Kelly fraction under uncertainty.
- **Volatility-Weighted Buffer:** Scales position size using a 95th percentile volatility multiplier.
- **Conditional Martingale Escalation:** Only applies position compounding when edge is statistically significant (e.g., probability > 0.97).
- **Stochastic Drawdown Barrier:** Uses an Ornstein-Uhlenbeck (OU) process to throttle risk when equity decays.

### Testing the Risk Model:
To test the risk model, run the test script from the root directory of the project:
```bash
pytest test_risk_engine.py
```
This script covers all components of the risk engine, including:
- Posterior Kelly fraction calculation
- Volatility multiplier
- OU drawdown barrier
- Position sizing constraints

For more details, refer to [adaptive_risk_model.md](adaptive_risk_model.md).

### Integration and Testing
The adaptive risk model is fully integrated with the trading module and trading model. The test suite (`test_risk_engine.py`) passes 100%, confirming that the risk model is robust and correctly implemented. This integration ensures that every trade is governed by the research-grade risk engine, providing dynamic risk allocation based on statistical confidence and market conditions.

## Redis Setup for Paper Trading Module

The Paper Trading module uses Redis to share live state (price, account balance, trades, etc.) between the trading engine and the web dashboard. **Redis must be running for the system to work.**

### 1. Download and Install Redis

#### **Windows:**
- Download the latest Redis for Windows from: https://github.com/tporadowski/redis/releases
- Extract the zip and run `redis-server.exe` (double-click or from command line).

#### **MacOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

#### **Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl enable redis-server.service
sudo systemctl start redis-server.service
```

### 2. Start the Redis Server
- Make sure the Redis server is running before you start the paper trading engine or web dashboard.
- By default, the system connects to `localhost:6379`.

### 3. Python Redis Client
- The required Python package (`redis`) is already included in `requirements.txt`.

### 4. Configuration
- No extra configuration is needed for local development. If you want to use a remote Redis server, edit the connection settings in `webapp.py` and `paper_trader.py`.

### 5. Troubleshooting
- If you see connection errors, make sure Redis is running and accessible on `localhost:6379`.
- You can test Redis with the command: `redis-cli ping` (should return `PONG`).

---
