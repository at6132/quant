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

## Running the ML Pipeline

### Prerequisites
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

2. Install dependencies (Currently only support python 3.10):
```bash
pip install -r requirements.txt
```

### Running the System

1. **Data Preparation** (Already completed from steps above)
   - Data has been collected and processed
   - Available in `processed_data/` directory

2. **Run the Pipeline**:
```bash
python main.py -c config.yaml
```

This will:
- Load and preprocess the data
- Mine trading rules
- Train LightGBM model (takes ~40-50 minutes)
- Train TFT model
- Generate predictions
- Save results

3. **Expected Output**:
   - In `logs/`: Detailed training logs
   - In `models/`: Trained model files
   - In `results/`: Performance metrics and predictions
   - In `artefacts/`: Generated trading rules and visualizations

### Using the Results

1. **Trading Rules** (`artefacts/rules/`):
   - Review mined rules in `rules.json`
   - Each rule contains:
     - Indicator combinations
     - Precision and recall metrics
     - Support level

2. **Model Predictions** (`results/predictions/`):
   - LightGBM predictions: `lgbm_predictions.csv`
   - TFT predictions: `tft_predictions.csv`
   - Combined predictions: `ensemble_predictions.csv`

3. **Performance Metrics** (`results/metrics/`):
   - Accuracy, precision, recall, F1 scores
   - Confusion matrices
   - ROC curves
   - Performance by timeframe

4. **Visualizations** (`artefacts/plots/`):
   - Rule performance plots
   - Model prediction plots
   - Feature importance plots
   - Performance comparison plots

## Data Processing Details

The system processes data in chunks to manage memory efficiently and includes:
- Automatic timezone handling (all timestamps in UTC)
- Memory optimization (float32 data types)
- Error handling and logging
- Progress tracking for long operations

## Customization

1. **Configuration** (`config.yml`):
   - Adjust model parameters
   - Modify rule mining thresholds
   - Change backtesting settings
   - Update logging levels

2. **Feature Engineering**:
   - Add new indicators in `src/features/`
   - Modify feature combinations
   - Adjust timeframes

3. **Model Training**:
   - Modify hyperparameter ranges
   - Add new models
   - Change evaluation metrics

## Troubleshooting

1. **Common Issues**:
   - Memory errors: Reduce batch size in config
   - Training time: Adjust number of trials
   - Data issues: Check data preprocessing

2. **Logs**:
   - Check `logs/` for detailed error messages
   - Review model training progress
   - Monitor system performance

## Dependencies

- Python 3.8+
- pandas
- numpy
- pyarrow
- requests (for data download)
- logging
- lightgbm
- pytorch
- pytorch-forecasting
- optuna
- scikit-learn

## Notes

- All timestamps are in UTC
- Data is processed in chunks for memory efficiency
- Each indicator's output is prefixed with its name (e.g., 'bb_' for Bollinger Bands)
- The system automatically handles data type conversions for parquet storage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under JT Capital LLC. All rights reserved.
