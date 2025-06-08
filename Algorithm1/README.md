# Algorithm1 - Quantitative Trading System

A comprehensive quantitative trading system for cryptocurrency (BTC/USDT) that analyzes multi-timeframe data to identify patterns preceding $500 price moves.

## Features

- **Multi-Timeframe Analysis**: Processes 15-second, 1-minute, 15-minute, 1-hour, and 4-hour timeframes
- **Advanced Labeling**: Identifies which direction hits $500 threshold first within a configurable horizon
- **Feature Engineering**: 
  - Price-based features (returns, volatility, z-scores)
  - Momentum indicators (RSI, MACD, Stochastic)
  - Market microstructure (spread proxy, Kyle's lambda)
  - Event-based features (time since signals, event counts)
  - Time features (sessions, cyclical encoding)
- **Pattern Mining**:
  - Rule-based pattern discovery
  - LightGBM with Bayesian optimization
  - Placeholder for Temporal Fusion Transformer
- **Backtesting**: Comprehensive backtesting with transaction costs and slippage

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place parquet files in `processed_data/` directory
   - Files should be named: `15second.parquet`, `1minute.parquet`, etc.
   - Each file should contain OHLCV data and indicator columns

3. Or use the test script to generate sample data:
```bash
python test_system.py
```

## Usage

### Run the complete system:
```bash
python main.py -c config.yaml
```

### Configuration

Edit `config.yaml` to adjust:
- Data directory path
- Labeling parameters (horizon, threshold)
- Train/test split
- Model hyperparameters

### Output

The system generates:
- `artefacts/feature_matrix.parquet`: Complete feature matrix
- `artefacts/backtest_results.pkl`: Backtest results for all models
- Console output with performance metrics
- Equity curves for each strategy

## System Architecture

```
Algorithm1/
├── quantsys/              # Core system modules
│   ├── data_loader.py     # Multi-timeframe data loading
│   ├── labeller.py        # Future move labeling
│   ├── backtester.py      # Backtesting engine
│   ├── features/          # Feature engineering
│   │   └── base.py        # Feature creation
│   └── miners/            # Pattern mining
│       ├── rule_miner.py  # Rule-based patterns
│       ├── lgbm_model.py  # LightGBM model
│       └── tft_model.py   # TFT placeholder
├── main.py                # Main orchestrator
├── test_system.py         # Test data generator
└── config.yaml            # Configuration

```

## Performance Metrics

The backtester calculates:
- Total Return & Annualized Return
- Sharpe Ratio & Calmar Ratio
- Maximum Drawdown
- Win Rate & Profit Factor
- Average Win/Loss
- Number of Trades

## Data Requirements

Each parquet file should contain:
- **Index**: DateTime index (UTC timezone)
- **OHLCV**: open, high, low, close, volume columns
- **Indicators**: Any additional indicator columns (will be automatically detected)

## Extending the System

1. **Add new features**: Modify `quantsys/features/base.py`
2. **Add new miners**: Create new modules in `quantsys/miners/`
3. **Modify labeling**: Update `quantsys/labeller.py`
4. **Custom backtesting**: Extend `quantsys/backtester.py`

## Notes

- The system automatically handles look-ahead bias when merging timeframes
- Features are forward-filled to handle missing data
- Columns with >95% NaN values are automatically dropped
- Boolean columns are converted to int8 for efficiency

## Future Improvements

- [ ] Implement walk-forward optimization
- [ ] Add portfolio optimization
- [ ] Implement Temporal Fusion Transformer
- [ ] Add real-time trading capabilities
- [ ] Add more sophisticated risk management
- [ ] Create interactive dashboard for results