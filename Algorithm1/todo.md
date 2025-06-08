# Algorithm1 Development Plan

## System Requirements

1. **Data Loading & Processing**
   - Load five parquet files (15 sec, 1 min, 15 min, 1 h, 4 h) with indicator columns
   - Clean and standardize data
   - Ensure UTC timestamps, no gaps, no NaNs
   - Standardize column names across files

2. **Multi-Timeframe Integration**
   - Blend timeframes together
   - For each 15-second candle, attach higher timeframe data:
     - 1 min / 15 min / 1 h / 4 h chart signals
     - Higher-TF trend flags
     - SMC/BOS tags
   - Create additional features:
     - Rolling statistics
     - Time-since-event counters
     - Cross-indicator combinations

3. **Labeling System**
   - Look ahead fixed horizon (30 minutes)
   - Labeling rules:
     - LongSuccess: Price moves +$500 before -$500
     - ShortSuccess: Price drops -$500 first
   - Track move size and duration

4. **Pattern Analysis**
   - Pattern hunting in features preceding labeled moves
   - Two approaches:
     - Rule-based: Market structure "if-this-and-that" combinations
     - ML-based: LightGBM for non-obvious combinations

5. **Backtesting & Reporting**
   - Backtest signals with costs/slippage
   - Generate equity curves & statistics
   - Create readable notebook/dashboard with:
     - Feature importance charts
     - Confusion matrix
     - P&L curve

## Implementation Tasks

### 1. Data Ingestion
- **File**: `src/ingestion/load_multitf.py`
- **Tasks**:
  - Helper to open all parquet files, return dict[str, DataFrame]
  - Ensure datetime index tz-aware UTC
  - Unit test: confirm shapes/columns match

### 2. Cleaning & Alignment
- **Files**:
  - `src/cleaning/standardise_columns.py`
    - Map indicator column names to snake_case
    - Convert BOOL/TEXT flags to int8
  - `src/cleaning/fill_gaps.py`
    - Detect missing 15s intervals
    - Forward-fill OHLCV except price columns
  - `src/cleaning/drop_nan_cols.py`
    - Drop columns > 95% NaN after fill
  - Unit tests for each utility

### 3. Multi-timeframe Join
- **File**: `src/feature_engineering/join_timeframes.py`
- **Tasks**:
  - Merge latest available row from higher TFs
  - Suffix high-TF columns (_1m, _15m, etc.)
  - Optional look-ahead leakage guard
  - Performance benchmark for stream-merge

### 4. Feature Generation
- **Files**:
  - `src/feature_engineering/rolling_stats.py`
    - Rolling mean, std, z-score (N = 20, 60, 120 bars)
  - `src/feature_engineering/event_counters.py`
    - Bars since last SMC BOS
    - Bars since last vector-candle
  - `src/feature_engineering/time_features.py`
    - Session encoding (Asia/EU/NY)
    - Cyclic time features
  - Config file: `config/features.yaml`

### 5. Move Labeling
- **File**: `src/labelling/label_moves.py`
- **Parameters**:
  - target_usd = 500
  - horizon_secs = 1800
  - slippage = 0
- **Tasks**:
  - Label generation (+1, -1, 0)
  - Track time-to-move and move size
  - Unit test with synthetic series

### 6. Dataset Builder
- **File**: `src/dataset/build_dataset.py`
- **Pipeline**:
  - ingest → clean → join TF → feature gen → label
  - Save compressed parquet
- **CLI**: `python -m dataset.build_dataset --config config/build.yml`

### 7. Exploratory Analysis
- **File**: `notebooks/01_eda.ipynb`
- **Analysis**:
  - Class balance
  - Correlations
  - Null heatmap
  - Target vs top-20 features

### 8. Rule-based Search
- **File**: `src/models/rule_miner.py`
- **Tasks**:
  - Use mlxtend.frequent_patterns
  - Mine association rules
  - Export to rules.json

### 9. ML Pipeline
- **File**: `src/models/train_lightgbm.py`
- **Tasks**:
  - Train/val/test split by date
  - LightGBM classifier with optuna optimization
  - Output: model.pkl, feature_importance.csv
  - Metrics: ROC-AUC, precision@top_K, PR-curve

### 10. Backtesting
- **File**: `src/backtest/simulate.py`
- **Tasks**:
  - Walk-forward on 15s stream
  - Position entry/exit logic
  - Track P&L, drawdown, Sharpe
  - Store trades to trades.parquet
  - Generate summary JSON

### 11. Reporting
- **Files**:
  - `notebooks/02_report.ipynb`
  - `src/reports/make_dashboard.py`
- **Outputs**:
  - Equity curve
  - Confusion matrix
  - Rule list table
  - SHAP bar chart