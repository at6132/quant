# Intelligent Trading System

An advanced quantitative trading system that learns trading intuition from Smart Money Concepts (SMC) indicators across multiple timeframes. The system uses a Temporal Fusion Transformer (TFT) to learn patterns and rules from data rather than relying on hardcoded thresholds.

## 🚀 Features

- **Multi-timeframe Analysis**: 15s, 1m, 5m, 15m, 1h, 4h timeframes
- **Smart Money Concepts**: IT Foundation, SMC Core, Breaker Signals, ICT Smart Money Trades
- **Intelligent Model**: Temporal Fusion Transformer with multi-output predictions
- **Adaptive Risk Management**: Bayesian Kelly sizing, volatility scaling, conditional Martingale
- **Dynamic Position Management**: One position at a time with dynamic TP/SL adjustments
- **Rule Extraction**: Interpretable trading rules from learned patterns

## 📋 Prerequisites

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (RTX 5070+ recommended)
- Windows 10/11 or Linux
- Git

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd quant
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎯 Training the Model

### Quick Start (7 days for testing)

```bash
cd Algorithm1
python trainmodel.py --symbol BTCUSDT --days 7
```

### Optimal Training (1-2 years for production)

```bash
# 1 year training (recommended)
python trainmodel.py --symbol BTCUSDT --days 365

# 2 years training (maximum performance)
python trainmodel.py --symbol BTCUSDT --days 730
```

### Multi-Symbol Training

```bash
python trainmodel.py --symbol BTCUSDT --days 365
python trainmodel.py --symbol ETHUSDT --days 365
python trainmodel.py --symbol SOLUSDT --days 365
python trainmodel.py --symbol ADAUSDT --days 365
```

## ⚙️ Configuration Options

### Command Line Arguments

```bash
python trainmodel.py [OPTIONS]

Options:
  --symbol TEXT     Trading symbol (default: BTCUSDT)
  --days INTEGER    Number of days to download (default: 30)
  --config TEXT     Config file path (default: config/intelligent_config.yaml)
```

### Supported Symbols

- **Crypto**: BTCUSDT, ETHUSDT, SOLUSDT, ADAUSDT, DOTUSDT, LINKUSDT

### Training Duration Guide

| Duration | Use Case | Training Time | Data Points |
|----------|----------|---------------|-------------|
| 7 days | Testing/Development | 30-60 min | ~400K |
| 30 days | Quick Validation | 2-4 hours | ~1.7M |
| 90 days | Initial Model | 6-8 hours | ~5M |
| 365 days | Production Ready | 8-12 hours | ~21M |
| 730 days | Maximum Performance | 16-24 hours | ~42M |

## 📊 Model Architecture

### Temporal Fusion Transformer (TFT)
- **Input**: Multi-timeframe features (200+ indicators)
- **Output**: 10 predictions per timestep
  - Entry probability & direction
  - Position size multiplier
  - Take profit distance
  - Stop loss distance
  - Trailing stop aggressiveness
  - Hold duration
  - Exit probability & urgency

### Training Configuration (Optimized for RTX 5070)
- **Hidden Size**: 512
- **Attention Heads**: 16
- **Batch Size**: 2048
- **Sequence Length**: 300
- **Learning Rate**: 0.0001
- **Max Epochs**: 500
- **Mixed Precision**: Enabled

## 📈 Expected Performance

With 1-2 years of training data:
- **Validation Accuracy**: 65-75%
- **Sharpe Ratio**: 1.5-2.5
- **Win Rate**: 55-65%
- **Maximum Drawdown**: <15%

## 📁 Project Structure

```
Algorithm1/
├── data/                   # Data processing modules
├── features/              # Feature engineering
├── models/                # Model definitions
├── indicators/            # SMC indicators
├── risk/                  # Risk management
├── artefacts/             # Training outputs
├── logs/                  # Training logs
├── config/                # Configuration files
└── trainmodel.py          # Main training script
```

## 🔍 Monitoring Training

### GPU Usage
```bash
nvidia-smi -l 1
```

### Training Progress
```bash
# Windows
Get-Content Algorithm1/logs/training.log -Tail 10 -Wait

# Linux
tail -f Algorithm1/logs/training.log
```

### Model Artifacts
```bash
# Check saved models
ls Algorithm1/artefacts/
```

## 🚧 Known Issues

- **Paper Trading Module**: Currently broken, will be fixed in next update
- **Live Trading**: Not yet implemented
- **Web Interface**: Under development

## 🛡️ Risk Disclaimer

This is experimental software for educational purposes. Trading involves substantial risk of loss. Never risk more than you can afford to lose. Past performance does not guarantee future results.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the logs in `Algorithm1/logs/`
- Review the configuration in `Algorithm1/config/`
- Ensure your GPU has sufficient VRAM (8GB+ recommended)

## 📄 License

This project is for educational purposes only. Use at your own risk. Licensed under JT Captial LLC
