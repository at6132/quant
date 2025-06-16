# Paper Trading System

A comprehensive paper trading system with real-time monitoring, risk management, and portfolio analytics.

## Features

- Real-time order management with Redis
- Comprehensive risk management system
- Portfolio analytics and performance metrics
- Real-time monitoring and alerting
- Web dashboard for visualization
- Configurable trading parameters
- Support for multiple trading strategies

## Prerequisites

- Python 3.8+
- Redis server
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd paper_trading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Redis:
- Install Redis server
- Start Redis server:
```bash
redis-server
```

5. Configure the system:
- Copy `paper_trading_config.yaml.example` to `paper_trading_config.yaml`
- Update configuration parameters as needed

## Usage

1. Start the paper trading system:
```bash
python start_all.py
```

This will start:
- Web dashboard on http://localhost:5000
- Paper trader
- Risk engine
- Data processor
- Monitoring system

2. Monitor the system:
- Web dashboard: http://localhost:5000
- Logs: `logs/paper_trading.log`
- Redis: Monitor using Redis CLI or GUI

3. Stop the system:
- Press Ctrl+C to gracefully shut down all components

## Configuration

The system is configured through `paper_trading_config.yaml`. Key sections:

- `redis`: Redis connection settings
- `account`: Account parameters (capital, leverage, etc.)
- `trading`: Trading parameters (slippage, commission, etc.)
- `risk_management`: Risk limits and position constraints
- `monitoring`: Logging and alert settings

## Components

### Order Manager
- Handles order creation and execution
- Manages positions and margin accounts
- Simulates realistic fills and slippage

### Risk Manager
- Enforces risk limits
- Manages position sizing
- Handles margin requirements

### Portfolio Analytics
- Calculates performance metrics
- Tracks risk metrics
- Generates reports

### Monitoring System
- Real-time logging
- Alert generation
- Performance monitoring
- Market regime detection

## Development

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

3. Code formatting:
```bash
black .
flake8
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 