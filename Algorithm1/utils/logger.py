"""
Logging utilities for the intelligent trading system.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = "intelligent_trading",
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # Main log file
        main_log_file = log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_path / f"{name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    return logger

def get_logger(name: str = "intelligent_trading") -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        setup_logger(name)
    
    return logger

def log_trade(logger: logging.Logger, trade_data: dict):
    """
    Log trade information in a structured way.
    
    Args:
        logger: Logger instance
        trade_data: Dictionary containing trade information
    """
    logger.info("=== TRADE EXECUTED ===")
    logger.info(f"Symbol: {trade_data.get('symbol', 'N/A')}")
    logger.info(f"Side: {trade_data.get('side', 'N/A')}")
    logger.info(f"Size: {trade_data.get('size', 'N/A')}")
    logger.info(f"Price: {trade_data.get('price', 'N/A')}")
    logger.info(f"Timestamp: {trade_data.get('timestamp', 'N/A')}")
    logger.info(f"Reason: {trade_data.get('reason', 'N/A')}")
    logger.info("=====================")

def log_position_update(logger: logging.Logger, position_data: dict):
    """
    Log position update information.
    
    Args:
        logger: Logger instance
        position_data: Dictionary containing position information
    """
    logger.info("=== POSITION UPDATE ===")
    logger.info(f"Symbol: {position_data.get('symbol', 'N/A')}")
    logger.info(f"Size: {position_data.get('size', 'N/A')}")
    logger.info(f"Entry Price: {position_data.get('entry_price', 'N/A')}")
    logger.info(f"Current Price: {position_data.get('current_price', 'N/A')}")
    logger.info(f"P&L: {position_data.get('pnl', 'N/A')}")
    logger.info(f"Stop Loss: {position_data.get('stop_loss', 'N/A')}")
    logger.info(f"Take Profit: {position_data.get('take_profit', 'N/A')}")
    logger.info("======================")

def log_model_prediction(logger: logging.Logger, prediction_data: dict):
    """
    Log model prediction information.
    
    Args:
        logger: Logger instance
        prediction_data: Dictionary containing prediction information
    """
    logger.info("=== MODEL PREDICTION ===")
    logger.info(f"Entry Probability: {prediction_data.get('entry_probability', 'N/A'):.4f}")
    logger.info(f"Entry Direction: {prediction_data.get('entry_direction', 'N/A')}")
    logger.info(f"Entry Confidence: {prediction_data.get('entry_confidence', 'N/A'):.4f}")
    logger.info(f"Position Size Multiplier: {prediction_data.get('position_size_multiplier', 'N/A'):.4f}")
    logger.info(f"TP Distance: {prediction_data.get('tp_distance', 'N/A'):.4f}")
    logger.info(f"SL Distance: {prediction_data.get('sl_distance', 'N/A'):.4f}")
    logger.info(f"Exit Probability: {prediction_data.get('exit_probability', 'N/A'):.4f}")
    logger.info("========================")

def log_risk_update(logger: logging.Logger, risk_data: dict):
    """
    Log risk management information.
    
    Args:
        logger: Logger instance
        risk_data: Dictionary containing risk information
    """
    logger.info("=== RISK UPDATE ===")
    logger.info(f"Account Balance: {risk_data.get('account_balance', 'N/A'):.2f}")
    logger.info(f"Position Size: {risk_data.get('position_size', 'N/A'):.6f}")
    logger.info(f"Risk Fraction: {risk_data.get('risk_fraction', 'N/A'):.4f}")
    logger.info(f"Volatility Multiplier: {risk_data.get('volatility_multiplier', 'N/A'):.4f}")
    logger.info(f"Drawdown: {risk_data.get('drawdown', 'N/A'):.4f}")
    logger.info(f"Max Drawdown: {risk_data.get('max_drawdown', 'N/A'):.4f}")
    logger.info("===================")

def log_performance_metrics(logger: logging.Logger, metrics: dict):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary containing performance metrics
    """
    logger.info("=== PERFORMANCE METRICS ===")
    logger.info(f"Total Return: {metrics.get('total_return', 'N/A'):.4f}")
    logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
    logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")
    logger.info(f"Win Rate: {metrics.get('win_rate', 'N/A'):.4f}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 'N/A'):.4f}")
    logger.info(f"Total Trades: {metrics.get('total_trades', 'N/A')}")
    logger.info("===========================")

class PerformanceLogger:
    """
    Logger specifically for performance tracking.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Performance log file
        self.performance_log = self.log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        handler = logging.FileHandler(self.performance_log)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def log_equity_curve(self, timestamp: datetime, equity: float, drawdown: float):
        """
        Log equity curve data point.
        
        Args:
            timestamp: Timestamp
            equity: Current equity
            drawdown: Current drawdown
        """
        self.logger.info(f"EQUITY_CURVE,{timestamp.isoformat()},{equity:.2f},{drawdown:.4f}")
    
    def log_trade_result(self, trade_id: str, entry_time: datetime, exit_time: datetime, 
                        entry_price: float, exit_price: float, size: float, pnl: float):
        """
        Log individual trade result.
        
        Args:
            trade_id: Trade identifier
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            pnl: Profit/loss
        """
        self.logger.info(f"TRADE_RESULT,{trade_id},{entry_time.isoformat()},{exit_time.isoformat()},"
                        f"{entry_price:.2f},{exit_price:.2f},{size:.6f},{pnl:.2f}")
    
    def log_daily_summary(self, date: datetime, daily_return: float, daily_pnl: float, 
                         trades_count: int, win_rate: float):
        """
        Log daily summary.
        
        Args:
            date: Date
            daily_return: Daily return
            daily_pnl: Daily P&L
            trades_count: Number of trades
            win_rate: Daily win rate
        """
        self.logger.info(f"DAILY_SUMMARY,{date.strftime('%Y-%m-%d')},{daily_return:.4f},"
                        f"{daily_pnl:.2f},{trades_count},{win_rate:.4f}") 