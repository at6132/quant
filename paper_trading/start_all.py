#!/usr/bin/env python3
"""
Start script for the complete paper trading system
"""

import os
import sys
import yaml
import logging
import time
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_trading.data_processor import DataProcessor
from paper_trading.risk_manager import RiskManager
from paper_trading.order_manager import OrderManager
from paper_trading.trading_loop import TradingLoop
from paper_trading.monitoring import MonitoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / 'paper_trading_config.yaml'
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def main():
    """Main function to start the trading system"""
    logger.info("Starting paper trading system...")
    
    # Load configuration
    config = load_config()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing trading components...")
        
        # Data processor
        data_processor = DataProcessor(config)
        logger.info("Data processor initialized")
        
        # Start data processor
        data_processor.start()
        logger.info("Data processor started")
        
        # Risk manager
        risk_manager = RiskManager(config)
        logger.info("Risk manager initialized")
        
        # Order manager
        order_manager = OrderManager(config)
        logger.info("Order manager initialized")
        
        # Trading loop
        trading_loop = TradingLoop(config, data_processor, risk_manager, order_manager)
        logger.info("Trading loop initialized")
        
        # Monitoring system
        monitoring = MonitoringSystem(config, data_processor, trading_loop)
        logger.info("Monitoring system initialized")
        
        # Start monitoring in background
        logger.info("Starting monitoring system...")
        monitoring.start()
        
        # Start trading loop
        logger.info("Starting trading loop...")
        trading_loop.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        # Cleanup
        try:
            if 'trading_loop' in locals():
                trading_loop.stop()
            if 'monitoring' in locals():
                monitoring.stop()
            if 'data_processor' in locals():
                data_processor.stop()
            logger.info("Trading system stopped")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == "__main__":
    main() 