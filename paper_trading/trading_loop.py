import logging
import time
from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import redis
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingLoop:
    def __init__(self, config: Dict, data_processor: Any = None,
                 risk_manager: Any = None, order_manager: Any = None):
        """Initialize trading loop with configuration and components
        
        Args:
            config: Configuration dictionary
            data_processor: Data processor instance
            risk_manager: Risk manager instance
            order_manager: Order manager instance
        """
        self.config = config
        self.data_processor = data_processor
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.running = False
        
        # Load trading settings
        self.trading_settings = self.config.get('trading', {})
        
        # Initialize state
        self.symbol = 'BTC/USD'  # Only trade BTC
        self.positions = {}
        self.trade_history = []
        
        # Initialize Redis connection for storing portfolio data
        redis_config = self.config.get('monitoring', {}).get('redis', {})
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0)
        )
        
        # Initialize portfolio tracking
        self.initial_balance = self.config.get('account', {}).get('initial_balance', 1000000.0)
        self.current_balance = self.initial_balance
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        
        logger.info("Trading loop initialized for BTC/USD")
        
    def start(self):
        """Start the trading loop"""
        if self.running:
            logger.warning("Trading loop is already running")
            return
            
        try:
            self.running = True
            logger.info("Starting trading loop...")
            
            # Initialize Redis with initial portfolio data
            self._update_redis_portfolio_data()
            
            # Start trading loop
            while self.running:
                try:
                    logger.info("\n=== Starting new trading scan ===")
                    
                    # Get latest data
                    price_data = self.data_processor.get_price_data(self.symbol)
                    if not price_data:
                        logger.warning("No price data available for BTC/USD")
                        time.sleep(1)
                        continue
                        
                    # Get latest bar
                    latest_bar = price_data[-1]
                    current_price = latest_bar['close']
                    logger.info(f"Price: {current_price:.2f}")
                    
                    # Get predictions
                    predictions = self.data_processor.get_predictions(self.symbol)
                    if not predictions:
                        logger.warning("No predictions available for BTC/USD")
                        time.sleep(1)
                        continue
                        
                    logger.info(f"Model prediction: {predictions['probability']:.2%}")
                    
                    # Calculate position size using account balance from configuration
                    position_size_usd = self.risk_manager.calculate_position_size(
                        self.symbol,
                        predictions,
                        self.current_balance  # Use current balance
                    )
                    logger.info(f"Risk model position size (USD): {position_size_usd:.4f}")
                    
                    # Convert USD position size to BTC quantity
                    position_size_btc = position_size_usd / current_price if current_price > 0 else 0.0
                    logger.info(f"Position size (BTC): {position_size_btc:.6f}")
                    
                    # Safety check: Limit position size to maximum 50% of account balance (with leverage this is reasonable)
                    max_position_usd = self.current_balance * 0.50  # 50% max position for leveraged account
                    max_position_btc = max_position_usd / current_price if current_price > 0 else 0.0
                    
                    if position_size_btc > max_position_btc:
                        logger.warning(f"Position size {position_size_btc:.6f} BTC exceeds maximum {max_position_btc:.6f} BTC (50% of account)")
                        position_size_btc = max_position_btc
                        logger.info(f"Position size capped at: {position_size_btc:.6f} BTC")
                    
                    # Get current position
                    current_position = self.positions.get(self.symbol, 0.0)
                    logger.info(f"Current position: {current_position:.6f}")
                    
                    # Calculate current PnL if we have a position
                    if current_position != 0:
                        # Get average entry price (simplified - you might want to track this properly)
                        entry_price = 105000.0  # Placeholder - should track actual entry price
                        current_pnl = current_position * (current_price - entry_price)
                        pnl_percentage = (current_pnl / (abs(current_position) * entry_price)) * 100
                        logger.info(f"Current PnL: ${current_pnl:,.2f} ({pnl_percentage:+.2f}%)")
                        
                        # Update total PnL
                        self.total_pnl = current_pnl
                        
                        # Update daily PnL
                        today = datetime.now().date()
                        if self.last_trade_date != today:
                            self.daily_pnl = current_pnl
                            self.last_trade_date = today
                        else:
                            self.daily_pnl = current_pnl
                    
                    # Check if we should trade
                    if abs(position_size_btc - current_position) > 0.0001:  # Small threshold to avoid dust trades
                        # Determine trade side
                        if position_size_btc > current_position:
                            side = 'buy'
                            size = position_size_btc - current_position
                        else:
                            side = 'sell'
                            size = current_position - position_size_btc
                        
                        # Minimum trade size check (0.01 BTC or $1000 equivalent for $1M account)
                        min_trade_btc = max(0.01, 1000.0 / current_price) if current_price > 0 else 0.01
                        
                        if size < min_trade_btc:
                            logger.info(f"Trade size {size:.6f} BTC below minimum {min_trade_btc:.6f} BTC, skipping trade")
                            continue
                            
                        logger.info(f"Signal: {side.upper()} {size:.6f} BTC")
                        
                        # Place order
                        order = self.order_manager.place_order(
                            self.symbol,
                            side,
                            size,
                            current_price
                        )
                        
                        if order:
                            # Update position
                            self.positions[self.symbol] = position_size_btc
                            
                            # Update trade history
                            self.trade_history.append(order)
                            
                            # Update portfolio data
                            self._update_portfolio_after_trade(order, current_price)
                            
                            logger.info(f"Order executed: {side.upper()} {size:.6f} BTC @ {current_price:.2f}")
                        else:
                            logger.warning("Order failed for BTC/USD")
                    else:
                        logger.info("No trade needed for BTC/USD")
                    
                    # Update Redis with current portfolio data
                    self._update_redis_portfolio_data()
                        
                    logger.info("\n=== Trading scan complete ===")
                    
                    # Sleep for trading interval
                    interval = self.trading_settings.get('interval', 15)
                    logger.info(f"Waiting {interval} seconds until next scan...")
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error("Error in trading loop: %s", str(e))
                    time.sleep(1)  # Brief pause on error
                    
        except Exception as e:
            logger.error("Error starting trading loop: %s", str(e))
            self.running = False
            raise
            
    def _update_portfolio_after_trade(self, order: Dict, current_price: float):
        """Update portfolio data after a trade"""
        try:
            # Calculate trade value - use 'size' field from order
            trade_value = order['size'] * current_price
            
            # Update current balance (simplified - in reality you'd account for fees, slippage, etc.)
            if order['side'] == 'buy':
                self.current_balance -= trade_value
            else:  # sell
                self.current_balance += trade_value
                
            # Update PnL
            if order.get('pnl'):
                self.total_pnl += order['pnl']
                self.daily_pnl += order['pnl']
                
        except Exception as e:
            logger.error("Error updating portfolio after trade: %s", str(e))
            
    def _update_redis_portfolio_data(self):
        """Update Redis with current portfolio data"""
        try:
            # Calculate portfolio value (current balance + position value)
            current_position = self.positions.get(self.symbol, 0.0)
            position_value = 0.0
            
            if current_position != 0:
                # Get current price for position valuation
                price_data = self.data_processor.get_price_data(self.symbol)
                if price_data:
                    current_price = price_data[-1]['close']
                    position_value = current_position * current_price
            
            portfolio_value = self.current_balance + position_value
            
            # Get true trade history (round-trips with realized PnL)
            trade_history = self.order_manager.get_trade_history()[-50:]
            
            # Store essential portfolio data in Redis
            portfolio_data = {
                'portfolio_value': portfolio_value,
                'account_balance': self.current_balance,
                'total_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'positions': self.positions,
                'trade_history': trade_history,  # Only round-trip trades
                'timestamp': datetime.now().isoformat()
            }
            
            # Store each key separately for easy access
            self.redis_client.set('portfolio_value', portfolio_value)
            self.redis_client.set('account_balance', self.current_balance)
            self.redis_client.set('total_pnl', self.total_pnl)
            self.redis_client.set('daily_pnl', self.daily_pnl)
            self.redis_client.set('positions', json.dumps(self.positions))
            self.redis_client.set('trade_history', json.dumps(trade_history))
            self.redis_client.set('last_update', datetime.now().isoformat())
            
            # Store complete portfolio data
            self.redis_client.set('portfolio_data', json.dumps(portfolio_data))
            
        except Exception as e:
            logger.error("Error updating Redis portfolio data: %s", str(e))
            
    def stop(self):
        """Stop the trading loop"""
        if not self.running:
            logger.warning("Trading loop is not running")
            return
            
        try:
            self.running = False
            logger.info("Trading loop stopped")
        except Exception as e:
            logger.error("Error stopping trading loop: %s", str(e))
            raise
            
    def get_positions(self) -> Dict[str, float]:
        """Get current positions
        
        Returns:
            Dictionary of symbol to position size
        """
        return self.positions.copy()
        
    def get_trade_history(self) -> List[Dict]:
        """Get trade history
        
        Returns:
            List of trade dictionaries
        """
        return self.trade_history.copy() 