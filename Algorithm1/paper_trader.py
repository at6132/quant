import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yaml
from risk_manager import AdaptiveRiskManager
from monitoring import MonitoringSystem
from portfolio_analytics import PortfolioAnalytics
from order_manager import OrderManager
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperTrader:
    def __init__(self, config_path: str = "paper_trading_config.yaml"):
        """Initialize paper trader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize components
        self.order_manager = OrderManager(config_path)
        self.risk_manager = RiskManager(config_path)
        self.monitoring = MonitoringSystem(config_path)
        self.portfolio_analytics = PortfolioAnalytics(config_path)
        
        # Initialize state
        self.positions = {}
        self.trades = []
        self.current_prices = {}
        
        # Load account settings
        self.initial_capital = self.config['account']['initial_capital']
        self.max_leverage = self.config['account']['max_leverage']
        self.margin_requirement = self.config['account']['margin_requirement']
        self.maintenance_margin = self.config['account']['maintenance_margin']
        self.liquidation_threshold = self.config['account']['liquidation_threshold']
        
        # Load trading parameters
        self.min_trade_interval = self.config['trading']['min_trade_interval']
        self.max_slippage = self.config['trading']['max_slippage']
        self.commission = self.config['trading']['commission']
        self.min_profit_threshold = self.config['trading']['min_profit_threshold']
        self.max_spread = self.config['trading']['max_spread']
        self.order_timeout = self.config['trading']['order_timeout']
        self.retry_attempts = self.config['trading']['retry_attempts']
        
        logger.info("Paper trader initialized with configuration from %s", config_path)
        
        # Initialize state
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.trade_history: List[Dict] = []
        self.current_prices: Dict[str, float] = {}
        self.returns_history: List[float] = []
        self.last_signal_time: Dict[str, datetime] = {}  # Track last signal time per symbol
        self.min_signal_interval = timedelta(seconds=15)  # Minimum time between signals
        
    def update_market_data(self, data: pd.DataFrame) -> None:
        """Update market data and calculate returns"""
        if len(data) < 2:
            return
            
        # Update current prices
        self.current_prices[self.symbol] = data['close'].iloc[-1]
        
        # Calculate returns
        returns = data['close'].pct_change().dropna().values
        self.returns_history.extend(returns)
        
        # Keep only last 1000 returns for memory efficiency
        if len(self.returns_history) > 1000:
            self.returns_history = self.returns_history[-1000:]
            
    def should_process_signal(self, symbol: str, timestamp: datetime) -> bool:
        """Check if we should process this signal based on timing"""
        if symbol not in self.last_signal_time:
            return True
            
        time_since_last = timestamp - self.last_signal_time[symbol]
        return time_since_last >= self.min_signal_interval
            
    def evaluate_signal(self, 
                       label: str, 
                       probability: float, 
                       current_position: Optional[Dict] = None,
                       timestamp: datetime = None) -> Tuple[bool, float, str, str]:
        """
        Evaluate a trading signal and determine if we should act on it
        
        Args:
            label: Model's prediction ('buy', 'sell', or 'hold')
            probability: Model's confidence (0-1)
            current_position: Current position details if any
            timestamp: Signal timestamp
            
        Returns:
            Tuple[bool, float, str, str]: (should_act, position_size_pct, action, reason)
        """
        if not self.current_prices:
            return False, 0.0, 'hold', "No market data available"
            
        if not self.should_process_signal(self.symbol, timestamp):
            return False, 0.0, 'hold', "Signal too soon after last one"
            
        price = self.current_prices[self.symbol]
        direction = 1 if label == 'buy' else -1
        
        # Handle hold signal
        if label == 'hold':
            if current_position:
                return False, 0.0, 'hold', "Holding current position"
            return False, 0.0, 'hold', "No position to hold"
            
        # Calculate base position size
        position_size, risk_fraction = self.risk_manager.calculate_position_size(
            price=price,
            probability=probability,
            direction=direction,
            returns=np.array(self.returns_history)
        )
        
        if position_size <= 0:
            return False, 0.0, 'hold', "Position size too small"
            
        # Convert to percentage of capital
        position_size_pct = (position_size * price / self.risk_manager.state.capital) * 100
        
        # Check if we have an existing position
        if current_position:
            current_direction = current_position['direction']
            current_size_pct = current_position['total_position_pct']
            
            # Same direction signal
            if direction == current_direction:
                # Check if we should pyramid
                if probability > self.risk_manager.confidence_threshold:
                    # Limit total position size
                    max_additional_pct = min(
                        position_size_pct,
                        self.risk_manager.max_position_size - current_size_pct
                    )
                    if max_additional_pct > 0:
                        return True, max_additional_pct, 'buy', "Pyramiding existing position"
                    return False, 0.0, 'hold', "Maximum position size reached"
                return False, 0.0, 'hold', "Signal confidence too low for pyramiding"
                
            # Opposite direction signal
            else:
                # Determine if we should partially or fully close
                if probability > 0.8:  # High confidence for reversal
                    if current_size_pct > position_size_pct:
                        # Partial close
                        return True, position_size_pct, 'sell', "Partially closing position"
                    else:
                        # Full close and reverse
                        return True, position_size_pct, 'sell', "Closing and reversing position"
                elif probability > 0.6:  # Medium confidence
                    # Partial close
                    close_pct = min(position_size_pct, current_size_pct * 0.5)
                    return True, close_pct, 'sell', "Partially reducing position"
                return False, 0.0, 'hold', "Signal confidence too low for action"
        else:
            # New position
            if probability > 0.6:  # Minimum confidence for new position
                return True, position_size_pct, 'buy', "Opening new position"
            return False, 0.0, 'hold', "Signal confidence too low for new position"
                
    def process_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Process trading signal with adaptive position sizing
        
        Args:
            signal: Dict containing:
                - label: 'buy', 'sell', or 'hold'
                - probability: model's confidence (0-1)
                - timestamp: signal time
                
        Returns:
            Optional[Dict]: Trade execution details if signal is acted upon
        """
        if not self.current_prices:
            logger.warning("No market data available")
            return None
            
        # Get current position if any
        current_position = self.positions.get(self.symbol)
        
        # Evaluate signal
        should_act, position_size_pct, action, reason = self.evaluate_signal(
            label=signal['label'],
            probability=signal['probability'],
            current_position=current_position,
            timestamp=signal['timestamp']
        )
        
        if not should_act:
            logger.info(f"Not acting on signal: {reason}")
            return None
            
        # Update last signal time
        self.last_signal_time[self.symbol] = signal['timestamp']
            
        price = self.current_prices[self.symbol]
        probability = signal['probability']
        direction = 1 if action == 'buy' else -1
        
        # Calculate position size using adaptive risk management
        position_size, risk_fraction = self.risk_manager.calculate_position_size(
            price=price,
            probability=probability,
            direction=direction,
            returns=np.array(self.returns_history)
        )
        
        # Adjust position size based on percentage
        if action == 'sell' and current_position:
            # For sells, use the percentage to determine how much to close
            current_size = current_position['size']
            if position_size_pct < current_position['total_position_pct']:
                position_size = current_size * (position_size_pct / current_position['total_position_pct'])
        
        # Generate trade ID
        trade_id = f"{self.symbol}_{action}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute trade
        trade = {
            'id': trade_id,
            'symbol': self.symbol,
            'action': action,
            'direction': direction,
            'size': position_size,
            'price': price,
            'timestamp': signal['timestamp'],
            'probability': probability,
            'risk_fraction': risk_fraction,
            'position_size_pct': position_size_pct,
            'reason': reason
        }
        
        # Update positions
        if action == 'buy':
            if self.symbol in self.positions:
                # Pyramiding existing position
                current_pos = self.positions[self.symbol]
                new_size = current_pos['size'] + position_size
                new_cost = (current_pos['size'] * current_pos['entry_price'] + 
                           position_size * price) / new_size
                self.positions[self.symbol] = {
                    'size': new_size,
                    'entry_price': new_cost,
                    'direction': 1,
                    'total_position_pct': (new_size * price / self.risk_manager.state.capital) * 100
                }
            else:
                # New position
                self.positions[self.symbol] = {
                    'size': position_size,
                    'entry_price': price,
                    'direction': 1,
                    'total_position_pct': position_size_pct
                }
        else:  # sell
            if self.symbol in self.positions:
                # Close or reduce position
                current_pos = self.positions[self.symbol]
                if current_pos['direction'] == -1:  # Already short
                    new_size = current_pos['size'] + position_size
                    new_cost = (current_pos['size'] * current_pos['entry_price'] + 
                               position_size * price) / new_size
                    self.positions[self.symbol] = {
                        'size': new_size,
                        'entry_price': new_cost,
                        'direction': -1,
                        'total_position_pct': (new_size * price / self.risk_manager.state.capital) * 100
                    }
                else:  # Close long position
                    pnl = (price - current_pos['entry_price']) * position_size
                    trade['pnl'] = pnl
                    
                    # Update or remove position
                    if position_size >= current_pos['size']:
                        del self.positions[self.symbol]
                    else:
                        remaining_size = current_pos['size'] - position_size
                        self.positions[self.symbol] = {
                            'size': remaining_size,
                            'entry_price': current_pos['entry_price'],
                            'direction': 1,
                            'total_position_pct': (remaining_size * price / self.risk_manager.state.capital) * 100
                        }
                    
        # Update risk manager state
        self.risk_manager.update_state(trade)
        
        # Record trade
        self.trade_history.append(trade)
        
        # Monitor and log
        self.monitoring.log_trade(trade)
        
        # Get risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        self.monitoring.log_risk_metrics(risk_metrics)
        
        # Check for alerts
        alerts = []
        alerts.extend(self.monitoring.check_risk_limits(risk_metrics))
        alerts.extend(self.monitoring.check_performance_degradation(risk_metrics))
        alerts.extend(self.monitoring.check_regime_change(risk_metrics))
        
        # Send alerts
        for alert in alerts:
            level = alert.split(':')[0]
            self.monitoring.send_alert(level, alert)
            
        # Save metrics
        self.monitoring.save_metrics({
            'trade': trade,
            'risk_metrics': risk_metrics
        })
        
        return trade
        
    def get_position_summary(self) -> Dict:
        """Get current position summary"""
        summary = {
            'positions': self.positions,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
        return summary
        
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history
        
    def get_daily_report(self) -> Dict:
        """Get daily performance report"""
        return self.monitoring.generate_daily_report()

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices and calculate metrics"""
        self.current_prices = prices
        
        # Calculate portfolio metrics
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            self.positions,
            self.trades,
            self.current_prices
        )
        
        # Check risk limits
        if metrics.leverage > self.config['risk_management']['max_leverage']:
            self.monitoring.send_alert(
                "Risk Limit Exceeded",
                f"Leverage {metrics.leverage:.2f} exceeds limit {self.config['risk_management']['max_leverage']}"
            )
            
        if metrics.margin_level < self.config['risk_management']['min_margin_level']:
            self.monitoring.send_alert(
                "Risk Limit Exceeded",
                f"Margin level {metrics.margin_level:.2f}% below limit {self.config['risk_management']['min_margin_level']}%"
            )
            
        if metrics.max_drawdown > self.config['risk_management']['max_drawdown']:
            self.monitoring.send_alert(
                "Risk Limit Exceeded",
                f"Max drawdown {metrics.max_drawdown:.2%} exceeds limit {self.config['risk_management']['max_drawdown']:.2%}"
            )
            
        if metrics.var_95 > self.config['risk_management']['max_var_95']:
            self.monitoring.send_alert(
                "Risk Limit Exceeded",
                f"VaR(95%) {metrics.var_95:.2%} exceeds limit {self.config['risk_management']['max_var_95']:.2%}"
            )
            
        # Check performance thresholds
        if metrics.sharpe_ratio < self.config['portfolio_analytics']['performance_thresholds']['min_sharpe']:
            self.monitoring.send_alert(
                "Performance Warning",
                f"Sharpe ratio {metrics.sharpe_ratio:.2f} below threshold {self.config['portfolio_analytics']['performance_thresholds']['min_sharpe']}"
            )
            
        if metrics.sortino_ratio < self.config['portfolio_analytics']['performance_thresholds']['min_sortino']:
            self.monitoring.send_alert(
                "Performance Warning",
                f"Sortino ratio {metrics.sortino_ratio:.2f} below threshold {self.config['portfolio_analytics']['performance_thresholds']['min_sortino']}"
            )
            
        if metrics.information_ratio < self.config['portfolio_analytics']['performance_thresholds']['min_information_ratio']:
            self.monitoring.send_alert(
                "Performance Warning",
                f"Information ratio {metrics.information_ratio:.2f} below threshold {self.config['portfolio_analytics']['performance_thresholds']['min_information_ratio']}"
            )
            
        if metrics.win_rate < self.config['portfolio_analytics']['performance_thresholds']['min_win_rate']:
            self.monitoring.send_alert(
                "Performance Warning",
                f"Win rate {metrics.win_rate:.2%} below threshold {self.config['portfolio_analytics']['performance_thresholds']['min_win_rate']:.2%}"
            )
            
        if metrics.profit_factor < self.config['portfolio_analytics']['performance_thresholds']['min_profit_factor']:
            self.monitoring.send_alert(
                "Performance Warning",
                f"Profit factor {metrics.profit_factor:.2f} below threshold {self.config['portfolio_analytics']['performance_thresholds']['min_profit_factor']}"
            )
            
        # Generate daily report if needed
        current_time = datetime.now().strftime("%H:%M:%S")
        if current_time == self.config['monitoring']['reporting']['daily_report_time']:
            report = self.portfolio_analytics.generate_portfolio_report(metrics)
            self.monitoring.log_metrics(report)
            
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float):
        """Execute a trade and update portfolio metrics"""
        # Check minimum trade interval
        last_trade_time = self.trades[-1]['timestamp'] if self.trades else None
        if last_trade_time:
            time_since_last_trade = (datetime.now() - datetime.fromisoformat(last_trade_time)).total_seconds()
            if time_since_last_trade < self.min_trade_interval:
                logger.warning("Trade rejected: Minimum trade interval not met")
                return
                
        # Check order size limits
        if quantity < self.config['order_management']['min_order_size']:
            logger.warning("Trade rejected: Order size below minimum")
            return
        if quantity > self.config['order_management']['max_order_size']:
            logger.warning("Trade rejected: Order size above maximum")
            return
            
        # Check position limits
        if side == 'buy' and len([p for p in self.positions.values() if p['side'] == 'long']) >= self.config['risk_management']['position_limits']['max_long_positions']:
            logger.warning("Trade rejected: Maximum long positions reached")
            return
        if side == 'sell' and len([p for p in self.positions.values() if p['side'] == 'short']) >= self.config['risk_management']['position_limits']['max_short_positions']:
            logger.warning("Trade rejected: Maximum short positions reached")
            return
            
        # Execute trade through order manager
        order = self.order_manager.create_order(symbol, side, quantity, price)
        filled_order = self.order_manager.process_order(order)
        
        if filled_order:
            # Update positions
            self.positions = self.order_manager.get_positions()
            
            # Add to trade history
            self.trades.append({
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().isoformat(),
                'pnl': filled_order.get('pnl', 0)
            })
            
            # Log trade
            self.monitoring.log_trade(filled_order)
            
            # Update portfolio metrics
            self.update_prices(self.current_prices)
            
    def get_portfolio_metrics(self) -> Dict:
        """Get current portfolio metrics"""
        metrics = self.portfolio_analytics.calculate_portfolio_metrics(
            self.positions,
            self.trades,
            self.current_prices
        )
        return self.portfolio_analytics.generate_portfolio_report(metrics) 