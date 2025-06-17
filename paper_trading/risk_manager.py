import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import yaml
import numpy as np
import pandas as pd
from .risk_engine import RiskEngine
from scipy.stats import beta
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config: Union[str, Dict] = "paper_trading_config.yaml"):
        """Initialize risk manager with configuration
        
        Args:
            config: Either a path to the config file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Load settings
        self.risk_settings = self.config.get('risk_management', {})
        self.account_settings = self.config.get('account', {})
        
        # Initialize state
        self.risk_metrics = {}
        self.alerts = []
        self.position_sizes = {}
        self.risk_tiers = {}
        self.trade_history = []
        self.position_history = []
        self.last_trade = None
        self.running = False
        
        # Initialize risk engine and account references (will be set later)
        self.risk_engine = None
        self.account = None
        
        # Load risk management settings with defaults
        self.max_leverage = self.risk_settings.get('max_leverage', 10.0)
        self.min_margin_level = self.risk_settings.get('min_margin_level', 150.0)
        self.max_drawdown = self.risk_settings.get('max_drawdown', 0.2)
        self.max_var_95 = self.risk_settings.get('max_var_95', 0.05)
        self.position_limits = self.risk_settings.get('position_limits', {
            'max_long_positions': 5,
            'max_short_positions': 5,
            'max_position_size': 1000.0
        })
        
        # Initialize state
        self.positions = {}
        self.trades = []
        self.current_prices = {}
        
        # Risk tier settings
        self.risk_tiers = {
            'tier1': {'kelly_fraction': 0.5, 'max_position': 0.1},  # Conservative
            'tier2': {'kelly_fraction': 0.75, 'max_position': 0.2},  # Moderate
            'tier3': {'kelly_fraction': 1.0, 'max_position': 0.3}   # Aggressive
        }
        
        # Bayesian Kelly parameters
        self.alpha = 1.0  # Beta prior alpha
        self.beta = 1.0   # Beta prior beta
        self.min_trades = 10  # Minimum trades for reliable posterior
        
        # Volatility scaling parameters
        self.vol_window = 30  # Days for volatility calculation
        self.vol_percentile = 95  # Percentile for volatility scaling
        
        # Martingale parameters
        self.pyramid_multiplier = 1.35  # Position size multiplier
        self.max_risk = 0.08  # Maximum risk per trade
        self.confidence_threshold = 0.97  # Required confidence for pyramiding
        
        # OU drawdown barrier parameters
        self.theta = 0.1  # Mean reversion speed
        self.mu = 0.1     # Long-term mean
        self.eta = 0.05   # Volatility of barrier
        self.gamma = 0.5  # Risk reduction factor during drawdown
        
        # State tracking
        self.high_water_mark = self.account_settings.get('initial_balance', 10000.0)
        self.current_drawdown = 0.0
        self.drawdown_barrier = self.mu
        self.consecutive_trades = {}  # Track consecutive trades per symbol
        
        logger.info("Risk manager initialized")
        
    def set_risk_engine(self, risk_engine):
        """Set the risk engine reference"""
        self.risk_engine = risk_engine
        
    def set_account(self, account):
        """Set the account reference"""
        self.account = account
        
    def start(self):
        """Start the risk manager"""
        if self.running:
            logger.warning("Risk manager is already running")
            return
            
        try:
            self.running = True
            logger.info("Risk manager started")
        except Exception as e:
            logger.error("Error starting risk manager: %s", str(e))
            self.running = False
            raise
            
    def stop(self):
        """Stop the risk manager"""
        if not self.running:
            logger.warning("Risk manager is not running")
            return
            
        try:
            self.running = False
            logger.info("Risk manager stopped")
        except Exception as e:
            logger.error("Error stopping risk manager: %s", str(e))
            raise
        
    def update_positions(self, positions: Dict):
        """Update current positions"""
        self.positions = positions
        
    def update_trades(self, trades: List[Dict]):
        """Update trade history"""
        self.trades = trades
        
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices"""
        self.current_prices = prices
        
    def check_risk_limits(self) -> List[str]:
        """Check if any risk limits are exceeded"""
        if not self.running:
            return []
            
        alerts = []
        
        try:
            # Calculate metrics
            metrics = self._calculate_risk_metrics()
            
            # Check leverage
            if metrics['leverage'] > self.max_leverage:
                alerts.append(f"Leverage {metrics['leverage']:.2f} exceeds limit {self.max_leverage}")
                
            # Check margin level
            if metrics['margin_level'] < self.min_margin_level:
                alerts.append(f"Margin level {metrics['margin_level']:.2f}% below limit {self.min_margin_level}%")
                
            # Check max drawdown
            if metrics['max_drawdown'] > self.max_drawdown:
                alerts.append(f"Max drawdown {metrics['max_drawdown']:.2%} exceeds limit {self.max_drawdown:.2%}")
                
            # Check VaR
            if metrics['var_95'] > self.max_var_95:
                alerts.append(f"VaR(95%) {metrics['var_95']:.2%} exceeds limit {self.max_var_95:.2%}")
                
            # Check position limits with safety checks
            long_positions = 0
            short_positions = 0
            
            for pos in self.positions.values():
                if isinstance(pos, dict):
                    pos_side = pos.get('side', 'unknown')
                    if pos_side == 'long':
                        long_positions += 1
                    elif pos_side == 'short':
                        short_positions += 1
            
            if long_positions > self.position_limits['max_long_positions']:
                alerts.append(f"Number of long positions ({long_positions}) exceeds limit ({self.position_limits['max_long_positions']})")
                
            if short_positions > self.position_limits['max_short_positions']:
                alerts.append(f"Number of short positions ({short_positions}) exceeds limit ({self.position_limits['max_short_positions']})")
                
            # Check position size limits with safety checks
            for symbol, position in self.positions.items():
                if isinstance(position, dict):
                    quantity = position.get('quantity', 0.0)
                    if isinstance(quantity, (int, float)) and quantity > self.position_limits['max_position_size']:
                        alerts.append(f"Position size for {symbol} ({quantity}) exceeds limit ({self.position_limits['max_position_size']})")
                    
        except Exception as e:
            logger.error("Error checking risk limits: %s", str(e))
            
        return alerts
        
    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk metrics"""
        try:
            # Calculate total position value with safety checks
            position_value = 0
            for symbol, pos in self.positions.items():
                if isinstance(pos, dict):
                    quantity = pos.get('quantity', 0.0)
                    price = self.current_prices.get(symbol, 0.0)
                    
                    # Ensure both are numbers
                    if isinstance(quantity, (int, float)) and isinstance(price, (int, float)):
                        position_value += float(quantity) * float(price)
            
            # Calculate margin used with safety checks
            margin_used = 0
            for symbol, pos in self.positions.items():
                if isinstance(pos, dict):
                    quantity = pos.get('quantity', 0.0)
                    price = self.current_prices.get(symbol, 0.0)
                    
                    # Ensure all values are numbers
                    if (isinstance(quantity, (int, float)) and 
                        isinstance(price, (int, float)) and 
                        isinstance(self.max_leverage, (int, float))):
                        margin_used += abs(float(quantity) * float(price) / float(self.max_leverage))
            
            # Calculate leverage
            leverage = margin_used / position_value if position_value > 0 else 0
            
            # Calculate margin level
            margin_level = (position_value / margin_used * 100) if margin_used > 0 else 100
            
            # Calculate max drawdown with safety checks
            pnl_values = []
            for trade in self.trades:
                if isinstance(trade, dict):
                    pnl = trade.get('pnl', 0)
                    # Ensure pnl is a number
                    if isinstance(pnl, (int, float)):
                        pnl_values.append(float(pnl))
                    else:
                        pnl_values.append(0.0)
            
            if pnl_values:
                cumulative_pnl = np.cumsum(pnl_values)
                max_dd = 0
                peak = cumulative_pnl[0] if len(cumulative_pnl) > 0 else 0
                
                for pnl in cumulative_pnl:
                    if pnl > peak:
                        peak = pnl
                    dd = (peak - pnl) / peak if peak > 0 else 0
                    max_dd = max(max_dd, dd)
            else:
                max_dd = 0
                
            # Calculate VaR with safety checks
            returns = []
            for trade in self.trades:
                if isinstance(trade, dict):
                    pnl = trade.get('pnl', 0)
                    if isinstance(pnl, (int, float)):
                        returns.append(float(pnl))
                    else:
                        returns.append(0.0)
            
            var_95 = np.percentile(returns, 5) if returns else 0
            
            return {
                'leverage': leverage,
                'margin_level': margin_level,
                'max_drawdown': max_dd,
                'var_95': abs(var_95)
            }
            
        except Exception as e:
            logger.error("Error calculating risk metrics: %s", str(e))
            return {
                'leverage': 0,
                'margin_level': 100,
                'max_drawdown': 0,
                'var_95': 0
            }
            
    def check_trade(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Check if a trade is allowed based on risk limits"""
        if not self.running:
            return False
            
        try:
            # Check position limits with safety checks
            long_positions = 0
            short_positions = 0
            
            for pos in self.positions.values():
                if isinstance(pos, dict):
                    pos_side = pos.get('side', 'unknown')
                    if pos_side == 'long':
                        long_positions += 1
                    elif pos_side == 'short':
                        short_positions += 1
                        
            if side == 'buy' and long_positions >= self.position_limits['max_long_positions']:
                logger.warning("Trade rejected: Maximum long positions reached")
                return False
                
            if side == 'sell' and short_positions >= self.position_limits['max_short_positions']:
                logger.warning("Trade rejected: Maximum short positions reached")
                return False
                
            # Check position size
            if quantity > self.position_limits['max_position_size']:
                logger.warning("Trade rejected: Position size exceeds limit")
                return False
                
            # Check margin requirements
            trade_value = quantity * price
            margin_required = trade_value / self.max_leverage
            
            # Calculate current margin used with safety checks
            current_margin = 0
            for symbol, pos in self.positions.items():
                if isinstance(pos, dict):
                    pos_quantity = pos.get('quantity', 0.0)
                    pos_price = self.current_prices.get(symbol, 0.0)
                    
                    # Ensure all values are numbers
                    if (isinstance(pos_quantity, (int, float)) and 
                        isinstance(pos_price, (int, float)) and 
                        isinstance(self.max_leverage, (int, float))):
                        current_margin += abs(float(pos_quantity) * float(pos_price) / float(self.max_leverage))
            
            # Check if adding this trade would exceed margin limits (with safe default)
            max_margin_used = self.risk_settings.get('max_margin_used', 10000.0)  # Default to 10000
            if current_margin + margin_required > max_margin_used:
                logger.warning("Trade rejected: Would exceed maximum margin used")
                return False
                
            return True
            
        except Exception as e:
            logger.error("Error checking trade: %s", str(e))
            return False

    def calculate_position_size(self, symbol: str, signal: Dict, account_value: float) -> float:
        """Calculate position size using the adaptive risk model
        
        Args:
            symbol: Trading symbol
            signal: Trading signal with probability
            account_value: Current account value
            
        Returns:
            Position size in base currency
        """
        try:
            logger.debug(f"[calculate_position_size] symbol={symbol}, signal={signal}, account_value={account_value}, type(signal)={type(signal)}")
            if not signal or 'probability' not in signal:
                return 0.0
                
            # Get win probability from model
            win_prob = signal['probability']
            logger.debug(f"[calculate_position_size] win_prob={win_prob}, type(win_prob)={type(win_prob)}")
            
            # 1. Calculate Bayesian Kelly fraction
            kelly = self._calculate_bayesian_kelly(symbol, win_prob)
            logger.debug(f"[calculate_position_size] kelly={kelly}, type(kelly)={type(kelly)}")
            
            # 2. Apply volatility scaling
            vol_scaled_kelly = self._apply_volatility_scaling(symbol, kelly)
            logger.debug(f"[calculate_position_size] vol_scaled_kelly={vol_scaled_kelly}, type(vol_scaled_kelly)={type(vol_scaled_kelly)}")
            
            # 3. Apply conditional Martingale
            martingale_kelly = self._apply_conditional_martingale(symbol, vol_scaled_kelly, win_prob)
            logger.debug(f"[calculate_position_size] martingale_kelly={martingale_kelly}, type(martingale_kelly)={type(martingale_kelly)}")
            
            # 4. Apply drawdown barrier
            final_kelly = self._apply_drawdown_barrier(martingale_kelly)
            logger.debug(f"[calculate_position_size] final_kelly={final_kelly}, type(final_kelly)={type(final_kelly)}")
            
            # Calculate position size
            position_size = account_value * final_kelly
            logger.debug(f"[calculate_position_size] position_size={position_size}, type(position_size)={type(position_size)}")
            
            # Store position size
            self.position_sizes[symbol] = position_size
            
            logger.info("Calculated position size for %s: %.2f", symbol, position_size)
            return position_size
            
        except Exception as e:
            logger.error("Error calculating position size: %s", str(e))
            return 0.0
            
    def _calculate_bayesian_kelly(self, symbol: str, win_prob: float) -> float:
        """Calculate Bayesian Kelly criterion
        
        Args:
            symbol: Trading symbol
            win_prob: Model's win probability
            
        Returns:
            Kelly fraction
        """
        try:
            logger.debug(f"[_calculate_bayesian_kelly] symbol={symbol}, win_prob={win_prob}, type(win_prob)={type(win_prob)}")
            trades = []
            for t in self.trade_history:
                if isinstance(t, dict) and t.get('symbol') == symbol:
                    trades.append(t)
            logger.debug(f"[_calculate_bayesian_kelly] trades={trades}, type(trades)={type(trades)}")
            if len(trades) < self.min_trades:
                # Use conservative prior if not enough data
                return 0.5 * win_prob
                
            # Update Beta posterior with safety checks
            wins = 0
            for t in trades:
                pnl = t.get('pnl', 0.0)
                if isinstance(pnl, (int, float)) and pnl > 0:
                    wins += 1
            
            losses = len(trades) - wins
            
            alpha_post = self.alpha + wins
            beta_post = self.beta + losses
            
            # Calculate posterior mean
            posterior_mean = alpha_post / (alpha_post + beta_post)
            
            # Calculate edge
            edge = 2 * posterior_mean - 1
            
            # Calculate win/loss ratio
            win_loss_ratio = self._calculate_win_loss_ratio(trades)
            
            # Kelly formula with posterior edge
            kelly = edge / win_loss_ratio if win_loss_ratio > 0 else 0.0
            
            # Ensure Kelly is between 0 and 1
            kelly = max(0.0, min(kelly, 1.0))
            
            logger.debug(f"[_calculate_bayesian_kelly] edge={edge}, win_loss_ratio={win_loss_ratio}")
            logger.debug(f"[_calculate_bayesian_kelly] kelly={kelly}")
            return kelly
            
        except Exception as e:
            logger.error("Error calculating Bayesian Kelly: %s", str(e))
            return 0.0
            
    def _apply_volatility_scaling(self, symbol: str, kelly: float) -> float:
        """Apply volatility-weighted scaling to Kelly fraction
        
        Args:
            symbol: Trading symbol
            kelly: Base Kelly fraction
            
        Returns:
            Volatility-scaled Kelly fraction
        """
        try:
            logger.debug(f"[_apply_volatility_scaling] symbol={symbol}, kelly={kelly}, type(kelly)={type(kelly)}")
            trades = []
            for t in self.trade_history:
                if isinstance(t, dict) and t.get('symbol') == symbol:
                    trades.append(t)
            logger.debug(f"[_apply_volatility_scaling] trades={trades}, type(trades)={type(trades)}")
            if not trades:
                return kelly
                
            # Calculate absolute returns with safety checks
            returns = []
            for t in trades:
                pnl = t.get('pnl', 0.0)
                position_size = t.get('position_size', 1.0)  # Default to 1.0 to avoid division by zero
                
                # Ensure both are numbers
                if not isinstance(pnl, (int, float)) or not isinstance(position_size, (int, float)):
                    continue
                
                if position_size > 0:
                    returns.append(abs(float(pnl) / float(position_size)))
                else:
                    returns.append(abs(float(pnl)))  # Use absolute PnL if position_size is zero
            
            logger.debug(f"[_apply_volatility_scaling] returns={returns}, type(returns)={type(returns)}")
            if not returns:
                return kelly
                
            # Calculate 95th percentile volatility
            vol_95 = np.percentile(returns, self.vol_percentile)
            
            # Calculate median volatility over window
            vol_median = np.median(returns[-self.vol_window:]) if len(returns) >= self.vol_window else vol_95
            
            # Calculate volatility multiplier
            vol_multiplier = vol_95 / vol_median if vol_median > 0 else 1.0
            
            # Scale Kelly by volatility
            scaled_kelly = kelly / (1 + vol_multiplier)
            
            logger.debug(f"[_apply_volatility_scaling] vol_95={vol_95}, vol_median={vol_median}, vol_multiplier={vol_multiplier}, scaled_kelly={scaled_kelly}")
            return scaled_kelly
            
        except Exception as e:
            logger.error("Error applying volatility scaling: %s", str(e))
            return kelly
            
    def _apply_conditional_martingale(self, symbol: str, kelly: float, win_prob: float) -> float:
        """Apply conditional Martingale scaling
        
        Args:
            symbol: Trading symbol
            kelly: Base Kelly fraction
            win_prob: Model's win probability
            
        Returns:
            Martingale-scaled Kelly fraction
        """
        try:
            logger.debug(f"[_apply_conditional_martingale] symbol={symbol}, kelly={kelly}, win_prob={win_prob}")
            # Only apply Martingale if confidence is high
            if win_prob < self.confidence_threshold:
                return kelly
                
            # Get consecutive trades in same direction with safety check
            k = self.consecutive_trades.get(symbol, 0)
            
            # Ensure k is an integer
            if not isinstance(k, (int, float)):
                k = 0
            else:
                k = int(k)  # Convert to integer for exponentiation
            
            # Calculate Martingale multiplier with safety checks
            if kelly > 0:
                max_multiplier = self.max_risk / kelly
            else:
                max_multiplier = float('inf')
                
            martingale_multiplier = min(
                self.pyramid_multiplier ** k,
                max_multiplier
            )
            
            # Apply multiplier
            martingale_kelly = kelly * martingale_multiplier
            
            logger.debug(f"[_apply_conditional_martingale] k={k}, martingale_multiplier={martingale_multiplier}, martingale_kelly={martingale_kelly}")
            return martingale_kelly
            
        except Exception as e:
            logger.error("Error applying conditional Martingale: %s", str(e))
            return kelly
            
    def _apply_drawdown_barrier(self, kelly: float) -> float:
        """Apply stochastic drawdown barrier
        
        Args:
            kelly: Base Kelly fraction
            
        Returns:
            Drawdown-adjusted Kelly fraction
        """
        try:
            logger.debug(f"[_apply_drawdown_barrier] kelly={kelly}, type(kelly)={type(kelly)}")
            # Update drawdown barrier using OU process
            self._update_drawdown_barrier()
            
            # Check if we're in drawdown
            if self.current_drawdown > self.drawdown_barrier:
                # Reduce risk during drawdown
                kelly *= self.gamma
                
            logger.debug(f"[_apply_drawdown_barrier] current_drawdown={self.current_drawdown}, drawdown_barrier={self.drawdown_barrier}, gamma={self.gamma}, adjusted_kelly={kelly}")
            return kelly
            
        except Exception as e:
            logger.error("Error applying drawdown barrier: %s", str(e))
            return kelly
            
    def _update_drawdown_barrier(self):
        """Update the stochastic drawdown barrier using OU process"""
        try:
            # Calculate current drawdown with safe defaults
            current_balance = self.account_settings.get('current_balance', self.high_water_mark)
            
            # Ensure both values are numbers
            if not isinstance(current_balance, (int, float)):
                current_balance = self.high_water_mark
            if not isinstance(self.high_water_mark, (int, float)):
                self.high_water_mark = 10000.0  # Default value
                
            current_balance = float(current_balance)
            self.high_water_mark = float(self.high_water_mark)
            
            # Calculate drawdown
            if self.high_water_mark > 0:
                self.current_drawdown = 1 - (current_balance / self.high_water_mark)
            else:
                self.current_drawdown = 0.0
            
            # Update high water mark
            self.high_water_mark = max(
                self.high_water_mark,
                current_balance
            )
            
            # Update barrier using OU process with safety checks
            dt = 1.0  # Time step
            
            # Ensure all parameters are numbers
            mu = float(self.mu) if isinstance(self.mu, (int, float)) else 0.2
            theta = float(self.theta) if isinstance(self.theta, (int, float)) else 0.1
            eta = float(self.eta) if isinstance(self.eta, (int, float)) else 0.05
            current_barrier = float(self.drawdown_barrier) if isinstance(self.drawdown_barrier, (int, float)) else mu
            
            # Calculate new barrier
            exp_term = np.exp(-theta * dt)
            sqrt_term = np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
            random_term = np.random.normal()
            
            self.drawdown_barrier = (
                mu + 
                (current_barrier - mu) * exp_term +
                eta * sqrt_term * random_term
            )
            
        except Exception as e:
            logger.error("Error updating drawdown barrier: %s", str(e))
            # Set safe defaults
            self.current_drawdown = 0.0
            self.drawdown_barrier = 0.2
            
    def _calculate_win_loss_ratio(self, trades: List[Dict]) -> float:
        """Calculate win/loss ratio from trades
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Win/loss ratio
        """
        try:
            if not trades:
                return 1.0
                
            # Calculate average win and loss with safety checks
            wins = []
            losses = []
            
            for t in trades:
                # Ensure t is a dictionary and has pnl
                if not isinstance(t, dict):
                    continue
                    
                pnl = t.get('pnl', 0.0)
                
                # Ensure pnl is a number
                if not isinstance(pnl, (int, float)):
                    continue
                    
                if pnl > 0:
                    wins.append(float(pnl))
                elif pnl < 0:
                    losses.append(abs(float(pnl)))
            
            avg_win = np.mean(wins) if wins else 1.0
            avg_loss = np.mean(losses) if losses else 1.0
            
            # Calculate ratio
            ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            return ratio
            
        except Exception as e:
            logger.error("Error calculating win/loss ratio: %s", str(e))
            return 1.0
            
    def update_trade_history(self, trade: Dict):
        """Update trade history with new trade
        
        Args:
            trade: Trade information
        """
        try:
            self.trade_history.append(trade)
            
            # Update consecutive trades counter
            symbol = trade.get('symbol', 'unknown')
            if symbol not in self.consecutive_trades:
                self.consecutive_trades[symbol] = 1
            else:
                # Increment if same direction, reset if opposite
                # Check if we have at least 2 trades to compare
                if len(self.trade_history) >= 2:
                    current_pnl = trade.get('pnl', 0)
                    previous_pnl = self.trade_history[-2].get('pnl', 0)
                    
                    # Ensure both are numbers
                    if isinstance(current_pnl, (int, float)) and isinstance(previous_pnl, (int, float)):
                        if (current_pnl > 0) == (previous_pnl > 0):
                            self.consecutive_trades[symbol] += 1
                        else:
                            self.consecutive_trades[symbol] = 1
                    else:
                        self.consecutive_trades[symbol] = 1
                else:
                    self.consecutive_trades[symbol] = 1
                    
            # Keep only recent trades
            max_history = self.risk_settings.get('max_trade_history', 100)
            if len(self.trade_history) > max_history:
                self.trade_history = self.trade_history[-max_history:]
                
        except Exception as e:
            logger.error("Error updating trade history: %s", str(e))
            
    def check_risk_limits(self, symbol: str, position_size: float, account_value: float) -> bool:
        """Check if position size violates risk limits
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            account_value: Current account value
            
        Returns:
            True if position size is acceptable, False otherwise
        """
        try:
            # Get position sizing settings with defaults
            position_sizing = self.risk_settings.get('position_sizing', {})
            max_position = position_sizing.get('max_position_size', 0.1)  # Default 10% of account
            max_exposure = position_sizing.get('max_total_exposure', 0.5)  # Default 50% of account
            
            # Get drawdown settings with defaults
            drawdown_settings = self.risk_settings.get('drawdown', {})
            max_drawdown = drawdown_settings.get('max_drawdown', 0.2)  # Default 20%
            
            # Check maximum position size
            if position_size > account_value * max_position:
                logger.warning("Position size %.2f exceeds maximum %.2f", 
                             position_size, account_value * max_position)
                return False
                
            # Check maximum total exposure
            current_exposure = sum(self.position_sizes.values())
            if current_exposure + position_size > account_value * max_exposure:
                logger.warning("Total exposure %.2f would exceed maximum %.2f",
                             current_exposure + position_size, account_value * max_exposure)
                return False
                
            # Check drawdown limits
            if self.current_drawdown > max_drawdown:
                logger.warning("Drawdown limit reached")
                return False
                
            return True
            
        except Exception as e:
            logger.error("Error checking risk limits: %s", str(e))
            return False

    def get_position_size(self, action_prob: float, account) -> Tuple[float, Dict]:
        """Calculate position size using the advanced risk engine."""
        try:
            # If risk engine is not set, use simple calculation
            if self.risk_engine is None:
                # Simple position sizing based on probability
                capital = account.get_capital() if hasattr(account, 'get_capital') else 10000.0
                price = account.get_last_price() if hasattr(account, 'get_last_price') else 50000.0
                
                # Simple Kelly-like calculation
                kelly_fraction = max(0.0, min(0.1, (action_prob - 0.5) * 0.2))
                position_size = capital * kelly_fraction / price
                
                debug_info = {
                    'method': 'simple_kelly',
                    'capital': capital,
                    'price': price,
                    'kelly_fraction': kelly_fraction,
                    'action_prob': action_prob
                }
                
                return position_size, debug_info
            
            # Use risk engine if available
            capital = account.get_capital()
            price = account.get_last_price()
            
            # Calculate position size using risk engine
            btc_size, debug_info = self.risk_engine.calculate_position_size(
                equity=capital,
                price=price,
                model_prob=action_prob
            )
            
            return btc_size, debug_info
            
        except Exception as e:
            logger.error("Error in get_position_size: %s", str(e))
            # Return safe defaults
            return 0.0, {'error': str(e), 'method': 'fallback'}
    
    def update_after_trade(self, trade_result: Dict):
        """Update risk engine state after a trade."""
        try:
            if trade_result:
                # Update risk engine with trade result if available
                if self.risk_engine is not None:
                    self.risk_engine.update_state(
                        price=trade_result['price'],
                        pnl=trade_result.get('net_pnl', 0.0)
                    )
                
                # Store trade for history
                self.position_history.append(trade_result)
                self.last_trade = trade_result
                
                # Keep only last 100 trades
                if len(self.position_history) > 100:
                    self.position_history = self.position_history[-100:]
                    
        except Exception as e:
            logger.error("Error updating after trade: %s", str(e))
    
    def execute_trade(self, action: str, action_prob: float, price: float) -> Optional[Dict]:
        """Execute a trade with proper risk management."""
        try:
            # Check if account is set
            if self.account is None:
                logger.warning("Account not set in risk manager, cannot execute trade")
                return None
            
            # Get position size from risk engine
            size, debug_info = self.get_position_size(action_prob, self.account)
            
            # Check if we have enough capital
            if size <= 0:
                return None
            
            # Execute trade based on action
            trade_result = None
            if action == 'OPEN_LONG':
                trade_result = self.account.open_trade(price, size, 'long')
            elif action == 'OPEN_SHORT':
                trade_result = self.account.open_trade(price, size, 'short')
            elif action == 'CLOSE_LONG':
                trade_result = self.account.close_long_trades(price)
            elif action == 'CLOSE_SHORT':
                trade_result = self.account.close_short_trades(price)
            elif action == 'ADD_LONG':
                trade_result = self.account.add_to_long_trades(price, size)
            elif action == 'ADD_SHORT':
                trade_result = self.account.add_to_short_trades(price, size)
            
            # Update risk engine if trade was executed
            if trade_result:
                self.update_after_trade(trade_result)
                # Add debug info to trade result
                trade_result['risk_debug'] = debug_info
            
            return trade_result
            
        except Exception as e:
            logger.error("Error executing trade: %s", str(e))
            return None 