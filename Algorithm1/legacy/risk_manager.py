import numpy as np
from scipy.stats import beta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import yaml
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RiskState:
    """Current state of risk management"""
    capital: float
    high_water_mark: float
    current_drawdown: float
    active_trades: List[Dict]
    posterior_alpha: float = 1.0
    posterior_beta: float = 1.0
    volatility_multiplier: float = 1.0
    drawdown_barrier: float = 0.0
    regime_score: float = 0.0
    last_update: datetime = None
    performance_metrics: Dict = None

class AdaptiveRiskManager:
    def __init__(self, config_path: str):
        """Initialize risk manager with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Risk parameters
        self.max_risk_per_trade = self.config['trading']['paper_trading']['risk_per_trade']
        self.max_position_size = self.config['trading']['paper_trading']['max_position_size']
        self.max_leverage = self.config['trading']['paper_trading']['max_leverage']
        
        # Pyramiding parameters
        self.pyramid_multiplier = 1.35  # c in the paper
        self.confidence_threshold = 0.97  # High confidence threshold
        
        # OU process parameters for drawdown barrier
        self.theta = 0.1  # Mean reversion speed
        self.mu = 0.05    # Long-term mean drawdown
        self.eta = 0.02   # Volatility of drawdown
        
        # Numerical stability parameters
        self.max_posterior_value = 1000.0  # Cap on alpha/beta values
        self.min_posterior_value = 0.1     # Floor on alpha/beta values
        self.vol_window = 30  # 30-day window for volatility calculation
        self.vol_quantile = 95  # 95th percentile for volatility scaling
        
        # Regime detection parameters
        self.regime_window = 100  # Window for regime detection
        self.regime_threshold = 0.7  # Threshold for regime change
        self.regime_metrics = {
            'returns': [],
            'volatility': [],
            'correlation': []
        }
        
        # Performance monitoring
        self.performance_window = 1000  # Window for performance metrics
        self.performance_metrics = {
            'returns': [],
            'drawdowns': [],
            'sharpe': None,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Initialize state
        self.state = RiskState(
            capital=self.config['trading']['paper_trading']['initial_capital'],
            high_water_mark=self.config['trading']['paper_trading']['initial_capital'],
            current_drawdown=0.0,
            active_trades=[],
            last_update=datetime.now(),
            performance_metrics=self.performance_metrics
        )
        
        # Add running flag
        self.running = True
        
    def stop(self):
        """Stop the risk manager"""
        self.running = False
        logger.info("Risk manager stopped")
        
    def update_posterior(self, trade_result: float) -> None:
        """Update Bayesian posterior for edge estimation with numerical stability"""
        # Clip trade result to prevent extreme values
        trade_result = np.clip(trade_result, -1.0, 1.0)
        
        if trade_result > 0:
            self.state.posterior_alpha = min(
                self.state.posterior_alpha + 1,
                self.max_posterior_value
            )
        else:
            self.state.posterior_beta = min(
                self.state.posterior_beta + 1,
                self.max_posterior_value
            )
            
        # Ensure minimum values
        self.state.posterior_alpha = max(self.state.posterior_alpha, self.min_posterior_value)
        self.state.posterior_beta = max(self.state.posterior_beta, self.min_posterior_value)
            
    def calculate_volatility_multiplier(self, returns: np.ndarray) -> float:
        """Calculate volatility multiplier with regime awareness"""
        if len(returns) < self.vol_window:
            return 1.0
            
        # Calculate recent and historical volatility
        recent_vol = np.percentile(np.abs(returns[-self.vol_window:]), self.vol_quantile)
        historical_vols = [np.percentile(np.abs(returns[i:i+self.vol_window]), self.vol_quantile) 
                          for i in range(len(returns)-self.vol_window)]
        
        # Check for regime change
        if len(historical_vols) > 0:
            vol_ratio = recent_vol / np.median(historical_vols)
            if vol_ratio > 2.0:  # Significant volatility increase
                logger.warning(f"Potential regime change detected: volatility ratio {vol_ratio:.2f}")
                return min(vol_ratio, 3.0)  # Cap the multiplier
                
        return recent_vol / np.median(historical_vols) if np.median(historical_vols) > 0 else 1.0
        
    def calculate_kelly_fraction(self, b: float = 2.0) -> float:
        """Calculate Bayesian Kelly fraction with posterior shrinkage and stability checks"""
        # Calculate posterior mean with numerical stability
        total = self.state.posterior_alpha + self.state.posterior_beta
        if total > self.max_posterior_value:
            # Normalize to prevent overflow
            p_mean = self.state.posterior_alpha / total
        else:
            p_mean = self.state.posterior_alpha / total
            
        edge = 2 * p_mean - 1
        
        # Apply shrinkage based on sample size
        shrinkage = min(1.0, total / 100.0)  # More shrinkage with fewer samples
        edge = edge * shrinkage
        
        return edge / b if b > 0 else 0.0
        
    def update_drawdown_barrier(self) -> None:
        """Update stochastic drawdown barrier using OU process with safety checks"""
        dt = 1.0  # Time step
        noise = np.random.normal(0, 1)
        
        # OU process update with bounds
        self.state.drawdown_barrier = np.clip(
            self.mu + 
            (self.state.drawdown_barrier - self.mu) * np.exp(-self.theta * dt) +
            self.eta * np.sqrt((1 - np.exp(-2 * self.theta * dt)) / (2 * self.theta)) * noise,
            0.01,  # Minimum barrier
            0.5    # Maximum barrier
        )
        
    def update_regime_metrics(self, returns: np.ndarray) -> None:
        """Update regime detection metrics"""
        if len(returns) < self.regime_window:
            return
            
        # Calculate regime metrics
        recent_returns = returns[-self.regime_window:]
        recent_vol = np.std(recent_returns)
        recent_corr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0,1]
        
        # Update metrics
        self.regime_metrics['returns'].append(np.mean(recent_returns))
        self.regime_metrics['volatility'].append(recent_vol)
        self.regime_metrics['correlation'].append(recent_corr)
        
        # Keep only recent history
        for key in self.regime_metrics:
            if len(self.regime_metrics[key]) > 10:
                self.regime_metrics[key] = self.regime_metrics[key][-10:]
                
        # Calculate regime score
        vol_change = recent_vol / np.mean(self.regime_metrics['volatility'])
        corr_change = recent_corr / np.mean(self.regime_metrics['correlation'])
        
        self.state.regime_score = (vol_change + corr_change) / 2
        
    def update_performance_metrics(self, trade_result: Dict) -> None:
        """Update performance metrics and check for degradation"""
        # Update metrics
        self.performance_metrics['returns'].append(trade_result.get('pnl', 0) / self.state.capital)
        self.performance_metrics['drawdowns'].append(self.state.current_drawdown)
        
        # Keep only recent history
        if len(self.performance_metrics['returns']) > self.performance_window:
            self.performance_metrics['returns'] = self.performance_metrics['returns'][-self.performance_window:]
            self.performance_metrics['drawdowns'] = self.performance_metrics['drawdowns'][-self.performance_window:]
            
        # Calculate metrics
        returns = np.array(self.performance_metrics['returns'])
        if len(returns) > 0:
            self.performance_metrics['sharpe'] = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
            self.performance_metrics['max_drawdown'] = max(self.performance_metrics['drawdowns'])
            self.performance_metrics['win_rate'] = np.mean(returns > 0)
            
        # Check for performance degradation
        if len(returns) >= 100:
            recent_sharpe = np.mean(returns[-100:]) / (np.std(returns[-100:]) + 1e-6) * np.sqrt(252)
            if recent_sharpe < 0.5 * self.performance_metrics['sharpe']:
                logger.warning("Performance degradation detected: recent Sharpe ratio significantly lower")
                
    def calculate_position_size(self, 
                              price: float, 
                              probability: float, 
                              direction: int,
                              returns: np.ndarray) -> Tuple[float, float]:
        """
        Calculate position size based on adaptive risk model with safety checks
        """
        # Update state
        self.state.current_drawdown = 1 - (self.state.capital / self.state.high_water_mark)
        self.update_drawdown_barrier()
        self.update_regime_metrics(returns)
        
        # Calculate base Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction()
        
        # Apply volatility scaling
        vol_multiplier = self.calculate_volatility_multiplier(returns)
        base_fraction = kelly_fraction / (1 + vol_multiplier)
        
        # Count active trades in same direction
        same_direction_trades = sum(1 for trade in self.state.active_trades 
                                  if trade['direction'] == direction)
        
        # Apply conditional pyramiding with regime awareness
        if (probability > self.confidence_threshold and 
            kelly_fraction > 0 and 
            self.state.regime_score < 1.5):  # Don't pyramid in extreme regimes
            risk_fraction = min(
                self.pyramid_multiplier ** same_direction_trades * base_fraction,
                self.max_risk_per_trade
            )
        else:
            risk_fraction = base_fraction
            
        # Apply drawdown barrier
        if self.state.current_drawdown > self.state.drawdown_barrier:
            risk_fraction *= 0.5  # Reduce risk by 50% when drawdown exceeds barrier
            
        # Apply performance-based adjustments
        if self.performance_metrics['sharpe'] is not None:
            if self.performance_metrics['sharpe'] < 0:
                risk_fraction *= 0.5  # Reduce risk when strategy is unprofitable
            elif self.performance_metrics['max_drawdown'] > 0.2:
                risk_fraction *= 0.7  # Reduce risk after large drawdowns
                
        # Calculate position size
        position_size = (self.state.capital * risk_fraction) / price
        
        # Apply position limits
        position_size = min(
            position_size,
            self.state.capital * self.max_position_size / price,
            self.state.capital * self.max_leverage / price
        )
        
        return position_size, risk_fraction
        
    def update_state(self, trade_result: Dict) -> None:
        """Update risk state after a trade with performance monitoring"""
        # Update capital
        self.state.capital += trade_result.get('pnl', 0)
        
        # Update high water mark
        self.state.high_water_mark = max(self.state.high_water_mark, self.state.capital)
        
        # Update posterior
        self.update_posterior(trade_result.get('pnl', 0))
        
        # Update performance metrics
        self.update_performance_metrics(trade_result)
        
        # Update active trades
        if trade_result.get('action') == 'close':
            self.state.active_trades = [t for t in self.state.active_trades 
                                      if t['id'] != trade_result['trade_id']]
        else:
            self.state.active_trades.append({
                'id': trade_result['trade_id'],
                'direction': trade_result['direction'],
                'size': trade_result['size'],
                'entry_price': trade_result['price']
            })
            
        # Update timestamp
        self.state.last_update = datetime.now()
            
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'capital': self.state.capital,
            'high_water_mark': self.state.high_water_mark,
            'current_drawdown': self.state.current_drawdown,
            'drawdown_barrier': self.state.drawdown_barrier,
            'posterior_edge': 2 * (self.state.posterior_alpha / 
                                 (self.state.posterior_alpha + self.state.posterior_beta)) - 1,
            'active_trades': len(self.state.active_trades),
            'regime_score': self.state.regime_score,
            'performance_metrics': self.performance_metrics
        } 