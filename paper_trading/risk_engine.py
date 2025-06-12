import numpy as np
from scipy.special import gamma as gamma_func
from typing import Dict, Optional, Tuple
import pandas as pd

class RiskEngine:
    def __init__(self, 
                 alpha: float = 1.7,  # BTC tail exponent
                 beta: float = 2.0,   # Bayesian prior strength
                 gamma: float = 0.35, # Martingale tempering
                 hurst: float = 0.12, # Rough volatility exponent
                 var_window: int = 252,  # Days for VaR
                 atr_window: int = 20,   # Days for ATR
                 ou_mu: float = 0.0,     # OU mean
                 ou_theta: float = 0.1,  # OU mean reversion
                 ou_eta: float = 0.1,    # OU volatility
                 gamma_dd: float = 0.5,  # Drawdown scaling factor
                 high_conf_threshold: float = 0.97):  # High confidence threshold for pyramiding
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.hurst = hurst
        self.var_window = var_window
        self.atr_window = atr_window
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_eta = ou_eta
        self.gamma_dd = gamma_dd
        self.high_conf_threshold = high_conf_threshold
        
        # State variables
        self.last_loss = 0.0
        self.rolling_signals = 0
        self.daily_returns = []
        self.price_history = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.ou_barrier = 0.0  # OU drawdown barrier
        self.high_water_mark = 0.0  # High water mark for drawdown calculation
        
        # Constants
        self.C_alpha = gamma_func(2 - alpha) / gamma_func(1 - alpha)
        self.max_leverage = 2.0
        self.max_position_pct = 0.08
        self.var_multiplier = 10.0
        self.drawdown_lock_threshold = 2.0
        
    def update_state(self, price: float, pnl: float = 0.0):
        """Update engine state with new price and PnL."""
        self.price_history.append(price)
        if len(self.price_history) > self.var_window:
            self.price_history = self.price_history[-self.var_window:]
            
        if pnl < 0:
            self.last_loss = pnl
            self.current_drawdown += abs(pnl)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        else:
            self.current_drawdown = max(0, self.current_drawdown - pnl)
            
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, price)
        
        # Update OU drawdown barrier
        self.update_ou_barrier()
            
    def calculate_atr(self) -> float:
        """Calculate Average True Range."""
        if len(self.price_history) < 2:
            return 0.0
            
        prices = np.array(self.price_history)
        high_low = np.max(prices[-self.atr_window:]) - np.min(prices[-self.atr_window:])
        close_close = np.abs(np.diff(prices[-self.atr_window-1:]))
        tr = np.maximum(high_low, close_close)
        return np.mean(tr)
        
    def calculate_var(self) -> float:
        """Calculate 95% Value at Risk."""
        if len(self.daily_returns) < self.var_window:
            return 0.0
        return np.percentile(self.daily_returns[-self.var_window:], 5)
        
    def calculate_volatility_multiplier(self) -> float:
        """Calculate 95th percentile volatility multiplier."""
        if len(self.daily_returns) < self.var_window:
            return 1.0
        sigma_95 = np.percentile(np.abs(self.daily_returns[-self.var_window:]), 95)
        sigma_median = np.median(np.abs(self.daily_returns[-self.var_window:]))
        return sigma_95 / sigma_median if sigma_median > 0 else 1.0
        
    def bayesian_shrink(self, p: float) -> float:
        """Apply Bayesian shrinkage to probability estimate using Beta posterior."""
        n = self.rolling_signals
        alpha_t = n * p + self.beta
        beta_t = n * (1 - p) + self.beta
        return alpha_t / (alpha_t + beta_t)
        
    def posterior_kelly(self, p: float, b: float) -> float:
        """Calculate Kelly fraction using Beta posterior mean."""
        p_bar = self.bayesian_shrink(p)
        e_bar = 2 * p_bar - 1
        return max(0, e_bar / b)
        
    def convexity_boost(self, p: float) -> float:
        """Apply probability-elastic convexity boost."""
        z = 10 * (p - 0.55)
        return 0.25 + 0.75 / (1 + np.exp(-z))
        
    def tempered_martingale(self, p: float) -> float:
        """Calculate tempered martingale multiplier only if high confidence."""
        if p < self.high_conf_threshold:
            return 1.0
            
        if self.last_loss >= 0:
            return 1.0
            
        atr = self.calculate_atr()
        if atr == 0:
            return 1.0
            
        loss_ratio = abs(self.last_loss) / (1.5 * atr)
        return 1 + self.gamma * min(2.0, loss_ratio)
        
    def update_ou_barrier(self):
        """Update OU drawdown barrier using Ornstein-Uhlenbeck process."""
        dt = 1.0  # Assuming daily updates
        epsilon = np.random.normal(0, 1)
        self.ou_barrier = self.ou_mu + (self.ou_barrier - self.ou_mu) * np.exp(-self.ou_theta * dt) + self.ou_eta * np.sqrt((1 - np.exp(-2 * self.ou_theta * dt)) / (2 * self.ou_theta)) * epsilon
        
    def calculate_position_size(self, 
                              equity: float,
                              price: float,
                              model_prob: float,
                              R: float = 2.0) -> Tuple[float, Dict]:
        """Calculate optimal position size using the full risk engine."""
        # Update rolling signals count
        self.rolling_signals += 1
        
        # Apply Bayesian shrinkage
        p_star = self.bayesian_shrink(model_prob)
        
        # Calculate components
        kelly = self.posterior_kelly(p_star, R)
        convexity = self.convexity_boost(p_star)
        martingale = self.tempered_martingale(p_star)
        atr = self.calculate_atr()
        var = self.calculate_var()
        vol_mult = self.calculate_volatility_multiplier()
        
        # Check drawdown barrier
        if self.current_drawdown > self.ou_barrier:
            martingale = 1.0
            kelly *= self.gamma_dd  # Scale down by gamma
        
        # Calculate raw position size
        raw_size = kelly * convexity * martingale * equity / (atr * vol_mult)
        
        # Apply brake-pad constraints
        max_var_size = self.var_multiplier * abs(var)
        max_leverage_size = equity * self.max_leverage / price
        max_position_size = equity * self.max_position_pct
        
        # Final position size
        position_size = min(raw_size, max_var_size, max_leverage_size, max_position_size)
        
        # Convert to BTC
        btc_size = position_size / price
        
        # Return size and debug info
        debug_info = {
            'p_star': p_star,
            'kelly': kelly,
            'convexity': convexity,
            'martingale': martingale,
            'atr': atr,
            'var': var,
            'vol_mult': vol_mult,
            'ou_barrier': self.ou_barrier,
            'raw_size': raw_size,
            'final_size': position_size,
            'btc_size': btc_size,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown
        }
        
        return btc_size, debug_info 