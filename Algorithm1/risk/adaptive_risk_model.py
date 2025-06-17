"""
Advanced Adaptive Risk Management System
Implements Bayesian Kelly Criterion, volatility-weighted position sizing, and dynamic risk adjustment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import logging
from pathlib import Path
from utils.logger import get_logger

# Fix imports for testing
try:
    from ..utils.logger import get_logger
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics for position sizing and management."""
    volatility: float
    sharpe: float
    drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy: float
    kelly_fraction: float
    max_position: float
    min_position: float
    dynamic_risk: float

class AdaptiveRiskModel:
    """
    Advanced adaptive risk management system for trading.
    Implements Bayesian Kelly, volatility-weighted sizing, and dynamic risk adjustment.
    """
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% default
        self.min_risk_per_trade = self.config.get('min_risk_per_trade', 0.001) # 0.1% default
        self.max_position_size = self.config.get('max_position_size', 1.0)     # 100% of capital
        self.min_position_size = self.config.get('min_position_size', 0.01)    # 1% of capital
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.kelly_lookback = self.config.get('kelly_lookback', 100)
        self.dynamic_risk_factor = self.config.get('dynamic_risk_factor', 1.0)
        logger.info("AdaptiveRiskModel initialized")

    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate risk metrics from a series of returns."""
        volatility = returns.rolling(self.volatility_lookback).std().iloc[-1]
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
        drawdown = self._max_drawdown(returns)
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        kelly_fraction = self._bayesian_kelly(win_rate, avg_win, avg_loss)
        max_position = self.max_position_size
        min_position = self.min_position_size
        dynamic_risk = self._dynamic_risk_adjustment(volatility, sharpe, drawdown)
        return RiskMetrics(
            volatility=volatility,
            sharpe=sharpe,
            drawdown=drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            kelly_fraction=kelly_fraction,
            max_position=max_position,
            min_position=min_position,
            dynamic_risk=dynamic_risk
        )

    def _max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    def _bayesian_kelly(self, win_rate, avg_win, avg_loss) -> float:
        """Bayesian Kelly Criterion for optimal fraction sizing."""
        if avg_loss == 0 or np.isnan(win_rate) or np.isnan(avg_win) or np.isnan(avg_loss):
            return 0.0
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p
        kelly = p - (q / b)
        # Bayesian shrinkage: blend with 0.5 (neutral) for stability
        shrinkage = 0.1
        kelly = (1 - shrinkage) * kelly + shrinkage * 0.5
        return max(0.0, min(kelly, 1.0))

    def _dynamic_risk_adjustment(self, volatility, sharpe, drawdown) -> float:
        """Dynamically adjust risk based on volatility, Sharpe, and drawdown."""
        # Lower risk if volatility or drawdown is high, or Sharpe is low
        risk = self.max_risk_per_trade
        if volatility > 0.03:
            risk *= 0.5
        if drawdown < -0.1:
            risk *= 0.5
        if sharpe < 1.0:
            risk *= 0.7
        risk *= self.dynamic_risk_factor
        return max(self.min_risk_per_trade, min(risk, self.max_risk_per_trade))

    def position_size(self, capital: float, risk_metrics: RiskMetrics, stop_distance: float) -> float:
        """Calculate position size based on risk metrics and stop distance."""
        # Volatility-weighted position sizing
        risk_per_trade = risk_metrics.dynamic_risk
        kelly_fraction = risk_metrics.kelly_fraction
        # Use Kelly as a multiplier, but cap at max risk
        effective_risk = min(risk_per_trade * kelly_fraction, self.max_risk_per_trade)
        # Position size = (capital * effective_risk) / stop_distance
        if stop_distance <= 0:
            stop_distance = 1e-4  # Prevent division by zero
        position = (capital * effective_risk) / stop_distance
        position = max(risk_metrics.min_position, min(position, risk_metrics.max_position))
        logger.info(f"Position size calculated: {position:.4f} (capital={capital}, risk={effective_risk}, stop={stop_distance})")
        return position

    def risk_report(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """Generate a risk report for logging or dashboard."""
        return {
            'volatility': risk_metrics.volatility,
            'sharpe': risk_metrics.sharpe,
            'drawdown': risk_metrics.drawdown,
            'win_rate': risk_metrics.win_rate,
            'avg_win': risk_metrics.avg_win,
            'avg_loss': risk_metrics.avg_loss,
            'expectancy': risk_metrics.expectancy,
            'kelly_fraction': risk_metrics.kelly_fraction,
            'dynamic_risk': risk_metrics.dynamic_risk,
            'max_position': risk_metrics.max_position,
            'min_position': risk_metrics.min_position
        }

    def optimize_risk_parameters(self, returns: pd.Series, capital: float, stop_distance: float) -> Dict[str, Any]:
        """Optimize risk parameters for best performance (optional)."""
        def objective(x):
            self.max_risk_per_trade, self.dynamic_risk_factor = x
            risk_metrics = self.calculate_risk_metrics(returns)
            position = self.position_size(capital, risk_metrics, stop_distance)
            # Objective: maximize expectancy * position (proxy for growth)
            return -risk_metrics.expectancy * position
        res = minimize(objective, [self.max_risk_per_trade, self.dynamic_risk_factor], bounds=[(0.001, 0.05), (0.5, 2.0)])
        return {'max_risk_per_trade': res.x[0], 'dynamic_risk_factor': res.x[1], 'fun': res.fun} 