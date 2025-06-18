"""
Position Sizer - Risk Integration and Position Sizing System
Integrates risk engine with trading agent, adds real-time risk monitoring, and implements risk limit enforcement
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import get_logger
from config.intelligent_config import load_config
from .adaptive_risk_engine import AdaptiveRiskEngine, RiskMetrics

logger = get_logger(__name__)


@dataclass
class PositionSizingResult:
    """Position sizing result data structure"""
    position_size: float
    risk_metrics: RiskMetrics
    risk_limits_respected: bool
    warnings: List[str]
    recommendations: List[str]


class PositionSizer:
    """
    Position Sizing and Risk Integration System
    """
    
    def __init__(self, config_path: str = "config/intelligent_config.yaml"):
        """Initialize position sizer"""
        self.config = load_config(config_path)
        self.logger = logger
        
        # Initialize risk engine
        self.risk_engine = AdaptiveRiskEngine(config_path)
        
        # Configuration
        self.sizing_config = self.config.get('position_sizing', {})
        self.limits_config = self.config.get('risk_limits', {})
        
        # Risk monitoring
        self.current_positions = {}
        self.total_exposure = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        
        # Risk limits
        self.max_total_exposure = self.limits_config.get('max_total_exposure', 0.5)
        self.max_daily_loss = self.limits_config.get('max_daily_loss', 0.05)
        self.max_weekly_loss = self.limits_config.get('max_weekly_loss', 0.15)
        self.max_monthly_loss = self.limits_config.get('max_monthly_loss', 0.30)
        self.max_correlation_exposure = self.limits_config.get('max_correlation_exposure', 0.3)
        
        # Performance tracking
        self.sizing_history = []
        
        self.logger.info("Position Sizer initialized successfully")
    
    async def initialize(self):
        """Initialize the position sizer (no-op for compatibility)"""
        pass
    
    async def calculate_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        model_predictions: Dict[str, float],
        market_context: Dict[str, Any],
        current_positions: Dict[str, Any] = None
    ) -> PositionSizingResult:
        """
        Calculate position size with risk integration and limit enforcement
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            account_balance: Current account balance
            model_predictions: Model predictions
            market_context: Market context information
            current_positions: Current open positions
            
        Returns:
            PositionSizingResult object with sizing and risk information
        """
        try:
            # Update current positions
            if current_positions:
                self.current_positions = current_positions
                self.total_exposure = sum(pos.get('exposure', 0) for pos in current_positions.values())
            
            # Get risk metrics from risk engine
            risk_metrics = await self.risk_engine.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                account_balance=account_balance,
                model_predictions=model_predictions,
                market_context=market_context,
                trade_history=self._get_trade_history()
            )
            
            # Apply risk limits
            adjusted_size, limits_respected, warnings = await self._apply_risk_limits(
                symbol, side, risk_metrics.position_size, account_balance, market_context
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                adjusted_size, risk_metrics, market_context, warnings
            )
            
            # Create result
            result = PositionSizingResult(
                position_size=adjusted_size,
                risk_metrics=risk_metrics,
                risk_limits_respected=limits_respected,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Store in history
            self.sizing_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'result': result
            })
            
            self.logger.info(f"Position size for {symbol}: {adjusted_size:.2f}, Risk limits: {limits_respected}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return await self._create_default_result(account_balance)
    
    async def _apply_risk_limits(
        self,
        symbol: str,
        side: str,
        proposed_size: float,
        account_balance: float,
        market_context: Dict[str, Any]
    ) -> Tuple[float, bool, List[str]]:
        """Apply risk limits and return adjusted size"""
        try:
            warnings = []
            adjusted_size = proposed_size
            limits_respected = True
            
            # Total exposure limit
            new_total_exposure = self.total_exposure + (proposed_size / account_balance)
            if new_total_exposure > self.max_total_exposure:
                max_additional_exposure = self.max_total_exposure - self.total_exposure
                adjusted_size = max_additional_exposure * account_balance
                warnings.append(f"Total exposure limit exceeded. Reduced size to {adjusted_size:.2f}")
                limits_respected = False
            
            # Daily loss limit
            if self.daily_pnl < -self.max_daily_loss * account_balance:
                adjusted_size *= 0.5  # Reduce size by 50%
                warnings.append("Daily loss limit approaching. Reduced position size.")
                limits_respected = False
            
            # Weekly loss limit
            if self.weekly_pnl < -self.max_weekly_loss * account_balance:
                adjusted_size *= 0.3  # Reduce size by 70%
                warnings.append("Weekly loss limit approaching. Significantly reduced position size.")
                limits_respected = False
            
            # Monthly loss limit
            if self.monthly_pnl < -self.max_monthly_loss * account_balance:
                adjusted_size = 0  # No new positions
                warnings.append("Monthly loss limit exceeded. No new positions allowed.")
                limits_respected = False
            
            # Volatility-based limits
            volatility_regime = market_context.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                adjusted_size *= 0.7  # Reduce size in high volatility
                warnings.append("High volatility regime. Reduced position size.")
            
            # Correlation limits
            correlation_exposure = await self._calculate_correlation_exposure(symbol, side)
            if correlation_exposure > self.max_correlation_exposure:
                adjusted_size *= 0.6  # Reduce size for high correlation
                warnings.append(f"High correlation exposure ({correlation_exposure:.2f}). Reduced position size.")
            
            # Minimum position size
            min_position_size = self.sizing_config.get('min_position_size', 10.0)
            if adjusted_size < min_position_size:
                adjusted_size = 0
                warnings.append("Position size below minimum threshold.")
                limits_respected = False
            
            return adjusted_size, limits_respected, warnings
            
        except Exception as e:
            self.logger.error(f"Error applying risk limits: {e}")
            return 0.0, False, [f"Error applying risk limits: {str(e)}"]
    
    async def _calculate_correlation_exposure(self, symbol: str, side: str) -> float:
        """Calculate correlation exposure with existing positions"""
        try:
            if not self.current_positions:
                return 0.0
            
            # Simple correlation calculation based on side
            same_side_positions = sum(
                1 for pos in self.current_positions.values() 
                if pos.get('side') == side
            )
            
            total_positions = len(self.current_positions)
            if total_positions == 0:
                return 0.0
            
            correlation_exposure = same_side_positions / total_positions
            
            return correlation_exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation exposure: {e}")
            return 0.0
    
    async def _generate_recommendations(
        self,
        position_size: float,
        risk_metrics: RiskMetrics,
        market_context: Dict[str, Any],
        warnings: List[str]
    ) -> List[str]:
        """Generate recommendations based on risk analysis"""
        try:
            recommendations = []
            
            # Risk score recommendations
            if risk_metrics.risk_score > 0.7:
                recommendations.append("High risk score - consider reducing position size")
            elif risk_metrics.risk_score < 0.3:
                recommendations.append("Low risk score - position size is conservative")
            
            # Kelly fraction recommendations
            if risk_metrics.kelly_fraction > 0.15:
                recommendations.append("High Kelly fraction - aggressive sizing")
            elif risk_metrics.kelly_fraction < 0.05:
                recommendations.append("Low Kelly fraction - very conservative")
            
            # Volatility recommendations
            volatility_regime = market_context.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                recommendations.append("High volatility - consider tighter stops")
            elif volatility_regime == 'low':
                recommendations.append("Low volatility - wider stops may be appropriate")
            
            # Market phase recommendations
            market_phase = market_context.get('market_phase', 'unknown')
            if market_phase == 'accumulation':
                recommendations.append("Accumulation phase - consider scaling in")
            elif market_phase == 'distribution':
                recommendations.append("Distribution phase - consider scaling out")
            
            # Drawdown recommendations
            if risk_metrics.drawdown_barrier < 0.8:
                recommendations.append("Drawdown protection active - reduced sizing")
            
            # Confidence recommendations
            if risk_metrics.confidence < 0.5:
                recommendations.append("Low confidence - consider waiting for better setup")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def monitor_risk_limits(self) -> Dict[str, Any]:
        """Monitor current risk limits and return status"""
        try:
            risk_status = {
                'total_exposure': self.total_exposure,
                'max_total_exposure': self.max_total_exposure,
                'exposure_utilization': self.total_exposure / self.max_total_exposure,
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'monthly_pnl': self.monthly_pnl,
                'daily_loss_limit': -self.max_daily_loss,
                'weekly_loss_limit': -self.max_weekly_loss,
                'monthly_loss_limit': -self.max_monthly_loss,
                'limits_respected': True,
                'warnings': []
            }
            
            # Check limits
            if self.total_exposure > self.max_total_exposure * 0.9:
                risk_status['warnings'].append("Total exposure approaching limit")
                risk_status['limits_respected'] = False
            
            if self.daily_pnl < -self.max_daily_loss * 0.8:
                risk_status['warnings'].append("Daily loss approaching limit")
                risk_status['limits_respected'] = False
            
            if self.weekly_pnl < -self.max_weekly_loss * 0.8:
                risk_status['warnings'].append("Weekly loss approaching limit")
                risk_status['limits_respected'] = False
            
            if self.monthly_pnl < -self.max_monthly_loss * 0.8:
                risk_status['warnings'].append("Monthly loss approaching limit")
                risk_status['limits_respected'] = False
            
            return risk_status
            
        except Exception as e:
            self.logger.error(f"Error monitoring risk limits: {e}")
            return {
                'error': str(e),
                'limits_respected': False,
                'warnings': [f"Error monitoring risk: {str(e)}"]
            }
    
    async def update_pnl(self, pnl: float, period: str = 'daily'):
        """Update P&L for different periods"""
        try:
            if period == 'daily':
                self.daily_pnl += pnl
            elif period == 'weekly':
                self.weekly_pnl += pnl
            elif period == 'monthly':
                self.monthly_pnl += pnl
            
            # Update risk engine
            self.risk_engine.update_trade_result(pnl, {
                'timestamp': datetime.now(),
                'pnl': pnl,
                'period': period
            })
            
        except Exception as e:
            self.logger.error(f"Error updating P&L: {e}")
    
    async def reset_daily_pnl(self):
        """Reset daily P&L (call at start of new day)"""
        self.daily_pnl = 0.0
    
    async def reset_weekly_pnl(self):
        """Reset weekly P&L (call at start of new week)"""
        self.weekly_pnl = 0.0
    
    async def reset_monthly_pnl(self):
        """Reset monthly P&L (call at start of new month)"""
        self.monthly_pnl = 0.0
    
    def _get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history for risk engine"""
        try:
            # Convert sizing history to trade history format
            trade_history = []
            for entry in self.sizing_history:
                if hasattr(entry['result'], 'risk_metrics'):
                    trade_history.append({
                        'timestamp': entry['timestamp'],
                        'symbol': entry['symbol'],
                        'side': entry['side'],
                        'position_size': entry['result'].position_size,
                        'pnl': 0.0  # Placeholder - would be updated with actual P&L
                    })
            
            return trade_history
            
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []
    
    async def _create_default_result(self, account_balance: float) -> PositionSizingResult:
        """Create default position sizing result when calculation fails"""
        default_risk_metrics = await self.risk_engine._create_default_metrics(account_balance)
        
        return PositionSizingResult(
            position_size=0.0,
            risk_metrics=default_risk_metrics,
            risk_limits_respected=False,
            warnings=["Error in position sizing calculation"],
            recommendations=["Review system configuration and try again"]
        )
    
    def get_sizing_history(self) -> List[Dict[str, Any]]:
        """Get position sizing history"""
        return self.sizing_history.copy()
    
    def get_risk_engine_metrics(self) -> Dict[str, Any]:
        """Get risk engine performance metrics"""
        return self.risk_engine.get_performance_metrics()
    
    def reset(self):
        """Reset position sizer (for testing)"""
        self.current_positions.clear()
        self.total_exposure = 0.0
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0
        self.sizing_history.clear()
        self.risk_engine.reset() 