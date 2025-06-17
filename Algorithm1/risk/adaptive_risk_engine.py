"""
Adaptive Risk Engine - Advanced Risk Management System
Implements Bayesian Kelly sizing, volatility-weighted buffer, conditional Martingale escalation, and stochastic drawdown barrier
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from scipy.optimize import minimize

from utils.logger import get_logger
from config.intelligent_config import load_config

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    """Risk metrics data structure"""
    kelly_fraction: float
    volatility_buffer: float
    drawdown_barrier: float
    position_size: float
    max_position_size: float
    risk_score: float
    confidence: float
    reasoning: List[str]


class AdaptiveRiskEngine:
    """
    Advanced Adaptive Risk Management System
    """
    
    def __init__(self, config_path: str = "config/intelligent_config.yaml"):
        """Initialize adaptive risk engine"""
        self.config = load_config(config_path)
        self.logger = logger
        
        # Configuration
        self.risk_config = self.config.get('risk_management', {})
        self.kelly_config = self.risk_config.get('kelly_criterion', {})
        self.volatility_config = self.risk_config.get('volatility_weighting', {})
        self.martingale_config = self.risk_config.get('martingale', {})
        self.drawdown_config = self.risk_config.get('drawdown_barrier', {})
        
        # Performance tracking
        self.trade_history = []
        self.risk_history = []
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Bayesian parameters
        self.win_rate_prior = 0.5
        self.avg_win_prior = 1.0
        self.avg_loss_prior = 1.0
        self.prior_weight = 0.1
        
        self.logger.info("Adaptive Risk Engine initialized successfully")
    
    async def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        model_predictions: Dict[str, float],
        market_context: Dict[str, Any],
        trade_history: List[Dict[str, Any]] = None
    ) -> RiskMetrics:
        """
        Calculate optimal position size using adaptive risk management
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            account_balance: Current account balance
            model_predictions: Model predictions
            market_context: Market context information
            trade_history: Historical trade data
            
        Returns:
            RiskMetrics object with position sizing results
        """
        try:
            # Update trade history
            if trade_history:
                self.trade_history = trade_history
            
            # Calculate basic risk parameters
            risk_per_unit = abs(entry_price - stop_loss)
            reward_per_unit = abs(take_profit - entry_price)
            
            if risk_per_unit == 0:
                return await self._create_default_metrics(account_balance)
            
            # Calculate Kelly fraction
            kelly_fraction = await self._calculate_kelly_fraction(
                model_predictions, market_context
            )
            
            # Calculate volatility buffer
            volatility_buffer = await self._calculate_volatility_buffer(
                market_context, risk_per_unit
            )
            
            # Calculate drawdown barrier
            drawdown_barrier = await self._calculate_drawdown_barrier(
                account_balance, market_context
            )
            
            # Calculate Martingale adjustment
            martingale_adjustment = await self._calculate_martingale_adjustment(
                model_predictions, market_context
            )
            
            # Combine all factors
            base_size = account_balance * kelly_fraction
            adjusted_size = base_size * volatility_buffer * drawdown_barrier * martingale_adjustment
            
            # Apply position limits
            max_position_pct = self.risk_config.get('max_position_pct', 0.1)
            max_position_size = account_balance * max_position_pct
            
            final_position_size = min(adjusted_size, max_position_size)
            
            # Calculate risk score
            risk_score = await self._calculate_risk_score(
                kelly_fraction, volatility_buffer, drawdown_barrier, martingale_adjustment
            )
            
            # Calculate confidence
            confidence = await self._calculate_confidence(
                model_predictions, market_context, risk_score
            )
            
            # Generate reasoning
            reasoning = await self._generate_reasoning(
                kelly_fraction, volatility_buffer, drawdown_barrier, 
                martingale_adjustment, risk_score, confidence
            )
            
            # Create risk metrics
            metrics = RiskMetrics(
                kelly_fraction=kelly_fraction,
                volatility_buffer=volatility_buffer,
                drawdown_barrier=drawdown_barrier,
                position_size=final_position_size,
                max_position_size=max_position_size,
                risk_score=risk_score,
                confidence=confidence,
                reasoning=reasoning
            )
            
            # Store in history
            self.risk_history.append(metrics)
            
            self.logger.info(f"Position size: {final_position_size:.2f}, Risk score: {risk_score:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return await self._create_default_metrics(account_balance)
    
    async def _calculate_kelly_fraction(
        self,
        model_predictions: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate Kelly fraction using Bayesian approach"""
        try:
            # Get model predictions
            win_prob = model_predictions.get('entry_probability', 0.5)
            entry_confidence = model_predictions.get('entry_confidence', 0.5)
            
            # Get historical performance
            if len(self.trade_history) > 0:
                wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
                total_trades = len(self.trade_history)
                historical_win_rate = wins / total_trades
                
                # Calculate average win and loss
                winning_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) > 0]
                losing_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) < 0]
                
                if winning_trades:
                    avg_win = np.mean(winning_trades)
                else:
                    avg_win = self.avg_win_prior
                
                if losing_trades:
                    avg_loss = abs(np.mean(losing_trades))
                else:
                    avg_loss = self.avg_loss_prior
            else:
                historical_win_rate = self.win_rate_prior
                avg_win = self.avg_win_prior
                avg_loss = self.avg_loss_prior
            
            # Bayesian combination of model and historical data
            combined_win_rate = (
                self.prior_weight * self.win_rate_prior +
                (1 - self.prior_weight) * (0.7 * win_prob + 0.3 * historical_win_rate)
            )
            
            # Adjust for model confidence
            confidence_adjustment = 0.5 + 0.5 * entry_confidence
            adjusted_win_rate = combined_win_rate * confidence_adjustment
            
            # Calculate Kelly fraction
            if avg_loss > 0:
                kelly_fraction = (adjusted_win_rate * avg_win - (1 - adjusted_win_rate) * avg_loss) / avg_win
            else:
                kelly_fraction = 0.0
            
            # Apply Kelly fraction limits
            max_kelly = self.kelly_config.get('max_fraction', 0.25)
            min_kelly = self.kelly_config.get('min_fraction', 0.01)
            
            kelly_fraction = np.clip(kelly_fraction, min_kelly, max_kelly)
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.02  # Default 2%
    
    async def _calculate_volatility_buffer(
        self,
        market_context: Dict[str, Any],
        risk_per_unit: float
    ) -> float:
        """Calculate volatility-weighted buffer"""
        try:
            volatility_regime = market_context.get('volatility_regime', 'normal')
            current_volatility = market_context.get('current_volatility', 0.02)
            
            # Base buffer
            base_buffer = self.volatility_config.get('base_buffer', 1.0)
            
            # Volatility adjustments
            if volatility_regime == 'high':
                vol_multiplier = self.volatility_config.get('high_vol_multiplier', 0.7)
            elif volatility_regime == 'low':
                vol_multiplier = self.volatility_config.get('low_vol_multiplier', 1.2)
            else:
                vol_multiplier = 1.0
            
            # Trend strength adjustment
            trend_strength = abs(market_context.get('trend_strength', 0.0))
            trend_multiplier = 1.0 + trend_strength * 0.3  # Stronger trend = larger position
            
            # Market phase adjustment
            market_phase = market_context.get('market_phase', 'unknown')
            phase_multiplier = self.volatility_config.get('phase_multipliers', {}).get(market_phase, 1.0)
            
            # Calculate final buffer
            volatility_buffer = base_buffer * vol_multiplier * trend_multiplier * phase_multiplier
            
            # Apply limits
            max_buffer = self.volatility_config.get('max_buffer', 1.5)
            min_buffer = self.volatility_config.get('min_buffer', 0.5)
            
            return np.clip(volatility_buffer, min_buffer, max_buffer)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility buffer: {e}")
            return 1.0
    
    async def _calculate_drawdown_barrier(
        self,
        account_balance: float,
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate stochastic drawdown barrier"""
        try:
            # Get drawdown configuration
            max_drawdown_pct = self.drawdown_config.get('max_drawdown_pct', 0.20)
            barrier_threshold = self.drawdown_config.get('barrier_threshold', 0.15)
            
            # Calculate current drawdown
            if len(self.trade_history) > 0:
                peak_balance = max(trade.get('cumulative_balance', account_balance) for trade in self.trade_history)
                current_drawdown = (peak_balance - account_balance) / peak_balance
            else:
                current_drawdown = 0.0
            
            self.current_drawdown = current_drawdown
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Calculate barrier
            if current_drawdown > barrier_threshold:
                # Reduce position size as drawdown increases
                reduction_factor = 1.0 - (current_drawdown - barrier_threshold) / (max_drawdown_pct - barrier_threshold)
                barrier = max(reduction_factor, 0.1)  # Minimum 10% position size
            else:
                barrier = 1.0
            
            # Volatility adjustment to barrier
            volatility_regime = market_context.get('volatility_regime', 'normal')
            if volatility_regime == 'high':
                barrier *= 0.8  # Further reduce in high volatility
            
            return barrier
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown barrier: {e}")
            return 1.0
    
    async def _calculate_martingale_adjustment(
        self,
        model_predictions: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> float:
        """Calculate conditional Martingale escalation"""
        try:
            # Get Martingale configuration
            max_consecutive_losses = self.martingale_config.get('max_consecutive_losses', 3)
            escalation_factor = self.martingale_config.get('escalation_factor', 1.5)
            de_escalation_factor = self.martingale_config.get('de_escalation_factor', 0.8)
            
            # Check if Martingale should be applied
            if self.consecutive_losses >= max_consecutive_losses:
                # High confidence model prediction
                entry_confidence = model_predictions.get('entry_confidence', 0.5)
                entry_prob = model_predictions.get('entry_probability', 0.5)
                
                if entry_confidence > 0.8 and entry_prob > 0.7:
                    # Apply Martingale escalation
                    adjustment = escalation_factor ** (self.consecutive_losses - max_consecutive_losses + 1)
                    
                    # Limit maximum escalation
                    max_escalation = self.martingale_config.get('max_escalation', 3.0)
                    adjustment = min(adjustment, max_escalation)
                    
                    return adjustment
            
            # De-escalation after wins
            if self.consecutive_wins > 0:
                adjustment = de_escalation_factor ** min(self.consecutive_wins, 3)
                return adjustment
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating Martingale adjustment: {e}")
            return 1.0
    
    async def _calculate_risk_score(
        self,
        kelly_fraction: float,
        volatility_buffer: float,
        drawdown_barrier: float,
        martingale_adjustment: float
    ) -> float:
        """Calculate overall risk score"""
        try:
            # Normalize factors
            kelly_score = kelly_fraction / 0.25  # Normalize to max Kelly
            vol_score = (volatility_buffer - 0.5) / 0.5  # Normalize to 0-1
            barrier_score = drawdown_barrier
            martingale_score = 1.0 / martingale_adjustment  # Inverse relationship
            
            # Weighted combination
            weights = self.risk_config.get('risk_score_weights', {
                'kelly': 0.3,
                'volatility': 0.2,
                'drawdown': 0.3,
                'martingale': 0.2
            })
            
            risk_score = (
                weights['kelly'] * kelly_score +
                weights['volatility'] * vol_score +
                weights['drawdown'] * barrier_score +
                weights['martingale'] * martingale_score
            )
            
            return np.clip(risk_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    async def _calculate_confidence(
        self,
        model_predictions: Dict[str, float],
        market_context: Dict[str, Any],
        risk_score: float
    ) -> float:
        """Calculate confidence in risk assessment"""
        try:
            # Model confidence
            entry_confidence = model_predictions.get('entry_confidence', 0.5)
            
            # Market context confidence
            context_confidence = market_context.get('confidence', 0.5)
            
            # Risk score confidence (lower risk = higher confidence)
            risk_confidence = 1.0 - risk_score
            
            # Historical confidence
            if len(self.trade_history) > 10:
                recent_trades = self.trade_history[-10:]
                win_rate = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0) / len(recent_trades)
                historical_confidence = win_rate
            else:
                historical_confidence = 0.5
            
            # Combine confidences
            confidence = (
                0.4 * entry_confidence +
                0.2 * context_confidence +
                0.2 * risk_confidence +
                0.2 * historical_confidence
            )
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _generate_reasoning(
        self,
        kelly_fraction: float,
        volatility_buffer: float,
        drawdown_barrier: float,
        martingale_adjustment: float,
        risk_score: float,
        confidence: float
    ) -> List[str]:
        """Generate reasoning for risk metrics"""
        reasoning = []
        
        try:
            reasoning.append(f"Kelly fraction: {kelly_fraction:.3f}")
            reasoning.append(f"Volatility buffer: {volatility_buffer:.3f}")
            reasoning.append(f"Drawdown barrier: {drawdown_barrier:.3f}")
            reasoning.append(f"Martingale adjustment: {martingale_adjustment:.3f}")
            reasoning.append(f"Risk score: {risk_score:.3f}")
            reasoning.append(f"Confidence: {confidence:.3f}")
            
            # Add specific reasoning
            if kelly_fraction < 0.05:
                reasoning.append("Low Kelly fraction - conservative sizing")
            elif kelly_fraction > 0.15:
                reasoning.append("High Kelly fraction - aggressive sizing")
            
            if volatility_buffer < 0.8:
                reasoning.append("Reduced size due to high volatility")
            elif volatility_buffer > 1.2:
                reasoning.append("Increased size due to low volatility")
            
            if drawdown_barrier < 0.8:
                reasoning.append("Reduced size due to drawdown protection")
            
            if martingale_adjustment > 1.5:
                reasoning.append("Martingale escalation applied")
            elif martingale_adjustment < 0.8:
                reasoning.append("Martingale de-escalation applied")
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return ["Error generating reasoning"]
    
    async def _create_default_metrics(self, account_balance: float) -> RiskMetrics:
        """Create default risk metrics when calculation fails"""
        return RiskMetrics(
            kelly_fraction=0.02,
            volatility_buffer=1.0,
            drawdown_barrier=1.0,
            position_size=account_balance * 0.02,
            max_position_size=account_balance * 0.1,
            risk_score=0.5,
            confidence=0.5,
            reasoning=["Default risk metrics due to calculation error"]
        )
    
    def update_trade_result(self, pnl: float, trade_data: Dict[str, Any]):
        """Update trade result for risk calculations"""
        try:
            # Update consecutive wins/losses
            if pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
            # Update trade history
            trade_data['pnl'] = pnl
            self.trade_history.append(trade_data)
            
            # Update Bayesian parameters
            if len(self.trade_history) > 0:
                wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
                total_trades = len(self.trade_history)
                self.win_rate_prior = wins / total_trades
                
                winning_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) > 0]
                losing_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) < 0]
                
                if winning_trades:
                    self.avg_win_prior = np.mean(winning_trades)
                if losing_trades:
                    self.avg_loss_prior = abs(np.mean(losing_trades))
            
        except Exception as e:
            self.logger.error(f"Error updating trade result: {e}")
    
    def get_risk_history(self) -> List[RiskMetrics]:
        """Get risk metrics history"""
        return self.risk_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if len(self.trade_history) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0
            }
        
        wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        total_trades = len(self.trade_history)
        win_rate = wins / total_trades
        
        winning_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) > 0]
        losing_trades = [trade['pnl'] for trade in self.trade_history if trade.get('pnl', 0) < 0]
        
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown
        }
    
    def reset(self):
        """Reset risk engine (for testing)"""
        self.trade_history.clear()
        self.risk_history.clear()
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0 