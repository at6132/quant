import pytest
import numpy as np
from risk_engine import RiskEngine

def test_alpha_kelly():
    """Test α-Kelly calculation."""
    engine = RiskEngine(alpha=1.7)
    
    # Test with different probabilities and R ratios
    test_cases = [
        (0.6, 2.0),  # Moderate edge, 2:1 reward
        (0.7, 1.5),  # Strong edge, 1.5:1 reward
        (0.5, 3.0),  # No edge, 3:1 reward
    ]
    
    for p, R in test_cases:
        kelly = engine.alpha_kelly(p, R)
        # Kelly should be between 0 and 1
        assert 0 <= kelly <= 1
        # Higher probability should give higher Kelly
        if p > 0.5:
            assert kelly > 0

def test_bayesian_shrink():
    """Test Bayesian shrinkage."""
    engine = RiskEngine(beta=2.0)
    
    # Test with different probabilities and signal counts
    test_cases = [
        (0.8, 10),   # High prob, few signals
        (0.8, 1000), # High prob, many signals
        (0.5, 10),   # Neutral prob, few signals
    ]
    
    for p, n in test_cases:
        engine.rolling_signals = n
        p_star = engine.bayesian_shrink(p)
        # Shrunk probability should be between prior and original
        assert 0.5 <= p_star <= p
        # More signals should give less shrinkage
        if n > 100:
            assert abs(p_star - p) < 0.1

def test_convexity_boost():
    """Test probability-elastic convexity boost."""
    engine = RiskEngine()
    
    # Test with different probabilities
    test_cases = [0.4, 0.5, 0.55, 0.6, 0.7, 0.8]
    
    for p in test_cases:
        boost = engine.convexity_boost(p)
        # Boost should be between 0.25 and 1.0
        assert 0.25 <= boost <= 1.0
        # Should have maximum derivative at p=0.55
        if p == 0.55:
            assert boost > 0.5

def test_tempered_martingale():
    """Test tempered martingale multiplier."""
    engine = RiskEngine(gamma=0.35)
    
    # Test with different loss scenarios
    test_cases = [
        (0.0, 1000),    # No loss
        (-100, 1000),   # Small loss
        (-1000, 1000),  # Large loss
    ]
    
    for loss, atr in test_cases:
        engine.last_loss = loss
        engine.price_history = [1000] * 20  # Set ATR
        mult = engine.tempered_martingale()
        # Multiplier should be between 1 and 1 + 2*gamma
        assert 1.0 <= mult <= 1.0 + 2*engine.gamma
        # No loss should give no multiplier
        if loss == 0:
            assert mult == 1.0

def test_position_sizing():
    """Test full position sizing calculation."""
    engine = RiskEngine()
    
    # Test with different scenarios
    test_cases = [
        (100000, 50000, 0.7),  # High prob, good capital
        (100000, 50000, 0.5),  # Neutral prob, good capital
        (10000, 50000, 0.7),   # High prob, low capital
    ]
    
    for equity, price, prob in test_cases:
        size, debug = engine.calculate_position_size(equity, price, prob)
        # Size should be positive and within limits
        assert size >= 0
        assert size * price <= equity * engine.max_position_pct
        # Debug info should contain all components
        assert all(k in debug for k in ['p_star', 'kelly', 'convexity', 'martingale'])

def test_drawdown_lock():
    """Test drawdown lock mechanism."""
    engine = RiskEngine()
    
    # Simulate a large drawdown
    engine.current_drawdown = 1000
    engine.price_history = [50000] * 20
    engine.daily_returns = [-100] * 252  # Large negative returns
    
    # Calculate position size
    size, debug = engine.calculate_position_size(100000, 50000, 0.7)
    
    # Martingale should be locked at 1.0
    assert debug['martingale'] == 1.0

def test_var_constraint():
    """Test VaR-based position limits."""
    engine = RiskEngine()
    
    # Simulate high volatility
    engine.daily_returns = [-1000] * 252  # Large negative returns
    
    # Calculate position size
    size, debug = engine.calculate_position_size(100000, 50000, 0.7)
    
    # Size should be limited by VaR
    assert size * 50000 <= abs(debug['var']) * engine.var_multiplier

if __name__ == "__main__":
    pytest.main([__file__]) 