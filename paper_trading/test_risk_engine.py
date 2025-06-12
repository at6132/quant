import pytest
import numpy as np
from .risk_engine import RiskEngine

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
        (0.0, 1000, 0.98),    # No loss, high prob
        (-100, 1000, 0.98),   # Small loss, high prob
        (-1000, 1000, 0.98),  # Large loss, high prob
        (-100, 1000, 0.8),    # Small loss, low prob (should not martingale)
    ]
    
    for loss, atr, prob in test_cases:
        engine.last_loss = loss
        engine.price_history = [1000] * 20  # Set ATR
        mult = engine.tempered_martingale(prob)
        # Multiplier should be between 1 and 1 + 2*gamma
        assert 1.0 <= mult <= 1.0 + 2*engine.gamma
        # No loss or low prob should give no multiplier
        if loss == 0 or prob < engine.high_conf_threshold:
            assert mult == 1.0

def test_position_sizing():
    """Test full position sizing calculation."""
    engine = RiskEngine()
    # Simulate some price history and returns to avoid zero ATR/vol_mult
    engine.price_history = [50000 + i for i in range(25)]
    engine.daily_returns = np.random.normal(0, 100, 252).tolist()
    
    # Test with different scenarios
    test_cases = [
        (100000, 50000, 0.7),  # High prob, good capital
        (100000, 50000, 0.5),  # Neutral prob, good capital
        (10000, 50000, 0.7),   # High prob, low capital
    ]
    
    for equity, price, prob in test_cases:
        size, debug = engine.calculate_position_size(equity, price, prob)
        # Size should be positive and within limits, and not nan/inf
        assert np.isfinite(size)
        assert size >= 0
        assert size * price <= equity * engine.max_position_pct
        # Debug info should contain all components
        assert all(k in debug for k in ['p_star', 'kelly', 'convexity', 'martingale'])

def test_drawdown_lock():
    """Test drawdown lock mechanism."""
    engine = RiskEngine()
    # Simulate price history and returns
    engine.price_history = [50000 + i for i in range(25)]
    engine.daily_returns = [-100] * 252  # Large negative returns
    # Simulate a large drawdown
    engine.current_drawdown = 1000
    # Calculate position size
    size, debug = engine.calculate_position_size(100000, 50000, 0.7)
    # Martingale should be locked at 1.0
    assert debug['martingale'] == 1.0

def test_var_constraint():
    """Test VaR-based position limits."""
    engine = RiskEngine()
    # Simulate price history and high volatility
    engine.price_history = [50000 + i for i in range(25)]
    engine.daily_returns = [-1000] * 252  # Large negative returns
    # Calculate position size
    size, debug = engine.calculate_position_size(100000, 50000, 0.7)
    # Size should be limited by VaR
    assert size * 50000 <= abs(debug['var']) * engine.var_multiplier

def test_posterior_kelly():
    """Test posterior Kelly fraction calculation."""
    engine = RiskEngine(beta=2.0)
    # Test with different probabilities and R ratios
    test_cases = [
        (0.6, 2.0),  # Moderate edge, 2:1 reward
        (0.7, 1.5),  # Strong edge, 1.5:1 reward
        (0.5, 3.0),  # No edge, 3:1 reward
    ]
    for p, R in test_cases:
        engine.rolling_signals = 100  # Set signal count
        kelly = engine.posterior_kelly(p, R)
        # Kelly should be between 0 and 1
        assert 0 <= kelly <= 1
        # Higher probability should give higher Kelly
        if p > 0.5:
            assert kelly > 0

def test_volatility_multiplier():
    """Test 95th percentile volatility multiplier."""
    engine = RiskEngine()
    # Simulate returns
    engine.daily_returns = np.random.normal(0, 1, 252)  # Random returns
    # Calculate volatility multiplier
    vol_mult = engine.calculate_volatility_multiplier()
    # Multiplier should be positive
    assert vol_mult > 0

def test_ou_barrier():
    """Test OU drawdown barrier update."""
    engine = RiskEngine(ou_mu=0.0, ou_theta=0.1, ou_eta=0.1)
    # Simulate a drawdown
    engine.current_drawdown = 1000
    engine.update_ou_barrier()
    # Barrier should be updated
    assert engine.ou_barrier != 0.0

if __name__ == "__main__":
    pytest.main([__file__]) 