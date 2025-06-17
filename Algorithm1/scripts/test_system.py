#!/usr/bin/env python3
"""
Test script to verify the intelligent trading system components.

How to run:
- From the parent directory (quant/):
    python -m Algorithm1.scripts.test_system
- Or directly (from Algorithm1/):
    python scripts/test_system.py
"""

import sys
import os
from pathlib import Path

# Ensure the parent directory is in sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    try:
        from data.data_processor import MultiTimeframeDataProcessor
        print("✓ MultiTimeframeDataProcessor imported successfully")
    except Exception as e:
        print(f"✗ Error importing MultiTimeframeDataProcessor: {e}")
        return False
    try:
        from data.label_generator import IntuitionLabelGenerator
        print("✓ IntuitionLabelGenerator imported successfully")
    except Exception as e:
        print(f"✗ Error importing IntuitionLabelGenerator: {e}")
        return False
    try:
        from models.intuition_model import IntuitionLearningModel
        print("✓ IntuitionLearningModel imported successfully")
    except Exception as e:
        print(f"✗ Error importing IntuitionLearningModel: {e}")
        return False
    try:
        from indicators import INDICATOR_REGISTRY, get_all_indicators
        print("✓ Indicators imported successfully")
        print(f"  Available indicators: {get_all_indicators()}")
    except Exception as e:
        print(f"✗ Error importing indicators: {e}")
        return False
    try:
        from utils.logger import setup_logger, get_logger
        print("✓ Logger imported successfully")
    except Exception as e:
        print(f"✗ Error importing logger: {e}")
        return False
    return True

def test_data_processor():
    """Test the data processor with sample data."""
    print("\nTesting data processor...")
    try:
        from data.data_processor import MultiTimeframeDataProcessor
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=1000, freq='15S', tz='UTC')
        sample_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 1000),
            'high': np.random.uniform(45000, 55000, 1000),
            'low': np.random.uniform(45000, 55000, 1000),
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(100, 1000, 1000)
        }, index=dates)
        os.makedirs('data', exist_ok=True)
        sample_data.to_csv('data/sample_data.csv')
        print("✓ Sample data created")
        config = {
            'data': {
                'timeframes': ['15s', '1m', '5m', '15m', '1h', '4h']
            }
        }
        processor = MultiTimeframeDataProcessor(config)
        raw_data = processor.load_data('data/sample_data.csv')
        print(f"✓ Data loaded: {len(raw_data)} rows")
        multi_tf_data = processor.create_multi_timeframe_data()
        print(f"✓ Multi-timeframe data created: {list(multi_tf_data.keys())}")
        feature_matrix = processor.create_feature_matrix()
        print(f"✓ Feature matrix created: {len(feature_matrix)} rows, {len(feature_matrix.columns)} columns")
        return True
    except Exception as e:
        print(f"✗ Error in data processor test: {e}")
        return False

def test_label_generator():
    """Test the label generator."""
    print("\nTesting label generator...")
    try:
        from data.label_generator import IntuitionLabelGenerator
        # Create sample feature matrix
        dates = pd.date_range('2024-01-01', periods=500, freq='15S', tz='UTC')
        sample_features = pd.DataFrame({
            'close': np.random.uniform(45000, 55000, 500),
            'smc_15s_trend': np.random.choice([-1, 0, 1], 500),
            'breaker_15s_signal': np.random.choice([-1, 0, 1], 500),
            'ict_15s_entry': np.random.uniform(0, 1, 500),
            'it_15s_align': np.random.choice([-1, 0, 1], 500)
        }, index=dates)
        config = {
            'labeling': {
                'lookforward_periods': [5, 10, 20, 30],
                'threshold': 0.002,
                'min_move_size': 0.002,
                'max_holding_time': 30,
                'min_holding_time': 5
            }
        }
        label_generator = IntuitionLabelGenerator(config)
        labels = label_generator.generate_labels(sample_features)
        print(f"✓ Labels generated: {len(labels.columns)} label columns")
        print(f"  Label columns: {list(labels.columns)}")
        is_valid = label_generator.validate_labels(labels)
        print(f"✓ Label validation: {'PASSED' if is_valid else 'FAILED'}")
        return is_valid
    except Exception as e:
        print(f"✗ Error in label generator test: {e}")
        return False

def test_model():
    """Test the intuition model."""
    print("\nTesting intuition model...")
    try:
        from models.intuition_model import IntuitionLearningModel
        config = {
            'model': {
                'input_dim': 10,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'sequence_length': 20,
                'prediction_horizon': 1,
                'learning_rate': 1e-4,
                'batch_size': 16,
                'num_epochs': 5,
                'patience': 3
            }
        }
        model = IntuitionLearningModel(config)
        print("✓ Model initialized")
        model.build_model(10)
        print("✓ Model built successfully")
        sample_features = pd.DataFrame(np.random.randn(100, 10))
        sample_labels = pd.DataFrame({
            'entry_probability': np.random.uniform(0, 1, 100),
            'entry_direction': np.random.choice([-1, 0, 1], 100),
            'entry_confidence': np.random.uniform(0, 1, 100),
            'position_size_multiplier': np.random.uniform(0.1, 2.0, 100),
            'tp_distance': np.random.uniform(0.005, 0.10, 100),
            'sl_distance': np.random.uniform(0.005, 0.08, 100),
            'trail_aggressiveness': np.random.uniform(0.1, 1.0, 100),
            'hold_duration': np.random.uniform(5, 30, 100),
            'exit_probability': np.random.uniform(0, 1, 100),
            'exit_urgency': np.random.uniform(0.1, 1.0, 100)
        })
        train_loader, val_loader = model.prepare_data(sample_features, sample_labels)
        print(f"✓ Data prepared: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples")
        return True
    except Exception as e:
        print(f"✗ Error in model test: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Intelligent Trading System Components")
    print("=" * 50)
    tests = [
        ("Imports", test_imports),
        ("Data Processor", test_data_processor),
        ("Label Generator", test_label_generator),
        ("Intuition Model", test_model)
    ]
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print("=" * 50)
    passed = 0
    total = len(results)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    print(f"\nOverall: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 All tests passed! System is ready for training.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 