"""
Simple Test Script - Basic Functionality Testing
Tests basic components without complex import dependencies
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    
    try:
        # Test utils imports
        from utils.logger import get_logger
        print("✅ Logger import successful")
        
        from utils.config_validator import ConfigValidator
        print("✅ Config validator import successful")
        
        from utils.performance_monitor import PerformanceMonitor
        print("✅ Performance monitor import successful")
        
        # Test config imports
        from config.intelligent_config import load_config
        print("✅ Config loader import successful")
        
        # Test data imports
        from data.data_processor import MultiTimeframeDataProcessor
        print("✅ Data processor import successful")
        
        from data.label_generator import IntuitionLabelGenerator
        print("✅ Label generator import successful")
        
        # Test models imports
        from models.intuition_model import IntuitionLearningModel
        print("✅ Intuition model import successful")
        
        from models.rule_extractor import RuleExtractor
        print("✅ Rule extractor import successful")
        
        # Test core imports
        from core.trading_agent import TradingAgent
        print("✅ Trading agent import successful")
        
        from core.position_manager import PositionManager
        print("✅ Position manager import successful")
        
        from core.entry_analyzer import EntryAnalyzer
        print("✅ Entry analyzer import successful")
        
        from core.exit_manager import ExitManager
        print("✅ Exit manager import successful")
        
        from core.market_context import MarketContextAnalyzer
        print("✅ Market context analyzer import successful")
        
        # Test risk imports
        from risk.position_sizer import PositionSizer
        print("✅ Position sizer import successful")
        
        from risk.adaptive_risk_engine import AdaptiveRiskEngine
        print("✅ Risk engine import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from config.intelligent_config import load_config
        
        config = load_config()
        print("✅ Configuration loaded successfully")
        print(f"   - Trading config: {'trading' in config}")
        print(f"   - Data config: {'data' in config}")
        print(f"   - Model config: {'model' in config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

def test_component_initialization():
    """Test component initialization"""
    print("\nTesting component initialization...")
    
    try:
        from config.intelligent_config import load_config
        from utils.logger import get_logger
        from utils.config_validator import ConfigValidator
        from utils.performance_monitor import PerformanceMonitor
        
        config = load_config()
        logger = get_logger(__name__)
        
        # Test config validator
        validator = ConfigValidator()
        print("✅ Config validator initialized")
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        print("✅ Performance monitor initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        return False

async def test_async_components():
    """Test async components"""
    print("\nTesting async components...")
    
    try:
        from utils.config_validator import ConfigValidator
        from utils.performance_monitor import PerformanceMonitor
        
        # Test config validation
        validator = ConfigValidator()
        results = await validator.validate_configs()
        print("✅ Config validation completed")
        print(f"   - Overall valid: {results.get('overall_valid', False)}")
        
        # Test performance monitor
        monitor = PerformanceMonitor()
        metrics = await monitor.update_metrics()
        print("✅ Performance metrics updated")
        
        return True
        
    except Exception as e:
        print(f"❌ Async component test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Simple System Test")
    print("=" * 50)
    
    # Test 1: Basic imports
    imports_ok = test_basic_imports()
    
    # Test 2: Configuration loading
    config_ok = test_config_loading()
    
    # Test 3: Component initialization
    init_ok = test_component_initialization()
    
    # Test 4: Async components
    async_ok = asyncio.run(test_async_components())
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Config Loading: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Component Init: {'✅ PASS' if init_ok else '❌ FAIL'}")
    print(f"Async Components: {'✅ PASS' if async_ok else '❌ FAIL'}")
    
    total_tests = 4
    passed_tests = sum([imports_ok, config_ok, init_ok, async_ok])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for use.")
    elif passed_tests >= total_tests // 2:
        print("⚠️ Most tests passed. Some components may need attention.")
    else:
        print("🚨 Multiple test failures. System needs fixes before use.")

if __name__ == "__main__":
    main() 