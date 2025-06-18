"""
Complete System Test - End-to-End Testing
Tests the complete intelligent trading system with all components integrated
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.intelligent_trading_system import IntelligentTradingSystem
from utils.logger import get_logger
from utils.config_validator import ConfigValidator

logger = get_logger(__name__)


class CompleteSystemTest:
    """
    Complete System Test Suite
    """
    
    def __init__(self):
        """Initialize test suite"""
        self.logger = logger
        self.test_results = []
        self.system = None
        
    async def run_all_tests(self):
        """Run all system tests"""
        try:
            self.logger.info("Starting Complete System Test Suite")
            self.logger.info("=" * 60)
            
            # Test 1: Configuration Validation
            await self._test_configuration_validation()
            
            # Test 2: System Initialization
            await self._test_system_initialization()
            
            # Test 3: Component Integration
            await self._test_component_integration()
            
            # Test 4: Data Processing Pipeline
            await self._test_data_pipeline()
            
            # Test 5: Model Prediction
            await self._test_model_prediction()
            
            # Test 6: Risk Management
            await self._test_risk_management()
            
            # Test 7: Trading Logic
            await self._test_trading_logic()
            
            # Test 8: Performance Monitoring
            await self._test_performance_monitoring()
            
            # Test 9: System Health
            await self._test_system_health()
            
            # Test 10: End-to-End Workflow
            await self._test_end_to_end_workflow()
            
            # Generate test report
            await self._generate_test_report()
            
        except Exception as e:
            self.logger.error(f"Error in test suite: {e}")
            await self._cleanup()
    
    async def _test_configuration_validation(self):
        """Test configuration validation"""
        try:
            self.logger.info("Test 1: Configuration Validation")
            
            validator = ConfigValidator()
            results = await validator.validate_configs()
            
            if results['overall_valid']:
                self.logger.info("✅ Configuration validation passed")
                self.test_results.append({
                    'test': 'Configuration Validation',
                    'status': 'PASS',
                    'details': 'All configuration files are valid'
                })
            else:
                self.logger.error("❌ Configuration validation failed")
                for config_name, result in results.items():
                    if config_name != 'overall_valid' and not result['valid']:
                        self.logger.error(f"  {config_name}: {result['errors']}")
                
                self.test_results.append({
                    'test': 'Configuration Validation',
                    'status': 'FAIL',
                    'details': f"Configuration errors: {results}"
                })
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation test failed: {e}")
            self.test_results.append({
                'test': 'Configuration Validation',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_system_initialization(self):
        """Test system initialization"""
        try:
            self.logger.info("Test 2: System Initialization")
            
            self.system = IntelligentTradingSystem()
            success = await self.system.initialize()
            
            if success:
                self.logger.info("✅ System initialization passed")
                self.test_results.append({
                    'test': 'System Initialization',
                    'status': 'PASS',
                    'details': 'All components initialized successfully'
                })
            else:
                self.logger.error("❌ System initialization failed")
                self.test_results.append({
                    'test': 'System Initialization',
                    'status': 'FAIL',
                    'details': 'Failed to initialize system components'
                })
            
        except Exception as e:
            self.logger.error(f"❌ System initialization test failed: {e}")
            self.test_results.append({
                'test': 'System Initialization',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_component_integration(self):
        """Test component integration"""
        try:
            self.logger.info("Test 3: Component Integration")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping component integration test - system not initialized")
                self.test_results.append({
                    'test': 'Component Integration',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Check all components are initialized
            components = [
                'data_processor', 'label_generator', 'intuition_model',
                'rule_extractor', 'trading_agent', 'position_manager',
                'entry_analyzer', 'exit_manager', 'market_context_analyzer',
                'position_sizer', 'performance_monitor'
            ]
            
            missing_components = []
            for component in components:
                if not hasattr(self.system, component) or getattr(self.system, component) is None:
                    missing_components.append(component)
            
            if not missing_components:
                self.logger.info("✅ Component integration passed")
                self.test_results.append({
                    'test': 'Component Integration',
                    'status': 'PASS',
                    'details': f'All {len(components)} components integrated'
                })
            else:
                self.logger.error(f"❌ Component integration failed - missing: {missing_components}")
                self.test_results.append({
                    'test': 'Component Integration',
                    'status': 'FAIL',
                    'details': f'Missing components: {missing_components}'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Component integration test failed: {e}")
            self.test_results.append({
                'test': 'Component Integration',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_data_pipeline(self):
        """Test data processing pipeline"""
        try:
            self.logger.info("Test 4: Data Processing Pipeline")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping data pipeline test - system not initialized")
                self.test_results.append({
                    'test': 'Data Processing Pipeline',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Test data processor
            timeframes = ['1m', '5m', '15m', '1h']
            test_data = {}
            
            for timeframe in timeframes:
                data = await self.system._get_market_data(timeframe)
                if data is not None and not data.empty:
                    test_data[timeframe] = data
            
            if len(test_data) == len(timeframes):
                self.logger.info("✅ Data pipeline passed")
                self.test_results.append({
                    'test': 'Data Processing Pipeline',
                    'status': 'PASS',
                    'details': f'Generated data for {len(timeframes)} timeframes'
                })
            else:
                self.logger.error(f"❌ Data pipeline failed - got {len(test_data)}/{len(timeframes)} timeframes")
                self.test_results.append({
                    'test': 'Data Processing Pipeline',
                    'status': 'FAIL',
                    'details': f'Missing data for timeframes: {set(timeframes) - set(test_data.keys())}'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Data pipeline test failed: {e}")
            self.test_results.append({
                'test': 'Data Processing Pipeline',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_model_prediction(self):
        """Test model prediction"""
        try:
            self.logger.info("Test 5: Model Prediction")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping model prediction test - system not initialized")
                self.test_results.append({
                    'test': 'Model Prediction',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Generate test data
            await self.system._update_market_data()
            await self.system._analyze_market_context()
            await self.system._generate_predictions()
            
            if self.system.model_predictions:
                self.logger.info("✅ Model prediction passed")
                self.test_results.append({
                    'test': 'Model Prediction',
                    'status': 'PASS',
                    'details': f'Generated predictions: {list(self.system.model_predictions.keys())}'
                })
            else:
                self.logger.error("❌ Model prediction failed - no predictions generated")
                self.test_results.append({
                    'test': 'Model Prediction',
                    'status': 'FAIL',
                    'details': 'No model predictions generated'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Model prediction test failed: {e}")
            self.test_results.append({
                'test': 'Model Prediction',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_risk_management(self):
        """Test risk management"""
        try:
            self.logger.info("Test 6: Risk Management")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping risk management test - system not initialized")
                self.test_results.append({
                    'test': 'Risk Management',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Test position sizing
            test_predictions = {
                'entry_probability': 0.7,
                'entry_confidence': 0.8,
                'direction': 'long'
            }
            
            test_context = {
                'volatility_regime': 'normal',
                'trend_direction': 'bullish',
                'confidence': 0.75
            }
            
            position_result = await self.system.position_sizer.calculate_position_size(
                symbol='BTC/USD',
                side='long',
                entry_price=50000.0,
                stop_loss=49000.0,
                take_profit=52000.0,
                account_balance=10000.0,
                model_predictions=test_predictions,
                market_context=test_context
            )
            
            if position_result and position_result.position_size >= 0:
                self.logger.info("✅ Risk management passed")
                self.test_results.append({
                    'test': 'Risk Management',
                    'status': 'PASS',
                    'details': f'Position size calculated: {position_result.position_size:.2f}'
                })
            else:
                self.logger.error("❌ Risk management failed - invalid position size")
                self.test_results.append({
                    'test': 'Risk Management',
                    'status': 'FAIL',
                    'details': 'Invalid position size calculation'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Risk management test failed: {e}")
            self.test_results.append({
                'test': 'Risk Management',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_trading_logic(self):
        """Test trading logic"""
        try:
            self.logger.info("Test 7: Trading Logic")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping trading logic test - system not initialized")
                self.test_results.append({
                    'test': 'Trading Logic',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Test entry analysis
            test_predictions = {
                'entry_probability': 0.8,
                'entry_confidence': 0.9,
                'direction': 'long'
            }
            
            test_context = {
                'volatility_regime': 'normal',
                'trend_direction': 'bullish',
                'confidence': 0.8
            }
            
            entry_analysis = await self.system.entry_analyzer.analyze_entry(
                self.system.market_data,
                test_predictions,
                {},
                test_context
            )
            
            if entry_analysis:
                self.logger.info("✅ Trading logic passed")
                self.test_results.append({
                    'test': 'Trading Logic',
                    'status': 'PASS',
                    'details': f'Entry analysis completed: should_enter={entry_analysis.should_enter}'
                })
            else:
                self.logger.error("❌ Trading logic failed - no entry analysis")
                self.test_results.append({
                    'test': 'Trading Logic',
                    'status': 'FAIL',
                    'details': 'Entry analysis failed'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Trading logic test failed: {e}")
            self.test_results.append({
                'test': 'Trading Logic',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_performance_monitoring(self):
        """Test performance monitoring"""
        try:
            self.logger.info("Test 8: Performance Monitoring")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping performance monitoring test - system not initialized")
                self.test_results.append({
                    'test': 'Performance Monitoring',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Test performance summary
            summary = await self.system.get_performance_summary()
            
            if summary and 'current_metrics' in summary:
                self.logger.info("✅ Performance monitoring passed")
                self.test_results.append({
                    'test': 'Performance Monitoring',
                    'status': 'PASS',
                    'details': 'Performance summary generated successfully'
                })
            else:
                self.logger.error("❌ Performance monitoring failed - no summary")
                self.test_results.append({
                    'test': 'Performance Monitoring',
                    'status': 'FAIL',
                    'details': 'Failed to generate performance summary'
                })
            
        except Exception as e:
            self.logger.error(f"❌ Performance monitoring test failed: {e}")
            self.test_results.append({
                'test': 'Performance Monitoring',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_system_health(self):
        """Test system health"""
        try:
            self.logger.info("Test 9: System Health")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping system health test - system not initialized")
                self.test_results.append({
                    'test': 'System Health',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Test system status
            status = await self.system.get_system_status()
            
            if status and hasattr(status, 'is_healthy'):
                self.logger.info("✅ System health passed")
                self.test_results.append({
                    'test': 'System Health',
                    'status': 'PASS',
                    'details': f'System healthy: {status.is_healthy}'
                })
            else:
                self.logger.error("❌ System health failed - no status")
                self.test_results.append({
                    'test': 'System Health',
                    'status': 'FAIL',
                    'details': 'Failed to get system status'
                })
            
        except Exception as e:
            self.logger.error(f"❌ System health test failed: {e}")
            self.test_results.append({
                'test': 'System Health',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _test_end_to_end_workflow(self):
        """Test end-to-end workflow"""
        try:
            self.logger.info("Test 10: End-to-End Workflow")
            
            if not self.system or not self.system.is_initialized:
                self.logger.warning("⚠️ Skipping end-to-end test - system not initialized")
                self.test_results.append({
                    'test': 'End-to-End Workflow',
                    'status': 'SKIP',
                    'details': 'System not initialized'
                })
                return
            
            # Simulate one complete trading cycle
            await self.system._update_market_data()
            await self.system._analyze_market_context()
            await self.system._generate_predictions()
            await self.system._check_entry_signals()
            await self.system._manage_positions()
            await self.system._update_performance()
            await self.system._check_system_health()
            
            self.logger.info("✅ End-to-end workflow passed")
            self.test_results.append({
                'test': 'End-to-End Workflow',
                'status': 'PASS',
                'details': 'Complete trading cycle executed successfully'
            })
            
        except Exception as e:
            self.logger.error(f"❌ End-to-end workflow test failed: {e}")
            self.test_results.append({
                'test': 'End-to-End Workflow',
                'status': 'ERROR',
                'details': str(e)
            })
    
    async def _generate_test_report(self):
        """Generate test report"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("TEST REPORT")
            self.logger.info("=" * 60)
            
            total_tests = len(self.test_results)
            passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
            failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
            error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
            skipped_tests = len([r for r in self.test_results if r['status'] == 'SKIP'])
            
            self.logger.info(f"Total Tests: {total_tests}")
            self.logger.info(f"Passed: {passed_tests}")
            self.logger.info(f"Failed: {failed_tests}")
            self.logger.info(f"Errors: {error_tests}")
            self.logger.info(f"Skipped: {skipped_tests}")
            self.logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
            
            self.logger.info("\nDetailed Results:")
            for result in self.test_results:
                status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️" if result['status'] == 'SKIP' else "💥"
                self.logger.info(f"{status_icon} {result['test']}: {result['status']}")
                if result['details']:
                    self.logger.info(f"    Details: {result['details']}")
            
            # Overall assessment
            if failed_tests == 0 and error_tests == 0:
                self.logger.info("\n🎉 ALL TESTS PASSED! System is ready for use.")
            elif passed_tests > failed_tests + error_tests:
                self.logger.info("\n⚠️ Most tests passed. Review failed tests before production use.")
            else:
                self.logger.info("\n🚨 Multiple test failures. System needs fixes before use.")
            
        except Exception as e:
            self.logger.error(f"Error generating test report: {e}")
    
    async def _cleanup(self):
        """Clean up test resources"""
        try:
            if self.system:
                await self.system.stop()
                self.system.reset()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main test function"""
    try:
        test_suite = CompleteSystemTest()
        await test_suite.run_all_tests()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test suite error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 