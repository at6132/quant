"""
Configuration Validator - Validates Configuration Files
Implements configuration validation for intelligent and risk configs
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import yaml
import jsonschema
import os

from .logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """
    Configuration Validation System
    """
    
    def __init__(self):
        """Initialize configuration validator"""
        self.logger = logger
        
        # Define schemas for validation
        self.intelligent_config_schema = {
            "type": "object",
            "properties": {
                "trading": {
                    "type": "object",
                    "properties": {
                        "model_confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "exit_confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_positions": {"type": "integer", "minimum": 1},
                        "default_stop_loss_pct": {"type": "number", "minimum": 0.001, "maximum": 0.5},
                        "default_take_profit_pct": {"type": "number", "minimum": 0.001, "maximum": 1.0}
                    },
                    "required": ["model_confidence_threshold", "exit_confidence_threshold"]
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "timeframes": {"type": "array", "items": {"type": "string"}},
                        "lookback_periods": {"type": "object"},
                        "feature_engineering": {"type": "object"}
                    }
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "model_path": {"type": "string"},
                        "prediction_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "entry_analysis": {
                    "type": "object",
                    "properties": {
                        "scoring": {"type": "object"},
                        "filtering": {"type": "object"},
                        "risk_parameters": {"type": "object"}
                    }
                },
                "position_management": {
                    "type": "object",
                    "properties": {
                        "trailing_stop": {"type": "object"},
                        "risk_management": {"type": "object"}
                    }
                },
                "market_context": {
                    "type": "object",
                    "properties": {
                        "volatility": {"type": "object"},
                        "trend": {"type": "object"},
                        "sessions": {"type": "object"}
                    }
                }
            },
            "required": ["trading", "data", "model"]
        }
        
        self.risk_config_schema = {
            "type": "object",
            "properties": {
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "kelly_criterion": {"type": "object"},
                        "volatility_weighting": {"type": "object"},
                        "martingale": {"type": "object"},
                        "drawdown_barrier": {"type": "object"}
                    }
                },
                "position_sizing": {
                    "type": "object",
                    "properties": {
                        "min_position_size": {"type": "number", "minimum": 0},
                        "max_positions_per_symbol": {"type": "integer", "minimum": 1}
                    }
                },
                "risk_limits": {
                    "type": "object",
                    "properties": {
                        "max_total_exposure": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_daily_loss": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_weekly_loss": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_monthly_loss": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            },
            "required": ["risk_management", "position_sizing", "risk_limits"]
        }
        
        self.logger.info("Configuration Validator initialized successfully")
    
    async def validate_configs(self, config_dir: str = None) -> Dict[str, Any]:
        """
        Validate all configuration files
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'intelligent_config': {'valid': False, 'errors': [], 'warnings': []},
                'risk_config': {'valid': False, 'errors': [], 'warnings': []},
                'overall_valid': False
            }
            
            if config_dir is None:
                config_path = Path(__file__).parent.parent / 'config'
            else:
                config_path = Path(config_dir)
            
            # Validate intelligent config
            intelligent_config_path = config_path / "intelligent_config.yaml"
            if intelligent_config_path.exists():
                intelligent_result = await self._validate_intelligent_config(intelligent_config_path)
                validation_results['intelligent_config'] = intelligent_result
            else:
                validation_results['intelligent_config']['errors'].append("File not found")
            
            # Validate risk config
            risk_config_path = config_path / "risk_config.yaml"
            if risk_config_path.exists():
                risk_result = await self._validate_risk_config(risk_config_path)
                validation_results['risk_config'] = risk_result
            else:
                validation_results['risk_config']['errors'].append("File not found")
            
            # Check overall validity
            validation_results['overall_valid'] = (
                validation_results['intelligent_config']['valid'] and
                validation_results['risk_config']['valid']
            )
            
            # Log results
            if validation_results['overall_valid']:
                self.logger.info("All configuration files are valid")
            else:
                self.logger.error("Configuration validation failed")
                for config_name, result in validation_results.items():
                    if config_name != 'overall_valid' and not result['valid']:
                        self.logger.error(f"{config_name} errors: {result['errors']}")
                        self.logger.warning(f"{config_name} warnings: {result['warnings']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating configurations: {e}")
            return {
                'intelligent_config': {'valid': False, 'errors': [str(e)], 'warnings': []},
                'risk_config': {'valid': False, 'errors': [str(e)], 'warnings': []},
                'overall_valid': False
            }
    
    async def _validate_intelligent_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate intelligent configuration file"""
        try:
            result = {'valid': False, 'errors': [], 'warnings': []}
            
            # Load and parse YAML
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            if config_data is None:
                result['errors'].append("Empty or invalid YAML file")
                return result
            
            # Validate against schema
            try:
                jsonschema.validate(instance=config_data, schema=self.intelligent_config_schema)
            except jsonschema.ValidationError as e:
                result['errors'].append(f"Schema validation error: {e.message}")
                return result
            
            # Additional custom validations
            custom_errors, custom_warnings = await self._validate_intelligent_config_custom(config_data)
            result['errors'].extend(custom_errors)
            result['warnings'].extend(custom_warnings)
            
            # Set validity
            result['valid'] = len(result['errors']) == 0
            
            return result
            
        except Exception as e:
            return {'valid': False, 'errors': [f"Error validating intelligent config: {str(e)}"], 'warnings': []}
    
    async def _validate_risk_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate risk configuration file"""
        try:
            result = {'valid': False, 'errors': [], 'warnings': []}
            
            # Load and parse YAML
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            if config_data is None:
                result['errors'].append("Empty or invalid YAML file")
                return result
            
            # Validate against schema
            try:
                jsonschema.validate(instance=config_data, schema=self.risk_config_schema)
            except jsonschema.ValidationError as e:
                result['errors'].append(f"Schema validation error: {e.message}")
                return result
            
            # Additional custom validations
            custom_errors, custom_warnings = await self._validate_risk_config_custom(config_data)
            result['errors'].extend(custom_errors)
            result['warnings'].extend(custom_warnings)
            
            # Set validity
            result['valid'] = len(result['errors']) == 0
            
            return result
            
        except Exception as e:
            return {'valid': False, 'errors': [f"Error validating risk config: {str(e)}"], 'warnings': []}
    
    async def _validate_intelligent_config_custom(self, config_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Custom validation for intelligent configuration"""
        errors = []
        warnings = []
        
        try:
            # Validate trading parameters
            trading = config_data.get('trading', {})
            
            # Check confidence thresholds
            model_threshold = trading.get('model_confidence_threshold', 0)
            exit_threshold = trading.get('exit_confidence_threshold', 0)
            
            if model_threshold < 0.5:
                warnings.append("Model confidence threshold is low (< 0.5)")
            
            if exit_threshold < 0.5:
                warnings.append("Exit confidence threshold is low (< 0.5)")
            
            if model_threshold < exit_threshold:
                warnings.append("Model confidence threshold is lower than exit threshold")
            
            # Validate data configuration
            data = config_data.get('data', {})
            timeframes = data.get('timeframes', [])
            
            if not timeframes:
                errors.append("No timeframes specified in data configuration")
            
            required_timeframes = ['1m', '5m', '15m', '1h']
            missing_timeframes = [tf for tf in required_timeframes if tf not in timeframes]
            if missing_timeframes:
                warnings.append(f"Missing recommended timeframes: {missing_timeframes}")
            
            # Validate model configuration
            model = config_data.get('model', {})
            model_path = model.get('model_path', '')
            
            if not model_path:
                warnings.append("No model path specified")
            
            # Validate entry analysis
            entry_analysis = config_data.get('entry_analysis', {})
            risk_params = entry_analysis.get('risk_parameters', {})
            
            max_risk_pct = risk_params.get('max_risk_pct', 0)
            if max_risk_pct > 0.05:
                warnings.append("Maximum risk per trade is high (> 5%)")
            
            # Validate position management
            pos_mgmt = config_data.get('position_management', {})
            trailing = pos_mgmt.get('trailing_stop', {})
            
            activation_threshold = trailing.get('activation_threshold', 0)
            if activation_threshold < 0.2:
                warnings.append("Trailing stop activation threshold is low (< 20%)")
            
        except Exception as e:
            errors.append(f"Error in custom intelligent config validation: {str(e)}")
        
        return errors, warnings
    
    async def _validate_risk_config_custom(self, config_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Custom validation for risk configuration"""
        errors = []
        warnings = []
        
        try:
            # Validate risk management
            risk_mgmt = config_data.get('risk_management', {})
            
            # Kelly criterion validation
            kelly = risk_mgmt.get('kelly_criterion', {})
            max_fraction = kelly.get('max_fraction', 0)
            min_fraction = kelly.get('min_fraction', 0)
            
            if max_fraction > 0.5:
                warnings.append("Maximum Kelly fraction is very high (> 50%)")
            
            if min_fraction > 0.05:
                warnings.append("Minimum Kelly fraction is high (> 5%)")
            
            if max_fraction <= min_fraction:
                errors.append("Maximum Kelly fraction must be greater than minimum")
            
            # Volatility weighting validation
            vol_weighting = risk_mgmt.get('volatility_weighting', {})
            max_buffer = vol_weighting.get('max_buffer', 1.0)
            min_buffer = vol_weighting.get('min_buffer', 1.0)
            
            if max_buffer <= min_buffer:
                errors.append("Maximum volatility buffer must be greater than minimum")
            
            if max_buffer > 2.0:
                warnings.append("Maximum volatility buffer is very high (> 2.0)")
            
            # Martingale validation
            martingale = risk_mgmt.get('martingale', {})
            max_escalation = martingale.get('max_escalation', 1.0)
            
            if max_escalation > 5.0:
                warnings.append("Maximum Martingale escalation is very high (> 5.0)")
            
            # Drawdown barrier validation
            drawdown = risk_mgmt.get('drawdown_barrier', {})
            max_drawdown = drawdown.get('max_drawdown_pct', 0)
            barrier_threshold = drawdown.get('barrier_threshold', 0)
            
            if max_drawdown <= barrier_threshold:
                errors.append("Maximum drawdown must be greater than barrier threshold")
            
            if max_drawdown > 0.5:
                warnings.append("Maximum drawdown is very high (> 50%)")
            
            # Risk limits validation
            risk_limits = config_data.get('risk_limits', {})
            max_exposure = risk_limits.get('max_total_exposure', 0)
            daily_loss = risk_limits.get('max_daily_loss', 0)
            weekly_loss = risk_limits.get('max_weekly_loss', 0)
            monthly_loss = risk_limits.get('max_monthly_loss', 0)
            
            if max_exposure > 0.8:
                warnings.append("Maximum total exposure is very high (> 80%)")
            
            if daily_loss > 0.1:
                warnings.append("Maximum daily loss is very high (> 10%)")
            
            if weekly_loss > 0.25:
                warnings.append("Maximum weekly loss is very high (> 25%)")
            
            if monthly_loss > 0.5:
                warnings.append("Maximum monthly loss is very high (> 50%)")
            
            # Validate loss limits hierarchy
            if daily_loss >= weekly_loss:
                errors.append("Daily loss limit must be less than weekly loss limit")
            
            if weekly_loss >= monthly_loss:
                errors.append("Weekly loss limit must be less than monthly loss limit")
            
        except Exception as e:
            errors.append(f"Error in custom risk config validation: {str(e)}")
        
        return errors, warnings
    
    async def generate_config_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a human-readable configuration validation report"""
        try:
            report = "Configuration Validation Report\n"
            report += "=" * 40 + "\n\n"
            
            # Overall status
            if validation_results['overall_valid']:
                report += "✅ All configuration files are valid\n\n"
            else:
                report += "❌ Configuration validation failed\n\n"
            
            # Intelligent config status
            intelligent_result = validation_results['intelligent_config']
            report += "Intelligent Configuration:\n"
            if intelligent_result['valid']:
                report += "  ✅ Valid\n"
            else:
                report += "  ❌ Invalid\n"
                for error in intelligent_result['errors']:
                    report += f"    - Error: {error}\n"
            
            for warning in intelligent_result['warnings']:
                report += f"    - Warning: {warning}\n"
            
            report += "\n"
            
            # Risk config status
            risk_result = validation_results['risk_config']
            report += "Risk Configuration:\n"
            if risk_result['valid']:
                report += "  ✅ Valid\n"
            else:
                report += "  ❌ Invalid\n"
                for error in risk_result['errors']:
                    report += f"    - Error: {error}\n"
            
            for warning in risk_result['warnings']:
                report += f"    - Warning: {warning}\n"
            
            report += "\n"
            
            # Recommendations
            if not validation_results['overall_valid']:
                report += "Recommendations:\n"
                report += "- Fix all errors before running the system\n"
                report += "- Review warnings and adjust parameters if needed\n"
                report += "- Test configuration with small position sizes first\n"
            
            return report
            
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def reset(self):
        """Reset configuration validator (for testing)"""
        pass 