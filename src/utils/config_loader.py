"""
Shared configuration loader utility to eliminate duplication across modules
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_filename: str = "config.yml") -> Dict[str, Any]:
    """
    Load configuration from YAML file in config directory
    
    Args:
        config_filename: Name of config file (defaults to config.yml)
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist in config directory
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    # Always use config directory from project root
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file '{config_filename}' not found at {config_path}"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML syntax in {config_path}: {e}")
    
    # Normalize moralis keys to always be a list for consistency
    if 'moralis_key' in config_data and 'moralis_keys' not in config_data:
        # Convert single key to list format
        config_data['moralis_keys'] = [config_data['moralis_key']]
    elif 'moralis_keys' in config_data:
        # Ensure it's a list even if single key provided
        if isinstance(config_data['moralis_keys'], str):
            config_data['moralis_keys'] = [config_data['moralis_keys']]
    
    # Normalize bitquery tokens to always be a list for consistency
    if 'bitquery_token' in config_data and 'bitquery_tokens' not in config_data:
        # Convert single token to list format
        config_data['bitquery_tokens'] = [config_data['bitquery_token']]
    elif 'bitquery_tokens' in config_data:
        # Ensure it's a list even if single token provided
        if isinstance(config_data['bitquery_tokens'], str):
            config_data['bitquery_tokens'] = [config_data['bitquery_tokens']]
    
    return config_data


def get_database_path(config: Dict[str, Any]) -> str:
    """Extract database path from config with proper defaults"""
    return config.get('database', {}).get('file', 'trades.db')


def get_log_path(config: Dict[str, Any]) -> str:
    """Extract log path from config with proper defaults"""
    return config.get('logging', {}).get('file', 'logs/trading.log')


def get_log_level(config: Dict[str, Any]) -> str:
    """Extract log level from config with proper defaults"""
    return config.get('logging', {}).get('level', 'INFO')


def validate_required_keys(config: Dict[str, Any]) -> None:
    """
    Validate that required configuration keys are present
    
    Raises:
        ValueError: If required keys are missing
    """
    required_keys = [
        'rpc_endpoint',
        'watched_wallets',
        'threshold_alpha_buys',
        'min_liquidity_usd',
        'time_window_sec',
        'max_trade_pct',
        'tp_multiplier',
        'stop_loss_pct'
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    # Check for either bitquery_token or bitquery_tokens
    if 'bitquery_token' not in config and 'bitquery_tokens' not in config:
        missing_keys.append('bitquery_token or bitquery_tokens')
    
    # Check for either moralis_key or moralis_keys
    if 'moralis_key' not in config and 'moralis_keys' not in config:
        missing_keys.append('moralis_key or moralis_keys')
    
    # Check nested required keys
    if 'trading' in config:
        trading_required = ['initial_capital', 'max_trades_per_day', 'min_time_between_trades']
        for key in trading_required:
            if key not in config['trading']:
                missing_keys.append(f'trading.{key}')
    else:
        missing_keys.append('trading section')
    
    if 'pump_fun' in config:
        pumpfun_required = ['program_address', 'api_endpoint']
        for key in pumpfun_required:
            if key not in config['pump_fun']:
                missing_keys.append(f'pump_fun.{key}')
    else:
        missing_keys.append('pump_fun section')
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")