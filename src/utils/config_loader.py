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
    
    # Set default realtime source if not specified
    if 'realtime_source' not in config_data:
        # Default to pumpportal if configured, otherwise bitquery
        if 'pumpportal' in config_data and config_data['pumpportal'].get('api_key'):
            config_data['realtime_source'] = 'pumpportal'
        else:
            config_data['realtime_source'] = 'bitquery'
    
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


def safe_update_config(updates: Dict[str, Any], config_path: str = None) -> bool:
    """
    Safely update config file with atomic write and validation
    
    Args:
        updates: Dictionary of config updates to apply
        config_path: Path to config file (auto-detected if None)
        
    Returns:
        True if update successful, False otherwise
    """
    import shutil
    import tempfile
    from pathlib import Path
    
    if not config_path:
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "config.yml")
    
    try:
        # Step 1: Create backup
        backup_path = f"{config_path}.backup"
        shutil.copy2(config_path, backup_path)
        
        # Step 2: Load current config
        current_config = load_config()
        
        # Step 3: Apply updates (deep merge)
        def deep_merge(base, updates):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(current_config, updates)
        
        # Step 4: Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp_file:
            yaml.safe_dump(current_config, tmp_file, default_flow_style=False, sort_keys=False)
            tmp_path = tmp_file.name
        
        # Step 5: Validate temp file has required keys
        validate_config_file(tmp_path)
        
        # Step 6: Atomically replace original
        shutil.move(tmp_path, config_path)
        
        return True
        
    except Exception as e:
        # Restore from backup on any error
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, config_path)
        raise RuntimeError(f"Config update failed, restored from backup: {e}")


def validate_config_file(config_path: str) -> None:
    """Validate that config file has all required keys"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Config file corrupted or unreadable: {e}")
    
    # Check for critical API keys
    critical_keys = ['moralis_keys', 'bitquery_tokens']
    missing_critical = [k for k in critical_keys if k not in config or not config[k]]
    
    if missing_critical:
        raise ValueError(f"Config corruption: Missing critical API keys: {missing_critical}")
    
    # Check for essential sections
    essential_sections = ['trading', 'pump_fun', 'rpc_endpoint', 'watched_wallets']
    missing_sections = [s for s in essential_sections if s not in config]
    
    if missing_sections:
        raise ValueError(f"Config corruption: Missing essential sections: {missing_sections}")


def restore_missing_config_sections() -> Dict[str, Any]:
    """Return default values for commonly missing config sections"""
    return {
        'rpc_endpoint': 'https://api.mainnet-beta.solana.com',
        'trading_mode': 'simulation',
        'time_window_sec': 300,
        'tp_multiplier': 1.25,
        'stop_loss_pct': 0.92,
        'trading': {
            'initial_capital': 500.0,
            'max_trades_per_day': 20,
            'min_time_between_trades': 120,
            'max_concurrent_positions': 3
        },
        'paper_trading': {
            'fee_bps': 30,
            'buy_slippage_bps': 75,
            'sell_slippage_bps': 100,
            'max_slippage_bps': 150
        },
        'pump_fun': {
            'program_address': '6EF8rrecthHAuSStzpf6aXr9HWs8jgPVr5S6fqF6P',
            'api_endpoint': 'https://pumpapi.fun'
        },
        'notifications': {
            'discord_webhook_url': '',
            'enabled': True
        }
    }