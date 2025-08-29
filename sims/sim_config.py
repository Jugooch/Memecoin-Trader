"""
Simplified configuration loader for simulation
Handles missing config gracefully and provides simulation-specific defaults
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class SimConfig:
    """Simplified config class for simulation"""
    
    def __init__(self, config_data: Dict[str, Any]):
        self.raw_data = config_data
        
        # Extract key fields with defaults
        self.watched_wallets = config_data.get('watched_wallets', [])
        self.pumpportal = config_data.get('pumpportal', {})
        self.moralis_keys = config_data.get('moralis_keys', [])
        self.discord_webhook_url = config_data.get('discord_webhook_url', '')
        
        # Handle legacy single key format
        if not self.moralis_keys and config_data.get('moralis_key'):
            self.moralis_keys = [config_data['moralis_key']]

def load_simulation_config(config_path: str = "config/config.yml") -> SimConfig:
    """
    Load configuration for simulation with fallback defaults
    
    Args:
        config_path: Path to config file
        
    Returns:
        SimConfig object with configuration data
    """
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        print("🔧 Creating minimal simulation config...")
        return create_minimal_config()
    
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
            
        return SimConfig(config_data)
        
    except Exception as e:
        print(f"⚠️ Error loading config: {e}")
        print("🔧 Using minimal simulation config...")
        return create_minimal_config()

def create_minimal_config() -> SimConfig:
    """Create a minimal config for simulation testing"""
    
    minimal_config = {
        'watched_wallets': [
            # Add some example wallets for testing
            'HKcsAKGp3ACUULmL3WwCr8Qd2WhzHqhbVPFKv8MiKJa',
            'DhA86Fv4JwgMWkD1DvDQRjEe9sdjzV5CqVh8uCbT3J4'
        ],
        'pumpportal': {
            'api_key': 'your_api_key_here',
            'ws_endpoint': 'wss://pumpportal.fun/api/data',
            'trade_endpoint': 'https://pumpportal.fun/api/trade'
        },
        'moralis_keys': [
            'your_moralis_key_here'
        ],
        'discord_webhook_url': ''
    }
    
    return SimConfig(minimal_config)

def save_example_config(output_path: str = "sims/configs/simulation_config.yml"):
    """Save an example configuration for simulation"""
    
    example_config = {
        'simulation_settings': {
            'duration_hours': 24,
            'save_interval_seconds': 60,
            'report_interval_minutes': 30
        },
        
        # Required for real operation
        'watched_wallets': [
            # Add your alpha wallet addresses here
            # 'wallet_address_1',
            # 'wallet_address_2',
        ],
        
        'pumpportal': {
            'api_key': 'your_pumpportal_api_key_here',
            'ws_endpoint': 'wss://pumpportal.fun/api/data',
            'trade_endpoint': 'https://pumpportal.fun/api/trade'
        },
        
        'moralis_keys': [
            # Add your Moralis API keys here
            # 'your_moralis_key_here'
        ],
        
        # Optional
        'discord_webhook_url': 'your_discord_webhook_url_here',
        
        # Strategy settings for simulation
        'strategy_settings': {
            'current': {
                'stop_loss_pct': -15.0,
                'take_profit_pct': 20.0,
                'max_hold_minutes': 30
            },
            'buffer': {
                'buffer_seconds': 60,
                'stop_loss_pct': -35.0,
                'tp1_pct': 20.0,
                'tp2_pct': 50.0,
                'tp3_pct': 100.0
            },
            'late_alpha': {
                'min_alpha_count': 4,
                'max_entry_delay_seconds': 300,
                'stop_loss_pct': -35.0
            }
        }
    }
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    print(f"📝 Example config saved to: {output_path}")
    return output_path