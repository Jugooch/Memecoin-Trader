"""
Configuration Manager for Frontrun Bot
Loads configuration from YAML files with environment variable support
"""

import os
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RPCEndpoint:
    """RPC endpoint configuration"""
    url: str
    websocket_url: str
    priority: int
    label: str
    max_connections: int = 10
    timeout_ms: int = 5000


@dataclass
class RPCConfig:
    """RPC manager configuration"""
    endpoints: List[RPCEndpoint]
    failover_threshold_errors: int = 3
    failover_threshold_latency_ms: int = 500
    health_check_interval_s: int = 10
    reconnect_backoff_base_ms: int = 100
    reconnect_backoff_max_ms: int = 5000


@dataclass
class LogConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "json"
    output_file: Optional[str] = None


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enable_prometheus: bool = False
    prometheus_port: int = 9090
    enable_histogram: bool = True
    histogram_buckets: List[float] = field(
        default_factory=lambda: [1, 5, 10, 50, 100, 500, 1000]
    )
    export_interval_s: int = 60


@dataclass
class TransactionConfig:
    """Transaction infrastructure configuration"""
    # Transaction Builder
    max_tx_size_bytes: int = 1232
    default_compute_units: int = 200000
    blockhash_cache_ttl_seconds: int = 30
    enable_blockhash_cache: bool = True

    # Transaction Signer
    enable_key_rotation: bool = True
    max_signatures_per_key: int = 10000
    key_rotation_interval_minutes: int = 60
    track_signature_count: bool = True

    # Transaction Submitter
    skip_preflight: bool = True
    max_retries: int = 3
    retry_delay_ms: int = 200
    confirmation_timeout_s: int = 30
    confirmation_poll_interval_s: float = 0.5
    enable_confirmation_tracking: bool = True

    # Priority Fees
    fee_update_interval_s: int = 10
    fee_lookback_slots: int = 150
    min_priority_fee: int = 1000
    max_priority_fee: int = 1000000
    fee_cache_ttl_seconds: int = 10


@dataclass
class BotConfig:
    """Complete bot configuration"""
    rpc_config: RPCConfig
    log_config: LogConfig
    metrics_config: MetricsConfig
    transaction_config: Optional[TransactionConfig] = None


class ConfigurationManager:
    """Manages bot configuration from YAML files and environment variables"""

    def __init__(self, config_path: str):
        """
        Initialize configuration manager

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config_data: Optional[Dict[str, Any]] = None
        self._bot_config: Optional[BotConfig] = None

    def load_config(self) -> BotConfig:
        """
        Load and validate configuration from file

        Returns:
            BotConfig: Validated configuration object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Substitute environment variables
        self._config_data = self._substitute_env_vars(raw_config)

        # Parse and validate
        self._bot_config = self._parse_config(self._config_data)

        return self._bot_config

    def reload_config(self) -> BotConfig:
        """
        Hot-reload configuration from file

        Returns:
            BotConfig: Updated configuration
        """
        return self.load_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key

        Args:
            key: Dot-notation key (e.g., "rpc.failover_threshold_errors")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config_data is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")

        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config

        Environment variables are specified as ${VAR_NAME}
        Supports both full-value substitution and embedded vars:
        - Full: "${API_KEY}" -> "abc123"
        - Embedded: "https://api.com/?key=${API_KEY}" -> "https://api.com/?key=abc123"

        Args:
            config: Configuration dictionary or value

        Returns:
            Configuration with substituted values
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Use regex to find and replace all ${VAR_NAME} patterns
            import re

            def replace_var(match):
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(
                        f"Environment variable {var_name} not found"
                    )
                return value

            # Replace all ${VAR_NAME} patterns
            return re.sub(r'\$\{([^}]+)\}', replace_var, config)
        else:
            return config

    def _parse_config(self, config: Dict[str, Any]) -> BotConfig:
        """
        Parse raw configuration into typed objects

        Args:
            config: Raw configuration dictionary

        Returns:
            BotConfig: Typed configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        # Parse RPC endpoints
        rpc_data = config.get('rpc', {})
        endpoints_data = rpc_data.get('endpoints', [])

        if not endpoints_data:
            raise ValueError("No RPC endpoints configured")

        endpoints = []
        for ep in endpoints_data:
            endpoints.append(RPCEndpoint(
                url=ep['url'],
                websocket_url=ep['websocket_url'],
                priority=ep['priority'],
                label=ep['label'],
                max_connections=ep.get('max_connections', 10),
                timeout_ms=ep.get('timeout_ms', 5000)
            ))

        # Sort by priority (0 = highest)
        endpoints.sort(key=lambda x: x.priority)

        rpc_config = RPCConfig(
            endpoints=endpoints,
            failover_threshold_errors=rpc_data.get('failover_threshold_errors', 3),
            failover_threshold_latency_ms=rpc_data.get('failover_threshold_latency_ms', 500),
            health_check_interval_s=rpc_data.get('health_check_interval_s', 10),
            reconnect_backoff_base_ms=rpc_data.get('reconnect_backoff_base_ms', 100),
            reconnect_backoff_max_ms=rpc_data.get('reconnect_backoff_max_ms', 5000)
        )

        # Parse logging config
        log_data = config.get('logging', {})
        log_config = LogConfig(
            level=log_data.get('level', 'INFO'),
            format=log_data.get('format', 'json'),
            output_file=log_data.get('output_file')
        )

        # Parse metrics config
        metrics_data = config.get('metrics', {})
        metrics_config = MetricsConfig(
            enable_prometheus=metrics_data.get('enable_prometheus', False),
            prometheus_port=metrics_data.get('prometheus_port', 9090),
            enable_histogram=metrics_data.get('enable_histogram', True),
            histogram_buckets=metrics_data.get('histogram_buckets', [1, 5, 10, 50, 100, 500, 1000]),
            export_interval_s=metrics_data.get('export_interval_s', 60)
        )

        # Parse transaction config (optional, for Phase 2)
        transaction_config = None
        if 'transactions' in config:
            tx_data = config['transactions']
            transaction_config = TransactionConfig(
                # Transaction Builder
                max_tx_size_bytes=tx_data.get('max_tx_size_bytes', 1232),
                default_compute_units=tx_data.get('default_compute_units', 200000),
                blockhash_cache_ttl_seconds=tx_data.get('blockhash_cache_ttl_seconds', 30),
                enable_blockhash_cache=tx_data.get('enable_blockhash_cache', True),
                # Transaction Signer
                enable_key_rotation=tx_data.get('enable_key_rotation', True),
                max_signatures_per_key=tx_data.get('max_signatures_per_key', 10000),
                key_rotation_interval_minutes=tx_data.get('key_rotation_interval_minutes', 60),
                track_signature_count=tx_data.get('track_signature_count', True),
                # Transaction Submitter
                skip_preflight=tx_data.get('skip_preflight', True),
                max_retries=tx_data.get('max_retries', 3),
                retry_delay_ms=tx_data.get('retry_delay_ms', 200),
                confirmation_timeout_s=tx_data.get('confirmation_timeout_s', 30),
                confirmation_poll_interval_s=tx_data.get('confirmation_poll_interval_s', 0.5),
                enable_confirmation_tracking=tx_data.get('enable_confirmation_tracking', True),
                # Priority Fees
                fee_update_interval_s=tx_data.get('fee_update_interval_s', 10),
                fee_lookback_slots=tx_data.get('fee_lookback_slots', 150),
                min_priority_fee=tx_data.get('min_priority_fee', 1000),
                max_priority_fee=tx_data.get('max_priority_fee', 1000000),
                fee_cache_ttl_seconds=tx_data.get('fee_cache_ttl_seconds', 10)
            )

        return BotConfig(
            rpc_config=rpc_config,
            log_config=log_config,
            metrics_config=metrics_config,
            transaction_config=transaction_config
        )


# Example usage
if __name__ == "__main__":
    config_manager = ConfigurationManager("config/config.yml")
    bot_config = config_manager.load_config()

    print(f"Loaded {len(bot_config.rpc_config.endpoints)} RPC endpoints:")
    for ep in bot_config.rpc_config.endpoints:
        print(f"  - {ep.label} (priority {ep.priority}): {ep.url}")

    print(f"\nLogging level: {bot_config.log_config.level}")
    print(f"Metrics enabled: {bot_config.metrics_config.enable_prometheus}")
