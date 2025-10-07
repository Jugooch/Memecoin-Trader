"""
Unit tests for Configuration Manager (core/config.py)

Tests:
- YAML parsing
- Environment variable substitution
- Configuration validation
- Type safety
- Error handling
"""

import pytest
import os
import yaml
from pathlib import Path

from core.config import (
    ConfigurationManager,
    RPCEndpoint,
    RPCConfig,
    LogConfig,
    MetricsConfig,
    BotConfig
)


class TestConfigurationManager:
    """Test configuration loading and validation"""

    def test_load_valid_config(self, test_config_file):
        """Test loading a valid configuration file"""
        config_manager = ConfigurationManager(test_config_file)
        bot_config = config_manager.load_config()

        # Verify config loaded
        assert bot_config is not None
        assert isinstance(bot_config, BotConfig)

        # Verify RPC config
        assert len(bot_config.rpc_config.endpoints) == 2
        assert bot_config.rpc_config.failover_threshold_errors == 3

        # Verify logging config
        assert bot_config.log_config.level == "DEBUG"
        assert bot_config.log_config.format == "json"

        # Verify metrics config
        assert bot_config.metrics_config.enable_prometheus is False

    def test_rpc_endpoint_priority_sorting(self, test_config_file):
        """Test that RPC endpoints are sorted by priority"""
        config_manager = ConfigurationManager(test_config_file)
        bot_config = config_manager.load_config()

        endpoints = bot_config.rpc_config.endpoints

        # Verify sorted by priority (0, 1, ...)
        assert endpoints[0].priority == 0
        assert endpoints[1].priority == 1
        assert endpoints[0].label == "solana_labs_devnet"

    def test_missing_config_file(self, tmp_path):
        """Test error handling for missing config file"""
        missing_file = tmp_path / "nonexistent.yml"
        config_manager = ConfigurationManager(str(missing_file))

        with pytest.raises(FileNotFoundError):
            config_manager.load_config()

    def test_invalid_yaml_syntax(self, tmp_path):
        """Test error handling for invalid YAML syntax"""
        bad_config = tmp_path / "bad.yml"
        with open(bad_config, 'w') as f:
            f.write("invalid: yaml: syntax:")

        config_manager = ConfigurationManager(str(bad_config))

        with pytest.raises(Exception):  # yaml.YAMLError
            config_manager.load_config()

    def test_missing_required_field(self, tmp_path):
        """Test validation fails when required field is missing"""
        incomplete_config = tmp_path / "incomplete.yml"
        with open(incomplete_config, 'w') as f:
            yaml.dump({"logging": {"level": "INFO"}}, f)  # Missing rpc section

        config_manager = ConfigurationManager(str(incomplete_config))

        with pytest.raises(ValueError, match="No RPC endpoints configured"):
            config_manager.load_config()

    def test_env_var_substitution(self, tmp_path):
        """Test environment variable substitution in config"""
        # Set test environment variable
        os.environ["TEST_API_KEY"] = "test-key-12345"

        config_data = {
            "rpc": {
                "endpoints": [{
                    "url": "https://api.test.com/?api-key=${TEST_API_KEY}",
                    "websocket_url": "wss://api.test.com",
                    "priority": 0,
                    "label": "test"
                }]
            },
            "logging": {"level": "INFO", "format": "json"},
            "metrics": {"enable_prometheus": False}
        }

        config_file = tmp_path / "env_test.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config_manager = ConfigurationManager(str(config_file))
        bot_config = config_manager.load_config()

        # Verify substitution
        assert "test-key-12345" in bot_config.rpc_config.endpoints[0].url
        assert "${TEST_API_KEY}" not in bot_config.rpc_config.endpoints[0].url

        # Cleanup
        del os.environ["TEST_API_KEY"]

    def test_missing_env_var(self, tmp_path):
        """Test error when environment variable is not found"""
        config_data = {
            "rpc": {
                "endpoints": [{
                    "url": "https://api.test.com/?api-key=${NONEXISTENT_VAR}",
                    "websocket_url": "wss://api.test.com",
                    "priority": 0,
                    "label": "test"
                }]
            },
            "logging": {"level": "INFO"},
            "metrics": {}
        }

        config_file = tmp_path / "missing_env.yml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config_manager = ConfigurationManager(str(config_file))

        with pytest.raises(ValueError, match="Environment variable NONEXISTENT_VAR not found"):
            config_manager.load_config()

    def test_dot_notation_get(self, test_config_file):
        """Test getting config values with dot notation"""
        config_manager = ConfigurationManager(test_config_file)
        bot_config = config_manager.load_config()

        # Test dot notation access
        value = config_manager.get("rpc.failover_threshold_errors")
        assert value == 3

        value = config_manager.get("logging.level")
        assert value == "DEBUG"

        # Test default value
        value = config_manager.get("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_reload_config(self, test_config_file):
        """Test hot-reloading configuration"""
        config_manager = ConfigurationManager(test_config_file)
        bot_config1 = config_manager.load_config()

        # Modify config file
        with open(test_config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['rpc']['failover_threshold_errors'] = 5

        with open(test_config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Reload
        bot_config2 = config_manager.reload_config()

        # Verify change
        assert bot_config2.rpc_config.failover_threshold_errors == 5

    def test_default_values(self, tmp_path):
        """Test that default values are applied correctly"""
        minimal_config = {
            "rpc": {
                "endpoints": [{
                    "url": "https://api.test.com",
                    "websocket_url": "wss://api.test.com",
                    "priority": 0,
                    "label": "test"
                    # Missing max_connections and timeout_ms
                }]
                # Missing failover settings
            },
            "logging": {},  # Empty, should use defaults
            "metrics": {}   # Empty, should use defaults
        }

        config_file = tmp_path / "defaults.yml"
        with open(config_file, 'w') as f:
            yaml.dump(minimal_config, f)

        config_manager = ConfigurationManager(str(config_file))
        bot_config = config_manager.load_config()

        # Verify defaults applied
        assert bot_config.rpc_config.endpoints[0].max_connections == 10
        assert bot_config.rpc_config.endpoints[0].timeout_ms == 5000
        assert bot_config.rpc_config.failover_threshold_errors == 3
        assert bot_config.log_config.level == "INFO"
        assert bot_config.metrics_config.enable_prometheus is False


class TestRPCEndpoint:
    """Test RPCEndpoint dataclass"""

    def test_rpc_endpoint_creation(self):
        """Test creating RPCEndpoint with all fields"""
        endpoint = RPCEndpoint(
            url="https://api.test.com",
            websocket_url="wss://api.test.com",
            priority=0,
            label="test_endpoint",
            max_connections=20,
            timeout_ms=10000
        )

        assert endpoint.url == "https://api.test.com"
        assert endpoint.websocket_url == "wss://api.test.com"
        assert endpoint.priority == 0
        assert endpoint.label == "test_endpoint"
        assert endpoint.max_connections == 20
        assert endpoint.timeout_ms == 10000

    def test_rpc_endpoint_defaults(self):
        """Test RPCEndpoint default values"""
        endpoint = RPCEndpoint(
            url="https://api.test.com",
            websocket_url="wss://api.test.com",
            priority=0,
            label="test"
        )

        # Verify defaults
        assert endpoint.max_connections == 10
        assert endpoint.timeout_ms == 5000


class TestRPCConfig:
    """Test RPCConfig dataclass"""

    def test_rpc_config_creation(self, test_rpc_config):
        """Test creating RPCConfig"""
        assert len(test_rpc_config.endpoints) == 2
        assert test_rpc_config.failover_threshold_errors == 3
        assert test_rpc_config.failover_threshold_latency_ms == 500
        assert test_rpc_config.health_check_interval_s == 10


class TestLogConfig:
    """Test LogConfig dataclass"""

    def test_log_config_defaults(self):
        """Test LogConfig default values"""
        log_config = LogConfig()

        assert log_config.level == "INFO"
        assert log_config.format == "json"
        assert log_config.output_file is None

    def test_log_config_custom(self):
        """Test LogConfig with custom values"""
        log_config = LogConfig(
            level="DEBUG",
            format="console",
            output_file="/tmp/test.log"
        )

        assert log_config.level == "DEBUG"
        assert log_config.format == "console"
        assert log_config.output_file == "/tmp/test.log"


class TestMetricsConfig:
    """Test MetricsConfig dataclass"""

    def test_metrics_config_defaults(self):
        """Test MetricsConfig default values"""
        metrics_config = MetricsConfig()

        assert metrics_config.enable_prometheus is False
        assert metrics_config.prometheus_port == 9090
        assert metrics_config.enable_histogram is True
        assert metrics_config.export_interval_s == 60
        assert len(metrics_config.histogram_buckets) == 7
