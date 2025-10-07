"""
Pytest configuration and shared fixtures
These fixtures are available to all test files
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

from core.config import ConfigurationManager, BotConfig, RPCConfig, RPCEndpoint, LogConfig, MetricsConfig
from core.metrics import MetricsCollector


@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config_dict() -> Dict[str, Any]:
    """
    Sample configuration dictionary for testing

    Returns valid config that can be modified per test
    """
    return {
        "rpc": {
            "endpoints": [
                {
                    "url": "https://api.devnet.solana.com",
                    "websocket_url": "wss://api.devnet.solana.com",
                    "priority": 0,
                    "label": "solana_labs_devnet",
                    "max_connections": 10,
                    "timeout_ms": 5000
                },
                {
                    "url": "https://api.testnet.solana.com",
                    "websocket_url": "wss://api.testnet.solana.com",
                    "priority": 1,
                    "label": "solana_labs_testnet",
                    "max_connections": 10,
                    "timeout_ms": 5000
                }
            ],
            "failover_threshold_errors": 3,
            "failover_threshold_latency_ms": 500,
            "health_check_interval_s": 10,
            "reconnect_backoff_base_ms": 100,
            "reconnect_backoff_max_ms": 5000
        },
        "logging": {
            "level": "DEBUG",
            "format": "json",
            "output_file": None
        },
        "metrics": {
            "enable_prometheus": False,
            "prometheus_port": 9090,
            "enable_histogram": True,
            "histogram_buckets": [1, 5, 10, 50, 100, 500, 1000],
            "export_interval_s": 60
        }
    }


@pytest.fixture
def test_config_file(test_config_dict, tmp_path):
    """
    Create a temporary config file for testing

    Returns path to temporary YAML config file
    """
    import yaml

    config_file = tmp_path / "test_config.yml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config_dict, f)

    return str(config_file)


@pytest.fixture
def test_rpc_config() -> RPCConfig:
    """
    Create test RPC configuration

    Returns RPCConfig with test endpoints
    """
    endpoints = [
        RPCEndpoint(
            url="https://api.devnet.solana.com",
            websocket_url="wss://api.devnet.solana.com",
            priority=0,
            label="test_endpoint_1"
        ),
        RPCEndpoint(
            url="https://api.testnet.solana.com",
            websocket_url="wss://api.testnet.solana.com",
            priority=1,
            label="test_endpoint_2"
        )
    ]

    return RPCConfig(
        endpoints=endpoints,
        failover_threshold_errors=3,
        failover_threshold_latency_ms=500,
        health_check_interval_s=10,
        reconnect_backoff_base_ms=100,
        reconnect_backoff_max_ms=5000
    )


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """
    Create fresh metrics collector for each test

    Returns clean MetricsCollector instance
    """
    collector = MetricsCollector(enable_histogram=True)
    yield collector
    # Clean up after test
    collector.reset()


@pytest.fixture
def temp_log_file(tmp_path):
    """
    Create temporary log file path

    Returns path to temporary log file
    """
    log_file = tmp_path / "test.log"
    return str(log_file)


# Integration test markers
def pytest_configure(config):
    """Register custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires network)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (>5 seconds)"
    )
    config.addinivalue_line(
        "markers", "rpc: mark test as requiring RPC access"
    )
