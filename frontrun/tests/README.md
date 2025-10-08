# Frontrun Bot - Testing Framework

## Overview

This testing framework provides **comprehensive regression testing** for all bot phases. Tests are organized by type (unit, integration) and phase, allowing you to verify that earlier phases still work correctly after adding new features.

**Current Status**: Phases 1-3 Complete
- **309 unit tests** (fast, no network required)
- **7 integration tests** (requires 2 SOL devnet funding)
- **316 total tests** - All passing ✅

**Quick Start**: See [`docs/TEST_EXECUTION_CHECKLIST.md`](../docs/TEST_EXECUTION_CHECKLIST.md) for step-by-step test execution guide.

---

## Test Structure

```
tests/
├── conftest.py                           # Shared pytest fixtures
├── unit/                                 # Unit tests (309 tests, no network)
│   ├── test_config.py                    # Configuration manager (Phase 1)
│   ├── test_metrics.py                   # Metrics system (Phase 1)
│   ├── test_rpc_manager.py               # RPC manager (Phase 1)
│   ├── test_health_monitor.py            # Health monitoring (Phase 1)
│   ├── test_tx_builder.py                # Transaction builder (Phase 2)
│   ├── test_tx_signer.py                 # Transaction signer (Phase 2)
│   ├── test_tx_submitter.py              # Transaction submitter (Phase 2)
│   ├── test_priority_fees.py             # Priority fees (Phase 2)
│   ├── test_wallet_manager.py            # Wallet manager (Phase 2)
│   ├── test_pumpfun_client.py            # Pump.fun client (Phase 3)
│   ├── test_bonding_curve.py             # Bonding curve (Phase 3)
│   ├── test_slippage.py                  # Slippage manager (Phase 3)
│   ├── test_pnl.py                       # PnL calculator (Phase 3)
│   └── test_position_tracker.py          # Position tracker (Phase 3)
├── integration/                          # Integration tests (7 tests)
│   ├── test_phase1_integration.py        # RPC connectivity
│   ├── test_phase2_integration.py        # Transaction submission
│   ├── test_devnet_rpc_connection.py     # Devnet RPC tests
│   ├── test_devnet_wallet_operations.py  # Wallet funding tests
│   ├── test_devnet_bonding_curve_read.py # Bonding curve on-chain reads
│   ├── test_devnet_transaction_submission.py  # Real transaction tests
│   └── test_devnet_full_trade_flow.py    # Complete buy/sell trades
├── load/                                 # Load tests (future - Phase 6)
└── e2e/                                  # End-to-end tests (future - Phase 6)
```

---

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### Unit Tests Only (Fast, No Network)

```bash
pytest tests/unit/ -v
```

### Integration Tests Only (Requires RPC Access)

```bash
pytest tests/integration/ -v
```

### Skip Integration Tests

```bash
pytest tests/ --ignore=tests/integration/ -v
```

### Specific Test File

```bash
pytest tests/unit/test_config.py -v
```

### Specific Test Function

```bash
pytest tests/unit/test_config.py::TestConfigurationManager::test_load_valid_config -v
```

### With Coverage Report

```bash
# Generate coverage report
pytest tests/ --cov=core --cov-report=html

# Open report
# Windows: start htmlcov/index.html
# Linux: xdg-open htmlcov/index.html
```

### Verbose Output with Print Statements

```bash
pytest tests/ -v -s
```

---

## Test Markers

Tests are marked with custom markers for selective running:

### `@pytest.mark.integration`
Integration tests requiring network access

```bash
# Run only integration tests
pytest -m integration -v

# Skip integration tests
pytest -m "not integration" -v
```

### `@pytest.mark.slow`
Slow-running tests (>5 seconds)

```bash
# Skip slow tests
pytest -m "not slow" -v
```

### `@pytest.mark.rpc`
Tests requiring RPC endpoint access

```bash
# Run only RPC tests
pytest -m rpc -v
```

### Combining Markers

```bash
# Run integration tests but skip slow ones
pytest -m "integration and not slow" -v
```

---

## Test Coverage Summary

| Phase | Tests | Coverage | Status |
|-------|-------|----------|--------|
| **Phase 1** | 37 tests (33 unit + 4 integration) | >85% | ✅ Complete |
| **Phase 2** | 46 tests (45 unit + 1 integration) | >90% | ✅ Complete |
| **Phase 3** | 233 tests (231 unit + 2 integration) | >85% | ✅ Complete |
| **Overall** | **316 tests** (309 unit + 7 integration) | **>85%** | ✅ **All Passing** |

### Component-Level Coverage

| Component | Phase | Tests | Coverage |
|-----------|-------|-------|----------|
| Configuration Manager | 1 | 3 | >90% |
| Metrics System | 1 | 3 | >90% |
| RPC Manager | 1 | 12 | >85% |
| Health Monitor | 1 | 19 | >85% |
| Transaction Builder | 2 | 9 | >90% |
| Transaction Signer | 2 | 9 | >90% |
| Transaction Submitter | 2 | 9 | >90% |
| Priority Fees | 2 | 9 | >90% |
| Wallet Manager | 2 | 9 | >90% |
| Pump.fun Client | 3 | 21 | >90% |
| Bonding Curve | 3 | 41 | >95% |
| Slippage Manager | 3 | 19 | >90% |
| PnL Calculator | 3 | 49 | >90% |
| Position Tracker | 3 | 24 | >90% |

---

## Writing New Tests

### Unit Test Template

```python
"""
Unit tests for Component X

Tests:
- Feature 1
- Feature 2
- Error handling
"""

import pytest
from core.component import Component


class TestComponent:
    """Test Component functionality"""

    def test_basic_functionality(self):
        """Test basic feature"""
        component = Component()
        result = component.do_something()
        assert result == expected_value

    def test_error_handling(self):
        """Test error case"""
        component = Component()
        with pytest.raises(ValueError, match="error message"):
            component.do_invalid_thing()
```

### Integration Test Template

```python
"""
Integration tests for Component X

Requires network access
"""

import pytest
import asyncio


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_real_world_scenario():
    """Test with real external services"""
    # Setup
    component = Component()

    try:
        # Test
        result = await component.call_external_api()
        assert result is not None

    finally:
        # Cleanup
        await component.shutdown()
```

---

## Fixtures

Shared fixtures are defined in `tests/conftest.py`:

### `test_config_dict`
Sample configuration dictionary

```python
def test_example(test_config_dict):
    # test_config_dict is a valid config dict
    assert "rpc" in test_config_dict
```

### `test_config_file`
Temporary config file path

```python
def test_example(test_config_file):
    config_manager = ConfigurationManager(test_config_file)
    config = config_manager.load_config()
```

### `test_rpc_config`
Pre-built RPCConfig object

```python
def test_example(test_rpc_config):
    rpc_manager = RPCManager(test_rpc_config)
```

### `metrics_collector`
Fresh MetricsCollector instance

```python
def test_example(metrics_collector):
    metrics_collector.increment_counter("test")
    assert metrics_collector.get_counter("test") == 1
```

---

## Continuous Integration (Future)

### GitHub Actions Workflow (Example)

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/unit/ -v --cov=core
      - run: pytest tests/integration/ -v
```

---

## Regression Testing Strategy

### After Adding Phase 2

```bash
# Run Phase 1 tests to ensure nothing broke
pytest tests/unit/test_config.py tests/unit/test_metrics.py -v

# Run Phase 1 integration tests
pytest tests/integration/test_phase1_integration.py -v

# Run Phase 2 tests
pytest tests/unit/test_phase2*.py -v
```

### Before Production Deployment

```bash
# Run ALL tests
pytest tests/ -v --cov=core --cov-report=html

# Ensure >80% coverage
# Ensure 0 failures
```

---

## Troubleshooting

### Tests Hanging

```bash
# Use timeout marker
@pytest.mark.timeout(10)  # Fail after 10 seconds
```

### RPC Connection Failures

```bash
# Check config.yml has valid endpoints
# Check network connectivity
# Skip integration tests: pytest -m "not integration"
```

### Import Errors

```bash
# Ensure you're in frontrun/ directory
cd frontrun

# Run as module
python -m pytest tests/
```

### Fixture Not Found

```bash
# Check tests/conftest.py exists
# Ensure fixture is defined in conftest.py
# Run from frontrun/ directory
```

---

## Performance Testing (Future)

### Latency Benchmarks

```python
@pytest.mark.benchmark
def test_transaction_build_performance(benchmark):
    result = benchmark(build_transaction, params)
    assert result.elapsed < 0.010  # <10ms
```

### Load Tests

```python
@pytest.mark.load
@pytest.mark.asyncio
async def test_1000_concurrent_connections():
    tasks = [connect_to_rpc() for _ in range(1000)]
    results = await asyncio.gather(*tasks)
    assert all(r.success for r in results)
```

---

## Best Practices

1. ✅ **Test one thing per test** - Easy to debug failures
2. ✅ **Use descriptive test names** - `test_config_loads_successfully` not `test1`
3. ✅ **Arrange-Act-Assert pattern** - Setup, execute, verify
4. ✅ **Use fixtures for setup** - DRY principle
5. ✅ **Test error cases** - Don't just test happy path
6. ✅ **Mock external dependencies in unit tests** - Fast and reliable
7. ✅ **Use real services in integration tests** - Catch real issues
8. ✅ **Add docstrings** - Explain what's being tested
9. ✅ **Keep tests fast** - Unit tests <100ms, integration <5s
10. ✅ **Clean up resources** - Use try/finally or fixtures

---

## Test Coverage Report

Generate HTML coverage report:

```bash
pytest tests/ --cov=core --cov-report=html --cov-report=term
```

View report:

```bash
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

---

## Adding Tests for New Phases

When adding Phase 2, create:

```
tests/
├── unit/
│   ├── test_tx_builder.py        # Unit tests
│   ├── test_tx_signer.py
│   └── test_tx_submitter.py
└── integration/
    └── test_phase2_integration.py  # Integration tests
```

Run Phase 1 regression tests:

```bash
# Ensure Phase 1 still works
pytest tests/unit/test_config.py tests/unit/test_metrics.py -v
pytest tests/integration/test_phase1_integration.py -v
```

---

## Useful Commands Cheat Sheet

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only (fast)
pytest tests/unit/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html

# Run specific test
pytest tests/unit/test_config.py::TestConfigurationManager::test_load_valid_config -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Skip integration tests
pytest tests/ -m "not integration" -v

# Run and show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Parallel execution (future, requires pytest-xdist)
pytest tests/ -n auto
```

---

## Questions?

If tests fail:
1. Read the error message carefully
2. Check if it's a network issue (integration tests)
3. Verify config/config.yml is valid
4. Try running just that test: `pytest path/to/test.py::test_name -v -s`
5. Check test docstring for requirements

Happy testing! 🧪
