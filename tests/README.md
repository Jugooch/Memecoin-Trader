# Testing Framework for Memecoin Trading Bot

## Overview
This directory contains all tests for the memecoin trading bot. Tests are organized by type and designed to ensure both paper and live trading modes work identically.

## Directory Structure
```
tests/
├── unit/           # Isolated component tests
├── integration/    # Multi-component interaction tests  
├── performance/    # Memory, latency, throughput tests
├── manual/         # Manual testing scripts and tools
├── fixtures/       # Test data and mocks
└── reports/        # Test results and coverage reports
```

## Installation

Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py all
```

### Run Specific Categories
```bash
# Unit tests only
python tests/run_all_tests.py unit

# Integration tests only
python tests/run_all_tests.py integration

# Performance tests only
python tests/run_all_tests.py performance
```

### Run Individual Test Files
```bash
# Run specific test file
python -m pytest tests/unit/test_wallet_scorer.py -v

# Run with coverage
python -m pytest tests/unit/test_wallet_scorer.py --cov=src.core.wallet_scorer

# Run specific test function
python -m pytest tests/unit/test_wallet_scorer.py::test_prior_score -v
```

### Continuous Testing During Development
```bash
# Auto-run tests on file changes
python -m pytest tests/ --watch

# Run with verbose output
python -m pytest tests/ -vv

# Run with print statements visible
python -m pytest tests/ -s
```

## Test Categories

### Unit Tests (`tests/unit/`)
Tests individual components in isolation.

**Example**: `test_wallet_scorer.py`
```python
def test_bayesian_prior():
    """Beta(3,5) prior should give 37.5% win rate"""
    scorer = WalletScorer(prior_alpha=3, prior_beta=5)
    assert scorer.get_wallet_score("new_wallet") == 0.375
```

**Expected Output**:
```
tests/unit/test_wallet_scorer.py::test_bayesian_prior PASSED [100%]
```

### Integration Tests (`tests/integration/`)
Tests component interactions and full workflows.

**Example**: `test_trading_flow.py`
```python
async def test_paper_trading_flow():
    """Test complete trade execution in paper mode"""
    engine = TradingEngine(paper_mode=True)
    signals = create_mock_signals(5)  # 5 wallets
    
    decision = await engine.should_enter_position(signals)
    assert decision == True
    
    result = await engine.execute_trade("mint_xxx", 25.0, "BUY")
    assert result['paper_mode'] == True
    assert result['fees'] > 0
```

**Expected Output**:
```
tests/integration/test_trading_flow.py::test_paper_trading_flow PASSED
Trade executed: paper_mode=True, fees=$0.73, slippage=0.5%
```

### Performance Tests (`tests/performance/`)
Tests system resource usage and performance metrics.

### Manual Tests (`tests/manual/`)
Scripts for manual testing and debugging, including:
- `manual_sell_gdp.py` - Manual token selling script
- `test_bitquery_rotation.py` - Bitquery API rotation testing
- `test_discord.py` - Discord notification testing

**Example**: `test_memory_usage.py`
```python
def test_memory_stability():
    """Memory should not grow unbounded"""
    initial_memory = get_memory_usage()
    
    # Process 1000 trades
    for i in range(1000):
        process_mock_trade()
    
    final_memory = get_memory_usage()
    growth = final_memory - initial_memory
    
    assert growth < 10_000_000  # Less than 10MB growth
    print(f"Memory growth: {growth/1_000_000:.2f} MB")
```

**Expected Output**:
```
Initial memory: 120.5 MB
After 1000 trades: 124.3 MB
Memory growth: 3.8 MB
✓ Memory usage stable
```

## Test Data & Fixtures

### Sample Trade Data (`fixtures/sample_trades.json`)
```json
{
  "trades": [
    {
      "mint": "So11111111111111111111111111111111111111112",
      "wallet": "8xY3Qz4nV2pL5mK9jF6wT1hR7dC3bA2gE4iU6oP9sN1x",
      "action": "buy",
      "price": 0.00000123,
      "amount": 1000000,
      "timestamp": 1704067200
    }
  ]
}
```

### Mock Stream Data (`fixtures/mock_stream_data.json`)
Contains realistic Bitquery websocket responses for testing stream processing.

## Coverage Reports

Generate coverage reports:
```bash
# HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
# Open: tests/reports/coverage/index.html

# Terminal coverage summary
python -m pytest tests/ --cov=src --cov-report=term-missing

# Fail if coverage below threshold
python -m pytest tests/ --cov=src --cov-fail-under=80
```

## Writing New Tests

### Test Template
```python
"""
Test [Component Name]

Run: python -m pytest tests/unit/test_[component].py -v

Expected output:
test_[scenario1] ✓ ([what it validates])
test_[scenario2] ✓ ([what it validates])
"""

import pytest
from src.module import Component

class TestComponent:
    """Test suite for Component"""
    
    @pytest.fixture
    def component(self):
        """Create component instance for testing"""
        return Component(test_config)
    
    def test_normal_operation(self, component):
        """Test normal operation scenario"""
        result = component.process(valid_input)
        assert result['success'] == True
        assert result['value'] > 0
    
    def test_edge_case(self, component):
        """Test edge case handling"""
        result = component.process(edge_input)
        assert result['handled'] == True
    
    def test_error_handling(self, component):
        """Test error scenarios"""
        with pytest.raises(ValueError):
            component.process(invalid_input)
```

### Testing Paper vs Live Consistency
Always test that paper and live modes produce identical decisions:

```python
@pytest.mark.parametrize("mode", ["paper", "live"])
def test_decision_consistency(mode):
    """Decisions must be identical in paper and live modes"""
    engine = TradingEngine(paper_mode=(mode=="paper"))
    
    signals = create_test_signals()
    decision = engine.should_enter_position(signals)
    
    # Decision logic should be identical
    assert decision == expected_decision
    
    # Only execution differs
    if mode == "paper":
        assert engine.paper_mode == True
    else:
        assert engine.paper_mode == False
```

## Common Test Patterns

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result is not None
```

### Testing with Mocks
```python
from unittest.mock import Mock, patch

def test_with_mock():
    mock_client = Mock()
    mock_client.get_price.return_value = 0.0001
    
    engine = TradingEngine(moralis_client=mock_client)
    result = engine.calculate_value()
    
    mock_client.get_price.assert_called_once()
    assert result > 0
```

### Testing Time-Dependent Code
```python
from freezegun import freeze_time

@freeze_time("2024-01-01 12:00:00")
def test_time_window():
    scorer = WalletScorer()
    scorer.update_wallet_outcome("wallet1", True, time.time())
    
    # Move time forward
    with freeze_time("2024-01-01 13:30:00"):
        # Should still be in 60-min window
        score = scorer.get_wallet_score("wallet1")
        assert score > 0.5
```

## Debugging Failed Tests

### Get More Information
```bash
# Verbose output
python -m pytest tests/unit/test_wallet_scorer.py -vv

# Show print statements
python -m pytest tests/unit/test_wallet_scorer.py -s

# Stop on first failure
python -m pytest tests/unit/test_wallet_scorer.py -x

# Enter debugger on failure
python -m pytest tests/unit/test_wallet_scorer.py --pdb
```

### Common Issues and Solutions

1. **Import Errors**
   - Ensure `PYTHONPATH` includes project root
   - Run from project root: `python -m pytest tests/`

2. **Async Test Failures**
   - Add `@pytest.mark.asyncio` decorator
   - Install: `pip install pytest-asyncio`

3. **Mock Data Issues**
   - Check fixture files exist in `tests/fixtures/`
   - Verify JSON is valid

4. **Flaky Tests**
   - Use fixed seeds: `random.seed(42)`
   - Mock time: `@freeze_time("2024-01-01")`
   - Mock external services

## Continuous Integration

### Pre-Commit Hook
Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python tests/run_all_tests.py unit
if [ $? -ne 0 ]; then
    echo "Unit tests failed. Commit aborted."
    exit 1
fi
```

### GitHub Actions (Optional)
`.github/workflows/test.yml`:
```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pip install -r tests/requirements-test.txt
      - run: python tests/run_all_tests.py all
```

## Test Maintenance

### Weekly Tasks
- Review and update test data
- Check coverage gaps
- Update expected outputs
- Remove obsolete tests

### Before Major Changes
- Run full test suite
- Save baseline metrics
- Document expected changes

### After Deployment
- Compare paper vs live results
- Validate assumptions
- Add tests for any bugs found

## Support

For test-related issues:
1. Check this README first
2. Review test output carefully
3. Check Discord for alerts
4. Review logs in `tests/reports/`