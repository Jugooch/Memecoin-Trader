"""
Unit tests for Priority Fee Calculator (core/priority_fees.py)

Tests:
- Fee calculation
- Urgency mapping
- Percentile calculations
- Caching
- Configuration
"""

import pytest
import time
from unittest.mock import MagicMock, AsyncMock

from core.priority_fees import (
    PriorityFeeCalculator,
    FeeConfig,
    FeeUrgency,
    FeeEstimate,
    PriorityFeeStats
)


class TestFeeConfig:
    """Test fee configuration"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = FeeConfig()

        assert config.update_interval_s == 10
        assert config.lookback_slots == 150
        assert config.min_priority_fee == 1_000
        assert config.max_priority_fee == 1_000_000
        assert config.cache_ttl_seconds == 10

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = FeeConfig(
            min_priority_fee=5_000,
            max_priority_fee=500_000,
            cache_ttl_seconds=30
        )

        assert config.min_priority_fee == 5_000
        assert config.max_priority_fee == 500_000
        assert config.cache_ttl_seconds == 30


class TestFeeUrgency:
    """Test fee urgency enum"""

    def test_urgency_values(self):
        """Test all urgency values"""
        assert FeeUrgency.LOW.value == "low"
        assert FeeUrgency.NORMAL.value == "normal"
        assert FeeUrgency.HIGH.value == "high"
        assert FeeUrgency.CRITICAL.value == "critical"


class TestFeeEstimate:
    """Test fee estimate data structure"""

    def test_estimate_creation(self):
        """Test creating fee estimate"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        assert estimate.p50 == 5_000
        assert estimate.p99 == 200_000
        assert estimate.sample_count == 100

    def test_get_fee_for_urgency_low(self):
        """Test getting fee for low urgency"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        fee = estimate.get_fee_for_urgency(FeeUrgency.LOW)
        assert fee == 5_000

    def test_get_fee_for_urgency_normal(self):
        """Test getting fee for normal urgency"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        fee = estimate.get_fee_for_urgency(FeeUrgency.NORMAL)
        assert fee == 10_000

    def test_get_fee_for_urgency_high(self):
        """Test getting fee for high urgency"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        fee = estimate.get_fee_for_urgency(FeeUrgency.HIGH)
        assert fee == 50_000

    def test_get_fee_for_urgency_critical(self):
        """Test getting fee for critical urgency"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        fee = estimate.get_fee_for_urgency(FeeUrgency.CRITICAL)
        assert fee == 200_000

    def test_is_expired_false(self):
        """Test estimate not expired when within TTL"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        assert estimate.is_expired(10) is False

    def test_is_expired_true(self):
        """Test estimate expires after TTL"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time() - 11  # 11 seconds ago
        )

        assert estimate.is_expired(10) is True

    def test_to_dict(self):
        """Test converting estimate to dictionary"""
        estimate = FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=100,
            cached_at=time.time()
        )

        est_dict = estimate.to_dict()

        assert "p50" in est_dict
        assert "p99" in est_dict
        assert "sample_count" in est_dict
        assert est_dict["p50"] == 5_000


class TestPriorityFeeStats:
    """Test priority fee stats"""

    def test_stats_creation(self):
        """Test creating priority fee stats"""
        stats = PriorityFeeStats(
            slot=1000,
            prioritization_fee=50_000,
            timestamp=time.time()
        )

        assert stats.slot == 1000
        assert stats.prioritization_fee == 50_000

    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = PriorityFeeStats(
            slot=1000,
            prioritization_fee=50_000,
            timestamp=time.time()
        )

        stats_dict = stats.to_dict()

        assert "slot" in stats_dict
        assert "prioritization_fee" in stats_dict
        assert stats_dict["slot"] == 1000


class TestPriorityFeeCalculator:
    """Test priority fee calculator functionality"""

    def test_calculator_initialization(self):
        """Test calculator initializes correctly"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        assert calculator.rpc_manager == rpc_manager
        assert calculator.config is not None
        assert calculator._cached_estimate is None

    def test_calculator_with_custom_config(self):
        """Test calculator with custom configuration"""
        rpc_manager = MagicMock()
        config = FeeConfig(min_priority_fee=5_000)
        calculator = PriorityFeeCalculator(rpc_manager, config)

        assert calculator.config.min_priority_fee == 5_000

    def test_invalidate_cache(self):
        """Test invalidating cache"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        # Set a cached estimate
        calculator._cached_estimate = FeeEstimate(
            p50=5_000, p75=10_000, p90=50_000, p95=100_000, p99=200_000,
            mean=25_000, min=1_000, max=500_000,
            sample_count=100, cached_at=time.time()
        )

        assert calculator._cached_estimate is not None

        calculator.invalidate_cache()

        assert calculator._cached_estimate is None

    def test_get_stats_empty_cache(self):
        """Test getting stats with empty cache"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        stats = calculator.get_stats()

        assert "cache_status" in stats
        assert stats["cache_status"] == "empty"

    def test_get_stats_valid_cache(self):
        """Test getting stats with valid cache"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        # Set valid cached estimate
        calculator._cached_estimate = FeeEstimate(
            p50=5_000, p75=10_000, p90=50_000, p95=100_000, p99=200_000,
            mean=25_000, min=1_000, max=500_000,
            sample_count=100, cached_at=time.time()
        )

        stats = calculator.get_stats()

        assert stats["cache_status"] == "valid"
        assert "cached_estimate" in stats

    def test_get_stats_expired_cache(self):
        """Test getting stats with expired cache"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        # Set expired cached estimate
        calculator._cached_estimate = FeeEstimate(
            p50=5_000, p75=10_000, p90=50_000, p95=100_000, p99=200_000,
            mean=25_000, min=1_000, max=500_000,
            sample_count=100, cached_at=time.time() - 11  # 11 seconds ago
        )

        stats = calculator.get_stats()

        assert stats["cache_status"] == "expired"

    def test_generate_sample_fees(self):
        """Test generating sample fees"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        fees = calculator._generate_sample_fees()

        assert len(fees) == 100  # 80 + 15 + 5
        assert all(fee > 0 for fee in fees)

    def test_calculate_percentiles(self):
        """Test calculating percentiles from fees"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        # Simple test data
        fees = [1_000, 5_000, 10_000, 50_000, 100_000]

        estimate = calculator._calculate_percentiles(fees)

        assert estimate.min == 1_000
        assert estimate.max == 100_000
        assert estimate.sample_count == 5
        assert estimate.p50 > 0

    def test_calculate_percentiles_empty(self):
        """Test calculating percentiles with no data"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        estimate = calculator._calculate_percentiles([])

        # Should return default estimate
        assert estimate.sample_count == 0

    def test_get_default_estimate(self):
        """Test getting default estimate"""
        rpc_manager = MagicMock()
        calculator = PriorityFeeCalculator(rpc_manager)

        estimate = calculator._get_default_estimate()

        assert estimate.sample_count == 0
        assert estimate.p50 > 0
        assert estimate.p99 > estimate.p50
