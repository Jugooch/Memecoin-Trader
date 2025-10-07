"""
Unit tests for Metrics System (core/metrics.py)

Tests:
- Latency recording and histogram calculation
- Counter increment
- Gauge set/get
- Percentile calculations (p50, p95, p99)
- Metrics export
- LatencyTimer context manager
"""

import pytest
import time
from core.metrics import MetricsCollector, LatencyTimer, HistogramStats


class TestMetricsCollector:
    """Test metrics collection functionality"""

    def test_record_latency(self, metrics_collector):
        """Test recording latency measurements"""
        # Record some latencies
        metrics_collector.record_latency("test_operation", 100.5)
        metrics_collector.record_latency("test_operation", 200.3)
        metrics_collector.record_latency("test_operation", 150.7)

        # Get histogram stats
        stats = metrics_collector.get_histogram_stats("test_operation")

        assert stats is not None
        assert stats.count == 3
        assert stats.min == pytest.approx(100.5, rel=0.01)
        assert stats.max == pytest.approx(200.3, rel=0.01)

    def test_latency_with_labels(self, metrics_collector):
        """Test latency recording with labels"""
        # Record with labels
        metrics_collector.record_latency(
            "rpc_call",
            150.0,
            labels={"endpoint": "helius"}
        )
        metrics_collector.record_latency(
            "rpc_call",
            200.0,
            labels={"endpoint": "quicknode"}
        )

        # Verify counters created with labels
        helius_count = metrics_collector.get_counter(
            "rpc_call_count",
            labels={"endpoint": "helius"}
        )
        quicknode_count = metrics_collector.get_counter(
            "rpc_call_count",
            labels={"endpoint": "quicknode"}
        )

        assert helius_count == 1
        assert quicknode_count == 1

    def test_percentile_calculations(self, metrics_collector):
        """Test p50, p95, p99 percentile calculations"""
        # Record 100 samples
        for i in range(100):
            metrics_collector.record_latency("test_op", float(i))

        stats = metrics_collector.get_histogram_stats("test_op")

        # Verify percentiles
        assert stats.p50 == pytest.approx(49.5, rel=0.1)  # Median
        assert stats.p95 == pytest.approx(94.05, rel=0.1)
        assert stats.p99 == pytest.approx(98.01, rel=0.1)
        assert stats.mean == pytest.approx(49.5, rel=0.1)

    def test_increment_counter(self, metrics_collector):
        """Test counter incrementing"""
        # Increment by 1 (default)
        metrics_collector.increment_counter("test_counter")
        assert metrics_collector.get_counter("test_counter") == 1

        # Increment by custom amount
        metrics_collector.increment_counter("test_counter", value=5)
        assert metrics_collector.get_counter("test_counter") == 6

        # Increment another counter
        metrics_collector.increment_counter("other_counter", value=10)
        assert metrics_collector.get_counter("other_counter") == 10

    def test_counter_with_labels(self, metrics_collector):
        """Test counters with label support"""
        metrics_collector.increment_counter("errors", labels={"endpoint": "helius"})
        metrics_collector.increment_counter("errors", labels={"endpoint": "helius"})
        metrics_collector.increment_counter("errors", labels={"endpoint": "quicknode"})

        helius_errors = metrics_collector.get_counter("errors", labels={"endpoint": "helius"})
        quicknode_errors = metrics_collector.get_counter("errors", labels={"endpoint": "quicknode"})

        assert helius_errors == 2
        assert quicknode_errors == 1

    def test_set_gauge(self, metrics_collector):
        """Test gauge setting"""
        metrics_collector.set_gauge("active_connections", 5.0)
        assert metrics_collector.get_gauge("active_connections") == 5.0

        # Update gauge
        metrics_collector.set_gauge("active_connections", 10.0)
        assert metrics_collector.get_gauge("active_connections") == 10.0

    def test_gauge_with_labels(self, metrics_collector):
        """Test gauges with labels"""
        metrics_collector.set_gauge("memory_usage", 50.5, labels={"process": "rpc_manager"})
        metrics_collector.set_gauge("memory_usage", 30.2, labels={"process": "tx_builder"})

        rpc_memory = metrics_collector.get_gauge("memory_usage", labels={"process": "rpc_manager"})
        tx_memory = metrics_collector.get_gauge("memory_usage", labels={"process": "tx_builder"})

        assert rpc_memory == 50.5
        assert tx_memory == 30.2

    def test_export_metrics(self, metrics_collector):
        """Test metrics export to JSON"""
        # Add some metrics
        metrics_collector.increment_counter("requests", value=100)
        metrics_collector.set_gauge("temperature", 75.5)
        metrics_collector.record_latency("api_call", 150.0)
        metrics_collector.record_latency("api_call", 200.0)

        # Export
        exported = metrics_collector.export_metrics()

        # Verify structure
        assert "counters" in exported
        assert "gauges" in exported
        assert "histograms" in exported

        # Verify data
        assert exported["counters"]["requests"] == 100
        assert exported["gauges"]["temperature"] == 75.5
        assert "api_call" in exported["histograms"]
        assert exported["histograms"]["api_call"]["count"] == 2

    def test_histogram_stats_empty(self, metrics_collector):
        """Test getting stats for non-existent operation"""
        stats = metrics_collector.get_histogram_stats("nonexistent")
        assert stats is None

    def test_reset_metrics(self, metrics_collector):
        """Test resetting all metrics"""
        # Add some metrics
        metrics_collector.increment_counter("test", value=10)
        metrics_collector.set_gauge("gauge", 5.0)
        metrics_collector.record_latency("latency", 100.0)

        # Reset
        metrics_collector.reset()

        # Verify all cleared
        assert metrics_collector.get_counter("test") == 0
        assert metrics_collector.get_gauge("gauge") == 0.0
        assert metrics_collector.get_histogram_stats("latency") is None

    def test_histogram_max_size(self, metrics_collector):
        """Test histogram size limit (maxlen=10000)"""
        # Record more than 10000 samples
        for i in range(15000):
            metrics_collector.record_latency("test_op", float(i))

        stats = metrics_collector.get_histogram_stats("test_op")

        # Should only keep last 10000
        assert stats.count == 10000
        assert stats.min >= 5000.0  # Oldest samples dropped


class TestLatencyTimer:
    """Test LatencyTimer context manager"""

    def test_latency_timer_basic(self, metrics_collector):
        """Test basic latency timer usage"""
        with LatencyTimer(metrics_collector, "test_operation") as timer:
            time.sleep(0.01)  # Sleep 10ms

        # Verify latency recorded
        stats = metrics_collector.get_histogram_stats("test_operation")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean >= 10.0  # At least 10ms
        assert timer.latency_ms >= 10.0

    def test_latency_timer_with_labels(self, metrics_collector):
        """Test latency timer with labels"""
        with LatencyTimer(metrics_collector, "api_call", labels={"endpoint": "test"}):
            time.sleep(0.005)  # Sleep 5ms

        # Verify recorded
        stats = metrics_collector.get_histogram_stats("api_call")
        assert stats.count == 1

        # Verify counter with labels
        count = metrics_collector.get_counter("api_call_count", labels={"endpoint": "test"})
        assert count == 1

    def test_latency_timer_exception(self, metrics_collector):
        """Test that latency is still recorded on exception"""
        try:
            with LatencyTimer(metrics_collector, "failing_op") as timer:
                time.sleep(0.005)
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify latency still recorded
        stats = metrics_collector.get_histogram_stats("failing_op")
        assert stats is not None
        assert stats.count == 1
        assert stats.mean >= 5.0

    def test_multiple_timers(self, metrics_collector):
        """Test multiple concurrent operations"""
        # Simulate multiple operations
        for i in range(5):
            with LatencyTimer(metrics_collector, "batch_op"):
                time.sleep(0.002 * (i + 1))  # Variable sleep

        stats = metrics_collector.get_histogram_stats("batch_op")
        assert stats.count == 5
        assert stats.min >= 2.0
        assert stats.max >= 10.0


class TestHistogramStats:
    """Test HistogramStats dataclass"""

    def test_histogram_stats_structure(self):
        """Test HistogramStats structure"""
        stats = HistogramStats(
            operation="test_op",
            count=100,
            p50=50.0,
            p95=95.0,
            p99=99.0,
            mean=55.0,
            min=10.0,
            max=100.0
        )

        assert stats.operation == "test_op"
        assert stats.count == 100
        assert stats.p50 == 50.0
        assert stats.p95 == 95.0
        assert stats.p99 == 99.0
        assert stats.mean == 55.0
        assert stats.min == 10.0
        assert stats.max == 100.0


class TestMetricsIntegration:
    """Integration tests for metrics system"""

    def test_realistic_workload(self, metrics_collector):
        """Test with realistic bot workload simulation"""
        # Simulate 100 RPC calls
        for i in range(100):
            # Vary latency
            latency = 50.0 + (i % 50)
            metrics_collector.record_latency(
                "rpc_call",
                latency,
                labels={"endpoint": "helius" if i % 2 == 0 else "quicknode"}
            )

            # Some errors
            if i % 10 == 0:
                metrics_collector.increment_counter("errors", labels={"type": "timeout"})

        # Verify stats
        stats = metrics_collector.get_histogram_stats("rpc_call")
        assert stats.count == 100
        assert 50.0 <= stats.mean <= 100.0

        # Verify error count
        errors = metrics_collector.get_counter("errors", labels={"type": "timeout"})
        assert errors == 10

    def test_export_with_all_metric_types(self, metrics_collector):
        """Test export with counters, gauges, and histograms"""
        # Add all types
        metrics_collector.increment_counter("total_requests", value=1000)
        metrics_collector.set_gauge("active_connections", 25.0)

        for i in range(50):
            metrics_collector.record_latency("operation", float(i * 10))

        # Export
        exported = metrics_collector.export_metrics()

        # Verify all present
        assert exported["counters"]["total_requests"] == 1000
        assert exported["gauges"]["active_connections"] == 25.0
        assert exported["histograms"]["operation"]["count"] == 50
        assert "p50" in exported["histograms"]["operation"]
        assert "p95" in exported["histograms"]["operation"]
        assert "p99" in exported["histograms"]["operation"]
