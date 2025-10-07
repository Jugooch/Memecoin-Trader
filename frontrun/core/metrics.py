"""
Metrics collection and export for Frontrun Bot
Tracks latencies, throughput, and system health
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics


@dataclass
class HistogramStats:
    """Statistical summary of histogram data"""
    operation: str
    count: int
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


class MetricsCollector:
    """Collects and aggregates performance metrics"""

    def __init__(self, enable_histogram: bool = True, histogram_buckets: Optional[List[float]] = None):
        """
        Initialize metrics collector

        Args:
            enable_histogram: Whether to collect histogram data
            histogram_buckets: Latency buckets for histogram
        """
        self.enable_histogram = enable_histogram
        self.histogram_buckets = histogram_buckets or [1, 5, 10, 50, 100, 500, 1000]

        # Latency histograms: operation -> list of latency values
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Counters: metric_name -> value
        self._counters: Dict[str, int] = defaultdict(int)

        # Gauges: metric_name -> value
        self._gauges: Dict[str, float] = defaultdict(float)

        # Labels cache: (metric_name, labels_tuple) -> value
        self._labeled_counters: Dict[tuple, int] = defaultdict(int)
        self._labeled_gauges: Dict[tuple, float] = defaultdict(float)

    def record_latency(
        self,
        operation: str,
        latency_ms: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record operation latency

        Args:
            operation: Operation name (e.g., "rpc_request", "tx_build")
            latency_ms: Latency in milliseconds
            labels: Optional labels for the metric
        """
        if self.enable_histogram:
            self._latencies[operation].append(latency_ms)

        # Also update counter for this operation
        if labels:
            label_key = (f"{operation}_count", tuple(sorted(labels.items())))
            self._labeled_counters[label_key] += 1
        else:
            self._counters[f"{operation}_count"] += 1

    def increment_counter(
        self,
        metric_name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric

        Args:
            metric_name: Name of the counter
            value: Amount to increment (default 1)
            labels: Optional labels for the metric
        """
        if labels:
            label_key = (metric_name, tuple(sorted(labels.items())))
            self._labeled_counters[label_key] += value
        else:
            self._counters[metric_name] += value

    def set_gauge(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value

        Args:
            metric_name: Name of the gauge
            value: Value to set
            labels: Optional labels for the metric
        """
        if labels:
            label_key = (metric_name, tuple(sorted(labels.items())))
            self._labeled_gauges[label_key] = value
        else:
            self._gauges[metric_name] = value

    def get_histogram_stats(self, operation: str) -> Optional[HistogramStats]:
        """
        Get histogram statistics for an operation

        Args:
            operation: Operation name

        Returns:
            HistogramStats or None if no data
        """
        latencies = list(self._latencies.get(operation, []))

        if not latencies:
            return None

        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)

        return HistogramStats(
            operation=operation,
            count=count,
            p50=self._percentile(sorted_latencies, 50),
            p95=self._percentile(sorted_latencies, 95),
            p99=self._percentile(sorted_latencies, 99),
            mean=statistics.mean(sorted_latencies),
            min=min(sorted_latencies),
            max=max(sorted_latencies)
        )

    def get_counter(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value"""
        if labels:
            label_key = (metric_name, tuple(sorted(labels.items())))
            return self._labeled_counters.get(label_key, 0)
        return self._counters.get(metric_name, 0)

    def get_gauge(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value"""
        if labels:
            label_key = (metric_name, tuple(sorted(labels.items())))
            return self._labeled_gauges.get(label_key, 0.0)
        return self._gauges.get(metric_name, 0.0)

    def export_metrics(self) -> Dict:
        """
        Export all metrics as JSON-serializable dict

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {}
        }

        # Export histogram stats
        for operation in self._latencies.keys():
            stats = self.get_histogram_stats(operation)
            if stats:
                metrics["histograms"][operation] = {
                    "count": stats.count,
                    "p50": stats.p50,
                    "p95": stats.p95,
                    "p99": stats.p99,
                    "mean": stats.mean,
                    "min": stats.min,
                    "max": stats.max
                }

        return metrics

    def reset(self) -> None:
        """Reset all metrics (useful for testing)"""
        self._latencies.clear()
        self._counters.clear()
        self._gauges.clear()
        self._labeled_counters.clear()
        self._labeled_gauges.clear()

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data"""
        if not sorted_data:
            return 0.0

        if len(sorted_data) == 1:
            return sorted_data[0]

        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1

        if upper >= len(sorted_data):
            return sorted_data[-1]

        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


class LatencyTimer:
    """Context manager for measuring operation latency"""

    def __init__(self, metrics: MetricsCollector, operation: str, labels: Optional[Dict[str, str]] = None):
        """
        Initialize latency timer

        Args:
            metrics: MetricsCollector instance
            operation: Operation name
            labels: Optional labels
        """
        self.metrics = metrics
        self.operation = operation
        self.labels = labels
        self.start_time: Optional[float] = None
        self.latency_ms: Optional[float] = None

    def __enter__(self):
        """Start timing"""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record latency"""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.latency_ms = elapsed * 1000  # Convert to milliseconds
            self.metrics.record_latency(self.operation, self.latency_ms, self.labels)


# Global metrics instance (initialized by main)
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def init_metrics(enable_histogram: bool = True, histogram_buckets: Optional[List[float]] = None) -> MetricsCollector:
    """Initialize global metrics collector"""
    global _global_metrics
    _global_metrics = MetricsCollector(enable_histogram, histogram_buckets)
    return _global_metrics


# Example usage
if __name__ == "__main__":
    # Initialize metrics
    metrics = MetricsCollector()

    # Simulate some RPC requests
    import random

    for i in range(100):
        latency = random.uniform(10, 500)
        metrics.record_latency("rpc_request", latency, labels={"endpoint": "helius"})

        if random.random() > 0.9:  # 10% error rate
            metrics.increment_counter("rpc_errors", labels={"endpoint": "helius"})

    # Get stats
    stats = metrics.get_histogram_stats("rpc_request")
    if stats:
        print(f"RPC Request Latency:")
        print(f"  Count: {stats.count}")
        print(f"  p50: {stats.p50:.2f}ms")
        print(f"  p95: {stats.p95:.2f}ms")
        print(f"  p99: {stats.p99:.2f}ms")
        print(f"  Mean: {stats.mean:.2f}ms")

    # Test latency timer context manager
    with LatencyTimer(metrics, "test_operation") as timer:
        time.sleep(0.1)  # Simulate work
    print(f"\nTest operation took: {timer.latency_ms:.2f}ms")

    # Export all metrics
    print("\nAll metrics:")
    import json
    print(json.dumps(metrics.export_metrics(), indent=2))
