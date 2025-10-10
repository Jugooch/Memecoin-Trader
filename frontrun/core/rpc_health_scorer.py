"""
RPC Health Scoring & Auto-Routing for Frontrun Bot
Scores RPC endpoints and automatically routes to best performer
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque

from core.rpc_manager import RPCManager, RPCConnection
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class HealthScorerConfig:
    """
    Configuration for RPC health scoring

    Scoring parameters:
    - base_score: Starting score for all endpoints (default: 100)
    - latency_threshold_p99_ms: p99 latency threshold (default: 500ms)
    - slot_lag_threshold: Max acceptable slot lag (default: 5)
    - error_rate_threshold: Max acceptable error rate (default: 0.1 = 10%)

    Deductions:
    - latency_deduction_per_100ms: Points lost per 100ms of p99 latency (default: 10)
    - slot_lag_deduction_per_slot: Points lost per slot of lag (default: 20)
    - error_rate_deduction: Points lost per 1% error rate (default: 5)

    Ejection:
    - fast_eject_slot_lag: Eject if slot lag > N for >5s (default: 5)
    - fast_eject_consecutive_errors: Eject after N consecutive errors (default: 2)

    Re-evaluation:
    - reeval_interval_s: Re-score endpoints every N seconds (default: 10)
    - brownout_mode_enabled: Keep degraded endpoints for read-only (default: True)
    """
    # Scoring parameters
    base_score: int = 100
    latency_threshold_p99_ms: float = 500.0
    slot_lag_threshold: int = 5
    error_rate_threshold: float = 0.1  # 10%

    # Deductions
    latency_deduction_per_100ms: int = 10
    slot_lag_deduction_per_slot: int = 20
    error_rate_deduction: int = 5

    # Fast ejection
    fast_eject_slot_lag: int = 5
    fast_eject_consecutive_errors: int = 2

    # Re-evaluation
    reeval_interval_s: int = 10
    brownout_mode_enabled: bool = True


@dataclass
class EndpointHealth:
    """
    Health score and metrics for an RPC endpoint

    Attributes:
        label: Endpoint label
        score: Health score (0-100)
        is_healthy: Whether endpoint is considered healthy
        is_ejected: Whether endpoint has been ejected
        p99_latency_ms: 99th percentile latency
        slot_lag: Current slot lag
        error_rate: Error rate (0.0-1.0)
        consecutive_errors: Consecutive error count
        last_score_update: When score was last calculated
    """
    label: str
    score: int = 100
    is_healthy: bool = True
    is_ejected: bool = False
    p99_latency_ms: float = 0.0
    slot_lag: int = 0
    error_rate: float = 0.0
    consecutive_errors: int = 0
    last_score_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "label": self.label,
            "score": self.score,
            "is_healthy": self.is_healthy,
            "is_ejected": self.is_ejected,
            "p99_latency_ms": self.p99_latency_ms,
            "slot_lag": self.slot_lag,
            "error_rate": self.error_rate,
            "consecutive_errors": self.consecutive_errors
        }


class RPCHealthScorer:
    """
    Scores RPC endpoints and routes to best performer

    Features:
    - Health score (0-100) per endpoint
    - Automatic routing to best-scoring endpoint
    - Fast eject on acute faults
    - Brownout mode for degraded endpoints
    - Continuous re-evaluation

    Usage:
        scorer = RPCHealthScorer(rpc_manager, config)
        await scorer.start()

        # Get best endpoint
        best = scorer.get_best_endpoint()

        # Record request result
        scorer.record_request(
            endpoint_label="helius",
            success=True,
            latency_ms=45.2,
            slot_lag=1
        )
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        config: Optional[HealthScorerConfig] = None
    ):
        """
        Initialize RPC health scorer

        Args:
            rpc_manager: RPC manager
            config: Health scorer configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or HealthScorerConfig()

        # Endpoint health tracking
        self._endpoint_health: Dict[str, EndpointHealth] = {}

        # Request history for error rate calculation (last 100 requests per endpoint)
        self._request_history: Dict[str, deque] = {}
        self._max_request_history = 100

        # Current best endpoint
        self._current_best: Optional[str] = None

        logger.info(
            "rpc_health_scorer_initialized",
            base_score=self.config.base_score,
            latency_threshold_ms=self.config.latency_threshold_p99_ms,
            slot_lag_threshold=self.config.slot_lag_threshold,
            reeval_interval_s=self.config.reeval_interval_s
        )

    async def start(self):
        """Initialize health tracking for all endpoints"""
        # Get RPC health stats
        rpc_stats = self.rpc_manager.get_health_stats()

        for label, stats in rpc_stats.items():
            self._endpoint_health[label] = EndpointHealth(
                label=label,
                score=self.config.base_score,
                is_healthy=stats.is_healthy,
                p99_latency_ms=stats.p99_latency_ms,
                error_rate=stats.error_rate
            )
            self._request_history[label] = deque(maxlen=self._max_request_history)

        # Initial scoring
        await self.update_all_scores()

        logger.info("rpc_health_scorer_started", endpoints=len(self._endpoint_health))

    def record_request(
        self,
        endpoint_label: str,
        success: bool,
        latency_ms: float,
        slot_lag: int = 0
    ):
        """
        Record request result for scoring

        Args:
            endpoint_label: Endpoint label
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            slot_lag: Slot lag (if applicable)

        Example:
            scorer.record_request(
                endpoint_label="helius",
                success=True,
                latency_ms=45.2,
                slot_lag=1
            )
        """
        if endpoint_label not in self._endpoint_health:
            logger.warning("unknown_endpoint", label=endpoint_label)
            return

        health = self._endpoint_health[endpoint_label]

        # Update request history
        if endpoint_label not in self._request_history:
            self._request_history[endpoint_label] = deque(maxlen=self._max_request_history)

        self._request_history[endpoint_label].append({
            "success": success,
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })

        # Update consecutive errors
        if success:
            health.consecutive_errors = 0
        else:
            health.consecutive_errors += 1

        # Check for fast ejection conditions
        if health.consecutive_errors >= self.config.fast_eject_consecutive_errors:
            self._fast_eject_endpoint(endpoint_label, "consecutive_errors")

        if slot_lag > self.config.fast_eject_slot_lag:
            self._fast_eject_endpoint(endpoint_label, "high_slot_lag")

        # Record metrics
        metrics.increment_counter(
            "rpc_requests",
            labels={
                "endpoint": endpoint_label,
                "success": str(success)
            }
        )

    async def update_all_scores(self):
        """Update health scores for all endpoints"""
        # Get latest RPC stats
        rpc_stats = self.rpc_manager.get_health_stats()

        for label, health in self._endpoint_health.items():
            # Skip ejected endpoints
            if health.is_ejected:
                continue

            # Get stats
            stats = rpc_stats.get(label)
            if not stats:
                continue

            # Calculate error rate from recent history
            if label in self._request_history and self._request_history[label]:
                recent_requests = list(self._request_history[label])
                errors = sum(1 for r in recent_requests if not r["success"])
                error_rate = errors / len(recent_requests) if recent_requests else 0.0
            else:
                error_rate = stats.error_rate

            # Calculate score
            score = self._calculate_score(
                p99_latency_ms=stats.p99_latency_ms,
                slot_lag=0,  # TODO: Get actual slot lag
                error_rate=error_rate
            )

            # Update health
            health.score = score
            health.is_healthy = score >= 50  # Healthy if score >= 50
            health.p99_latency_ms = stats.p99_latency_ms
            health.error_rate = error_rate
            health.last_score_update = time.time()

            logger.debug(
                "endpoint_score_updated",
                label=label,
                score=score,
                p99_latency_ms=stats.p99_latency_ms,
                error_rate=error_rate
            )

            metrics.set_gauge(
                "rpc_health_score",
                score,
                labels={"endpoint": label}
            )

        # Update best endpoint
        self._update_best_endpoint()

    def _calculate_score(
        self,
        p99_latency_ms: float,
        slot_lag: int,
        error_rate: float
    ) -> int:
        """
        Calculate health score for endpoint

        Args:
            p99_latency_ms: 99th percentile latency
            slot_lag: Slot lag
            error_rate: Error rate (0.0-1.0)

        Returns:
            Health score (0-100)

        Calculation:
        - Start with base_score (100)
        - Deduct for high latency (10 points per 100ms over threshold)
        - Deduct for slot lag (20 points per slot)
        - Deduct for errors (5 points per 1% error rate)
        """
        score = self.config.base_score

        # Latency deduction
        if p99_latency_ms > self.config.latency_threshold_p99_ms:
            overage_ms = p99_latency_ms - self.config.latency_threshold_p99_ms
            overage_100ms = overage_ms / 100.0
            latency_deduction = int(overage_100ms * self.config.latency_deduction_per_100ms)
            score -= latency_deduction

        # Slot lag deduction
        if slot_lag > 0:
            score -= slot_lag * self.config.slot_lag_deduction_per_slot

        # Error rate deduction
        error_rate_pct = error_rate * 100
        error_deduction = int(error_rate_pct * self.config.error_rate_deduction)
        score -= error_deduction

        # Clamp to 0-100
        return max(0, min(100, score))

    def _fast_eject_endpoint(self, endpoint_label: str, reason: str):
        """
        Eject endpoint immediately due to acute fault

        Args:
            endpoint_label: Endpoint to eject
            reason: Reason for ejection
        """
        if endpoint_label not in self._endpoint_health:
            return

        health = self._endpoint_health[endpoint_label]
        health.is_ejected = True
        health.is_healthy = False
        health.score = 0

        logger.error(
            "endpoint_ejected",
            label=endpoint_label,
            reason=reason,
            consecutive_errors=health.consecutive_errors
        )

        metrics.increment_counter(
            "rpc_endpoints_ejected",
            labels={"endpoint": endpoint_label, "reason": reason}
        )

        # Update best endpoint if current best was ejected
        if self._current_best == endpoint_label:
            self._update_best_endpoint()

    def _update_best_endpoint(self):
        """Update current best endpoint based on scores"""
        # Get all healthy endpoints
        healthy = [
            (label, health)
            for label, health in self._endpoint_health.items()
            if health.is_healthy and not health.is_ejected
        ]

        if not healthy:
            logger.error("no_healthy_endpoints_available")
            self._current_best = None
            return

        # Sort by score (highest first)
        healthy.sort(key=lambda x: x[1].score, reverse=True)

        # Get best endpoint
        best_label = healthy[0][0]
        best_score = healthy[0][1].score

        if self._current_best != best_label:
            logger.info(
                "best_endpoint_changed",
                previous=self._current_best,
                new=best_label,
                score=best_score
            )
            self._current_best = best_label

            metrics.increment_counter("best_endpoint_changes")

    def get_best_endpoint(self) -> Optional[str]:
        """
        Get current best endpoint label

        Returns:
            Best endpoint label or None if no healthy endpoints

        Example:
            best = scorer.get_best_endpoint()
            if best:
                # Use this endpoint for submissions
                pass
        """
        return self._current_best

    def get_endpoint_health(self, label: str) -> Optional[EndpointHealth]:
        """
        Get health info for specific endpoint

        Args:
            label: Endpoint label

        Returns:
            EndpointHealth or None if not found

        Example:
            health = scorer.get_endpoint_health("helius")
            print(f"Score: {health.score}")
        """
        return self._endpoint_health.get(label)

    def get_all_health(self) -> Dict[str, EndpointHealth]:
        """
        Get health info for all endpoints

        Returns:
            Dictionary of endpoint_label -> EndpointHealth

        Example:
            all_health = scorer.get_all_health()
            for label, health in all_health.items():
                print(f"{label}: score={health.score}")
        """
        return self._endpoint_health.copy()

    def restore_endpoint(self, endpoint_label: str):
        """
        Restore an ejected endpoint (for manual recovery)

        Args:
            endpoint_label: Endpoint to restore

        Example:
            scorer.restore_endpoint("helius")
        """
        if endpoint_label not in self._endpoint_health:
            logger.warning("unknown_endpoint_for_restore", label=endpoint_label)
            return

        health = self._endpoint_health[endpoint_label]
        health.is_ejected = False
        health.consecutive_errors = 0
        health.score = self.config.base_score

        logger.info("endpoint_restored", label=endpoint_label)

        # Trigger re-scoring
        self._update_best_endpoint()

    def get_stats(self) -> Dict:
        """
        Get scorer statistics

        Returns:
            Dictionary with statistics

        Example:
            stats = scorer.get_stats()
            print(f"Best endpoint: {stats['best_endpoint']}")
            print(f"Healthy endpoints: {stats['healthy_count']}")
        """
        healthy = [h for h in self._endpoint_health.values() if h.is_healthy]
        ejected = [h for h in self._endpoint_health.values() if h.is_ejected]

        avg_score = sum(h.score for h in healthy) / len(healthy) if healthy else 0

        return {
            "total_endpoints": len(self._endpoint_health),
            "healthy_count": len(healthy),
            "ejected_count": len(ejected),
            "best_endpoint": self._current_best,
            "avg_score": avg_score
        }


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging
    from core.config import ConfigurationManager
    import asyncio

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Load config
        config_mgr = ConfigurationManager("config/config.yml")
        bot_config = config_mgr.load_config()

        # Create RPC manager
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        try:
            # Create health scorer
            scorer_config = HealthScorerConfig(
                latency_threshold_p99_ms=500.0,
                fast_eject_consecutive_errors=2
            )
            scorer = RPCHealthScorer(rpc_manager, scorer_config)
            await scorer.start()

            # Simulate requests
            for i in range(20):
                scorer.record_request(
                    endpoint_label="helius_devnet",
                    success=True,
                    latency_ms=50.0 + (i * 10),
                    slot_lag=0
                )

            # Update scores
            await scorer.update_all_scores()

            # Get best endpoint
            best = scorer.get_best_endpoint()
            logger.info(f"Best endpoint: {best}")

            # Get stats
            stats = scorer.get_stats()
            logger.info("scorer_stats", **stats)

            # Get all health
            all_health = scorer.get_all_health()
            for label, health in all_health.items():
                logger.info("endpoint_health", **health.to_dict())

        finally:
            await rpc_manager.stop()

    asyncio.run(main())
