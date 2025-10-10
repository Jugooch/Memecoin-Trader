"""
Hard Latency Budget Enforcer for Frontrun Bot
Enforces strict latency budgets with ABORT semantics to prevent fee burn
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


class TradeStage(Enum):
    """Trading pipeline stages"""
    DETECT = "detect"  # Event detection
    BUILD_SIGN = "build_sign"  # Transaction building + signing
    SUBMIT = "submit"  # RPC submission
    TOTAL = "total"  # End-to-end


@dataclass
class LatencyConfig:
    """
    Latency budget configuration

    Hard budget limits (ABORT if exceeded):
    - detect_stage_budget_ms: Event detection (default: 30ms)
    - build_sign_budget_ms: Build + sign combined (default: 15ms)
    - submit_stage_budget_ms: RPC submission (default: 50ms)
    - total_budget_ms: End-to-end hard limit (default: 100ms)

    Circuit breaker settings:
    - circuit_breaker_enabled: Enable circuit breaker (default: True)
    - violations_before_trip: Trip after N consecutive violations (default: 5)
    - circuit_reset_after_s: Reset circuit after N seconds (default: 60)
    """
    # Hard budget limits (ABORT if exceeded)
    detect_stage_budget_ms: float = 30.0
    build_sign_budget_ms: float = 15.0
    submit_stage_budget_ms: float = 50.0
    total_budget_ms: float = 100.0

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    violations_before_trip: int = 5
    circuit_reset_after_s: int = 60


@dataclass
class OperationTimer:
    """
    Timer for tracking operation stages

    Attributes:
        operation_id: Unique identifier for this operation
        start_time: Start time (perf_counter in seconds)
        stage_times: Stage completion times (stage_name -> elapsed_ms)
        total_elapsed_ms: Total time elapsed
    """
    operation_id: str
    start_time: float  # perf_counter() in seconds
    stage_times: Dict[str, float] = field(default_factory=dict)

    def elapsed_ms(self) -> float:
        """Get total elapsed time in milliseconds"""
        return (time.perf_counter() - self.start_time) * 1000.0

    def stage_elapsed_ms(self, stage: str) -> float:
        """Get elapsed time for a specific stage"""
        return self.stage_times.get(stage, 0.0)


@dataclass
class BudgetAttribution:
    """
    Detailed attribution for post-mortem analysis

    Records:
    - Which stage exceeded budget
    - How much it exceeded by
    - Context (slot, leader, RPC endpoint)
    """
    operation_id: str
    stage: str
    elapsed_ms: float
    budget_ms: float
    exceeded: bool
    slot: Optional[int] = None
    leader: Optional[str] = None
    rpc_endpoint: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "operation_id": self.operation_id,
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "exceeded": self.exceeded,
            "slot": self.slot,
            "leader": self.leader,
            "rpc_endpoint": self.rpc_endpoint,
            "timestamp": self.timestamp
        }


class LatencyBudgetEnforcer:
    """
    Enforces hard latency budgets with ABORT semantics

    Features:
    - Per-stage hard budgets with abort on violation
    - Circuit breaker on repeated violations
    - Detailed attribution tracking
    - Stop-fast failure mode (late trades killed, not submitted)

    Usage:
        enforcer = LatencyBudgetEnforcer(config)

        # Start operation
        timer = enforcer.start_operation("trade_abc123")

        # Check after each stage
        if enforcer.check_and_abort(timer, TradeStage.DETECT):
            logger.info("Aborted - detection too slow")
            return  # STOP - don't continue

        # ... build and sign ...

        if enforcer.check_and_abort(timer, TradeStage.BUILD_SIGN):
            logger.info("Aborted - build/sign too slow")
            return  # STOP - don't submit

        # Submit only if within budget
        await submit_transaction()
    """

    def __init__(self, config: Optional[LatencyConfig] = None):
        """
        Initialize latency budget enforcer

        Args:
            config: Latency budget configuration (optional)
        """
        self.config = config or LatencyConfig()

        # Circuit breaker state
        self.consecutive_violations = 0
        self.circuit_open_until: Optional[float] = None

        # Attribution tracking (last 100 operations)
        self._attribution_buffer: list[BudgetAttribution] = []
        self._max_attribution_buffer = 100

        logger.info(
            "latency_budget_enforcer_initialized",
            detect_budget_ms=self.config.detect_stage_budget_ms,
            build_sign_budget_ms=self.config.build_sign_budget_ms,
            submit_budget_ms=self.config.submit_stage_budget_ms,
            total_budget_ms=self.config.total_budget_ms,
            circuit_breaker_enabled=self.config.circuit_breaker_enabled
        )

    def start_operation(self, operation_id: str) -> OperationTimer:
        """
        Start timing an operation with microsecond precision

        Args:
            operation_id: Unique identifier for this operation

        Returns:
            OperationTimer for tracking stages

        Example:
            timer = enforcer.start_operation("trade_abc123")
        """
        timer = OperationTimer(
            operation_id=operation_id,
            start_time=time.perf_counter()
        )

        logger.debug("operation_timer_started", operation_id=operation_id)

        return timer

    def check_and_abort(
        self,
        timer: OperationTimer,
        stage: TradeStage,
        slot: Optional[int] = None,
        leader: Optional[str] = None,
        rpc_endpoint: Optional[str] = None
    ) -> bool:
        """
        Check if stage exceeded budget. Returns True if should ABORT.

        Args:
            timer: Operation timer
            stage: Current stage
            slot: Current slot (for attribution)
            leader: Current leader (for attribution)
            rpc_endpoint: RPC endpoint used (for attribution)

        Returns:
            True if should ABORT (budget exceeded), False if OK to continue

        Example:
            if enforcer.check_and_abort(timer, TradeStage.DETECT):
                logger.info("Aborted - detection too slow")
                return  # STOP - don't continue
        """
        # Calculate elapsed time
        elapsed = timer.elapsed_ms()

        # Get budget for this stage
        budget = self._get_budget_for_stage(stage)

        # Check if exceeded
        exceeded = elapsed > budget

        # Record stage time
        timer.stage_times[stage.value] = elapsed

        # Create attribution record
        attribution = BudgetAttribution(
            operation_id=timer.operation_id,
            stage=stage.value,
            elapsed_ms=elapsed,
            budget_ms=budget,
            exceeded=exceeded,
            slot=slot,
            leader=leader,
            rpc_endpoint=rpc_endpoint
        )

        # Store attribution
        self._record_attribution(attribution)

        if exceeded:
            # Log detailed attribution
            logger.warning(
                "latency_budget_exceeded",
                operation_id=timer.operation_id,
                stage=stage.value,
                elapsed_ms=elapsed,
                budget_ms=budget,
                overage_ms=elapsed - budget,
                slot=slot,
                leader=leader,
                rpc_endpoint=rpc_endpoint
            )

            # Increment metrics
            metrics.increment_counter(
                "latency_budget_violations",
                labels={"stage": stage.value}
            )

            # Automatically track violation for circuit breaker
            self.consecutive_violations += 1
            logger.debug(
                "violation_recorded",
                consecutive_violations=self.consecutive_violations
            )

            # Check if circuit should trip
            if (self.config.circuit_breaker_enabled and
                self.consecutive_violations >= self.config.violations_before_trip):
                self._trip_circuit_breaker()

            return True  # ABORT

        # Budget OK - don't reset violations here
        # Violations only reset when complete operation succeeds
        logger.debug(
            "latency_budget_ok",
            operation_id=timer.operation_id,
            stage=stage.value,
            elapsed_ms=elapsed,
            budget_ms=budget,
            remaining_ms=budget - elapsed
        )

        return False  # Continue

    def record_stage(
        self,
        timer: OperationTimer,
        stage: TradeStage
    ):
        """
        Record stage completion time for attribution tracking

        Args:
            timer: Operation timer
            stage: Completed stage

        Example:
            enforcer.record_stage(timer, TradeStage.DETECT)
        """
        elapsed = timer.elapsed_ms()
        timer.stage_times[stage.value] = elapsed

        logger.debug(
            "stage_completed",
            operation_id=timer.operation_id,
            stage=stage.value,
            elapsed_ms=elapsed
        )

    def record_operation_success(self):
        """
        Record successful operation completion - resets violation counter

        This should be called by the orchestrator when a COMPLETE operation
        succeeds (all stages passed within budget). This is the ONLY place
        where consecutive_violations gets reset to prevent false resets.

        Example:
            # In orchestrator after successful frontrun
            enforcer.record_operation_success()
        """
        if self.consecutive_violations > 0:
            logger.debug(
                "operation_success_reset_violations",
                previous_violations=self.consecutive_violations
            )
            self.consecutive_violations = 0

    def record_violation(self):
        """
        Record operation failure due to latency violation

        This is called when an operation is aborted due to latency.
        Increments consecutive violations for circuit breaker tracking.

        Example:
            # In orchestrator when operation aborted
            enforcer.record_violation()
        """
        self.consecutive_violations += 1
        logger.debug(
            "violation_recorded",
            consecutive_violations=self.consecutive_violations
        )

        # Check if circuit should trip
        if (self.config.circuit_breaker_enabled and
            self.consecutive_violations >= self.config.violations_before_trip):
            self._trip_circuit_breaker()

    def _get_budget_for_stage(self, stage: TradeStage) -> float:
        """Get budget in milliseconds for specified stage"""
        if stage == TradeStage.DETECT:
            return self.config.detect_stage_budget_ms
        elif stage == TradeStage.BUILD_SIGN:
            return self.config.build_sign_budget_ms
        elif stage == TradeStage.SUBMIT:
            return self.config.submit_stage_budget_ms
        elif stage == TradeStage.TOTAL:
            return self.config.total_budget_ms
        return 0.0

    def _trip_circuit_breaker(self):
        """Pause trading temporarily on repeated budget violations"""
        self.circuit_open_until = time.time() + self.config.circuit_reset_after_s

        logger.error(
            "circuit_breaker_tripped",
            consecutive_violations=self.consecutive_violations,
            reset_after_s=self.config.circuit_reset_after_s
        )

        metrics.increment_counter("circuit_breaker_trips")

    def is_circuit_open(self) -> bool:
        """
        Check if circuit breaker is currently open (trading paused)

        Returns:
            True if circuit is open (trading paused), False otherwise

        Example:
            if enforcer.is_circuit_open():
                logger.info("Circuit breaker open - skipping trade")
                return
        """
        if self.circuit_open_until is None:
            return False

        if time.time() > self.circuit_open_until:
            # Circuit has timed out, reset
            logger.info("circuit_breaker_reset")
            self.circuit_open_until = None
            self.consecutive_violations = 0
            metrics.increment_counter("circuit_breaker_resets")
            return False

        return True  # Still open

    def _record_attribution(self, attribution: BudgetAttribution):
        """Record attribution for post-mortem analysis"""
        self._attribution_buffer.append(attribution)

        # Keep buffer size limited
        if len(self._attribution_buffer) > self._max_attribution_buffer:
            self._attribution_buffer.pop(0)

    def get_attribution_stats(self) -> Dict:
        """
        Get attribution statistics

        Returns:
            Dictionary with attribution stats

        Example:
            stats = enforcer.get_attribution_stats()
            print(f"Total violations: {stats['total_violations']}")
            print(f"Most violated stage: {stats['most_violated_stage']}")
        """
        if not self._attribution_buffer:
            return {
                "total_operations": 0,
                "total_violations": 0,
                "most_violated_stage": None,
                "avg_overage_ms": 0.0
            }

        total_ops = len(self._attribution_buffer)
        violations = [a for a in self._attribution_buffer if a.exceeded]
        total_violations = len(violations)

        # Count violations by stage
        stage_violations: Dict[str, int] = {}
        total_overage = 0.0

        for attribution in violations:
            stage = attribution.stage
            stage_violations[stage] = stage_violations.get(stage, 0) + 1
            total_overage += (attribution.elapsed_ms - attribution.budget_ms)

        # Find most violated stage
        most_violated_stage = None
        max_violations = 0
        for stage, count in stage_violations.items():
            if count > max_violations:
                max_violations = count
                most_violated_stage = stage

        avg_overage = total_overage / total_violations if total_violations > 0 else 0.0

        return {
            "total_operations": total_ops,
            "total_violations": total_violations,
            "violation_rate": total_violations / total_ops if total_ops > 0 else 0.0,
            "most_violated_stage": most_violated_stage,
            "stage_violations": stage_violations,
            "avg_overage_ms": avg_overage
        }

    def get_recent_attributions(self, limit: int = 10) -> list[BudgetAttribution]:
        """
        Get recent attribution records

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent attribution records (most recent first)

        Example:
            recent = enforcer.get_recent_attributions(limit=5)
            for attr in recent:
                print(f"{attr.operation_id}: {attr.stage} - {attr.elapsed_ms}ms")
        """
        return list(reversed(self._attribution_buffer[-limit:]))


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging

    # Setup logging
    setup_logging(level="INFO", format="console")

    # Create enforcer with custom config
    config = LatencyConfig(
        detect_stage_budget_ms=30.0,
        build_sign_budget_ms=15.0,
        submit_stage_budget_ms=50.0,
        total_budget_ms=100.0,
        circuit_breaker_enabled=True,
        violations_before_trip=3
    )

    enforcer = LatencyBudgetEnforcer(config)

    # Simulate trade operation
    timer = enforcer.start_operation("trade_example")

    # Simulate detection stage (within budget)
    time.sleep(0.02)  # 20ms
    if enforcer.check_and_abort(timer, TradeStage.DETECT):
        logger.info("Aborted at detect stage")
    else:
        logger.info("Detect stage OK")

    # Simulate build/sign stage (within budget)
    time.sleep(0.01)  # 10ms
    if enforcer.check_and_abort(timer, TradeStage.BUILD_SIGN):
        logger.info("Aborted at build/sign stage")
    else:
        logger.info("Build/sign stage OK")

    # Simulate submit stage (exceeds budget)
    time.sleep(0.08)  # 80ms (total now >100ms)
    if enforcer.check_and_abort(timer, TradeStage.TOTAL):
        logger.info("Aborted - total time exceeded")
    else:
        logger.info("Submit stage OK")

    # Get attribution stats
    stats = enforcer.get_attribution_stats()
    logger.info("attribution_stats", **stats)
