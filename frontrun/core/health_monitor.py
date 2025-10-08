"""
Health Monitor for Frontrun Bot
Monitors system health, RPC availability, and slot lag detection
"""

import asyncio
import psutil
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class HealthConfig:
    """Health monitoring configuration"""
    check_interval_s: int = 10
    rpc_timeout_ms: int = 5000
    max_slot_lag: int = 50
    cpu_threshold_pct: float = 80.0
    memory_threshold_pct: float = 80.0


@dataclass
class RPCHealth:
    """Health status for a single RPC endpoint"""
    endpoint: str
    is_healthy: bool
    latency_ms: float
    slot: int
    slot_lag: int
    last_check: datetime
    error: Optional[str] = None


@dataclass
class ResourceUsage:
    """System resource usage statistics"""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    thread_count: int


@dataclass
class SystemHealth:
    """Overall system health status"""
    is_healthy: bool
    issues: List[str]
    rpcs_healthy: int
    rpcs_total: int
    cpu_usage_pct: float
    memory_usage_pct: float
    uptime_seconds: int
    last_check: datetime


class HealthMonitor:
    """
    Monitors system health and RPC availability

    Features:
    - RPC health checks via getHealth/getSlot methods
    - Slot lag detection across endpoints
    - System resource monitoring (CPU, memory)
    - Automatic health status aggregation
    - Continuous health check loop

    Usage:
        monitor = HealthMonitor(rpc_manager, config)
        await monitor.start_monitoring()
        health = await monitor.get_system_health()
        await monitor.stop_monitoring()
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        config: Optional[HealthConfig] = None
    ):
        """
        Initialize health monitor

        Args:
            rpc_manager: RPC manager instance
            config: Health monitoring configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or HealthConfig()

        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = time.time()

        # Cache latest health data
        self._rpc_health_cache: Dict[str, RPCHealth] = {}
        self._system_health_cache: Optional[SystemHealth] = None
        self._resource_usage_cache: Optional[ResourceUsage] = None

        logger.info(
            "health_monitor_initialized",
            check_interval_s=self.config.check_interval_s,
            max_slot_lag=self.config.max_slot_lag
        )

    async def start_monitoring(self) -> None:
        """Start health monitoring loop"""
        if self._running:
            logger.warning("health_monitor_already_running")
            return

        self._running = True
        self._start_time = time.time()

        logger.info("health_monitor_starting")

        # Run initial health check
        await self._perform_health_checks()

        # Start continuous monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("health_monitor_started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring loop"""
        if not self._running:
            return

        logger.info("health_monitor_stopping")
        self._running = False

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("health_monitor_stopped")

    async def get_system_health(self) -> SystemHealth:
        """
        Get overall system health status

        Returns:
            SystemHealth with aggregated status
        """
        if self._system_health_cache is None:
            # No cached data - run health check now
            await self._perform_health_checks()

        return self._system_health_cache

    async def check_rpc_health(self) -> Dict[str, RPCHealth]:
        """
        Check health of all RPC endpoints

        Returns:
            Dictionary of endpoint label -> RPCHealth
        """
        rpc_health = {}

        # Get highest slot seen across all RPCs
        max_slot = 0

        # First pass: collect all slots
        temp_results = {}

        for label, conn in self.rpc_manager.connections.items():
            try:
                start_time = time.perf_counter()

                # Try to get slot via RPC call
                try:
                    result = await asyncio.wait_for(
                        self.rpc_manager.call_rpc("getSlot", []),
                        timeout=self.config.rpc_timeout_ms / 1000
                    )
                    slot = result if isinstance(result, int) else result.get("result", 0)
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    error = None

                    max_slot = max(max_slot, slot)

                except Exception as e:
                    slot = 0
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    error = str(e)

                temp_results[label] = {
                    "slot": slot,
                    "latency_ms": latency_ms,
                    "error": error
                }

            except Exception as e:
                logger.warning(
                    "rpc_health_check_failed",
                    endpoint=label,
                    error=str(e)
                )
                temp_results[label] = {
                    "slot": 0,
                    "latency_ms": 0,
                    "error": str(e)
                }

        # Second pass: calculate lag and health
        for label, result in temp_results.items():
            slot = result["slot"]
            latency_ms = result["latency_ms"]
            error = result["error"]

            slot_lag = max_slot - slot if max_slot > 0 else 0

            # Determine if healthy
            is_healthy = (
                error is None and
                slot > 0 and
                slot_lag <= self.config.max_slot_lag and
                latency_ms < self.config.rpc_timeout_ms
            )

            rpc_health[label] = RPCHealth(
                endpoint=label,
                is_healthy=is_healthy,
                latency_ms=latency_ms,
                slot=slot,
                slot_lag=slot_lag,
                last_check=datetime.now(timezone.utc),
                error=error
            )

            # Update metrics
            metrics.set_gauge(
                "rpc_slot",
                float(slot),
                labels={"endpoint": label}
            )
            metrics.set_gauge(
                "rpc_slot_lag",
                float(slot_lag),
                labels={"endpoint": label}
            )
            metrics.set_gauge(
                "rpc_health_check_latency_ms",
                latency_ms,
                labels={"endpoint": label}
            )

        # Cache results
        self._rpc_health_cache = rpc_health

        return rpc_health

    def get_resource_usage(self) -> ResourceUsage:
        """
        Get current system resource usage

        Returns:
            ResourceUsage with CPU, memory, and thread stats
        """
        try:
            process = psutil.Process()

            # Get CPU percentage (non-blocking)
            cpu_percent = process.cpu_percent(interval=0.1)

            # Get memory info
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB

            # Get system memory percentage
            system_memory = psutil.virtual_memory()
            memory_percent = system_memory.percent

            # Get thread count
            thread_count = process.num_threads()

            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                thread_count=thread_count
            )

            # Update metrics
            metrics.set_gauge("system_cpu_percent", cpu_percent)
            metrics.set_gauge("system_memory_percent", memory_percent)
            metrics.set_gauge("system_memory_mb", memory_mb)
            metrics.set_gauge("system_thread_count", float(thread_count))

            # Cache result
            self._resource_usage_cache = usage

            return usage

        except Exception as e:
            logger.error("resource_usage_check_failed", error=str(e))
            return ResourceUsage(
                cpu_percent=0,
                memory_percent=0,
                memory_mb=0,
                thread_count=0
            )

    async def _perform_health_checks(self) -> None:
        """Perform all health checks and update system health"""
        try:
            # Check RPC health
            rpc_health = await self.check_rpc_health()

            # Check resource usage
            resource_usage = self.get_resource_usage()

            # Aggregate system health
            issues = []

            # Count healthy RPCs
            healthy_rpcs = sum(1 for h in rpc_health.values() if h.is_healthy)
            total_rpcs = len(rpc_health)

            if healthy_rpcs == 0:
                issues.append("No healthy RPC endpoints available")
            elif healthy_rpcs < total_rpcs:
                issues.append(f"Only {healthy_rpcs}/{total_rpcs} RPC endpoints healthy")

            # Check for slot lag
            lagging_rpcs = [
                label for label, h in rpc_health.items()
                if h.slot_lag > self.config.max_slot_lag
            ]
            if lagging_rpcs:
                issues.append(f"RPCs lagging: {', '.join(lagging_rpcs)}")

            # Check resource usage
            if resource_usage.cpu_percent > self.config.cpu_threshold_pct:
                issues.append(
                    f"High CPU usage: {resource_usage.cpu_percent:.1f}% "
                    f"(threshold: {self.config.cpu_threshold_pct}%)"
                )

            if resource_usage.memory_percent > self.config.memory_threshold_pct:
                issues.append(
                    f"High memory usage: {resource_usage.memory_percent:.1f}% "
                    f"(threshold: {self.config.memory_threshold_pct}%)"
                )

            # Calculate uptime
            uptime_seconds = int(time.time() - self._start_time)

            # Determine overall health
            is_healthy = (
                healthy_rpcs > 0 and
                resource_usage.cpu_percent < self.config.cpu_threshold_pct and
                resource_usage.memory_percent < self.config.memory_threshold_pct
            )

            # Create system health object
            self._system_health_cache = SystemHealth(
                is_healthy=is_healthy,
                issues=issues,
                rpcs_healthy=healthy_rpcs,
                rpcs_total=total_rpcs,
                cpu_usage_pct=resource_usage.cpu_percent,
                memory_usage_pct=resource_usage.memory_percent,
                uptime_seconds=uptime_seconds,
                last_check=datetime.now(timezone.utc)
            )

            # Update metrics
            metrics.set_gauge("system_healthy", 1.0 if is_healthy else 0.0)
            metrics.set_gauge("system_issues_count", float(len(issues)))
            metrics.set_gauge("system_uptime_seconds", float(uptime_seconds))

            # Log health status
            if is_healthy:
                logger.debug(
                    "health_check_passed",
                    healthy_rpcs=healthy_rpcs,
                    total_rpcs=total_rpcs,
                    cpu_percent=resource_usage.cpu_percent,
                    memory_percent=resource_usage.memory_percent
                )
            else:
                logger.warning(
                    "health_check_failed",
                    issues=issues,
                    healthy_rpcs=healthy_rpcs,
                    total_rpcs=total_rpcs
                )

        except Exception as e:
            logger.error("health_check_error", error=str(e), exc_info=True)

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_s)
                await self._perform_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e), exc_info=True)


# Example usage
if __name__ == "__main__":
    from core.config import ConfigurationManager
    from core.logger import setup_logging

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Load config
        config_manager = ConfigurationManager("config/config.yml")
        bot_config = config_manager.load_config()

        # Initialize RPC manager
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        # Initialize health monitor
        health_config = HealthConfig(
            check_interval_s=5,
            max_slot_lag=50
        )
        monitor = HealthMonitor(rpc_manager, health_config)

        try:
            # Start monitoring
            await monitor.start_monitoring()

            logger.info("health_monitor_test_started")

            # Monitor for 30 seconds
            for i in range(6):
                await asyncio.sleep(5)

                # Get system health
                system_health = await monitor.get_system_health()

                logger.info(
                    "system_health_status",
                    is_healthy=system_health.is_healthy,
                    healthy_rpcs=f"{system_health.rpcs_healthy}/{system_health.rpcs_total}",
                    cpu_usage=f"{system_health.cpu_usage_pct:.1f}%",
                    memory_usage=f"{system_health.memory_usage_pct:.1f}%",
                    uptime=f"{system_health.uptime_seconds}s",
                    issues=system_health.issues
                )

                # Get RPC health details
                rpc_health = await monitor.check_rpc_health()
                for label, health in rpc_health.items():
                    logger.info(
                        "rpc_health_detail",
                        endpoint=label,
                        healthy=health.is_healthy,
                        slot=health.slot,
                        lag=health.slot_lag,
                        latency_ms=f"{health.latency_ms:.2f}ms"
                    )

            logger.info("health_monitor_test_completed")

        except KeyboardInterrupt:
            logger.info("test_interrupted_by_user")
        finally:
            await monitor.stop_monitoring()
            await rpc_manager.stop()

    asyncio.run(main())
