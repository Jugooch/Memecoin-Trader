"""
Priority Fee Calculator for Frontrun Bot
Calculates optimal compute unit prices for transaction inclusion
"""

import time
import statistics
import aiohttp
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


class FeeUrgency(Enum):
    """Transaction urgency level"""
    LOW = "low"  # p50 fees (normal inclusion)
    NORMAL = "normal"  # p75 fees (faster inclusion)
    HIGH = "high"  # p90 fees (priority inclusion)
    CRITICAL = "critical"  # p99 fees (maximum priority)


@dataclass
class FeeConfig:
    """Configuration for fee calculator"""
    update_interval_s: int = 10
    lookback_slots: int = 150
    min_priority_fee: int = 1_000  # micro-lamports
    max_priority_fee: int = 1_000_000  # micro-lamports (0.001 SOL)
    cache_ttl_seconds: int = 10


@dataclass
class PriorityFeeStats:
    """Priority fee statistics"""
    slot: int
    prioritization_fee: int  # micro-lamports
    timestamp: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "slot": self.slot,
            "prioritization_fee": self.prioritization_fee,
            "timestamp": self.timestamp
        }


@dataclass
class FeeEstimate:
    """Fee estimate with percentiles"""
    p50: int  # median
    p75: int
    p90: int
    p95: int
    p99: int
    mean: int
    min: int
    max: int
    sample_count: int
    cached_at: float

    def get_fee_for_urgency(self, urgency: FeeUrgency) -> int:
        """Get fee for specified urgency level"""
        if urgency == FeeUrgency.LOW:
            return self.p50
        elif urgency == FeeUrgency.NORMAL:
            return self.p75
        elif urgency == FeeUrgency.HIGH:
            return self.p90
        elif urgency == FeeUrgency.CRITICAL:
            return self.p99
        return self.p75  # Default to normal

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if estimate has expired"""
        return time.time() - self.cached_at > ttl_seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "sample_count": self.sample_count,
            "cached_at": self.cached_at
        }


class PriorityFeeCalculator:
    """
    Calculates optimal priority fees based on network conditions

    Features:
    - Fetches recent prioritization fees from RPC
    - Calculates percentiles (p50, p75, p90, p95, p99)
    - Maps urgency levels to fee amounts
    - Caches fee data to reduce RPC calls
    - Clamps fees between min/max bounds

    Usage:
        calculator = PriorityFeeCalculator(rpc_manager)
        fee = await calculator.calculate_priority_fee(
            compute_units=200_000,
            urgency=FeeUrgency.HIGH
        )
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        config: Optional[FeeConfig] = None
    ):
        """
        Initialize priority fee calculator

        Args:
            rpc_manager: RPC manager for network queries
            config: Fee calculator configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or FeeConfig()
        self._cached_estimate: Optional[FeeEstimate] = None

        logger.info(
            "priority_fee_calculator_initialized",
            min_fee=self.config.min_priority_fee,
            max_fee=self.config.max_priority_fee,
            update_interval_s=self.config.update_interval_s
        )

    async def calculate_priority_fee(
        self,
        compute_units: int = 200_000,
        urgency: FeeUrgency = FeeUrgency.NORMAL
    ) -> int:
        """
        Calculate priority fee (compute unit price) for transaction

        Args:
            compute_units: Compute units for transaction (used for logging)
            urgency: Transaction urgency level

        Returns:
            Compute unit price in micro-lamports

        Example:
            # Normal priority
            fee = await calculator.calculate_priority_fee(
                compute_units=200_000,
                urgency=FeeUrgency.NORMAL
            )
            # Returns ~p75 fee (e.g., 50,000 micro-lamports)
        """
        # Get fee estimate (from cache or fetch new)
        estimate = await self.get_fee_estimate()

        # Get fee for urgency level
        fee = estimate.get_fee_for_urgency(urgency)

        # Clamp between min and max
        fee = max(self.config.min_priority_fee, min(fee, self.config.max_priority_fee))

        # Calculate total cost for logging
        total_lamports = (fee * compute_units) // 1_000_000  # Convert to lamports

        logger.debug(
            "priority_fee_calculated",
            urgency=urgency.value,
            fee_micro_lamports=fee,
            compute_units=compute_units,
            total_cost_lamports=total_lamports
        )

        metrics.increment_counter(
            "priority_fees_calculated",
            labels={"urgency": urgency.value}
        )

        return fee

    async def get_fee_estimate(self) -> FeeEstimate:
        """
        Get current fee estimate (cached or fresh)

        Returns:
            FeeEstimate with percentiles

        Raises:
            Exception: If unable to fetch fees
        """
        # Return cached estimate if valid
        if self._cached_estimate and not self._cached_estimate.is_expired(
            self.config.cache_ttl_seconds
        ):
            metrics.increment_counter("priority_fee_cache_hit")
            return self._cached_estimate

        # Fetch fresh estimate
        metrics.increment_counter("priority_fee_cache_miss")
        estimate = await self._fetch_fee_estimate()
        self._cached_estimate = estimate

        return estimate

    async def _fetch_fee_estimate(self) -> FeeEstimate:
        """
        Fetch fresh fee estimate from RPC via getRecentPrioritizationFees

        Returns:
            FeeEstimate calculated from recent fees

        Raises:
            Returns default estimate on error
        """
        try:
            # Get healthy RPC connection
            connection = await self.rpc_manager.get_healthy_connection()
            if connection is None:
                logger.warning("no_healthy_rpc_for_fees")
                return self._get_default_estimate()

            # Build RPC request
            # getRecentPrioritizationFees returns fees for recent slots
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentPrioritizationFees",
                "params": []  # Can optionally filter by accounts
            }

            # Send HTTP POST request
            timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(connection.endpoint.url, json=payload) as response:
                    result = await response.json()

                    # Check for RPC errors
                    if "error" in result:
                        logger.error(
                            "fee_estimate_rpc_error",
                            error=result["error"]
                        )
                        return self._get_default_estimate()

                    # Parse fee data
                    fee_data = result.get("result", [])
                    if not fee_data:
                        logger.warning("no_fee_data_from_rpc")
                        return self._get_default_estimate()

                    # Extract prioritization fees
                    fees = [
                        item["prioritizationFee"]
                        for item in fee_data
                        if "prioritizationFee" in item
                    ]

                    if not fees:
                        logger.warning("no_fees_extracted")
                        return self._get_default_estimate()

                    # Calculate percentiles
                    estimate = self._calculate_percentiles(fees)

                    logger.info(
                        "fee_estimate_fetched",
                        p50=estimate.p50,
                        p75=estimate.p75,
                        p90=estimate.p90,
                        p99=estimate.p99,
                        sample_count=estimate.sample_count,
                        rpc=connection.endpoint.label
                    )

                    return estimate

        except Exception as e:
            logger.error(
                "fee_estimate_fetch_failed",
                error=str(e)
            )

            # Return safe default on error
            return self._get_default_estimate()

    def _generate_sample_fees(self) -> List[int]:
        """
        Generate sample fee data for testing

        Returns:
            List of sample fees

        Note:
            Production would fetch real data from RPC
        """
        # Generate realistic fee distribution
        # Most fees are low, some are high (typical network pattern)
        import random

        fees = []

        # 80% low fees (1k-10k)
        fees.extend([random.randint(1_000, 10_000) for _ in range(80)])

        # 15% medium fees (10k-100k)
        fees.extend([random.randint(10_000, 100_000) for _ in range(15)])

        # 5% high fees (100k-500k)
        fees.extend([random.randint(100_000, 500_000) for _ in range(5)])

        return fees

    def _calculate_percentiles(self, fees: List[int]) -> FeeEstimate:
        """
        Calculate fee percentiles from sample

        Args:
            fees: List of recent priority fees

        Returns:
            FeeEstimate with percentiles
        """
        if not fees:
            return self._get_default_estimate()

        # Sort fees
        sorted_fees = sorted(fees)

        # Calculate percentiles
        def percentile(data: List[int], p: float) -> int:
            """Calculate percentile"""
            if not data:
                return 0
            k = (len(data) - 1) * (p / 100)
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            d0 = data[f] * (c - k)
            d1 = data[c] * (k - f)
            return int(d0 + d1)

        return FeeEstimate(
            p50=percentile(sorted_fees, 50),
            p75=percentile(sorted_fees, 75),
            p90=percentile(sorted_fees, 90),
            p95=percentile(sorted_fees, 95),
            p99=percentile(sorted_fees, 99),
            mean=int(statistics.mean(fees)),
            min=min(fees),
            max=max(fees),
            sample_count=len(fees),
            cached_at=time.time()
        )

    def _get_default_estimate(self) -> FeeEstimate:
        """
        Get default/fallback fee estimate

        Returns:
            Conservative fee estimate
        """
        return FeeEstimate(
            p50=5_000,
            p75=10_000,
            p90=50_000,
            p95=100_000,
            p99=200_000,
            mean=25_000,
            min=1_000,
            max=500_000,
            sample_count=0,
            cached_at=time.time()
        )

    def invalidate_cache(self) -> None:
        """Invalidate cached fee estimate"""
        self._cached_estimate = None
        logger.debug("priority_fee_cache_invalidated")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get calculator statistics

        Returns:
            Dictionary with calculator stats
        """
        cache_status = "empty"
        if self._cached_estimate:
            if self._cached_estimate.is_expired(self.config.cache_ttl_seconds):
                cache_status = "expired"
            else:
                cache_status = "valid"

        return {
            "cache_status": cache_status,
            "cache_ttl_seconds": self.config.cache_ttl_seconds,
            "min_priority_fee": self.config.min_priority_fee,
            "max_priority_fee": self.config.max_priority_fee,
            "cached_estimate": self._cached_estimate.to_dict() if self._cached_estimate else None
        }
