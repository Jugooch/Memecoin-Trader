"""
Race Failure Detector for Frontrun Bot
Identifies when bot lost frontrun race to dev buy
"""

from typing import Optional, Dict
from dataclasses import dataclass

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class RaceConfig:
    """
    Configuration for race detection

    Parameters:
    - price_deviation_threshold_pct: Max acceptable price deviation (default: 5%)
    - max_acceptable_slot_delay: Max slots we can be behind dev (default: 2)
    """
    price_deviation_threshold_pct: float = 5.0
    max_acceptable_slot_delay: int = 2


@dataclass
class RaceResult:
    """
    Result of race analysis

    Attributes:
        won_race: Whether we won the race
        reason: Human-readable reason for outcome
        price_deviation_pct: Price deviation percentage
        slot_delay: How many slots behind dev buy
    """
    won_race: bool
    reason: str
    price_deviation_pct: float
    slot_delay: int

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "won_race": self.won_race,
            "reason": self.reason,
            "price_deviation_pct": self.price_deviation_pct,
            "slot_delay": self.slot_delay
        }


class RaceFailureDetector:
    """
    Detects when bot lost frontrun race

    Detection methods:
    1. Price deviation: If fill price >5% higher → lost race
    2. Slot timing: If our fill slot > dev buy slot + 2 → lost race

    Usage:
        detector = RaceFailureDetector(config)

        # Analyze by price
        result = detector.detect_race_loss_by_price(
            expected_price=0.000030,
            actual_fill_price=0.000035
        )

        # Analyze by timing
        result = detector.analyze_fill_timing(
            dev_buy_slot=12345,
            our_fill_slot=12348
        )
    """

    def __init__(self, config: Optional[RaceConfig] = None):
        """
        Initialize race detector

        Args:
            config: Race detection configuration (optional)
        """
        self.config = config or RaceConfig()

        logger.info(
            "race_detector_initialized",
            price_threshold_pct=self.config.price_deviation_threshold_pct,
            max_slot_delay=self.config.max_acceptable_slot_delay
        )

    def detect_race_loss_by_price(
        self,
        expected_price: float,
        actual_fill_price: float,
        threshold_pct: Optional[float] = None
    ) -> RaceResult:
        """
        Detect race loss based on price deviation

        Args:
            expected_price: Expected price before dev buy
            actual_fill_price: Actual fill price we got
            threshold_pct: Override default threshold (optional)

        Returns:
            RaceResult indicating win/loss

        Logic:
        - If actual > expected by >5% → Lost race (entered after dev)
        - Otherwise → Won race (entered before/with dev)

        Example:
            result = detector.detect_race_loss_by_price(
                expected_price=0.000030,
                actual_fill_price=0.000035
            )
            if not result.won_race:
                logger.warning("Lost race", reason=result.reason)
        """
        if expected_price <= 0:
            logger.warning("invalid_expected_price", price=expected_price)
            return RaceResult(
                won_race=False,
                reason="invalid_expected_price",
                price_deviation_pct=0.0,
                slot_delay=0
            )

        # Calculate deviation
        deviation = actual_fill_price - expected_price
        deviation_pct = (deviation / expected_price) * 100.0

        threshold = threshold_pct or self.config.price_deviation_threshold_pct

        # Determine outcome
        if deviation_pct > threshold:
            # Filled at much higher price → Lost race
            won_race = False
            reason = f"price_too_high (expected {expected_price:.8f}, got {actual_fill_price:.8f}, +{deviation_pct:.1f}%)"
        else:
            # Filled at expected price → Won race
            won_race = True
            reason = f"price_acceptable (deviation {deviation_pct:.1f}%)"

        result = RaceResult(
            won_race=won_race,
            reason=reason,
            price_deviation_pct=deviation_pct,
            slot_delay=0
        )

        logger.debug(
            "race_analyzed_by_price",
            won_race=won_race,
            expected_price=expected_price,
            actual_price=actual_fill_price,
            deviation_pct=deviation_pct
        )

        metrics.increment_counter(
            "race_outcomes",
            labels={
                "method": "price",
                "won_race": str(won_race)
            }
        )

        return result

    def analyze_fill_timing(
        self,
        dev_buy_slot: int,
        our_fill_slot: int
    ) -> RaceResult:
        """
        Analyze race outcome based on slot timing

        Args:
            dev_buy_slot: Slot when dev buy landed
            our_fill_slot: Slot when our buy landed

        Returns:
            RaceResult indicating win/loss

        Logic:
        - If our_fill_slot <= dev_buy_slot → Won race (same slot or before)
        - If our_fill_slot == dev_buy_slot + 1 → Won race (next slot, likely same batch)
        - If our_fill_slot > dev_buy_slot + 2 → Lost race (too late)

        Example:
            result = detector.analyze_fill_timing(
                dev_buy_slot=12345,
                our_fill_slot=12348
            )
        """
        slot_delay = our_fill_slot - dev_buy_slot

        # Determine outcome
        if slot_delay <= 0:
            # We filled before or same slot as dev → Won race
            won_race = True
            reason = f"filled_before_dev (our_slot={our_fill_slot}, dev_slot={dev_buy_slot})"
        elif slot_delay <= self.config.max_acceptable_slot_delay:
            # Within acceptable delay → Won race
            won_race = True
            reason = f"filled_close_to_dev (delay={slot_delay} slots)"
        else:
            # Too late → Lost race
            won_race = False
            reason = f"filled_too_late (delay={slot_delay} slots > threshold {self.config.max_acceptable_slot_delay})"

        result = RaceResult(
            won_race=won_race,
            reason=reason,
            price_deviation_pct=0.0,
            slot_delay=slot_delay
        )

        logger.debug(
            "race_analyzed_by_timing",
            won_race=won_race,
            dev_buy_slot=dev_buy_slot,
            our_fill_slot=our_fill_slot,
            slot_delay=slot_delay
        )

        metrics.increment_counter(
            "race_outcomes",
            labels={
                "method": "timing",
                "won_race": str(won_race)
            }
        )

        return result

    def analyze_combined(
        self,
        expected_price: float,
        actual_fill_price: float,
        dev_buy_slot: int,
        our_fill_slot: int
    ) -> RaceResult:
        """
        Analyze race using both price and timing

        Args:
            expected_price: Expected price before dev buy
            actual_fill_price: Actual fill price we got
            dev_buy_slot: Slot when dev buy landed
            our_fill_slot: Slot when our buy landed

        Returns:
            RaceResult (won if both methods agree we won)

        Logic:
        - Both methods must indicate win for overall win
        - If either indicates loss → Overall loss
        - Reason combines both analyses

        Example:
            result = detector.analyze_combined(
                expected_price=0.000030,
                actual_fill_price=0.000032,
                dev_buy_slot=12345,
                our_fill_slot=12346
            )
        """
        # Analyze by price
        price_result = self.detect_race_loss_by_price(expected_price, actual_fill_price)

        # Analyze by timing
        timing_result = self.analyze_fill_timing(dev_buy_slot, our_fill_slot)

        # Combine results
        won_race = price_result.won_race and timing_result.won_race

        # Combine reasons
        if won_race:
            reason = f"won_race (price: {price_result.reason}, timing: {timing_result.reason})"
        else:
            reasons = []
            if not price_result.won_race:
                reasons.append(f"price: {price_result.reason}")
            if not timing_result.won_race:
                reasons.append(f"timing: {timing_result.reason}")
            reason = f"lost_race ({', '.join(reasons)})"

        result = RaceResult(
            won_race=won_race,
            reason=reason,
            price_deviation_pct=price_result.price_deviation_pct,
            slot_delay=timing_result.slot_delay
        )

        logger.info(
            "race_analyzed_combined",
            won_race=won_race,
            reason=reason,
            price_deviation_pct=price_result.price_deviation_pct,
            slot_delay=timing_result.slot_delay
        )

        metrics.increment_counter(
            "race_outcomes",
            labels={
                "method": "combined",
                "won_race": str(won_race)
            }
        )

        return result


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging

    # Setup logging
    setup_logging(level="INFO", format="console")

    # Create detector
    config = RaceConfig(
        price_deviation_threshold_pct=5.0,
        max_acceptable_slot_delay=2
    )
    detector = RaceFailureDetector(config)

    # Test 1: Won race by price
    logger.info("=== Test 1: Won Race by Price ===")
    result = detector.detect_race_loss_by_price(
        expected_price=0.000030,
        actual_fill_price=0.000031  # Only 3.3% higher
    )
    logger.info("result", **result.to_dict())

    # Test 2: Lost race by price
    logger.info("\n=== Test 2: Lost Race by Price ===")
    result = detector.detect_race_loss_by_price(
        expected_price=0.000030,
        actual_fill_price=0.000035  # 16.7% higher
    )
    logger.info("result", **result.to_dict())

    # Test 3: Won race by timing
    logger.info("\n=== Test 3: Won Race by Timing ===")
    result = detector.analyze_fill_timing(
        dev_buy_slot=12345,
        our_fill_slot=12346  # 1 slot later
    )
    logger.info("result", **result.to_dict())

    # Test 4: Lost race by timing
    logger.info("\n=== Test 4: Lost Race by Timing ===")
    result = detector.analyze_fill_timing(
        dev_buy_slot=12345,
        our_fill_slot=12350  # 5 slots later
    )
    logger.info("result", **result.to_dict())

    # Test 5: Combined analysis - won both
    logger.info("\n=== Test 5: Combined - Won Both ===")
    result = detector.analyze_combined(
        expected_price=0.000030,
        actual_fill_price=0.000031,
        dev_buy_slot=12345,
        our_fill_slot=12346
    )
    logger.info("result", **result.to_dict())

    # Test 6: Combined analysis - lost on price
    logger.info("\n=== Test 6: Combined - Lost on Price ===")
    result = detector.analyze_combined(
        expected_price=0.000030,
        actual_fill_price=0.000040,
        dev_buy_slot=12345,
        our_fill_slot=12346
    )
    logger.info("result", **result.to_dict())
