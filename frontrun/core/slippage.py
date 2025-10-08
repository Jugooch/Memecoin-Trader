"""
Slippage Manager for Frontrun Bot
Enforces slippage tolerance and calculates minimum outputs for trades
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


class TradeUrgency(Enum):
    """Trade urgency level affecting slippage tolerance"""
    LOW = "low"  # 1% slippage - patient entries
    NORMAL = "normal"  # 5% slippage - standard trades
    HIGH = "high"  # 10% slippage - urgent entries
    CRITICAL = "critical"  # 20% slippage - emergency exits, frontrun competition


# Slippage tolerance in basis points (1 bps = 0.01%)
SLIPPAGE_TOLERANCE_BPS = {
    TradeUrgency.LOW: 100,  # 1%
    TradeUrgency.NORMAL: 500,  # 5%
    TradeUrgency.HIGH: 1000,  # 10%
    TradeUrgency.CRITICAL: 2000,  # 20%
}

BPS_DENOMINATOR = 10_000


@dataclass
class SlippageConfig:
    """Slippage configuration"""
    default_urgency: TradeUrgency = TradeUrgency.NORMAL
    custom_slippage_bps: Optional[int] = None  # Override default slippage


@dataclass
class SlippageCheck:
    """Result of slippage validation"""
    is_valid: bool
    expected_amount: int
    actual_amount: int
    slippage_pct: float
    tolerance_pct: float
    message: str


class SlippageManager:
    """
    Manages slippage tolerance for trades

    Features:
    - Maps urgency levels to slippage tolerance
    - Calculates minimum output amounts
    - Validates actual vs expected amounts
    - Conservative rounding (always favors user)

    Usage:
        manager = SlippageManager()
        min_tokens = manager.calculate_min_amount_out(
            expected_amount=1_000_000,
            urgency=TradeUrgency.NORMAL
        )
        # Returns 950_000 (5% slippage)

        # Later, validate actual fill
        check = manager.validate_slippage(
            expected_amount=1_000_000,
            actual_amount=980_000,
            urgency=TradeUrgency.NORMAL
        )
        assert check.is_valid  # True if within tolerance
    """

    def __init__(self, config: Optional[SlippageConfig] = None):
        """
        Initialize slippage manager

        Args:
            config: Slippage configuration (optional)
        """
        self.config = config or SlippageConfig()

        logger.info(
            "slippage_manager_initialized",
            default_urgency=self.config.default_urgency.value,
            custom_slippage_bps=self.config.custom_slippage_bps
        )

    def calculate_min_amount_out(
        self,
        expected_amount: int,
        urgency: Optional[TradeUrgency] = None,
        custom_slippage_bps: Optional[int] = None
    ) -> int:
        """
        Calculate minimum acceptable output amount

        Args:
            expected_amount: Expected output amount (tokens or lamports)
            urgency: Trade urgency level (optional, uses config default)
            custom_slippage_bps: Custom slippage in BPS (overrides urgency)

        Returns:
            Minimum amount out (rounded down conservatively)

        Example:
            # 5% slippage (normal urgency)
            min_out = manager.calculate_min_amount_out(1_000_000, TradeUrgency.NORMAL)
            # Returns 950_000
        """
        if expected_amount <= 0:
            raise ValueError("Expected amount must be positive")

        # Determine slippage tolerance (priority order):
        # 1. Function parameter (most specific)
        # 2. Config-level custom slippage
        # 3. Urgency parameter
        # 4. Config default urgency
        if custom_slippage_bps is not None:
            slippage_bps = custom_slippage_bps
        elif self.config.custom_slippage_bps is not None:
            slippage_bps = self.config.custom_slippage_bps
        elif urgency is not None:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[urgency]
        else:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[self.config.default_urgency]

        # Calculate min amount: expected * (1 - slippage)
        # Using integer math: expected * (10000 - slippage_bps) / 10000
        min_amount = (expected_amount * (BPS_DENOMINATOR - slippage_bps)) // BPS_DENOMINATOR

        slippage_pct = slippage_bps / BPS_DENOMINATOR * 100

        logger.debug(
            "min_amount_calculated",
            expected_amount=expected_amount,
            min_amount=min_amount,
            slippage_pct=slippage_pct,
            urgency=urgency.value if urgency else "default"
        )

        metrics.increment_counter(
            "slippage_calculations",
            labels={"urgency": urgency.value if urgency else "default"}
        )

        return min_amount

    def validate_slippage(
        self,
        expected_amount: int,
        actual_amount: int,
        urgency: Optional[TradeUrgency] = None,
        custom_slippage_bps: Optional[int] = None
    ) -> SlippageCheck:
        """
        Validate actual amount is within slippage tolerance

        Args:
            expected_amount: Expected output amount
            actual_amount: Actual received amount
            urgency: Trade urgency level (optional)
            custom_slippage_bps: Custom slippage tolerance (optional)

        Returns:
            SlippageCheck with validation result

        Example:
            check = manager.validate_slippage(
                expected_amount=1_000_000,
                actual_amount=980_000,
                urgency=TradeUrgency.NORMAL
            )
            print(check.is_valid)  # True
            print(check.slippage_pct)  # 2.0%
        """
        if expected_amount <= 0:
            return SlippageCheck(
                is_valid=False,
                expected_amount=expected_amount,
                actual_amount=actual_amount,
                slippage_pct=0.0,
                tolerance_pct=0.0,
                message="Expected amount must be positive"
            )

        # Determine slippage tolerance (priority order):
        # 1. Function parameter (most specific)
        # 2. Config-level custom slippage
        # 3. Urgency parameter
        # 4. Config default urgency
        if custom_slippage_bps is not None:
            slippage_bps = custom_slippage_bps
        elif self.config.custom_slippage_bps is not None:
            slippage_bps = self.config.custom_slippage_bps
        elif urgency is not None:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[urgency]
        else:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[self.config.default_urgency]

        # Calculate minimum acceptable amount
        min_amount = self.calculate_min_amount_out(expected_amount, urgency, custom_slippage_bps)

        # Calculate actual slippage
        if actual_amount >= expected_amount:
            # Positive slippage (got more than expected)
            actual_slippage_pct = 0.0
        else:
            # Negative slippage
            actual_slippage_pct = ((expected_amount - actual_amount) / expected_amount) * 100

        tolerance_pct = slippage_bps / BPS_DENOMINATOR * 100

        # Validate
        is_valid = actual_amount >= min_amount

        if is_valid:
            message = f"Slippage {actual_slippage_pct:.2f}% within tolerance {tolerance_pct:.2f}%"
        else:
            message = f"Slippage {actual_slippage_pct:.2f}% exceeds tolerance {tolerance_pct:.2f}%"

        logger.debug(
            "slippage_validated",
            expected=expected_amount,
            actual=actual_amount,
            slippage_pct=actual_slippage_pct,
            tolerance_pct=tolerance_pct,
            is_valid=is_valid
        )

        metrics.increment_counter(
            "slippage_validations",
            labels={"valid": str(is_valid), "urgency": urgency.value if urgency else "default"}
        )

        return SlippageCheck(
            is_valid=is_valid,
            expected_amount=expected_amount,
            actual_amount=actual_amount,
            slippage_pct=actual_slippage_pct,
            tolerance_pct=tolerance_pct,
            message=message
        )

    def get_slippage_tolerance_bps(
        self,
        urgency: Optional[TradeUrgency] = None
    ) -> int:
        """
        Get slippage tolerance in basis points for urgency level

        Args:
            urgency: Trade urgency level (optional, uses config default)

        Returns:
            Slippage tolerance in BPS
        """
        if self.config.custom_slippage_bps is not None:
            return self.config.custom_slippage_bps

        urgency = urgency or self.config.default_urgency
        return SLIPPAGE_TOLERANCE_BPS[urgency]

    def get_slippage_tolerance_pct(
        self,
        urgency: Optional[TradeUrgency] = None
    ) -> float:
        """
        Get slippage tolerance as percentage

        Args:
            urgency: Trade urgency level (optional)

        Returns:
            Slippage tolerance as percentage (e.g., 5.0 for 5%)
        """
        bps = self.get_slippage_tolerance_bps(urgency)
        return bps / BPS_DENOMINATOR * 100

    def calculate_max_input_amount(
        self,
        expected_output: int,
        urgency: Optional[TradeUrgency] = None,
        custom_slippage_bps: Optional[int] = None
    ) -> int:
        """
        Calculate maximum input amount for expected output (inverse calculation)

        Useful for sell orders: given desired SOL output, calculate max tokens to sell

        Args:
            expected_output: Desired output amount
            urgency: Trade urgency level (optional)
            custom_slippage_bps: Custom slippage (optional)

        Returns:
            Maximum input amount (rounded up conservatively)

        Example:
            # Want to receive at least 1 SOL, how many tokens to sell?
            max_tokens = manager.calculate_max_input_amount(
                expected_output=1_000_000_000,
                urgency=TradeUrgency.NORMAL
            )
        """
        if expected_output <= 0:
            raise ValueError("Expected output must be positive")

        # Determine slippage tolerance (priority order):
        # 1. Function parameter (most specific)
        # 2. Config-level custom slippage
        # 3. Urgency parameter
        # 4. Config default urgency
        if custom_slippage_bps is not None:
            slippage_bps = custom_slippage_bps
        elif self.config.custom_slippage_bps is not None:
            slippage_bps = self.config.custom_slippage_bps
        elif urgency is not None:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[urgency]
        else:
            slippage_bps = SLIPPAGE_TOLERANCE_BPS[self.config.default_urgency]

        # Calculate max input: expected / (1 - slippage)
        # Using integer math: expected * 10000 / (10000 - slippage_bps)
        # Round up to be conservative
        max_input = (expected_output * BPS_DENOMINATOR + (BPS_DENOMINATOR - slippage_bps - 1)) // (BPS_DENOMINATOR - slippage_bps)

        return max_input

    def get_stats(self) -> dict:
        """
        Get slippage manager statistics

        Returns:
            Dictionary with stats
        """
        return {
            "default_urgency": self.config.default_urgency.value,
            "default_slippage_pct": self.get_slippage_tolerance_pct(),
            "urgency_mappings": {
                urgency.value: SLIPPAGE_TOLERANCE_BPS[urgency] / BPS_DENOMINATOR * 100
                for urgency in TradeUrgency
            }
        }


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging

    # Setup logging
    setup_logging(level="DEBUG", format="console")

    # Create manager
    manager = SlippageManager()

    # Calculate minimum amounts for different urgency levels
    expected_tokens = 1_000_000

    print("Expected tokens out: 1,000,000\n")

    for urgency in TradeUrgency:
        min_out = manager.calculate_min_amount_out(expected_tokens, urgency)
        tolerance = manager.get_slippage_tolerance_pct(urgency)

        print(f"{urgency.value.upper()}:")
        print(f"  Tolerance: {tolerance}%")
        print(f"  Min amount: {min_out:,}")
        print(f"  Loss: {expected_tokens - min_out:,} tokens\n")

    # Validate some scenarios
    print("\nValidation scenarios:")

    # Good fill (within tolerance)
    check1 = manager.validate_slippage(
        expected_amount=1_000_000,
        actual_amount=980_000,
        urgency=TradeUrgency.NORMAL
    )
    print(f"Expected: 1,000,000, Actual: 980,000")
    print(f"  Valid: {check1.is_valid}")
    print(f"  Slippage: {check1.slippage_pct:.2f}%")
    print(f"  Message: {check1.message}\n")

    # Bad fill (exceeds tolerance)
    check2 = manager.validate_slippage(
        expected_amount=1_000_000,
        actual_amount=900_000,
        urgency=TradeUrgency.NORMAL
    )
    print(f"Expected: 1,000,000, Actual: 900,000")
    print(f"  Valid: {check2.is_valid}")
    print(f"  Slippage: {check2.slippage_pct:.2f}%")
    print(f"  Message: {check2.message}\n")

    # Better than expected
    check3 = manager.validate_slippage(
        expected_amount=1_000_000,
        actual_amount=1_020_000,
        urgency=TradeUrgency.NORMAL
    )
    print(f"Expected: 1,000,000, Actual: 1,020,000")
    print(f"  Valid: {check3.is_valid}")
    print(f"  Slippage: {check3.slippage_pct:.2f}%")
    print(f"  Message: {check3.message}")
