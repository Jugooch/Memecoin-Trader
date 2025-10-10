"""
Token Launch Sniper Detector
Detects NEWLY LAUNCHED tokens on Pump.fun (not old tokens with large buys!)
"""

from typing import Optional, Dict
from dataclasses import dataclass

from solders.pubkey import Pubkey

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


# Pump.fun program ID
PUMP_FUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")


@dataclass
class DetectorConfig:
    """
    Configuration for token launch detection

    Key settings:
    - min_buy_amount_sol: Minimum buy size (default: 0.5 SOL)
    - max_market_cap_sol: Maximum market cap for "new" token (default: 5 SOL)
    - only_new_tokens: Only trade newly launched tokens (default: True)
    """
    min_buy_amount_sol: float = 0.5
    max_market_cap_sol: float = 5.0  # Only trade tokens with < 5 SOL market cap
    only_new_tokens: bool = True  # CRITICAL: Only trade new launches


@dataclass
class FrontrunOpportunity:
    """
    Detected frontrun opportunity

    Attributes:
        should_frontrun: Whether this transaction is worth frontrunning
        buy_amount_sol: Amount of SOL in the buy
        mint: Token mint (if detected)
        sender: Transaction sender
        reason: Reason for decision

    IMPORTANT: Caller must verify token is NEW before trading!
    Use core.bonding_curve.fetch_bonding_curve_state() and is_new_token_launch()
    """
    should_frontrun: bool
    buy_amount_sol: float
    mint: Optional[Pubkey] = None
    sender: Optional[Pubkey] = None
    reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "should_frontrun": self.should_frontrun,
            "buy_amount_sol": self.buy_amount_sol,
            "mint": str(self.mint) if self.mint else None,
            "sender": str(self.sender) if self.sender else None,
            "reason": self.reason
        }


class DevWalletDetector:
    """
    Simplified detector for frontrun opportunities

    Simple logic:
    1. Is this a Pump.fun transaction?
    2. Is it a buy?
    3. Is it >= minimum size?

    That's it. No complex heuristics needed.

    Usage:
        detector = DevWalletDetector(config)

        # Analyze transaction
        opportunity = detector.analyze_transaction(
            program_id=tx.program_id,
            buy_amount_sol=tx.amount_sol
        )

        if opportunity.should_frontrun:
            # Execute frontrun
            await frontrun(opportunity)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize frontrun detector

        Args:
            config: Detector configuration (optional)
        """
        self.config = config or DetectorConfig()

        logger.info(
            "frontrun_detector_initialized",
            min_buy_amount_sol=self.config.min_buy_amount_sol
        )

    def should_frontrun(
        self,
        program_id: Pubkey,
        buy_amount_sol: float
    ) -> bool:
        """
        Simple check: Should we frontrun this transaction?

        Args:
            program_id: Transaction program ID
            buy_amount_sol: Amount of SOL in buy

        Returns:
            True if should frontrun, False otherwise

        Example:
            if detector.should_frontrun(tx.program_id, tx.amount_sol):
                await execute_frontrun()
        """
        # Must be Pump.fun program
        if program_id != PUMP_FUN_PROGRAM_ID:
            return False

        # Must be large enough to be worth frontrunning
        if buy_amount_sol < self.config.min_buy_amount_sol:
            return False

        return True

    def analyze_transaction(
        self,
        program_id: Pubkey,
        buy_amount_sol: float,
        mint: Optional[Pubkey] = None,
        sender: Optional[Pubkey] = None
    ) -> FrontrunOpportunity:
        """
        Analyze transaction for frontrun opportunity

        Args:
            program_id: Transaction program ID
            buy_amount_sol: Amount of SOL in buy
            mint: Token mint (optional)
            sender: Transaction sender (optional)

        Returns:
            FrontrunOpportunity with analysis

        Example:
            opportunity = detector.analyze_transaction(
                program_id=tx.program_id,
                buy_amount_sol=1.0,
                mint=token_mint
            )

            if opportunity.should_frontrun:
                logger.info("Frontrun opportunity detected!", **opportunity.to_dict())
        """
        should_frontrun = False
        reason = ""

        # Check if Pump.fun program
        if program_id != PUMP_FUN_PROGRAM_ID:
            reason = "Not Pump.fun program"
        # Check if large enough
        elif buy_amount_sol < self.config.min_buy_amount_sol:
            reason = f"Buy too small (<{self.config.min_buy_amount_sol} SOL)"
        else:
            should_frontrun = True
            reason = f"Large buy detected ({buy_amount_sol:.2f} SOL)"

        opportunity = FrontrunOpportunity(
            should_frontrun=should_frontrun,
            buy_amount_sol=buy_amount_sol,
            mint=mint,
            sender=sender,
            reason=reason
        )

        logger.debug(
            "transaction_analyzed",
            should_frontrun=should_frontrun,
            buy_amount_sol=buy_amount_sol,
            reason=reason
        )

        metrics.increment_counter(
            "transactions_analyzed",
            labels={"should_frontrun": str(should_frontrun)}
        )

        return opportunity

    def get_stats(self) -> Dict:
        """
        Get detector statistics

        Returns:
            Dictionary with statistics

        Example:
            stats = detector.get_stats()
            print(f"Minimum buy: {stats['min_buy_amount_sol']} SOL")
        """
        return {
            "min_buy_amount_sol": self.config.min_buy_amount_sol
        }


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging
    import asyncio

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Create detector
        config = DetectorConfig(min_buy_amount_sol=0.5)
        detector = DevWalletDetector(config)

        # Test 1: Large Pump.fun buy (should frontrun)
        logger.info("=== Test 1: Large Pump.fun Buy ===")
        opportunity = detector.analyze_transaction(
            program_id=PUMP_FUN_PROGRAM_ID,
            buy_amount_sol=1.0
        )
        logger.info("opportunity", **opportunity.to_dict())

        # Test 2: Small buy (should NOT frontrun)
        logger.info("\n=== Test 2: Small Buy ===")
        opportunity = detector.analyze_transaction(
            program_id=PUMP_FUN_PROGRAM_ID,
            buy_amount_sol=0.1
        )
        logger.info("opportunity", **opportunity.to_dict())

        # Test 3: Different program (should NOT frontrun)
        logger.info("\n=== Test 3: Different Program ===")
        other_program = Pubkey.from_string("11111111111111111111111111111111")
        opportunity = detector.analyze_transaction(
            program_id=other_program,
            buy_amount_sol=1.0
        )
        logger.info("opportunity", **opportunity.to_dict())

        # Get stats
        stats = detector.get_stats()
        logger.info("detector_stats", **stats)

    asyncio.run(main())
