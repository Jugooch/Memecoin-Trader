"""
Dev Buy Confirmation Detector for Frontrun Bot
Detects exact moment when dev buy transaction confirms on-chain
"""

import asyncio
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime

from solders.pubkey import Pubkey

from clients.pumpfun_client import PumpFunClient
from core.bonding_curve import BondingCurveState
from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class DevBuyConfirmation:
    """
    Dev buy confirmation details

    Attributes:
        mint: Token mint
        dev_buy_signature: Transaction signature (if known)
        confirmed_slot: Slot when confirmed
        confirmed_at: Timestamp when confirmed
        bonding_curve_state: Bonding curve state after dev buy
        dev_buy_amount_sol: Estimated dev buy amount
        reserve_ratio_change: Change in reserve ratio
    """
    mint: Pubkey
    dev_buy_signature: Optional[str]
    confirmed_slot: int
    confirmed_at: datetime
    bonding_curve_state: BondingCurveState
    dev_buy_amount_sol: float
    reserve_ratio_change: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "mint": str(self.mint),
            "dev_buy_signature": self.dev_buy_signature,
            "confirmed_slot": self.confirmed_slot,
            "confirmed_at": self.confirmed_at.isoformat(),
            "dev_buy_amount_sol": self.dev_buy_amount_sol,
            "reserve_ratio_change": self.reserve_ratio_change,
            "virtual_sol_reserves": self.bonding_curve_state.virtual_sol_reserves,
            "virtual_token_reserves": self.bonding_curve_state.virtual_token_reserves
        }


@dataclass
class BondingCurveChange:
    """
    Bonding curve state change

    Attributes:
        mint: Token mint
        old_state: Previous bonding curve state
        new_state: New bonding curve state
        slot: Slot when change occurred
        reserve_ratio_change: % change in reserve ratio
    """
    mint: Pubkey
    old_state: BondingCurveState
    new_state: BondingCurveState
    slot: int
    reserve_ratio_change: float


class DevBuyConfirmationDetector:
    """
    Detects when dev buy transaction confirms

    Detection method:
    - Subscribe to bonding curve account updates
    - Calculate reserve ratio: sol_reserves / token_reserves
    - Large ratio change (>10%) indicates dev buy
    - Track timestamp with microsecond precision

    Usage:
        detector = DevBuyConfirmationDetector(rpc_manager, pumpfun_client)

        # Wait for dev buy confirmation
        confirmation = await detector.wait_for_dev_buy_confirmation(
            mint=token_mint,
            timeout_seconds=30
        )

        if confirmation:
            logger.info("Dev buy confirmed!", **confirmation.to_dict())

        # Or subscribe to changes
        async def on_change(change: BondingCurveChange):
            if change.reserve_ratio_change > 10.0:
                logger.info("Large buy detected!")

        await detector.subscribe_bonding_curve_changes(
            mint=token_mint,
            callback=on_change
        )
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        pumpfun_client: PumpFunClient
    ):
        """
        Initialize dev buy confirmation detector

        Args:
            rpc_manager: RPC manager for subscriptions
            pumpfun_client: Pump.fun client for bonding curve access
        """
        self.rpc_manager = rpc_manager
        self.pumpfun_client = pumpfun_client

        # Track previous states for comparison
        self._previous_states: dict[str, BondingCurveState] = {}

        logger.info("dev_buy_confirmation_detector_initialized")

    async def wait_for_dev_buy_confirmation(
        self,
        mint: Pubkey,
        timeout_seconds: int = 30,
        threshold_pct: float = 10.0
    ) -> Optional[DevBuyConfirmation]:
        """
        Wait for dev buy to confirm

        Args:
            mint: Token mint to monitor
            timeout_seconds: Max time to wait
            threshold_pct: Reserve ratio change threshold (default: 10%)

        Returns:
            DevBuyConfirmation if detected, None if timeout

        Example:
            confirmation = await detector.wait_for_dev_buy_confirmation(
                mint=token_mint,
                timeout_seconds=30
            )

            if confirmation:
                logger.info("Dev buy detected at slot", slot=confirmation.confirmed_slot)
        """
        # Get initial bonding curve state
        try:
            initial_state = await self.pumpfun_client.get_bonding_curve_account(mint)
        except Exception as e:
            logger.error(
                "failed_to_get_initial_bonding_curve",
                mint=str(mint),
                error=str(e)
            )
            return None

        # Store initial state
        self._previous_states[str(mint)] = initial_state
        initial_ratio = self._calculate_reserve_ratio(initial_state)

        logger.info(
            "waiting_for_dev_buy",
            mint=str(mint),
            timeout_seconds=timeout_seconds,
            initial_ratio=initial_ratio
        )

        # Poll for changes (in production, would use account subscription)
        start_time = asyncio.get_event_loop().time()

        while True:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(
                    "dev_buy_wait_timeout",
                    mint=str(mint),
                    elapsed_seconds=elapsed
                )
                metrics.increment_counter("dev_buy_wait_timeouts")
                return None

            # Get current state
            try:
                current_state = await self.pumpfun_client.get_bonding_curve_account(mint)
            except Exception as e:
                logger.error(
                    "failed_to_poll_bonding_curve",
                    mint=str(mint),
                    error=str(e)
                )
                await asyncio.sleep(0.5)
                continue

            # Calculate reserve ratio change
            current_ratio = self._calculate_reserve_ratio(current_state)
            ratio_change_pct = ((current_ratio - initial_ratio) / initial_ratio) * 100.0 if initial_ratio > 0 else 0.0

            logger.debug(
                "bonding_curve_polled",
                mint=str(mint),
                current_ratio=current_ratio,
                ratio_change_pct=ratio_change_pct
            )

            # Check if threshold exceeded
            if ratio_change_pct >= threshold_pct:
                # Dev buy detected!
                dev_buy_amount_sol = (current_state.virtual_sol_reserves - initial_state.virtual_sol_reserves) / 1e9

                confirmation = DevBuyConfirmation(
                    mint=mint,
                    dev_buy_signature=None,  # TODO: Track actual signature
                    confirmed_slot=0,  # TODO: Get current slot
                    confirmed_at=datetime.now(),
                    bonding_curve_state=current_state,
                    dev_buy_amount_sol=dev_buy_amount_sol,
                    reserve_ratio_change=ratio_change_pct
                )

                logger.info(
                    "dev_buy_confirmed",
                    mint=str(mint),
                    dev_buy_amount_sol=dev_buy_amount_sol,
                    ratio_change_pct=ratio_change_pct,
                    elapsed_ms=(elapsed * 1000)
                )

                metrics.increment_counter("dev_buys_confirmed")
                metrics.record_latency(
                    "dev_buy_confirmation_latency",
                    elapsed * 1000,
                    labels={"mint": str(mint)}
                )

                return confirmation

            # Wait before next poll (500ms)
            await asyncio.sleep(0.5)

    async def subscribe_bonding_curve_changes(
        self,
        mint: Pubkey,
        callback: Callable[[BondingCurveChange], Awaitable[None]]
    ):
        """
        Subscribe to bonding curve account changes

        Args:
            mint: Token mint to monitor
            callback: Async function to call on changes

        Example:
            async def on_change(change: BondingCurveChange):
                logger.info(
                    "Bonding curve changed",
                    ratio_change=change.reserve_ratio_change
                )

            await detector.subscribe_bonding_curve_changes(
                mint=token_mint,
                callback=on_change
            )
        """
        # Get initial state
        try:
            initial_state = await self.pumpfun_client.get_bonding_curve_account(mint)
            self._previous_states[str(mint)] = initial_state
        except Exception as e:
            logger.error(
                "failed_to_subscribe_bonding_curve",
                mint=str(mint),
                error=str(e)
            )
            return

        logger.info("subscribed_to_bonding_curve", mint=str(mint))

        # Poll for changes (TODO: Use actual RPC subscription)
        while True:
            await asyncio.sleep(0.5)  # Poll every 500ms

            try:
                current_state = await self.pumpfun_client.get_bonding_curve_account(mint)
            except Exception as e:
                logger.error(
                    "failed_to_poll_bonding_curve",
                    mint=str(mint),
                    error=str(e)
                )
                continue

            # Check for changes
            previous_state = self._previous_states.get(str(mint))
            if not previous_state:
                self._previous_states[str(mint)] = current_state
                continue

            # Calculate reserve ratio change
            prev_ratio = self._calculate_reserve_ratio(previous_state)
            curr_ratio = self._calculate_reserve_ratio(current_state)
            ratio_change_pct = ((curr_ratio - prev_ratio) / prev_ratio) * 100.0 if prev_ratio > 0 else 0.0

            # If significant change, call callback
            if abs(ratio_change_pct) > 1.0:  # >1% change
                change = BondingCurveChange(
                    mint=mint,
                    old_state=previous_state,
                    new_state=current_state,
                    slot=0,  # TODO: Get current slot
                    reserve_ratio_change=ratio_change_pct
                )

                logger.debug(
                    "bonding_curve_changed",
                    mint=str(mint),
                    ratio_change_pct=ratio_change_pct
                )

                # Call callback
                await callback(change)

                # Update previous state
                self._previous_states[str(mint)] = current_state

    def _calculate_reserve_ratio(self, state: BondingCurveState) -> float:
        """
        Calculate reserve ratio (sol / tokens)

        Args:
            state: Bonding curve state

        Returns:
            Reserve ratio

        Formula:
            ratio = virtual_sol_reserves / virtual_token_reserves
        """
        if state.virtual_token_reserves == 0:
            return 0.0

        return state.virtual_sol_reserves / state.virtual_token_reserves


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
            # Create pump.fun client
            pumpfun_client = PumpFunClient(rpc_manager)

            # Create detector
            detector = DevBuyConfirmationDetector(rpc_manager, pumpfun_client)

            # Example: Wait for dev buy on a specific mint
            # (Replace with actual mint address)
            example_mint = Pubkey.from_string("11111111111111111111111111111111")

            logger.info("Waiting for dev buy confirmation...")
            confirmation = await detector.wait_for_dev_buy_confirmation(
                mint=example_mint,
                timeout_seconds=10  # Short timeout for example
            )

            if confirmation:
                logger.info("Dev buy confirmed!", **confirmation.to_dict())
            else:
                logger.info("No dev buy detected within timeout")

        finally:
            await rpc_manager.stop()

    asyncio.run(main())
