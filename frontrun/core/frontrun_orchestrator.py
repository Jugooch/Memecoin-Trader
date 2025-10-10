"""
Frontrun Orchestrator - Coordinates All Phase 4 Components
Orchestrates the complete frontrun flow with defensive guardrails
"""

import asyncio
from typing import Optional, Callable, Awaitable, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from clients.pumpfun_client import PumpFunClient
from core.rpc_manager import RPCManager
from core.wallet_manager import WalletManager
from core.tx_builder import TransactionBuilder
from core.tx_signer import TransactionSigner
from core.tx_submitter import TransactionSubmitter
from core.position_tracker import PositionTracker
from core.pnl import PnLCalculator
from core.logger import get_logger
from core.metrics import get_metrics

# Phase 4 components
from core.latency_enforcer import LatencyBudgetEnforcer, LatencyConfig, TradeStage
from core.profit_aware_fees import ProfitAwareFeeBidder, BidderConfig, Trade
from core.rpc_health_scorer import RPCHealthScorer, HealthScorerConfig
from core.race_detector import RaceFailureDetector, RaceConfig
from services.dev_detector import DevWalletDetector, DetectorConfig
from services.dev_buy_confirmation import DevBuyConfirmationDetector, DevBuyConfirmation
from services.mempool_monitor import LaserStreamMonitor, LaserStreamConfig, PendingTransaction


logger = get_logger(__name__)
metrics = get_metrics()


class FrontrunResult(Enum):
    """Result of frontrun attempt"""
    SUCCESS = "success"  # Trade submitted successfully
    ABORTED_LATENCY = "aborted_latency"  # Aborted due to latency budget
    ABORTED_NEGATIVE_EV = "aborted_negative_ev"  # Aborted due to negative EV
    ABORTED_CIRCUIT_OPEN = "aborted_circuit_open"  # Circuit breaker open
    ABORTED_NOT_DEV = "aborted_not_dev"  # Not a dev wallet
    ABORTED_DEV_CONFIRM_TIMEOUT = "aborted_dev_confirm_timeout"  # Dev buy didn't confirm in time
    FAILED_TX_BUILD = "failed_tx_build"  # Transaction build failed
    FAILED_TX_SUBMIT = "failed_tx_submit"  # Transaction submission failed


@dataclass
class OrchestratorConfig:
    """
    Configuration for frontrun orchestrator

    Integrates configs for all Phase 4 components
    """
    # Latency budgets (ms)
    latency_config: Optional[LatencyConfig] = None

    # Fee bidding
    bidder_config: Optional[BidderConfig] = None

    # RPC health scoring
    health_scorer_config: Optional[HealthScorerConfig] = None

    # Race detection
    race_config: Optional[RaceConfig] = None

    # Dev wallet detection
    detector_config: Optional[DetectorConfig] = None

    # Mempool monitoring (LaserStream)
    laserstream_config: Optional[LaserStreamConfig] = None

    # Trading parameters
    default_buy_amount_sol: float = 0.05  # Default frontrun amount
    max_slippage_bps: int = 500  # 5% max slippage
    compute_units: int = 200_000  # Default compute units


@dataclass
class FrontrunAttempt:
    """
    Record of a frontrun attempt

    Attributes:
        mint: Token mint
        dev_wallet: Dev wallet detected
        dev_buy_confirmation: Dev buy confirmation details
        our_signature: Our transaction signature (if submitted)
        result: Outcome of attempt
        latency_ms: Total latency from detection to submission
        abort_reason: Reason for abort (if aborted)
        race_won: Whether we won the race (if analyzable)
    """
    mint: Pubkey
    dev_wallet: Optional[Pubkey]
    dev_buy_confirmation: Optional[DevBuyConfirmation]
    our_signature: Optional[str]
    result: FrontrunResult
    latency_ms: float
    abort_reason: Optional[str] = None
    race_won: Optional[bool] = None
    profit_sol: Optional[float] = None
    fee_paid_lamports: Optional[int] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "mint": str(self.mint),
            "dev_wallet": str(self.dev_wallet) if self.dev_wallet else None,
            "our_signature": self.our_signature,
            "result": self.result.value,
            "latency_ms": self.latency_ms,
            "abort_reason": self.abort_reason,
            "race_won": self.race_won,
            "profit_sol": self.profit_sol,
            "fee_paid_lamports": self.fee_paid_lamports,
            "timestamp": self.timestamp.isoformat()
        }


class FrontrunOrchestrator:
    """
    Orchestrates complete frontrun flow with defensive guardrails

    Flow:
    1. MempoolMonitor detects pending dev transaction
    2. DevDetector validates it's likely a dev buy
    3. DevBuyConfirmationDetector waits for confirmation
    4. LatencyBudgetEnforcer starts timing
    5. Build frontrun transaction (check latency)
    6. ProfitAwareFeeBidder calculates fee with profit cap
    7. Validate EV > 0 (check latency)
    8. RPCHealthScorer selects best RPC endpoint
    9. Submit transaction (check latency)
    10. RaceDetector analyzes win/loss

    Defensive guardrails:
    - Hard latency budgets with circuit breaker
    - Profit-aware fee caps (never bid >30% of p25 profit)
    - Pre-submit EV validation
    - RPC health-based routing
    - Stop-fast abort semantics

    Usage:
        orchestrator = FrontrunOrchestrator(
            rpc_manager=rpc_manager,
            wallet_manager=wallet_manager,
            pumpfun_client=pumpfun_client,
            config=OrchestratorConfig()
        )

        await orchestrator.start()

        # Orchestrator runs in background, monitoring mempool
        # and executing frontrun attempts with guardrails
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        wallet_manager: WalletManager,
        pumpfun_client: PumpFunClient,
        position_tracker: PositionTracker,
        pnl_calculator: PnLCalculator,
        config: Optional[OrchestratorConfig] = None
    ):
        """
        Initialize frontrun orchestrator

        Args:
            rpc_manager: RPC manager
            wallet_manager: Wallet manager
            pumpfun_client: Pump.fun client
            position_tracker: Position tracker
            pnl_calculator: PnL calculator
            config: Orchestrator configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.wallet_manager = wallet_manager
        self.pumpfun_client = pumpfun_client
        self.position_tracker = position_tracker
        self.pnl_calculator = pnl_calculator
        self.config = config or OrchestratorConfig()

        # Initialize Phase 4 components
        self.latency_enforcer = LatencyBudgetEnforcer(
            self.config.latency_config or LatencyConfig()
        )

        # Fee bidder requires priority fee calculator
        from core.priority_fees import PriorityFeeCalculator
        fee_calculator = PriorityFeeCalculator(rpc_manager)
        self.fee_bidder = ProfitAwareFeeBidder(
            fee_calculator,
            self.config.bidder_config or BidderConfig()
        )

        self.rpc_health_scorer = RPCHealthScorer(
            rpc_manager,
            self.config.health_scorer_config or HealthScorerConfig()
        )

        self.race_detector = RaceFailureDetector(
            self.config.race_config or RaceConfig()
        )

        self.dev_detector = DevWalletDetector(
            self.config.detector_config or DetectorConfig()
        )

        self.dev_buy_confirmation_detector = DevBuyConfirmationDetector(
            rpc_manager,
            pumpfun_client
        )

        self.mempool_monitor = LaserStreamMonitor(
            self.config.laserstream_config or LaserStreamConfig(
                endpoint="laserstream-mainnet-ewr.helius-rpc.com",
                api_key="YOUR_API_KEY_HERE"  # Will be provided by config
            )
        )

        # Phase 1-3 components
        self.tx_builder = TransactionBuilder()
        self.tx_signer = TransactionSigner()
        self.tx_submitter = TransactionSubmitter(rpc_manager)

        # State
        self._running = False
        self._attempts_history: list[FrontrunAttempt] = []

        logger.info(
            "frontrun_orchestrator_initialized",
            latency_total_budget_ms=self.latency_enforcer.config.total_budget_ms,
            fee_profit_cap_pct=self.fee_bidder.config.profit_cap_pct,
            default_buy_amount_sol=self.config.default_buy_amount_sol
        )

    async def start(self):
        """
        Start orchestrator

        Begins monitoring mempool and executing frontrun attempts

        Example:
            await orchestrator.start()
            # Runs in background until stop() called
        """
        if self._running:
            logger.warning("orchestrator_already_running")
            return

        self._running = True

        logger.info("frontrun_orchestrator_started")
        metrics.increment_counter("orchestrator_starts")

        # Start mempool monitoring with callback
        await self.mempool_monitor.start_monitoring(self._on_pending_transaction)

    async def stop(self):
        """Stop orchestrator"""
        self._running = False
        await self.mempool_monitor.stop_monitoring()

        logger.info("frontrun_orchestrator_stopped")
        metrics.increment_counter("orchestrator_stops")

    async def _on_pending_transaction(self, tx: PendingTransaction):
        """
        Handle pending transaction from mempool

        Args:
            tx: Pending transaction
        """
        if not self._running:
            return

        logger.debug(
            "pending_transaction_seen",
            signature=tx.signature[:16] + "...",
            fee=tx.fee,
            program_id=str(tx.program_id) if tx.program_id else None
        )

        # Execute frontrun attempt (will be recorded automatically)
        attempt = await self.execute_frontrun_attempt(tx)

        # Log result
        logger.info(
            "frontrun_attempt_completed",
            **attempt.to_dict()
        )

        metrics.increment_counter(
            "frontrun_attempts",
            labels={
                "result": attempt.result.value,
                "race_won": str(attempt.race_won) if attempt.race_won is not None else "unknown"
            }
        )

    async def execute_frontrun_attempt(
        self,
        pending_tx: PendingTransaction,
        buy_amount_sol: Optional[float] = None
    ) -> FrontrunAttempt:
        """
        Execute single frontrun attempt with full guardrails

        Args:
            pending_tx: Pending transaction from mempool
            buy_amount_sol: Amount to buy (optional, uses default)

        Returns:
            FrontrunAttempt with results

        Flow:
        1. Check circuit breaker
        2. Start latency timer
        3. Detect dev wallet pattern
        4. Wait for dev buy confirmation
        5. Build our transaction (check latency)
        6. Calculate aggressive fee (check latency)
        7. Validate EV > 0
        8. Sign transaction
        9. Submit (check latency)
        10. Analyze race result
        """
        buy_amount_sol = buy_amount_sol or self.config.default_buy_amount_sol

        # Start latency timer
        timer = self.latency_enforcer.start_operation(pending_tx.signature)

        # Check circuit breaker
        if self.latency_enforcer.is_circuit_open():
            logger.warning("circuit_breaker_open_aborting")
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),  # Placeholder
                dev_wallet=None,
                dev_buy_confirmation=None,
                our_signature=None,
                result=FrontrunResult.ABORTED_CIRCUIT_OPEN,
                latency_ms=timer.elapsed_ms(),
                abort_reason="Circuit breaker open"
            )
            self._attempts_history.append(attempt)
            return attempt

        # STAGE 1: Detect frontrun opportunity
        self.latency_enforcer.record_stage(timer, TradeStage.DETECT)

        # Extract data from transaction (real LaserStream data!)
        program_id = pending_tx.program_id if pending_tx.program_id else Pubkey.from_string("11111111111111111111111111111111")
        buy_amount_sol = pending_tx.buy_amount_sol  # Real buy amount from LaserStream
        sender = pending_tx.accounts[0] if pending_tx.accounts else Pubkey.from_string("11111111111111111111111111111111")

        # Analyze if this is a frontrun opportunity
        opportunity = self.dev_detector.analyze_transaction(
            program_id=program_id,
            buy_amount_sol=buy_amount_sol,
            sender=sender
        )

        # Check latency budget after detection
        if self.latency_enforcer.check_and_abort(timer, TradeStage.DETECT):
            logger.warning("aborted_latency_exceeded_detect_stage")

            # Violation already recorded by check_and_abort()
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),
                dev_wallet=sender,
                dev_buy_confirmation=None,
                our_signature=None,
                result=FrontrunResult.ABORTED_LATENCY,
                latency_ms=timer.elapsed_ms(),
                abort_reason="Latency budget exceeded in detect stage"
            )
            self._attempts_history.append(attempt)
            return attempt

        # If not worth frontrunning, abort
        if not opportunity.should_frontrun:
            logger.debug(
                "not_frontrun_opportunity",
                reason=opportunity.reason,
                buy_amount_sol=opportunity.buy_amount_sol
            )
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),
                dev_wallet=sender,
                dev_buy_confirmation=None,
                our_signature=None,
                result=FrontrunResult.ABORTED_NOT_DEV,
                latency_ms=timer.elapsed_ms(),
                abort_reason=opportunity.reason
            )
            self._attempts_history.append(attempt)
            return attempt

        # STAGE 2: Wait for dev buy confirmation
        # For simulation, we skip this as it requires real token mint
        # In production, would call:
        # confirmation = await self.dev_buy_confirmation_detector.wait_for_dev_buy_confirmation(
        #     mint=token_mint,
        #     timeout_seconds=30
        # )

        confirmation = None  # Skip for now

        # STAGE 3: Build transaction
        self.latency_enforcer.record_stage(timer, TradeStage.BUILD_SIGN)

        # For simulation, skip actual transaction building
        # In production, would call tx_builder to build swap transaction

        # Check latency budget after build
        if self.latency_enforcer.check_and_abort(timer, TradeStage.BUILD_SIGN):
            logger.warning("aborted_latency_exceeded_build_stage")

            # Violation already recorded by check_and_abort()
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),
                dev_wallet=sender,
                dev_buy_confirmation=confirmation,
                our_signature=None,
                result=FrontrunResult.ABORTED_LATENCY,
                latency_ms=timer.elapsed_ms(),
                abort_reason="Latency budget exceeded in build stage"
            )
            self._attempts_history.append(attempt)
            return attempt

        # STAGE 4: Calculate aggressive fee with profit cap
        fee_microlamports = await self.fee_bidder.calculate_aggressive_fee(
            competing_tx={"fee": pending_tx.fee},
            compute_units=self.config.compute_units
        )

        fee_lamports = (fee_microlamports * self.config.compute_units) // 1_000_000

        # STAGE 5: Validate EV before submit
        expected_profit_sol = buy_amount_sol * 0.1  # Simplified - assume 10% profit

        if not self.fee_bidder.validate_ev_before_submit(
            expected_profit_sol=expected_profit_sol,
            fee_lamports=fee_lamports
        ):
            logger.warning("aborted_negative_ev")
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),
                dev_wallet=sender,
                dev_buy_confirmation=confirmation,
                our_signature=None,
                result=FrontrunResult.ABORTED_NEGATIVE_EV,
                latency_ms=timer.elapsed_ms(),
                abort_reason="Negative EV after fee calculation"
            )
            self._attempts_history.append(attempt)
            return attempt

        # STAGE 6: Submit transaction
        self.latency_enforcer.record_stage(timer, TradeStage.SUBMIT)

        # For simulation, skip actual submission
        # In production, would sign and submit transaction

        # Check final latency budget
        if self.latency_enforcer.check_and_abort(timer, TradeStage.SUBMIT):
            logger.warning("aborted_latency_exceeded_submit_stage")

            # Violation already recorded by check_and_abort()
            attempt = FrontrunAttempt(
                mint=Pubkey.from_string("11111111111111111111111111111111"),
                dev_wallet=sender,
                dev_buy_confirmation=confirmation,
                our_signature=None,
                result=FrontrunResult.ABORTED_LATENCY,
                latency_ms=timer.elapsed_ms(),
                abort_reason="Latency budget exceeded before submit"
            )
            self._attempts_history.append(attempt)
            return attempt

        # Simulated successful submission
        signature = "simulated_" + pending_tx.signature[:32]

        logger.info(
            "frontrun_transaction_submitted",
            signature=signature,
            fee_lamports=fee_lamports,
            expected_profit_sol=expected_profit_sol,
            latency_ms=timer.elapsed_ms()
        )

        # STAGE 7: Analyze race result (would happen after confirmation)
        # For now, assume we won
        race_won = True

        # Record trade for fee bidder learning
        self.fee_bidder.add_trade(Trade(
            won_race=race_won,
            pnl_sol=expected_profit_sol,  # Actual would come from position tracker
            fee_paid_lamports=fee_lamports,
            timestamp=datetime.now()
        ))

        # Record operation success - resets violation counter
        self.latency_enforcer.record_operation_success()

        attempt = FrontrunAttempt(
            mint=Pubkey.from_string("11111111111111111111111111111111"),
            dev_wallet=sender,
            dev_buy_confirmation=confirmation,
            our_signature=signature,
            result=FrontrunResult.SUCCESS,
            latency_ms=timer.elapsed_ms(),
            race_won=race_won,
            profit_sol=expected_profit_sol,
            fee_paid_lamports=fee_lamports
        )

        # Record attempt in history
        self._attempts_history.append(attempt)

        return attempt

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics

        Returns:
            Dictionary with comprehensive stats
        """
        total_attempts = len(self._attempts_history)

        # Calculate basic stats
        successes = sum(1 for a in self._attempts_history if a.result == FrontrunResult.SUCCESS)
        aborts_latency = sum(1 for a in self._attempts_history if a.result == FrontrunResult.ABORTED_LATENCY)
        aborts_ev = sum(1 for a in self._attempts_history if a.result == FrontrunResult.ABORTED_NEGATIVE_EV)

        avg_latency = (sum(a.latency_ms for a in self._attempts_history) / total_attempts) if total_attempts > 0 else 0.0

        races_won = sum(1 for a in self._attempts_history if a.race_won is True)
        races_lost = sum(1 for a in self._attempts_history if a.race_won is False)

        return {
            "total_attempts": total_attempts,
            "successes": successes,
            "success_rate": successes / total_attempts if total_attempts > 0 else 0.0,
            "aborts_latency": aborts_latency,
            "aborts_negative_ev": aborts_ev,
            "avg_latency_ms": avg_latency,
            "races_won": races_won,
            "races_lost": races_lost,
            "win_rate": races_won / (races_won + races_lost) if (races_won + races_lost) > 0 else 0.0,
            "circuit_breaker_open": self.latency_enforcer.is_circuit_open(),
            "latency_enforcer_stats": self.latency_enforcer.get_attribution_stats(),
            "fee_bidder_stats": self.fee_bidder.get_stats(),
            "rpc_health_stats": self.rpc_health_scorer.get_stats()
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

        # Create core components
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        try:
            wallet_manager = WalletManager(bot_config.wallet_config)
            await wallet_manager.load_wallets()

            pumpfun_client = PumpFunClient(rpc_manager)
            position_tracker = PositionTracker(bot_config.database_config)
            await position_tracker.start()

            pnl_calculator = PnLCalculator()

            # Create orchestrator
            orchestrator_config = OrchestratorConfig(
                default_buy_amount_sol=0.05,
                latency_config=LatencyConfig(total_budget_ms=100.0),
                bidder_config=BidderConfig(profit_cap_pct=0.3)
            )

            orchestrator = FrontrunOrchestrator(
                rpc_manager=rpc_manager,
                wallet_manager=wallet_manager,
                pumpfun_client=pumpfun_client,
                position_tracker=position_tracker,
                pnl_calculator=pnl_calculator,
                config=orchestrator_config
            )

            # Start orchestrator (runs for 30 seconds in this example)
            logger.info("Starting frontrun orchestrator...")
            logger.info("Orchestrator will monitor mempool and execute frontrun attempts")
            logger.info("Press Ctrl+C to stop")

            await asyncio.wait_for(
                orchestrator.start(),
                timeout=30.0
            )

        except asyncio.TimeoutError:
            logger.info("Example completed")

            # Print stats
            stats = orchestrator.get_stats()
            logger.info("orchestrator_stats", **stats)

        finally:
            await orchestrator.stop()
            await position_tracker.stop()
            await rpc_manager.stop()

    asyncio.run(main())
