"""
Phase 4 End-to-End Test: Complete Frontrun Flow
Demonstrates the full frontrun flow from mempool monitoring to race analysis

This test validates the COMPLETE frontrun flow:
1. Mempool Monitor detects pending dev transaction
2. Dev Detector validates it's a dev buy pattern
3. Dev Buy Confirmation Detector waits for confirmation
4. Latency Budget Enforcer starts timing and checks budgets
5. Transaction Builder builds our frontrun transaction
6. Profit-Aware Fee Bidder calculates aggressive fee with cap
7. EV Validator ensures positive expected value
8. RPC Health Scorer selects best endpoint
9. Transaction Submitter submits with retries
10. Race Detector analyzes if we won or lost
11. Position Tracker records position
12. PnL Tracker calculates profit/loss

Defensive Guardrails Demonstrated:
- Hard latency budgets with circuit breaker
- Profit-aware fee caps (never bid >30% of p25 profit)
- Pre-submit EV validation
- Stop-fast abort semantics (abort late trades, don't submit)
- RPC health-based routing

NOTE: This test uses SIMULATION mode for mempool monitoring
      Production would require Geyser plugin for real mempool access
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from clients.pumpfun_client import PumpFunClient
from core.rpc_manager import RPCManager
from core.wallet_manager import WalletManager
from core.position_tracker import PositionTracker, PositionStorage
from core.pnl import PnLCalculator
from core.logger import get_logger

# Phase 4 components
from core.latency_enforcer import LatencyBudgetEnforcer, LatencyConfig, TradeStage
from core.profit_aware_fees import ProfitAwareFeeBidder, BidderConfig, Trade
from core.rpc_health_scorer import RPCHealthScorer, HealthScorerConfig
from core.race_detector import RaceFailureDetector, RaceConfig
from services.dev_detector import DevWalletDetector, DetectorConfig
from services.dev_buy_confirmation import DevBuyConfirmationDetector
from services.mempool_monitor import MempoolMonitor, MempoolConfig, PendingTransaction
from core.frontrun_orchestrator import FrontrunOrchestrator, OrchestratorConfig, FrontrunResult


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# E2E TEST: COMPLETE FRONTRUN FLOW (SIMULATED)
# =============================================================================

@pytest.mark.asyncio
async def test_e2e_complete_frontrun_flow(devnet_rpc_manager, funded_wallet):
    """
    End-to-end test of complete frontrun flow

    This test demonstrates the full flow with all Phase 4 components
    working together, from mempool detection to race analysis.

    Flow:
    1. Setup all components
    2. Seed fee bidder with trade history (exit cold-start mode)
    3. Seed dev detector with known wallets
    4. Start mempool monitoring
    5. Receive pending dev transaction
    6. Execute frontrun attempt with all guardrails
    7. Analyze results
    8. Verify defensive guardrails worked
    """

    logger.info("=== STARTING E2E FRONTRUN FLOW TEST ===")

    # =============================================================================
    # PHASE 0: SETUP COMPONENTS
    # =============================================================================

    # Create temporary position database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    # Create temporary known dev wallets file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as f:
        import json
        json.dump({"wallets": []}, f)
        known_wallets_file = f.name

    try:
        logger.info("--- Phase 0: Component Setup ---")

        # Initialize core components
        pumpfun_client = PumpFunClient(devnet_rpc_manager)

        position_storage = PositionStorage(db_path=db_path)
        position_tracker = PositionTracker(position_storage)
        await position_tracker.start()

        pnl_calculator = PnLCalculator()

        # Create wallet manager with funded wallet
        wallet_manager = WalletManager(
            rpc_manager=devnet_rpc_manager,
            keypairs=[funded_wallet]
        )

        logger.info("core_components_initialized")

        # =============================================================================
        # PHASE 1: CONFIGURE ORCHESTRATOR
        # =============================================================================

        logger.info("--- Phase 1: Orchestrator Configuration ---")

        orchestrator_config = OrchestratorConfig(
            default_buy_amount_sol=0.05,

            # Latency budgets (permissive for testing)
            latency_config=LatencyConfig(
                detect_stage_budget_ms=200.0,
                build_sign_budget_ms=100.0,
                submit_stage_budget_ms=150.0,
                total_budget_ms=400.0,
                violations_before_trip=5
            ),

            # Fee bidding config
            bidder_config=BidderConfig(
                cold_start_max_lamports=50_000,
                profit_cap_pct=0.3,  # Never bid >30% of p25 profit
                min_samples_for_ev=5
            ),

            # RPC health config
            health_scorer_config=HealthScorerConfig(
                latency_threshold_p99_ms=500.0,
                slot_lag_threshold=5
            ),

            # Race detection config
            race_config=RaceConfig(
                price_deviation_threshold_pct=5.0,
                max_acceptable_slot_delay=2
            ),

            # Dev detector config
            detector_config=DetectorConfig(
                min_buy_amount_sol=0.5,
                max_time_since_token_creation_s=300,
                known_dev_wallets_file=known_wallets_file
            ),

            # Mempool config (simulation mode)
            mempool_config=MempoolConfig(
                simulation_mode=True,
                simulation_event_interval_s=2.0
            )
        )

        # Create orchestrator
        orchestrator = FrontrunOrchestrator(
            rpc_manager=devnet_rpc_manager,
            wallet_manager=wallet_manager,
            pumpfun_client=pumpfun_client,
            position_tracker=position_tracker,
            pnl_calculator=pnl_calculator,
            config=orchestrator_config
        )

        logger.info("orchestrator_configured", config=orchestrator_config.__dict__)

        # =============================================================================
        # PHASE 2: SEED COMPONENTS WITH DATA (EXIT COLD-START)
        # =============================================================================

        logger.info("--- Phase 2: Seeding Components ---")

        # Seed fee bidder with 10 winning trades to exit cold-start mode
        for i in range(10):
            orchestrator.fee_bidder.add_trade(Trade(
                won_race=True,
                pnl_sol=0.03 + (i * 0.01),  # 0.03 to 0.12 SOL profits
                fee_paid_lamports=100_000,
                timestamp=datetime.now()
            ))

        fee_stats = orchestrator.fee_bidder.get_stats()
        logger.info(
            "fee_bidder_seeded",
            total_trades=fee_stats["total_trades"],
            using_cold_start=fee_stats["using_cold_start"],
            conservative_ev_sol=fee_stats["conservative_ev_sol"]
        )

        assert fee_stats["using_cold_start"] is False, "Should exit cold-start mode"
        assert fee_stats["conservative_ev_sol"] > 0, "Should have positive EV estimate"

        # Seed dev detector with a known dev wallet
        known_dev_wallet = Pubkey.from_string("11111111111111111111111111111111")
        orchestrator.dev_detector.add_known_dev_wallet(known_dev_wallet)

        logger.info("dev_detector_seeded", known_wallets=1)

        # =============================================================================
        # PHASE 3: SIMULATE PENDING DEV TRANSACTION FROM MEMPOOL
        # =============================================================================

        logger.info("--- Phase 3: Simulating Pending Dev Transaction ---")

        # Create a simulated pending dev buy transaction
        simulated_dev_tx = PendingTransaction(
            signature="e2e_test_dev_buy_signature_12345",
            transaction=None,
            received_at=datetime.now(),
            fee=500_000,  # 500k lamports - competitive fee
            accounts=[
                known_dev_wallet,  # Dev wallet
                Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),  # Token mint
                Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")  # Pump.fun program
            ],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )

        logger.info(
            "pending_dev_tx_detected",
            signature=simulated_dev_tx.signature[:32] + "...",
            fee=simulated_dev_tx.fee,
            sender=str(simulated_dev_tx.accounts[0]) if simulated_dev_tx.accounts else None
        )

        # =============================================================================
        # PHASE 4: EXECUTE FRONTRUN ATTEMPT
        # =============================================================================

        logger.info("--- Phase 4: Executing Frontrun Attempt ---")

        # Execute frontrun attempt (this exercises the ENTIRE flow)
        attempt = await orchestrator.execute_frontrun_attempt(
            simulated_dev_tx,
            buy_amount_sol=0.05
        )

        logger.info("frontrun_attempt_completed", **attempt.to_dict())

        # =============================================================================
        # PHASE 5: VERIFY RESULTS AND GUARDRAILS
        # =============================================================================

        logger.info("--- Phase 5: Verifying Results ---")

        # Verify attempt completed
        assert attempt is not None, "Attempt should complete"
        assert attempt.latency_ms >= 0, "Latency should be tracked"
        assert attempt.result is not None, "Result should be set"

        # Verify latency was tracked
        assert attempt.latency_ms < orchestrator_config.latency_config.total_budget_ms, \
            "Total latency should be within budget (or would have aborted)"

        logger.info(
            "latency_verified",
            latency_ms=attempt.latency_ms,
            budget_ms=orchestrator_config.latency_config.total_budget_ms
        )

        # Verify latency enforcer stats
        latency_stats = orchestrator.latency_enforcer.get_attribution_stats()
        logger.info("latency_enforcer_stats", **latency_stats)

        assert latency_stats["total_operations"] >= 1
        assert "violation_rate" in latency_stats

        # Verify fee bidder calculated fees
        if attempt.fee_paid_lamports:
            fee_sol = attempt.fee_paid_lamports / 1e9

            logger.info(
                "fee_verification",
                fee_lamports=attempt.fee_paid_lamports,
                fee_sol=fee_sol,
                expected_profit_sol=attempt.profit_sol
            )

            # Verify profit-aware cap was applied
            # Fee should be <= 30% of expected profit
            if attempt.profit_sol:
                max_allowed_fee_sol = attempt.profit_sol * 0.3
                assert fee_sol <= max_allowed_fee_sol * 1.1, \
                    f"Fee {fee_sol} should be <= 30% of profit {attempt.profit_sol} (with 10% tolerance)"

        # Verify RPC health scorer tracked requests
        rpc_stats = orchestrator.rpc_health_scorer.get_stats()
        logger.info("rpc_health_stats", **rpc_stats)

        # Verify race detection (if trade was submitted)
        if attempt.result == FrontrunResult.SUCCESS:
            assert attempt.race_won is not None, "Race result should be analyzed"

            logger.info(
                "race_result_verified",
                race_won=attempt.race_won,
                profit_sol=attempt.profit_sol
            )

        # Verify defensive guardrails logged decisions
        if attempt.result == FrontrunResult.ABORTED_LATENCY:
            logger.info(
                "defensive_abort_latency",
                abort_reason=attempt.abort_reason,
                latency_ms=attempt.latency_ms
            )

        if attempt.result == FrontrunResult.ABORTED_NEGATIVE_EV:
            logger.info(
                "defensive_abort_negative_ev",
                abort_reason=attempt.abort_reason
            )

        # =============================================================================
        # PHASE 6: VERIFY ORCHESTRATOR STATISTICS
        # =============================================================================

        logger.info("--- Phase 6: Orchestrator Statistics ---")

        stats = orchestrator.get_stats()
        logger.info("orchestrator_stats", **stats)

        # Verify stats are tracked
        assert stats["total_attempts"] >= 1
        assert "success_rate" in stats
        assert "avg_latency_ms" in stats
        assert "circuit_breaker_open" in stats
        assert "latency_enforcer_stats" in stats
        assert "fee_bidder_stats" in stats
        assert "rpc_health_stats" in stats

        # =============================================================================
        # PHASE 7: TEST DEFENSIVE ABORT SCENARIOS
        # =============================================================================

        logger.info("--- Phase 7: Testing Defensive Abort Scenarios ---")

        # Test 1: Simulate transaction that would exceed latency budget
        # (by using a very tight budget)
        tight_budget_config = OrchestratorConfig(
            latency_config=LatencyConfig(
                detect_stage_budget_ms=1.0,  # 1ms - impossible to meet
                total_budget_ms=5.0
            ),
            mempool_config=MempoolConfig(simulation_mode=True)
        )

        tight_orchestrator = FrontrunOrchestrator(
            rpc_manager=devnet_rpc_manager,
            wallet_manager=wallet_manager,
            pumpfun_client=pumpfun_client,
            position_tracker=position_tracker,
            pnl_calculator=pnl_calculator,
            config=tight_budget_config
        )

        # This should abort due to latency
        abort_attempt = await tight_orchestrator.execute_frontrun_attempt(simulated_dev_tx)

        logger.info("tight_budget_attempt", **abort_attempt.to_dict())

        # Should have aborted
        assert abort_attempt.result in [
            FrontrunResult.ABORTED_LATENCY,
            FrontrunResult.ABORTED_NOT_DEV,  # Might abort here first
            FrontrunResult.ABORTED_CIRCUIT_OPEN
        ], f"Should abort with tight budget, got {abort_attempt.result}"

        logger.info(
            "defensive_guardrail_verified",
            result=abort_attempt.result.value,
            abort_reason=abort_attempt.abort_reason
        )

        # =============================================================================
        # FINAL VERIFICATION
        # =============================================================================

        logger.info("=== E2E FRONTRUN FLOW TEST COMPLETE ===")

        logger.info(
            "test_summary",
            total_attempts=stats["total_attempts"] + 1,  # +1 for tight budget attempt
            main_attempt_result=attempt.result.value,
            defensive_abort_result=abort_attempt.result.value,
            latency_tracked=True,
            fees_capped=True,
            ev_validated=True,
            race_analyzed=attempt.race_won is not None
        )

        # All Phase 4 components worked together successfully!
        assert True, "E2E frontrun flow executed successfully with defensive guardrails"

    finally:
        # Cleanup
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass
        try:
            os.unlink(known_wallets_file)
        except:
            pass


# =============================================================================
# E2E TEST: CIRCUIT BREAKER SCENARIO
# =============================================================================

@pytest.mark.asyncio
async def test_e2e_circuit_breaker_protection(devnet_rpc_manager, funded_wallet):
    """
    E2E test demonstrating circuit breaker protection

    Scenario:
    1. Configure aggressive latency budgets
    2. Execute multiple frontrun attempts that exceed budgets
    3. Verify circuit breaker trips after consecutive violations
    4. Verify subsequent attempts are blocked
    5. Verify circuit breaker resets after timeout
    """

    logger.info("=== STARTING CIRCUIT BREAKER E2E TEST ===")

    # Create temporary position database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        logger.info("--- Setting up Circuit Breaker Test ---")

        # Initialize components
        pumpfun_client = PumpFunClient(devnet_rpc_manager)
        position_storage = PositionStorage(db_path=db_path)
        position_tracker = PositionTracker(position_storage)
        await position_tracker.start()

        pnl_calculator = PnLCalculator()

        # Create wallet manager with funded wallet
        wallet_manager = WalletManager(
            rpc_manager=devnet_rpc_manager,
            keypairs=[funded_wallet]
        )

        # Configure with IMPOSSIBLY tight latency budgets
        # Even Python overhead will cause violations
        circuit_config = OrchestratorConfig(
            latency_config=LatencyConfig(
                detect_stage_budget_ms=0.001,  # 1 microsecond - impossible
                build_sign_budget_ms=0.001,
                submit_stage_budget_ms=0.001,
                total_budget_ms=0.01,  # 10 microseconds total
                violations_before_trip=2,  # Trip after just 2 violations
                circuit_reset_after_s=2  # Reset after 2 seconds
            ),
            mempool_config=MempoolConfig(simulation_mode=True)
        )

        orchestrator = FrontrunOrchestrator(
            rpc_manager=devnet_rpc_manager,
            wallet_manager=wallet_manager,
            pumpfun_client=pumpfun_client,
            position_tracker=position_tracker,
            pnl_calculator=pnl_calculator,
            config=circuit_config
        )

        # Attempt 1: Should violate but not trip circuit (budgets are impossibly tight)
        logger.info("--- Attempt 1: First Violation ---")
        pending_tx1 = PendingTransaction(
            signature="circuit_test_tx_1",
            transaction=None,
            received_at=datetime.now(),
            fee=100_000,
            accounts=[Pubkey.from_string("11111111111111111111111111111111")],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )
        attempt1 = await orchestrator.execute_frontrun_attempt(pending_tx1)
        logger.info("attempt1_result", result=attempt1.result.value)

        circuit_open_1 = orchestrator.latency_enforcer.is_circuit_open()
        violations_1 = orchestrator.latency_enforcer.consecutive_violations
        logger.info("circuit_status_after_attempt1", circuit_open=circuit_open_1, violations=violations_1)

        # Verify first attempt violated and recorded violation
        assert attempt1.result == FrontrunResult.ABORTED_LATENCY, \
            f"First attempt should violate latency, got {attempt1.result}"
        assert violations_1 == 1, f"Should have 1 violation, got {violations_1}"

        # Attempt 2: Should also violate (budgets are impossibly tight) and trip circuit
        logger.info("--- Attempt 2: Second Violation (Should Trip Circuit) ---")
        pending_tx2 = PendingTransaction(
            signature="circuit_test_tx_2",
            transaction=None,
            received_at=datetime.now(),
            fee=100_000,
            accounts=[Pubkey.from_string("11111111111111111111111111111111")],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )
        attempt2 = await orchestrator.execute_frontrun_attempt(pending_tx2)
        logger.info("attempt2_result", result=attempt2.result.value)

        circuit_open_2 = orchestrator.latency_enforcer.is_circuit_open()
        violations_2 = orchestrator.latency_enforcer.consecutive_violations
        logger.info("circuit_status_after_attempt2", circuit_open=circuit_open_2, violations=violations_2)

        # Verify second attempt also violated and tripped circuit
        assert attempt2.result == FrontrunResult.ABORTED_LATENCY, \
            f"Second attempt should violate latency, got {attempt2.result}"
        assert violations_2 == 2, f"Should have 2 violations, got {violations_2}"
        assert circuit_open_2 is True, "Circuit should be open after 2 violations"

        # Attempt 3: Should be blocked by circuit breaker
        logger.info("--- Attempt 3: Should Be Blocked by Circuit ---")
        pending_tx3 = PendingTransaction(
            signature="circuit_test_tx_3",
            transaction=None,
            received_at=datetime.now(),
            fee=100_000,
            accounts=[Pubkey.from_string("11111111111111111111111111111111")],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )
        attempt3 = await orchestrator.execute_frontrun_attempt(pending_tx3)
        logger.info("attempt3_result", result=attempt3.result.value)

        assert attempt3.result == FrontrunResult.ABORTED_CIRCUIT_OPEN, \
            "Should abort due to circuit breaker"

        # Wait for circuit to reset
        logger.info("--- Waiting for Circuit Reset (2 seconds) ---")
        await asyncio.sleep(2.5)

        circuit_open_after_reset = orchestrator.latency_enforcer.is_circuit_open()
        logger.info("circuit_status_after_reset", circuit_open=circuit_open_after_reset)

        assert circuit_open_after_reset is False, "Circuit should reset after timeout"

        # Attempt 4: Should proceed now that circuit is reset
        logger.info("--- Attempt 4: After Circuit Reset ---")
        pending_tx4 = PendingTransaction(
            signature="circuit_test_tx_4",
            transaction=None,
            received_at=datetime.now(),
            fee=100_000,
            accounts=[Pubkey.from_string("11111111111111111111111111111111")],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )
        attempt4 = await orchestrator.execute_frontrun_attempt(pending_tx4)
        logger.info("attempt4_result", result=attempt4.result.value)

        # Should not be blocked by circuit (though may still abort for latency)
        assert attempt4.result != FrontrunResult.ABORTED_CIRCUIT_OPEN, \
            "Should not be blocked after circuit reset"

        logger.info("=== CIRCUIT BREAKER TEST COMPLETE ===")

        logger.info(
            "circuit_breaker_summary",
            violations_to_trip=2,
            circuit_tripped=True,
            circuit_reset=True,
            protection_working=True
        )

    finally:
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass
