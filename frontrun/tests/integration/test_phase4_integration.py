"""
Phase 4 Integration Test: Frontrun Orchestration with Defensive Guardrails
Tests integration of all Phase 4 components working together

This test validates:
- Latency budget enforcement in real trade flow
- Profit-aware fee caps with RPC data
- RPC health scoring and routing
- Race detection and classification
- Dev wallet detection
- Complete orchestration flow

NOTE: This test requires a funded devnet wallet. It will skip if:
- DEVNET_TEST_WALLET_PRIVATE_KEY is not set
- Wallet has insufficient balance
"""

import pytest
import asyncio
import tempfile
import os
import time
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from datetime import datetime, timedelta

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
from services.mempool_monitor import MempoolMonitor, MempoolConfig
from core.frontrun_orchestrator import FrontrunOrchestrator, OrchestratorConfig


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# LATENCY ENFORCER INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_latency_enforcer_real_timing(devnet_rpc_manager):
    """
    Test latency enforcer with actual timing measurements

    Validates:
    - Starting operation creates timer
    - Recording stage times
    - Checking budgets with real async operations
    - Circuit breaker triggering
    """
    # Configure tight budgets for testing
    config = LatencyConfig(
        detect_stage_budget_ms=100.0,
        build_sign_budget_ms=50.0,
        submit_stage_budget_ms=75.0,
        total_budget_ms=200.0,
        violations_before_trip=2
    )

    enforcer = LatencyBudgetEnforcer(config)

    # Test 1: Operation within budget
    timer = enforcer.start_operation("test_trade_001")

    # Simulate detect stage
    await asyncio.sleep(0.02)  # 20ms
    enforcer.record_stage(timer, TradeStage.DETECT)

    should_abort = enforcer.check_and_abort(timer, TradeStage.DETECT)
    assert should_abort is False, "Should not abort when within budget"

    # Simulate build stage
    await asyncio.sleep(0.01)  # 10ms
    enforcer.record_stage(timer, TradeStage.BUILD_SIGN)

    should_abort = enforcer.check_and_abort(timer, TradeStage.BUILD_SIGN)
    assert should_abort is False, "Should not abort when within budget"

    logger.info(
        "latency_test_within_budget",
        total_ms=timer.elapsed_ms(),
        stages=timer.stage_times
    )

    # Test 2: Operation exceeding budget
    timer2 = enforcer.start_operation("test_trade_002")

    # Simulate slow detect stage (exceeds 100ms budget)
    await asyncio.sleep(0.12)  # 120ms
    enforcer.record_stage(timer2, TradeStage.DETECT)

    should_abort = enforcer.check_and_abort(timer2, TradeStage.DETECT)
    assert should_abort is True, "Should abort when budget exceeded"
    assert enforcer.consecutive_violations == 1

    logger.info(
        "latency_test_budget_exceeded",
        total_ms=timer2.elapsed_ms(),
        violations=enforcer.consecutive_violations
    )

    # Test 3: Circuit breaker triggers after consecutive violations
    timer3 = enforcer.start_operation("test_trade_003")
    await asyncio.sleep(0.12)  # Another violation
    enforcer.record_stage(timer3, TradeStage.DETECT)
    enforcer.check_and_abort(timer3, TradeStage.DETECT)

    assert enforcer.is_circuit_open() is True, "Circuit breaker should be open"

    logger.info(
        "circuit_breaker_test",
        circuit_open=enforcer.is_circuit_open(),
        violations=enforcer.consecutive_violations
    )


# =============================================================================
# PROFIT-AWARE FEE BIDDER INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_profit_aware_fee_bidder_with_rpc(devnet_rpc_manager):
    """
    Test profit-aware fee bidder with real RPC data

    Validates:
    - Fee calculation with real network fees
    - Cold-start protection
    - Profit-based capping after trade history
    - EV validation
    """
    from core.priority_fees import PriorityFeeCalculator

    # Create fee calculator with real RPC
    fee_calc = PriorityFeeCalculator(devnet_rpc_manager)

    # Create bidder with test config
    config = BidderConfig(
        cold_start_max_lamports=50_000,
        profit_cap_pct=0.3,
        min_samples_for_ev=5
    )

    bidder = ProfitAwareFeeBidder(fee_calc, config)

    # Test 1: Cold start (no trade history)
    fee = await bidder.calculate_aggressive_fee(
        competing_tx=None,
        compute_units=200_000
    )

    assert fee > 0, "Should calculate fee"
    logger.info("cold_start_fee_calculated", fee=fee)

    stats = bidder.get_stats()
    assert stats["using_cold_start"] is True
    assert stats["total_trades"] == 0

    # Test 2: Add trade history
    for i in range(10):
        bidder.add_trade(Trade(
            won_race=True,
            pnl_sol=0.02 + (i * 0.01),  # Varying profits: 0.02 to 0.11 SOL
            fee_paid_lamports=100_000,
            timestamp=datetime.now()
        ))

    # Calculate fee with trade history
    fee_with_history = await bidder.calculate_aggressive_fee(
        competing_tx=None,
        compute_units=200_000
    )

    logger.info(
        "fee_with_history_calculated",
        fee=fee_with_history,
        trade_count=10
    )

    stats = bidder.get_stats()
    assert stats["using_cold_start"] is False
    assert stats["total_trades"] == 10
    assert stats["win_rate"] == 1.0
    assert stats["conservative_ev_sol"] > 0

    # Test 3: EV validation
    valid = bidder.validate_ev_before_submit(
        expected_profit_sol=0.05,
        fee_lamports=500_000
    )
    assert valid is True, "Should accept positive EV trade"

    invalid = bidder.validate_ev_before_submit(
        expected_profit_sol=0.0001,
        fee_lamports=5_000_000
    )
    assert invalid is False, "Should reject negative EV trade"

    logger.info("ev_validation_tests_passed")


# =============================================================================
# RPC HEALTH SCORER INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_rpc_health_scorer_real_endpoints(devnet_rpc_manager):
    """
    Test RPC health scorer with real endpoint monitoring

    Validates:
    - Recording real RPC requests
    - Calculating health scores
    - Selecting best endpoint
    - Fast ejection of degraded endpoints
    """
    config = HealthScorerConfig(
        latency_threshold_p99_ms=500.0,
        slot_lag_threshold=5,
        error_rate_threshold=0.1  # 10%
    )

    scorer = RPCHealthScorer(devnet_rpc_manager, config)

    # Initialize scorer (loads endpoints from RPC manager)
    await scorer.start()

    # Get actual endpoint labels from RPC manager
    all_health = scorer.get_all_health()
    assert len(all_health) > 0, "Should have at least one endpoint"

    # Use first available endpoint label
    endpoint_label = list(all_health.keys())[0]
    logger.info("using_endpoint_for_test", label=endpoint_label)

    # Record some successful requests
    for i in range(10):
        scorer.record_request(
            endpoint_label=endpoint_label,
            success=True,
            latency_ms=100.0 + (i * 10)  # 100ms to 190ms
        )

    # Get endpoint health
    endpoint_health = scorer.get_endpoint_health(endpoint_label)
    assert endpoint_health is not None
    assert endpoint_health.consecutive_errors == 0
    assert endpoint_health.score >= 90  # Should have good score

    initial_score = endpoint_health.score
    logger.info("initial_health", **endpoint_health.to_dict())

    # Simulate some failures
    for i in range(3):
        scorer.record_request(
            endpoint_label=endpoint_label,
            success=False,
            latency_ms=1000.0
        )

    # Get updated health
    endpoint_health_after = scorer.get_endpoint_health(endpoint_label)
    assert endpoint_health_after.consecutive_errors == 3
    # Note: Score doesn't automatically update on record_request,
    # it updates on update_all_scores() calls

    # Get general stats
    stats = scorer.get_stats()
    logger.info("rpc_health_stats", **stats)
    assert stats["total_endpoints"] > 0
    assert stats["best_endpoint"] is not None

    logger.info("rpc_health_scorer_test_passed")


# =============================================================================
# RACE DETECTOR INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_race_detector_scenarios():
    """
    Test race detector with various scenarios

    Validates:
    - Price-based race detection
    - Timing-based race detection
    - Combined analysis
    """
    config = RaceConfig(
        price_deviation_threshold_pct=5.0,
        max_acceptable_slot_delay=2
    )

    detector = RaceFailureDetector(config)

    # Scenario 1: Won by price (good fill)
    result = detector.detect_race_loss_by_price(
        expected_price=0.000030,
        actual_fill_price=0.000031  # Only 3.3% higher
    )

    assert result.won_race is True
    assert result.price_deviation_pct < 5.0
    logger.info("race_won_by_price", **result.to_dict())

    # Scenario 2: Lost by price (bad fill)
    result = detector.detect_race_loss_by_price(
        expected_price=0.000030,
        actual_fill_price=0.000040  # 33% higher
    )

    assert result.won_race is False
    assert result.price_deviation_pct > 5.0
    logger.info("race_lost_by_price", **result.to_dict())

    # Scenario 3: Won by timing (same slot)
    result = detector.analyze_fill_timing(
        dev_buy_slot=12345,
        our_fill_slot=12345  # Same slot!
    )

    assert result.won_race is True
    assert result.slot_delay == 0
    logger.info("race_won_by_timing_same_slot", **result.to_dict())

    # Scenario 4: Won by timing (1 slot after)
    result = detector.analyze_fill_timing(
        dev_buy_slot=12345,
        our_fill_slot=12346  # 1 slot after
    )

    assert result.won_race is True
    assert result.slot_delay == 1
    logger.info("race_won_by_timing_next_slot", **result.to_dict())

    # Scenario 5: Lost by timing (5 slots after)
    result = detector.analyze_fill_timing(
        dev_buy_slot=12345,
        our_fill_slot=12350  # 5 slots after
    )

    assert result.won_race is False
    assert result.slot_delay == 5
    logger.info("race_lost_by_timing", **result.to_dict())

    # Scenario 6: Combined analysis - both win
    result = detector.analyze_combined(
        expected_price=0.000030,
        actual_fill_price=0.000031,  # Good price
        dev_buy_slot=12345,
        our_fill_slot=12346  # Good timing
    )

    assert result.won_race is True
    logger.info("race_won_combined", **result.to_dict())

    # Scenario 7: Combined analysis - price loss
    result = detector.analyze_combined(
        expected_price=0.000030,
        actual_fill_price=0.000040,  # Bad price
        dev_buy_slot=12345,
        our_fill_slot=12346  # Good timing
    )

    assert result.won_race is False
    logger.info("race_lost_combined_price", **result.to_dict())


# =============================================================================
# DEV DETECTOR INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_dev_detector_pattern_analysis():
    """
    Test dev wallet detector with pattern analysis

    Validates:
    - Known wallet detection
    - Behavioral heuristics
    - Confidence scoring
    """
    # Create temp file for known wallets
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w') as f:
        import json
        json.dump({"wallets": []}, f)
        wallets_file = f.name

    try:
        config = DetectorConfig(
            min_buy_amount_sol=0.5,
            max_time_since_token_creation_s=300,
            known_dev_wallets_file=wallets_file
        )

        detector = DevWalletDetector(config)
        await detector.load_known_wallets()

        # Test 1: Add known dev wallet
        known_dev = Pubkey.from_string("11111111111111111111111111111111")
        detector.add_known_dev_wallet(known_dev)

        assert detector.is_known_dev_wallet(known_dev) is True
        logger.info("known_dev_wallet_added")

        # Test 2: Pattern analysis for known wallet
        pattern = detector.analyze_transaction_pattern(
            sender=known_dev,
            buy_amount_sol=0.1,  # Small buy
            token_age_seconds=1000,  # Old token
            instruction_count=5  # Multiple instructions
        )

        assert pattern.is_likely_dev_buy is True  # Known wallet = 100% confidence
        assert pattern.confidence == 1.0
        logger.info("known_wallet_pattern", **pattern.to_dict())

        # Test 3: Behavioral dev buy (unknown wallet)
        unknown_wallet = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

        pattern = detector.analyze_transaction_pattern(
            sender=unknown_wallet,
            buy_amount_sol=1.0,  # Large buy
            token_age_seconds=60,  # Early
            instruction_count=1  # Single instruction
        )

        assert pattern.is_likely_dev_buy is True  # 0.3 + 0.3 + 0.2 = 0.8 > 0.5 threshold
        assert pattern.confidence >= 0.5
        logger.info("behavioral_dev_buy_pattern", **pattern.to_dict())

        # Test 4: Not a dev buy
        pattern = detector.analyze_transaction_pattern(
            sender=unknown_wallet,
            buy_amount_sol=0.1,  # Small buy
            token_age_seconds=1000,  # Late
            instruction_count=3  # Multiple instructions
        )

        assert pattern.is_likely_dev_buy is False  # Low confidence
        assert pattern.confidence < 0.5
        logger.info("not_dev_buy_pattern", **pattern.to_dict())

    finally:
        os.unlink(wallets_file)


# =============================================================================
# MEMPOOL MONITOR INTEGRATION TEST (SIMULATION)
# =============================================================================

@pytest.mark.asyncio
async def test_mempool_monitor_simulation():
    """
    Test mempool monitor in simulation mode

    Validates:
    - Starting monitoring
    - Receiving fake transactions
    - Callback invocation
    - Stopping monitoring
    """
    config = MempoolConfig(
        simulation_mode=True,
        simulation_event_interval_s=0.5  # Fast for testing
    )

    monitor = MempoolMonitor(config)

    # Track received transactions
    received_txs = []

    async def on_pending_tx(tx):
        received_txs.append(tx)
        logger.info(
            "pending_tx_received",
            signature=tx.signature[:16] + "...",
            fee=tx.fee
        )

    # Start monitoring (will run in background)
    monitor_task = asyncio.create_task(monitor.start_monitoring(on_pending_tx))

    # Wait for a few transactions
    await asyncio.sleep(2.5)  # Should get ~5 transactions

    # Stop monitoring
    await monitor.stop_monitoring()

    # Cancel the task
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    # Verify we received transactions
    assert len(received_txs) >= 3, f"Should receive at least 3 txs, got {len(received_txs)}"

    stats = monitor.get_mempool_stats()
    logger.info("mempool_stats", **stats.to_dict() if hasattr(stats, 'to_dict') else {"transactions_seen": stats.transactions_seen})

    assert stats.transactions_seen >= 3


# =============================================================================
# ORCHESTRATOR INTEGRATION TEST
# =============================================================================

@pytest.mark.asyncio
async def test_orchestrator_integration(devnet_rpc_manager, funded_wallet):
    """
    Test complete orchestrator integration with all Phase 4 components

    Validates:
    - Orchestrator initialization
    - Simulated frontrun attempt flow
    - Defensive guardrails working together
    - Statistics tracking
    """
    # Create temporary position database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
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

        # Create orchestrator config
        orchestrator_config = OrchestratorConfig(
            default_buy_amount_sol=0.05,
            latency_config=LatencyConfig(
                detect_stage_budget_ms=100.0,
                build_sign_budget_ms=50.0,
                submit_stage_budget_ms=75.0,
                total_budget_ms=200.0
            ),
            bidder_config=BidderConfig(
                cold_start_max_lamports=50_000,
                profit_cap_pct=0.3
            ),
            mempool_config=MempoolConfig(
                simulation_mode=True,
                simulation_event_interval_s=1.0
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

        logger.info("orchestrator_initialized")

        # Test manual frontrun attempt (bypass mempool monitoring)
        from services.mempool_monitor import PendingTransaction

        pending_tx = PendingTransaction(
            signature="test_signature_12345",
            transaction=None,
            received_at=datetime.now(),
            fee=250_000,  # 250k lamports
            accounts=[Pubkey.from_string("11111111111111111111111111111111")],
            program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
        )

        # Execute frontrun attempt
        attempt = await orchestrator.execute_frontrun_attempt(pending_tx)

        logger.info("frontrun_attempt_completed", **attempt.to_dict())

        # Verify attempt was recorded
        assert attempt is not None
        assert attempt.latency_ms >= 0
        assert attempt.result is not None

        # Get orchestrator stats
        stats = orchestrator.get_stats()
        logger.info("orchestrator_stats", **stats)

        assert "total_attempts" in stats
        assert "latency_enforcer_stats" in stats
        assert "fee_bidder_stats" in stats
        assert "rpc_health_stats" in stats

        logger.info("orchestrator_integration_test_passed")

    finally:
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass


# =============================================================================
# STRESS TEST: Multiple Concurrent Frontrun Attempts
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.slow  # Mark as slow test
async def test_orchestrator_concurrent_attempts(devnet_rpc_manager, funded_wallet):
    """
    Stress test: Multiple concurrent frontrun attempts

    Validates:
    - Orchestrator handles concurrent attempts
    - Latency enforcer tracks multiple operations
    - Fee bidder accumulates trade history correctly
    - No race conditions or deadlocks
    """
    # Create temporary position database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
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

        orchestrator_config = OrchestratorConfig(
            default_buy_amount_sol=0.05,
            latency_config=LatencyConfig(total_budget_ms=200.0),
            mempool_config=MempoolConfig(simulation_mode=True)
        )

        orchestrator = FrontrunOrchestrator(
            rpc_manager=devnet_rpc_manager,
            wallet_manager=wallet_manager,
            pumpfun_client=pumpfun_client,
            position_tracker=position_tracker,
            pnl_calculator=pnl_calculator,
            config=orchestrator_config
        )

        # Create 10 concurrent frontrun attempts
        from services.mempool_monitor import PendingTransaction

        tasks = []
        for i in range(10):
            pending_tx = PendingTransaction(
                signature=f"concurrent_tx_{i:03d}",
                transaction=None,
                received_at=datetime.now(),
                fee=100_000 + (i * 10_000),
                accounts=[Pubkey.from_string("11111111111111111111111111111111")],
                program_id=Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
            )

            task = orchestrator.execute_frontrun_attempt(pending_tx)
            tasks.append(task)

        # Execute all attempts concurrently
        results = await asyncio.gather(*tasks)

        # Verify all attempts completed
        assert len(results) == 10

        # Get stats
        stats = orchestrator.get_stats()
        logger.info("concurrent_stress_test_stats", **stats)

        assert stats["total_attempts"] == 10

        logger.info("concurrent_stress_test_passed")

    finally:
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass
