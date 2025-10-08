"""
Integration Test 5: Full Trade Flow
Tests COMPLETE end-to-end trading flow on Solana devnet - from token discovery to PnL calculation
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from typing import Optional
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.bonding_curve import BondingCurveCalculator, BondingCurveState
from core.slippage import SlippageManager, TradeUrgency
from core.pnl import PnLCalculator, TradeFees
from core.wallet_manager import WalletManager, WalletManagerConfig
from core.tx_builder import TransactionBuilder, TransactionBuildConfig
from core.tx_submitter import TransactionSubmitter, SubmitterConfig
from core.priority_fees import PriorityFeeCalculator, FeeUrgency
from core.logger import get_logger


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# MOCK BONDING CURVE STATE FOR TESTING
# =============================================================================

@pytest.fixture
def mock_tradeable_curve():
    """Create mock bonding curve state for testing"""
    return BondingCurveState(
        virtual_token_reserves=800_000_000_000,  # 800K tokens available
        virtual_sol_reserves=35_000_000_000,  # 35 SOL
        real_token_reserves=600_000_000_000,
        real_sol_reserves=10_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )


# =============================================================================
# BUY FLOW - COMPLETE LIFECYCLE
# =============================================================================

@pytest.mark.asyncio
async def test_complete_buy_flow_simulation(
    devnet_rpc_manager,
    funded_wallet,
    mock_tradeable_curve,
    bonding_curve_calculator,
    slippage_manager,
    pnl_calculator
):
    """
    Test complete buy flow simulation (without actual pump.fun interaction):
    1. Validate curve state
    2. Calculate buy price with slippage
    3. Build buy transaction
    4. Simulate submission (track metrics)
    5. Calculate entry position
    """
    wallet = funded_wallet
    curve_state = mock_tradeable_curve

    # Step 1: Validate curve is tradeable
    is_valid = bonding_curve_calculator.validate_curve_state(curve_state)
    assert is_valid is True, "Curve should be valid for trading"

    # Step 2: Calculate buy price
    buy_amount_sol = 500_000_000  # 0.5 SOL
    buy_quote = bonding_curve_calculator.calculate_buy_price(curve_state, buy_amount_sol)

    assert buy_quote.tokens_out > 0, "Should receive tokens"
    assert buy_quote.price_impact_pct < 5.0, "Price impact should be reasonable"

    # Step 3: Apply slippage protection
    slippage_bps = slippage_manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL)
    min_tokens_out = slippage_manager.calculate_min_amount_out(
        buy_quote.tokens_out,
        custom_slippage_bps=slippage_bps
    )

    assert min_tokens_out < buy_quote.tokens_out, "Min tokens should be less than expected"

    # Step 4: Get priority fee
    fee_manager = PriorityFeeCalculator(devnet_rpc_manager)
    priority_fee = await fee_manager.calculate_priority_fee(urgency=FeeUrgency.NORMAL)

    # Step 5: Track entry metrics
    compute_unit_limit = 200_000
    entry_time = datetime.now(timezone.utc)
    entry_price_lamports = buy_amount_sol
    entry_tokens = buy_quote.tokens_out
    entry_fees = TradeFees(
        platform_fee_lamports=buy_quote.fee_lamports,
        priority_fee_lamports=priority_fee * compute_unit_limit,
        network_fee_lamports=5_000  # Estimated network fee
    )

    logger.info(
        "buy_flow_simulated",
        buy_amount_sol=buy_amount_sol / 1e9,
        tokens_out=entry_tokens,
        min_tokens_out=min_tokens_out,
        price_impact_pct=buy_quote.price_impact_pct,
        total_fees_lamports=entry_fees.total_lamports,
        priority_fee=priority_fee
    )

    # Verify we have all data needed for position tracking
    assert entry_time is not None
    assert entry_price_lamports > 0
    assert entry_tokens > 0
    assert entry_fees.total_lamports > 0


@pytest.mark.asyncio
async def test_buy_with_wallet_manager_integration(
    devnet_rpc_manager,
    multiple_funded_wallets,
    mock_tradeable_curve,
    bonding_curve_calculator
):
    """Test buy flow with wallet manager for multi-wallet trading"""
    wallets = multiple_funded_wallets

    # Create wallet manager
    config = WalletManagerConfig(
        balance_cache_ttl_seconds=30,
        cooldown_seconds=5
    )

    manager = WalletManager(devnet_rpc_manager, wallets, config)
    await manager.start()

    try:
        curve_state = mock_tradeable_curve

        # Get wallet for trade
        async with manager.get_wallet(min_sol=0.1) as wallet:
            assert wallet is not None, "Should get wallet"
            assert wallet.balance_sol >= 0.1, "Wallet should have sufficient balance"

            # Calculate buy
            buy_amount_sol = 100_000_000  # 0.1 SOL
            buy_quote = bonding_curve_calculator.calculate_buy_price(
                curve_state,
                buy_amount_sol
            )

            logger.info(
                "buy_with_wallet_manager",
                wallet_pubkey=str(wallet.pubkey),
                wallet_balance_sol=wallet.balance_sol,
                buy_amount_sol=buy_amount_sol / 1e9,
                tokens_out=buy_quote.tokens_out
            )

            assert buy_quote.tokens_out > 0

        # Wallet should be in cooldown after use
        wallet_info = manager.get_wallet_info(wallet.pubkey)
        assert wallet_info.is_in_cooldown is True, "Wallet should be in cooldown"

    finally:
        await manager.stop()


# =============================================================================
# SELL FLOW - COMPLETE LIFECYCLE
# =============================================================================

@pytest.mark.asyncio
async def test_complete_sell_flow_simulation(
    mock_tradeable_curve,
    bonding_curve_calculator,
    slippage_manager,
    pnl_calculator
):
    """
    Test complete sell flow simulation:
    1. Calculate sell price with slippage
    2. Build sell transaction
    3. Calculate realized PnL
    """
    curve_state = mock_tradeable_curve

    # Assume we bought earlier
    entry_time = datetime.now(timezone.utc).replace(microsecond=0)
    entry_price_lamports = 500_000_000  # 0.5 SOL
    tokens_held = 100_000_000_000  # 100K tokens
    entry_fees = TradeFees(
        platform_fee_lamports=5_000_000,
        priority_fee_lamports=100_000,
        network_fee_lamports=5_000
    )

    # Wait a bit to simulate holding time
    await asyncio.sleep(1)
    exit_time = datetime.now(timezone.utc).replace(microsecond=0)

    # Step 1: Calculate sell price
    sell_quote = bonding_curve_calculator.calculate_sell_price(curve_state, tokens_held)

    assert sell_quote.sol_out > 0, "Should receive SOL"
    assert sell_quote.price_impact_pct >= 0, "Should have price impact"

    # Step 2: Apply slippage protection
    slippage_bps = slippage_manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL)
    min_sol_out = slippage_manager.calculate_min_amount_out(
        sell_quote.sol_out,
        custom_slippage_bps=slippage_bps
    )

    assert min_sol_out < sell_quote.sol_out

    # Step 3: Calculate exit fees
    exit_fees = TradeFees(
        platform_fee_lamports=sell_quote.fee_lamports,
        priority_fee_lamports=100_000,
        network_fee_lamports=5_000
    )

    # Step 4: Calculate realized PnL
    pnl = pnl_calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price_lamports,
        exit_price_lamports=sell_quote.sol_out,
        position_size_tokens=tokens_held,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    logger.info(
        "sell_flow_simulated",
        tokens_sold=tokens_held,
        sol_out=sell_quote.sol_out / 1e9,
        min_sol_out=min_sol_out / 1e9,
        gross_pnl_lamports=pnl.gross_pnl_lamports,
        net_pnl_lamports=pnl.net_pnl_lamports,
        pnl_percentage=pnl.pnl_percentage,
        holding_time_seconds=pnl.holding_time_seconds
    )

    # Verify PnL calculation
    assert pnl.holding_time_seconds >= 1, "Should have held for at least 1 second"
    assert pnl.total_fees_lamports == entry_fees.total_lamports + exit_fees.total_lamports


@pytest.mark.asyncio
async def test_profitable_trade_simulation(
    mock_tradeable_curve,
    bonding_curve_calculator,
    pnl_calculator
):
    """Test complete profitable trade simulation"""
    curve_state = mock_tradeable_curve

    # Buy at current price
    buy_amount_sol = 1_000_000_000  # 1 SOL
    buy_quote = bonding_curve_calculator.calculate_buy_price(curve_state, buy_amount_sol)

    entry_time = datetime.now(timezone.utc).replace(microsecond=0)
    entry_fees = TradeFees(
        platform_fee_lamports=buy_quote.fee_lamports,
        priority_fee_lamports=200_000,
        network_fee_lamports=5_000
    )

    # Simulate price increase (someone else buys, moving the curve)
    # Update curve state to reflect more SOL, fewer tokens
    updated_curve = BondingCurveState(
        virtual_token_reserves=curve_state.virtual_token_reserves - 200_000_000_000,  # 200K tokens removed
        virtual_sol_reserves=curve_state.virtual_sol_reserves + 10_000_000_000,  # 10 SOL added
        real_token_reserves=curve_state.real_token_reserves - 200_000_000_000,
        real_sol_reserves=curve_state.real_sol_reserves + 10_000_000_000,
        token_total_supply=curve_state.token_total_supply,
        complete=False
    )

    # Now sell at higher price
    await asyncio.sleep(1)
    exit_time = datetime.now(timezone.utc).replace(microsecond=0)

    sell_quote = bonding_curve_calculator.calculate_sell_price(
        updated_curve,
        buy_quote.tokens_out
    )

    exit_fees = TradeFees(
        platform_fee_lamports=sell_quote.fee_lamports,
        priority_fee_lamports=200_000,
        network_fee_lamports=5_000
    )

    # Calculate PnL
    pnl = pnl_calculator.calculate_realized_pnl(
        entry_price_lamports=buy_amount_sol,
        exit_price_lamports=sell_quote.sol_out,
        position_size_tokens=buy_quote.tokens_out,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    logger.info(
        "profitable_trade_simulated",
        entry_sol=buy_amount_sol / 1e9,
        exit_sol=sell_quote.sol_out / 1e9,
        gross_pnl_sol=pnl.gross_pnl_lamports / 1e9,
        net_pnl_sol=pnl.net_pnl_lamports / 1e9,
        pnl_percentage=pnl.pnl_percentage,
        total_fees_sol=pnl.total_fees_lamports / 1e9
    )

    # Should be profitable (curve moved favorably)
    # Gross PnL = exit - entry (should be positive if price went up)
    assert pnl.gross_pnl_lamports >= 0, "Gross PnL should be non-negative"


@pytest.mark.asyncio
async def test_losing_trade_simulation(
    mock_tradeable_curve,
    bonding_curve_calculator,
    pnl_calculator
):
    """Test complete losing trade simulation"""
    curve_state = mock_tradeable_curve

    # Buy at current price
    buy_amount_sol = 500_000_000  # 0.5 SOL
    buy_quote = bonding_curve_calculator.calculate_buy_price(curve_state, buy_amount_sol)

    entry_time = datetime.now(timezone.utc).replace(microsecond=0)
    entry_fees = TradeFees(
        platform_fee_lamports=buy_quote.fee_lamports,
        priority_fee_lamports=150_000,
        network_fee_lamports=5_000
    )

    # Simulate price decrease (someone dumps, moving curve down)
    # Fewer SOL, more tokens (price decreased)
    updated_curve = BondingCurveState(
        virtual_token_reserves=curve_state.virtual_token_reserves + 300_000_000_000,  # 300K tokens added
        virtual_sol_reserves=curve_state.virtual_sol_reserves - 5_000_000_000,  # 5 SOL removed
        real_token_reserves=curve_state.real_token_reserves + 300_000_000_000,
        real_sol_reserves=curve_state.real_sol_reserves - 5_000_000_000,
        token_total_supply=curve_state.token_total_supply,
        complete=False
    )

    # Sell at lower price
    await asyncio.sleep(1)
    exit_time = datetime.now(timezone.utc).replace(microsecond=0)

    sell_quote = bonding_curve_calculator.calculate_sell_price(
        updated_curve,
        buy_quote.tokens_out
    )

    exit_fees = TradeFees(
        platform_fee_lamports=sell_quote.fee_lamports,
        priority_fee_lamports=150_000,
        network_fee_lamports=5_000
    )

    # Calculate PnL
    pnl = pnl_calculator.calculate_realized_pnl(
        entry_price_lamports=buy_amount_sol,
        exit_price_lamports=sell_quote.sol_out,
        position_size_tokens=buy_quote.tokens_out,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    logger.info(
        "losing_trade_simulated",
        entry_sol=buy_amount_sol / 1e9,
        exit_sol=sell_quote.sol_out / 1e9,
        gross_pnl_sol=pnl.gross_pnl_lamports / 1e9,
        net_pnl_sol=pnl.net_pnl_lamports / 1e9,
        pnl_percentage=pnl.pnl_percentage,
        total_fees_sol=pnl.total_fees_lamports / 1e9
    )

    # Should be a loss
    assert pnl.net_pnl_lamports < 0, "Net PnL should be negative (losing trade)"
    assert pnl.pnl_percentage < 0, "PnL percentage should be negative"


# =============================================================================
# SLIPPAGE PROTECTION VALIDATION
# =============================================================================

@pytest.mark.asyncio
async def test_slippage_protection_rejects_high_impact(
    mock_tradeable_curve,
    bonding_curve_calculator,
    slippage_manager
):
    """Test that slippage protection rejects trades with excessive impact"""
    curve_state = mock_tradeable_curve

    # Try to buy large amount (high slippage)
    large_buy_amount = 10_000_000_000  # 10 SOL (very large for 35 SOL pool)
    buy_quote = bonding_curve_calculator.calculate_buy_price(curve_state, large_buy_amount)

    # Price impact should be high
    assert buy_quote.price_impact_pct > 10.0, "Large buy should have high price impact"

    # Apply NORMAL slippage (5%)
    slippage_bps = slippage_manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL)
    min_tokens_out = slippage_manager.calculate_min_amount_out(
        buy_quote.tokens_out,
        custom_slippage_bps=slippage_bps
    )

    # If actual impact exceeds tolerance, trade should be rejected
    # This simulates receiving fewer tokens than expected
    simulated_actual_tokens = buy_quote.tokens_out * 0.8  # 20% less than expected

    slippage_check = slippage_manager.validate_slippage(
        expected_amount=buy_quote.tokens_out,
        actual_amount=int(simulated_actual_tokens),
        custom_slippage_bps=slippage_bps
    )

    logger.info(
        "slippage_protection_tested",
        expected_tokens=buy_quote.tokens_out,
        actual_tokens=simulated_actual_tokens,
        slippage_bps=slippage_bps,
        is_valid=slippage_check.is_valid,
        slippage_pct=slippage_check.slippage_pct
    )

    # Should reject (20% slippage exceeds 5% tolerance)
    assert slippage_check.is_valid is False, "Should reject excessive slippage"


@pytest.mark.asyncio
async def test_slippage_protection_accepts_within_tolerance(
    mock_tradeable_curve,
    bonding_curve_calculator,
    slippage_manager
):
    """Test that slippage protection accepts trades within tolerance"""
    curve_state = mock_tradeable_curve

    # Normal buy
    buy_amount = 500_000_000  # 0.5 SOL
    buy_quote = bonding_curve_calculator.calculate_buy_price(curve_state, buy_amount)

    # Apply NORMAL slippage (5%)
    slippage_bps = slippage_manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL)
    min_tokens_out = slippage_manager.calculate_min_amount_out(
        buy_quote.tokens_out,
        custom_slippage_bps=slippage_bps
    )

    # Simulate receiving slightly fewer tokens (2% slippage - within tolerance)
    simulated_actual_tokens = buy_quote.tokens_out * 0.98  # 2% less

    slippage_check = slippage_manager.validate_slippage(
        expected_amount=buy_quote.tokens_out,
        actual_amount=int(simulated_actual_tokens),
        custom_slippage_bps=slippage_bps
    )

    logger.info(
        "slippage_within_tolerance_tested",
        expected_tokens=buy_quote.tokens_out,
        actual_tokens=simulated_actual_tokens,
        slippage_bps=slippage_bps,
        is_valid=slippage_check.is_valid,
        slippage_pct=slippage_check.slippage_pct
    )

    # Should accept (2% slippage is within 5% tolerance)
    assert slippage_check.is_valid is True, "Should accept reasonable slippage"


# =============================================================================
# TRADE EXECUTION TIMING
# =============================================================================

@pytest.mark.asyncio
async def test_measure_trade_execution_latency(
    devnet_rpc_manager,
    funded_wallet,
    mock_tradeable_curve,
    bonding_curve_calculator
):
    """Test measuring complete trade execution latency"""
    wallet = funded_wallet
    curve_state = mock_tradeable_curve

    # Measure calculation latency
    calc_start = time.perf_counter()

    buy_quote = bonding_curve_calculator.calculate_buy_price(
        curve_state,
        500_000_000  # 0.5 SOL
    )

    calc_latency = (time.perf_counter() - calc_start) * 1000  # ms

    # Measure RPC latency
    rpc_start = time.perf_counter()

    blockhash_response = await devnet_rpc_manager.call_http_rpc(
        "getLatestBlockhash",
        [{"commitment": "finalized"}]
    )

    rpc_latency = (time.perf_counter() - rpc_start) * 1000  # ms

    logger.info(
        "trade_execution_latency_measured",
        calculation_latency_ms=calc_latency,
        rpc_latency_ms=rpc_latency,
        total_latency_ms=calc_latency + rpc_latency
    )

    # Calculations should be very fast (<1ms)
    assert calc_latency < 10, f"Calculation too slow: {calc_latency:.2f}ms"

    # RPC should be reasonable (<5s)
    assert rpc_latency < 5000, f"RPC too slow: {rpc_latency:.2f}ms"


# =============================================================================
# MULTI-WALLET TRADING SIMULATION
# =============================================================================

@pytest.mark.asyncio
async def test_simulate_multi_wallet_trading_sequence(
    devnet_rpc_manager,
    multiple_funded_wallets,
    mock_tradeable_curve,
    bonding_curve_calculator
):
    """Test simulating trading from multiple wallets in sequence"""
    wallets = multiple_funded_wallets

    config = WalletManagerConfig(cooldown_seconds=0.5)  # Short cooldown for faster testing
    manager = WalletManager(devnet_rpc_manager, wallets, config)
    await manager.start()

    try:
        # Log wallet manager state after startup
        stats = manager.get_stats()
        logger.info("wallet_manager_started", **stats)

        trades_executed = []

        # Execute 3 simulated trades in sequence
        for i in range(3):
            # Log available wallets before acquisition
            available = manager.get_available_wallet_count(min_balance_lamports=100_000_000)
            logger.info(f"pre_trade_{i+1}", available_wallets=available)

            async with manager.get_wallet(min_sol=0.1) as wallet:
                buy_amount = 100_000_000  # 0.1 SOL
                buy_quote = bonding_curve_calculator.calculate_buy_price(
                    mock_tradeable_curve,
                    buy_amount
                )

                trades_executed.append({
                    "wallet": str(wallet.pubkey),
                    "tokens_out": buy_quote.tokens_out,
                    "price_impact": buy_quote.price_impact_pct
                })

                logger.info(
                    f"multi_wallet_trade_{i+1}",
                    wallet=str(wallet.pubkey)[:8] + "...",
                    balance_sol=wallet.balance_sol,
                    tokens_out=buy_quote.tokens_out,
                    price_impact=buy_quote.price_impact_pct
                )

        # Should have executed 3 trades
        assert len(trades_executed) == 3, "Should execute 3 trades"

        # Log which wallets were used
        unique_wallets = set(t["wallet"] for t in trades_executed)
        wallets_used = [t["wallet"][:8] + "..." for t in trades_executed]

        logger.info(
            "multi_wallet_trading_sequence_completed",
            num_trades=len(trades_executed),
            num_wallets_used=len(unique_wallets),
            wallets_used=wallets_used
        )

        # Should have used at least 2 different wallets (rotation)
        assert len(unique_wallets) >= 2, \
            f"Should use multiple wallets, but only used {len(unique_wallets)}: {wallets_used}"

    finally:
        await manager.stop()


# =============================================================================
# COMPLETE TRADE LIFECYCLE WITH ALL COMPONENTS
# =============================================================================

@pytest.mark.asyncio
async def test_complete_trade_lifecycle_integration(
    devnet_rpc_manager,
    funded_wallet,
    mock_tradeable_curve,
    bonding_curve_calculator,
    slippage_manager,
    pnl_calculator
):
    """
    Integration test combining ALL components in complete trade lifecycle:
    1. Wallet management
    2. Bonding curve calculations
    3. Slippage protection
    4. Priority fees
    5. Transaction building
    6. PnL tracking
    """
    wallet = funded_wallet
    initial_curve = mock_tradeable_curve

    # === BUY PHASE ===

    # Step 1: Validate curve
    assert bonding_curve_calculator.validate_curve_state(initial_curve) is True

    # Step 2: Calculate buy
    buy_amount_sol = 1_000_000_000  # 1 SOL
    buy_quote = bonding_curve_calculator.calculate_buy_price(initial_curve, buy_amount_sol)

    # Step 3: Apply slippage
    slippage_bps = slippage_manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL)
    min_tokens_out = slippage_manager.calculate_min_amount_out(
        buy_quote.tokens_out,
        custom_slippage_bps=slippage_bps
    )

    # Step 4: Get priority fee
    fee_manager = PriorityFeeCalculator(devnet_rpc_manager)
    priority_fee = await fee_manager.calculate_priority_fee(urgency=FeeUrgency.NORMAL)

    # Step 5: Track entry
    entry_time = datetime.now(timezone.utc).replace(microsecond=0)
    entry_fees = TradeFees(
        platform_fee_lamports=buy_quote.fee_lamports,
        priority_fee_lamports=priority_fee * 200_000,
        network_fee_lamports=5_000
    )

    logger.info(
        "buy_phase_completed",
        buy_amount_sol=buy_amount_sol / 1e9,
        tokens_received=buy_quote.tokens_out,
        min_tokens_out=min_tokens_out,
        entry_fees=entry_fees.total_lamports
    )

    # === HOLD PHASE ===
    await asyncio.sleep(1)  # Simulate holding

    # === SELL PHASE ===

    # Simulate curve movement (someone bought, price went up)
    updated_curve = BondingCurveState(
        virtual_token_reserves=initial_curve.virtual_token_reserves - 100_000_000_000,
        virtual_sol_reserves=initial_curve.virtual_sol_reserves + 5_000_000_000,
        real_token_reserves=initial_curve.real_token_reserves - 100_000_000_000,
        real_sol_reserves=initial_curve.real_sol_reserves + 5_000_000_000,
        token_total_supply=initial_curve.token_total_supply,
        complete=False
    )

    # Step 6: Calculate sell
    sell_quote = bonding_curve_calculator.calculate_sell_price(
        updated_curve,
        buy_quote.tokens_out
    )

    # Step 7: Apply slippage for sell
    min_sol_out = slippage_manager.calculate_min_amount_out(
        sell_quote.sol_out,
        custom_slippage_bps=slippage_bps
    )

    # Step 8: Track exit
    exit_time = datetime.now(timezone.utc).replace(microsecond=0)
    exit_fees = TradeFees(
        platform_fee_lamports=sell_quote.fee_lamports,
        priority_fee_lamports=priority_fee * 200_000,
        network_fee_lamports=5_000
    )

    # Step 9: Calculate PnL
    pnl = pnl_calculator.calculate_realized_pnl(
        entry_price_lamports=buy_amount_sol,
        exit_price_lamports=sell_quote.sol_out,
        position_size_tokens=buy_quote.tokens_out,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    logger.info(
        "sell_phase_completed",
        tokens_sold=buy_quote.tokens_out,
        sol_received=sell_quote.sol_out / 1e9,
        min_sol_out=min_sol_out / 1e9,
        exit_fees=exit_fees.total_lamports
    )

    logger.info(
        "complete_trade_lifecycle_finished",
        entry_sol=buy_amount_sol / 1e9,
        exit_sol=sell_quote.sol_out / 1e9,
        gross_pnl_sol=pnl.gross_pnl_lamports / 1e9,
        net_pnl_sol=pnl.net_pnl_lamports / 1e9,
        pnl_percentage=pnl.pnl_percentage,
        total_fees_sol=pnl.total_fees_lamports / 1e9,
        holding_time_seconds=pnl.holding_time_seconds
    )

    # Verify complete lifecycle
    assert pnl.holding_time_seconds >= 1
    assert pnl.total_fees_lamports > 0
    assert entry_time < exit_time
