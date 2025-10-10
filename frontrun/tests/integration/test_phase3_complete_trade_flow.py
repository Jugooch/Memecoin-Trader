"""
Phase 3 Integration Test: Complete Trade Flow
Tests the full buy → position tracking → sell flow using all Phase 3 components

This test validates:
- Pump.fun instruction encoding (buy/sell)
- Bonding curve calculations
- Position tracking with PnL
- Transaction building (simulated)

NOTE: This test requires a funded devnet wallet. It will skip if:
- DEVNET_TEST_WALLET_PRIVATE_KEY is not set
- Wallet has insufficient balance
"""

import pytest
import asyncio
import tempfile
import os
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from clients.pumpfun_client import PumpFunClient
from core.bonding_curve import BondingCurveCalculator, BondingCurveState
from core.slippage import SlippageManager
from core.pnl import PnLCalculator
from core.position_tracker import PositionTracker, PositionStorage
from core.logger import get_logger


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# COMPLETE TRADE FLOW TEST
# =============================================================================

@pytest.mark.asyncio
async def test_complete_buy_sell_flow_simulated(
    devnet_rpc_manager,
    funded_wallet,
    bonding_curve_calculator,
    slippage_manager,
    pnl_calculator
):
    """
    Test complete trade flow from buy → position tracking → sell (SIMULATED)

    This test doesn't submit real transactions (to conserve SOL), but validates
    that all Phase 3 components work together correctly.
    """
    # =============================================================================
    # SETUP
    # =============================================================================

    # Create a temporary position database
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Initialize components
        pumpfun_client = PumpFunClient(devnet_rpc_manager)
        position_storage = PositionStorage(db_path=db_path)
        position_tracker = PositionTracker(position_storage)
        await position_tracker.start()

        # Use funded wallet
        wallet_keypair = funded_wallet
        wallet_pubkey = wallet_keypair.pubkey()

        # Example mint (this would be a real token in production)
        # For this test, we'll use a placeholder mint
        example_mint = Pubkey.from_string("So11111111111111111111111111111111111111112")

        logger.info(
            "test_setup_complete",
            wallet=str(wallet_pubkey),
            mint=str(example_mint)
        )

        # =============================================================================
        # PHASE 1: BUY INSTRUCTION
        # =============================================================================

        buy_amount_sol = 0.1  # 100M lamports
        buy_amount_lamports = int(buy_amount_sol * 1e9)
        max_slippage_bps = 500  # 5%

        logger.info("building_buy_instruction", amount_sol=buy_amount_sol)

        # Build buy instruction
        buy_instruction = await pumpfun_client.build_buy_instruction(
            mint=example_mint,
            user=wallet_pubkey,
            amount_sol=buy_amount_lamports,
            max_slippage_bps=max_slippage_bps
        )

        # Verify buy instruction was built correctly
        assert buy_instruction is not None, "Buy instruction should be created"
        assert buy_instruction.program_id == pumpfun_client.PROGRAM_ID

        logger.info(
            "buy_instruction_built",
            accounts=len(buy_instruction.accounts),
            data_size=len(buy_instruction.data)
        )

        # =============================================================================
        # PHASE 2: BONDING CURVE CALCULATION (SIMULATED)
        # =============================================================================

        # Simulate bonding curve state (in production, this would be fetched from chain)
        simulated_curve = BondingCurveState(
            virtual_token_reserves=1_000_000_000_000,  # 1T tokens
            virtual_sol_reserves=30_000_000_000,  # 30 SOL
            real_token_reserves=800_000_000_000,
            real_sol_reserves=5_000_000_000,
            token_total_supply=1_000_000_000_000,
            complete=False
        )

        logger.info(
            "simulated_bonding_curve",
            virtual_sol=simulated_curve.virtual_sol_reserves / 1e9,
            virtual_tokens=simulated_curve.virtual_token_reserves
        )

        # Calculate expected tokens out
        buy_quote = bonding_curve_calculator.calculate_buy_price(
            simulated_curve,
            buy_amount_lamports
        )

        assert buy_quote.tokens_out > 0, "Should receive tokens"
        assert buy_quote.price_impact_pct < 10, "Price impact should be reasonable (<10%)"

        logger.info(
            "buy_quote_calculated",
            tokens_out=buy_quote.tokens_out,
            price_per_token=buy_quote.price_per_token_sol,
            price_impact_pct=buy_quote.price_impact_pct
        )

        # Validate slippage
        simulated_actual_tokens = buy_quote.tokens_out * 0.98  # 2% slippage

        slippage_check = slippage_manager.validate_slippage(
            expected_amount=buy_quote.tokens_out,
            actual_amount=int(simulated_actual_tokens),
            custom_slippage_bps=max_slippage_bps
        )

        assert slippage_check.is_valid, "Slippage should be valid"

        logger.info(
            "slippage_validated",
            expected=buy_quote.tokens_out,
            actual=int(simulated_actual_tokens),
            slippage_pct=slippage_check.slippage_pct
        )

        # =============================================================================
        # PHASE 3: OPEN POSITION
        # =============================================================================

        entry_price_sol = buy_amount_sol  # Simplified: total SOL spent
        entry_slot = 12345  # Would be from transaction confirmation
        tokens_bought = int(simulated_actual_tokens)

        logger.info("opening_position", tokens=tokens_bought)

        position = await position_tracker.open_position(
            wallet=wallet_pubkey,
            mint=example_mint,
            amount_tokens=tokens_bought,
            entry_price_sol=entry_price_sol,
            entry_slot=entry_slot,
            strategy="frontrun"
        )

        assert position is not None, "Position should be created"
        assert position.amount_tokens == tokens_bought
        assert position.entry_price_sol == entry_price_sol

        logger.info(
            "position_opened",
            position_id=position.position_id,
            tokens=position.amount_tokens,
            entry_price=position.entry_price_sol
        )

        # Verify position is tracked
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        assert len(open_positions) == 1
        assert open_positions[0].position_id == position.position_id

        # =============================================================================
        # PHASE 4: SELL INSTRUCTION
        # =============================================================================

        # Simulate some time passing and price movement
        await asyncio.sleep(0.5)

        # Sell 100% of position
        sell_amount_tokens = position.amount_tokens
        min_sol_output = int(buy_amount_lamports * 0.9)  # Accept 10% slippage

        logger.info("building_sell_instruction", tokens=sell_amount_tokens)

        # Build sell instruction
        sell_instruction = await pumpfun_client.build_sell_instruction(
            mint=example_mint,
            user=wallet_pubkey,
            amount_tokens=sell_amount_tokens,
            min_sol_output=min_sol_output
        )

        # Verify sell instruction was built correctly
        assert sell_instruction is not None, "Sell instruction should be created"
        assert sell_instruction.program_id == pumpfun_client.PROGRAM_ID

        logger.info(
            "sell_instruction_built",
            accounts=len(sell_instruction.accounts),
            data_size=len(sell_instruction.data)
        )

        # =============================================================================
        # PHASE 5: BONDING CURVE CALCULATION FOR SELL (SIMULATED)
        # =============================================================================

        # Simulate updated bonding curve state after our buy
        simulated_curve_after_buy = BondingCurveState(
            virtual_token_reserves=simulated_curve.virtual_token_reserves - tokens_bought,
            virtual_sol_reserves=simulated_curve.virtual_sol_reserves + buy_amount_lamports,
            real_token_reserves=simulated_curve.real_token_reserves,
            real_sol_reserves=simulated_curve.real_sol_reserves,
            token_total_supply=simulated_curve.token_total_supply,
            complete=False
        )

        # Calculate expected SOL out
        sell_quote = bonding_curve_calculator.calculate_sell_price(
            simulated_curve_after_buy,
            sell_amount_tokens
        )

        assert sell_quote.sol_out > 0, "Should receive SOL"

        logger.info(
            "sell_quote_calculated",
            sol_out=sell_quote.sol_out / 1e9,
            price_per_token=sell_quote.price_per_token_sol,
            price_impact_pct=sell_quote.price_impact_pct
        )

        # =============================================================================
        # PHASE 6: CLOSE POSITION AND CALCULATE PNL
        # =============================================================================

        exit_price_sol = sell_quote.sol_out / 1e9
        exit_slot = entry_slot + 100  # Simulated slot progression

        logger.info("closing_position", exit_price_sol=exit_price_sol)

        closed_position = await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=exit_price_sol,
            exit_slot=exit_slot
        )

        assert closed_position is not None, "Position should be closed"
        assert closed_position.pnl_sol is not None, "PnL should be calculated"
        assert closed_position.pnl_pct is not None, "PnL % should be calculated"
        assert closed_position.holding_time_seconds >= 0, "Holding time should be tracked"

        logger.info(
            "position_closed",
            position_id=closed_position.position_id,
            pnl_sol=closed_position.pnl_sol,
            pnl_pct=closed_position.pnl_pct,
            holding_time_seconds=closed_position.holding_time_seconds
        )

        # Verify PnL makes sense
        # If we sold for more than we bought, PnL should be positive
        # If we sold for less than we bought, PnL should be negative
        if exit_price_sol > entry_price_sol:
            assert closed_position.pnl_sol > 0, "Profitable trade should have positive PnL"
        elif exit_price_sol < entry_price_sol:
            assert closed_position.pnl_sol < 0, "Losing trade should have negative PnL"

        # Verify position is now closed
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        assert len(open_positions) == 0, "Should have no open positions"

        closed_positions = await position_tracker.get_closed_positions(wallet=wallet_pubkey)
        assert len(closed_positions) == 1, "Should have one closed position"
        assert closed_positions[0].position_id == position.position_id

        # =============================================================================
        # FINAL VERIFICATION
        # =============================================================================

        logger.info(
            "trade_flow_complete",
            entry_price=entry_price_sol,
            exit_price=exit_price_sol,
            pnl_sol=closed_position.pnl_sol,
            pnl_pct=closed_position.pnl_pct,
            holding_time=closed_position.holding_time_seconds
        )

        # All Phase 3 components worked together successfully!
        assert True, "Complete trade flow executed successfully"

    finally:
        # Cleanup
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass


# =============================================================================
# MULTI-POSITION TRADE FLOW TEST
# =============================================================================

@pytest.mark.asyncio
async def test_multiple_positions_trade_flow(
    devnet_rpc_manager,
    funded_wallet,
    bonding_curve_calculator
):
    """
    Test managing multiple positions across different tokens

    Validates:
    - Opening multiple positions simultaneously
    - Tracking positions independently
    - Closing positions in different order
    - PnL calculation for each position
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

        wallet_pubkey = funded_wallet.pubkey()

        # Simulate trading 3 different tokens
        mints = [
            Pubkey.from_string("So11111111111111111111111111111111111111112"),
            Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T"),
            Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
        ]

        positions = []

        # Open 3 positions
        for i, mint in enumerate(mints):
            # Build buy instruction
            buy_instruction = await pumpfun_client.build_buy_instruction(
                mint=mint,
                user=wallet_pubkey,
                amount_sol=100_000_000,  # 0.1 SOL
                max_slippage_bps=500
            )

            assert buy_instruction is not None

            # Open position
            position = await position_tracker.open_position(
                wallet=wallet_pubkey,
                mint=mint,
                amount_tokens=1_000_000_000 * (i + 1),
                entry_price_sol=0.1,
                entry_slot=12345 + i,
                strategy="frontrun"
            )

            positions.append(position)

            logger.info(
                f"position_{i+1}_opened",
                mint=str(mint),
                tokens=position.amount_tokens
            )

        # Verify all positions are open
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        assert len(open_positions) == 3

        # Close positions in reverse order with different P/Ls
        exit_prices = [0.15, 0.08, 0.12]  # +50%, -20%, +20%

        for i, (position, exit_price) in enumerate(zip(reversed(positions), exit_prices)):
            # Build sell instruction
            sell_instruction = await pumpfun_client.build_sell_instruction(
                mint=position.mint,
                user=wallet_pubkey,
                amount_tokens=position.amount_tokens,
                min_sol_output=50_000_000  # 0.05 SOL min
            )

            assert sell_instruction is not None

            # Close position
            closed_position = await position_tracker.close_position(
                position_id=position.position_id,
                exit_price_sol=exit_price,
                exit_slot=12400 + i
            )

            logger.info(
                f"position_{i+1}_closed",
                pnl_sol=closed_position.pnl_sol,
                pnl_pct=closed_position.pnl_pct
            )

        # Verify all positions are closed
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        closed_positions = await position_tracker.get_closed_positions(wallet=wallet_pubkey)

        assert len(open_positions) == 0
        assert len(closed_positions) == 3

        # Verify PnL variety (some wins, some losses)
        pnls = [p.pnl_pct for p in closed_positions]
        assert any(pnl > 0 for pnl in pnls), "Should have at least one profitable trade"
        assert any(pnl < 0 for pnl in pnls), "Should have at least one losing trade"

        logger.info(
            "multi_position_flow_complete",
            total_positions=len(closed_positions),
            pnls=pnls
        )

    finally:
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass


# =============================================================================
# PARTIAL EXIT TRADE FLOW TEST
# =============================================================================

@pytest.mark.asyncio
async def test_partial_exit_trade_flow(
    devnet_rpc_manager,
    funded_wallet,
    bonding_curve_calculator
):
    """
    Test partial position exits

    Validates:
    - Opening a position
    - Selling partial amount (update position size)
    - Selling remaining amount (close position)
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

        wallet_pubkey = funded_wallet.pubkey()
        mint = Pubkey.from_string("So11111111111111111111111111111111111111112")

        # Open position with 1B tokens
        initial_tokens = 1_000_000_000

        position = await position_tracker.open_position(
            wallet=wallet_pubkey,
            mint=mint,
            amount_tokens=initial_tokens,
            entry_price_sol=0.1,
            entry_slot=12345,
            strategy="frontrun"
        )

        logger.info("initial_position_opened", tokens=initial_tokens)

        # Partial exit: sell 40%
        partial_sell_amount = int(initial_tokens * 0.4)
        remaining_amount = initial_tokens - partial_sell_amount

        # Build sell instruction for partial amount
        partial_sell_ix = await pumpfun_client.build_sell_instruction(
            mint=mint,
            user=wallet_pubkey,
            amount_tokens=partial_sell_amount,
            min_sol_output=30_000_000  # 0.03 SOL min
        )

        assert partial_sell_ix is not None

        # Update position size
        updated_position = await position_tracker.update_position_size(
            position_id=position.position_id,
            new_amount=remaining_amount
        )

        assert updated_position.amount_tokens == remaining_amount

        logger.info(
            "partial_exit_complete",
            sold=partial_sell_amount,
            remaining=remaining_amount
        )

        # Position should still be open
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        assert len(open_positions) == 1

        # Final exit: sell remaining 60%
        final_sell_ix = await pumpfun_client.build_sell_instruction(
            mint=mint,
            user=wallet_pubkey,
            amount_tokens=remaining_amount,
            min_sol_output=50_000_000  # 0.05 SOL min
        )

        assert final_sell_ix is not None

        # Close position
        closed_position = await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.12,  # +20% on remaining position
            exit_slot=12350
        )

        logger.info(
            "final_exit_complete",
            pnl_sol=closed_position.pnl_sol,
            pnl_pct=closed_position.pnl_pct
        )

        # Position should be closed
        open_positions = await position_tracker.get_open_positions(wallet=wallet_pubkey)
        assert len(open_positions) == 0

    finally:
        await position_tracker.stop()
        try:
            os.unlink(db_path)
        except:
            pass
