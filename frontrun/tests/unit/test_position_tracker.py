"""
Unit tests for Position Tracker
Tests position management, PnL calculation, and SQLite persistence
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from solders.pubkey import Pubkey

from core.position_tracker import (
    PositionTracker,
    Position,
    ClosedPosition,
    PositionStatus,
    PositionStorage,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest_asyncio.fixture
async def position_tracker(temp_db_path):
    """Create PositionTracker with temporary database"""
    storage = PositionStorage(db_path=temp_db_path)
    tracker = PositionTracker(storage)
    await tracker.start()
    yield tracker
    await tracker.stop()


@pytest.fixture
def example_wallet():
    """Example wallet pubkey"""
    return Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T")


@pytest.fixture
def example_mint():
    """Example mint pubkey"""
    return Pubkey.from_string("So11111111111111111111111111111111111111112")


# =============================================================================
# POSITION OPENING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_open_position(position_tracker, example_wallet, example_mint):
    """Test opening a new position"""
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Verify position properties
    assert isinstance(position, Position)
    assert position.wallet == example_wallet
    assert position.mint == example_mint
    assert position.amount_tokens == 1_000_000_000
    assert position.entry_price_sol == 0.5
    assert position.entry_slot == 12345
    assert position.strategy == "frontrun"
    assert position.status == PositionStatus.OPEN

    # Should have a position ID
    assert len(position.position_id) > 0

    # Should have timestamp
    assert isinstance(position.entry_timestamp, datetime)


@pytest.mark.asyncio
async def test_open_multiple_positions(position_tracker, example_wallet, example_mint):
    """Test opening multiple positions"""
    positions = []

    for i in range(3):
        position = await position_tracker.open_position(
            wallet=example_wallet,
            mint=example_mint,
            amount_tokens=1_000_000_000 * (i + 1),
            entry_price_sol=0.5 + (i * 0.1),
            entry_slot=12345 + i,
            strategy="frontrun"
        )
        positions.append(position)

    # All should have unique IDs
    position_ids = [p.position_id for p in positions]
    assert len(set(position_ids)) == 3, "All positions should have unique IDs"

    # All should be OPEN
    assert all(p.status == PositionStatus.OPEN for p in positions)


@pytest.mark.asyncio
async def test_open_position_different_strategies(position_tracker, example_wallet, example_mint):
    """Test opening positions with different strategies"""
    frontrun_position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    copytrading_position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=2_000_000_000,
        entry_price_sol=0.6,
        entry_slot=12346,
        strategy="copytrading"
    )

    assert frontrun_position.strategy == "frontrun"
    assert copytrading_position.strategy == "copytrading"


# =============================================================================
# POSITION CLOSING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_close_position_with_profit(position_tracker, example_wallet, example_mint):
    """Test closing a position with profit"""
    # Open position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Wait a moment so holding time > 0
    await asyncio.sleep(0.1)

    # Close position with profit
    exit_price = 0.75  # +50% profit
    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=exit_price,
        exit_slot=12450
    )

    # Verify closed position
    assert isinstance(closed_position, ClosedPosition)
    assert closed_position.position_id == position.position_id
    assert closed_position.status == PositionStatus.CLOSED
    assert closed_position.exit_price_sol == exit_price
    assert closed_position.exit_slot == 12450

    # Verify PnL calculation
    expected_pnl_sol = exit_price - position.entry_price_sol  # 0.25
    expected_pnl_pct = (expected_pnl_sol / position.entry_price_sol) * 100  # 50%

    assert closed_position.pnl_sol == pytest.approx(expected_pnl_sol, rel=1e-9)
    assert closed_position.pnl_pct == pytest.approx(expected_pnl_pct, rel=1e-9)

    # Should have holding time
    assert closed_position.holding_time_seconds >= 0


@pytest.mark.asyncio
async def test_close_position_with_loss(position_tracker, example_wallet, example_mint):
    """Test closing a position with loss"""
    # Open position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Close position with loss
    exit_price = 0.3  # -40% loss
    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=exit_price,
        exit_slot=12350
    )

    # Verify PnL calculation (negative)
    expected_pnl_sol = exit_price - position.entry_price_sol  # -0.2
    expected_pnl_pct = (expected_pnl_sol / position.entry_price_sol) * 100  # -40%

    assert closed_position.pnl_sol == pytest.approx(expected_pnl_sol, rel=1e-9)
    assert closed_position.pnl_pct == pytest.approx(expected_pnl_pct, rel=1e-9)
    assert closed_position.pnl_sol < 0, "Should have negative PnL"


@pytest.mark.asyncio
async def test_close_position_breakeven(position_tracker, example_wallet, example_mint):
    """Test closing a position at breakeven"""
    # Open position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Close at same price
    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=0.5,  # Same as entry
        exit_slot=12350
    )

    # Should be breakeven
    assert closed_position.pnl_sol == pytest.approx(0.0, abs=1e-9)
    assert closed_position.pnl_pct == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_close_nonexistent_position(position_tracker):
    """Test closing a position that doesn't exist"""
    with pytest.raises(ValueError, match="Position .* not found"):
        await position_tracker.close_position(
            position_id="nonexistent-id",
            exit_price_sol=0.5,
            exit_slot=12345
        )


@pytest.mark.asyncio
async def test_close_already_closed_position(position_tracker, example_wallet, example_mint):
    """Test closing a position that's already closed"""
    # Open and close position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=0.6,
        exit_slot=12350
    )

    # Try to close again
    with pytest.raises(ValueError, match="Position .* already closed"):
        await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.7,
            exit_slot=12355
        )


# =============================================================================
# POSITION QUERYING TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_get_open_positions(position_tracker, example_wallet, example_mint):
    """Test querying open positions"""
    # Open multiple positions
    position1 = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    position2 = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=2_000_000_000,
        entry_price_sol=0.6,
        entry_slot=12346,
        strategy="copytrading"
    )

    # Close one position
    await position_tracker.close_position(
        position_id=position1.position_id,
        exit_price_sol=0.7,
        exit_slot=12350
    )

    # Get open positions
    open_positions = await position_tracker.get_open_positions()

    # Should only have position2
    assert len(open_positions) == 1
    assert open_positions[0].position_id == position2.position_id
    assert open_positions[0].status == PositionStatus.OPEN


@pytest.mark.asyncio
async def test_get_open_positions_by_wallet(position_tracker, example_mint):
    """Test querying open positions filtered by wallet"""
    wallet1 = Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T")
    wallet2 = Pubkey.from_string("So11111111111111111111111111111111111111112")

    # Open positions for different wallets
    await position_tracker.open_position(
        wallet=wallet1,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    await position_tracker.open_position(
        wallet=wallet2,
        mint=example_mint,
        amount_tokens=2_000_000_000,
        entry_price_sol=0.6,
        entry_slot=12346,
        strategy="frontrun"
    )

    # Query wallet1's positions
    wallet1_positions = await position_tracker.get_open_positions(wallet=wallet1)

    assert len(wallet1_positions) == 1
    assert wallet1_positions[0].wallet == wallet1


@pytest.mark.asyncio
async def test_get_open_positions_by_mint(position_tracker, example_wallet):
    """Test querying open positions filtered by mint"""
    mint1 = Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T")
    mint2 = Pubkey.from_string("So11111111111111111111111111111111111111112")

    # Open positions for different mints
    await position_tracker.open_position(
        wallet=example_wallet,
        mint=mint1,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    await position_tracker.open_position(
        wallet=example_wallet,
        mint=mint2,
        amount_tokens=2_000_000_000,
        entry_price_sol=0.6,
        entry_slot=12346,
        strategy="frontrun"
    )

    # Query mint1's positions
    mint1_positions = await position_tracker.get_open_positions(mint=mint1)

    assert len(mint1_positions) == 1
    assert mint1_positions[0].mint == mint1


@pytest.mark.asyncio
async def test_get_closed_positions(position_tracker, example_wallet, example_mint):
    """Test querying closed positions"""
    # Open and close multiple positions
    for i in range(3):
        position = await position_tracker.open_position(
            wallet=example_wallet,
            mint=example_mint,
            amount_tokens=1_000_000_000 * (i + 1),
            entry_price_sol=0.5 + (i * 0.1),
            entry_slot=12345 + i,
            strategy="frontrun"
        )

        await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.6 + (i * 0.1),
            exit_slot=12350 + i
        )

    # Get closed positions
    closed_positions = await position_tracker.get_closed_positions()

    assert len(closed_positions) == 3
    assert all(p.status == PositionStatus.CLOSED for p in closed_positions)
    assert all(isinstance(p, ClosedPosition) for p in closed_positions)


@pytest.mark.asyncio
async def test_get_closed_positions_with_limit(position_tracker, example_wallet, example_mint):
    """Test querying closed positions with limit"""
    # Open and close 5 positions
    for i in range(5):
        position = await position_tracker.open_position(
            wallet=example_wallet,
            mint=example_mint,
            amount_tokens=1_000_000_000,
            entry_price_sol=0.5,
            entry_slot=12345 + i,
            strategy="frontrun"
        )

        await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.6,
            exit_slot=12350 + i
        )

    # Get only 2 most recent
    closed_positions = await position_tracker.get_closed_positions(limit=2)

    assert len(closed_positions) == 2


@pytest.mark.asyncio
async def test_get_closed_positions_by_wallet(position_tracker, example_mint):
    """Test querying closed positions filtered by wallet"""
    wallet1 = Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T")
    wallet2 = Pubkey.from_string("So11111111111111111111111111111111111111112")

    # Open and close positions for different wallets
    for wallet in [wallet1, wallet2]:
        position = await position_tracker.open_position(
            wallet=wallet,
            mint=example_mint,
            amount_tokens=1_000_000_000,
            entry_price_sol=0.5,
            entry_slot=12345,
            strategy="frontrun"
        )

        await position_tracker.close_position(
            position_id=position.position_id,
            exit_price_sol=0.6,
            exit_slot=12350
        )

    # Query wallet1's closed positions
    wallet1_closed = await position_tracker.get_closed_positions(wallet=wallet1)

    assert len(wallet1_closed) == 1
    assert wallet1_closed[0].wallet == wallet1


# =============================================================================
# POSITION UPDATE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_update_position_size(position_tracker, example_wallet, example_mint):
    """Test updating position size (partial exit)"""
    # Open position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Reduce position size by 50%
    new_amount = 500_000_000
    updated_position = await position_tracker.update_position_size(
        position_id=position.position_id,
        new_amount=new_amount
    )

    assert updated_position.amount_tokens == new_amount
    assert updated_position.status == PositionStatus.OPEN


@pytest.mark.asyncio
async def test_update_position_size_to_zero(position_tracker, example_wallet, example_mint):
    """Test updating position size to zero (should close position)"""
    # Open position
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Update to zero
    updated_position = await position_tracker.update_position_size(
        position_id=position.position_id,
        new_amount=0
    )

    assert updated_position.amount_tokens == 0
    # Note: This doesn't change status to CLOSED, just reduces amount
    # Actual closing requires calling close_position with exit price


@pytest.mark.asyncio
async def test_update_nonexistent_position(position_tracker):
    """Test updating a position that doesn't exist"""
    with pytest.raises(ValueError, match="Position .* not found"):
        await position_tracker.update_position_size(
            position_id="nonexistent-id",
            new_amount=500_000_000
        )


# =============================================================================
# PNL CALCULATION EDGE CASES
# =============================================================================

@pytest.mark.asyncio
async def test_pnl_calculation_100x_gain(position_tracker, example_wallet, example_mint):
    """Test PnL calculation with 100x gain"""
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.01,
        entry_slot=12345,
        strategy="frontrun"
    )

    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=1.0,  # 100x gain
        exit_slot=12350
    )

    expected_pnl_pct = ((1.0 - 0.01) / 0.01) * 100  # 9900%
    assert closed_position.pnl_pct == pytest.approx(expected_pnl_pct, rel=1e-6)


@pytest.mark.asyncio
async def test_pnl_calculation_total_loss(position_tracker, example_wallet, example_mint):
    """Test PnL calculation with total loss (exit price = 0)"""
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=0.0,  # Total loss
        exit_slot=12350
    )

    assert closed_position.pnl_sol == -0.5
    assert closed_position.pnl_pct == -100.0


# =============================================================================
# HOLDING TIME TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_holding_time_calculation(position_tracker, example_wallet, example_mint):
    """Test that holding time is calculated correctly"""
    position = await position_tracker.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    # Wait 2 seconds
    await asyncio.sleep(2)

    closed_position = await position_tracker.close_position(
        position_id=position.position_id,
        exit_price_sol=0.6,
        exit_slot=12350
    )

    # Should be at least 2 seconds
    assert closed_position.holding_time_seconds >= 2


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_positions_persist_across_restarts(temp_db_path, example_wallet, example_mint):
    """Test that positions are persisted to database"""
    # Create tracker and open position
    storage1 = PositionStorage(db_path=temp_db_path)
    tracker1 = PositionTracker(storage1)
    await tracker1.start()

    position = await tracker1.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )
    position_id = position.position_id

    await tracker1.stop()

    # Create new tracker with same database
    storage2 = PositionStorage(db_path=temp_db_path)
    tracker2 = PositionTracker(storage2)
    await tracker2.start()

    # Should be able to retrieve position
    open_positions = await tracker2.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].position_id == position_id

    await tracker2.stop()


@pytest.mark.asyncio
async def test_closed_positions_persist(temp_db_path, example_wallet, example_mint):
    """Test that closed positions persist across restarts"""
    # Create tracker, open and close position
    storage1 = PositionStorage(db_path=temp_db_path)
    tracker1 = PositionTracker(storage1)
    await tracker1.start()

    position = await tracker1.open_position(
        wallet=example_wallet,
        mint=example_mint,
        amount_tokens=1_000_000_000,
        entry_price_sol=0.5,
        entry_slot=12345,
        strategy="frontrun"
    )

    closed_position = await tracker1.close_position(
        position_id=position.position_id,
        exit_price_sol=0.6,
        exit_slot=12350
    )
    expected_pnl = closed_position.pnl_sol

    await tracker1.stop()

    # Create new tracker with same database
    storage2 = PositionStorage(db_path=temp_db_path)
    tracker2 = PositionTracker(storage2)
    await tracker2.start()

    # Should be able to retrieve closed position
    closed_positions = await tracker2.get_closed_positions()
    assert len(closed_positions) == 1
    assert closed_positions[0].pnl_sol == pytest.approx(expected_pnl, rel=1e-9)

    await tracker2.stop()


# =============================================================================
# MULTIPLE POSITIONS PER MINT TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_multiple_positions_same_mint(position_tracker, example_wallet, example_mint):
    """Test opening multiple positions for the same mint"""
    # Open 3 positions on same mint
    positions = []
    for i in range(3):
        position = await position_tracker.open_position(
            wallet=example_wallet,
            mint=example_mint,
            amount_tokens=1_000_000_000,
            entry_price_sol=0.5,
            entry_slot=12345 + i,
            strategy="frontrun"
        )
        positions.append(position)

    # All should be tracked independently
    open_positions = await position_tracker.get_open_positions(mint=example_mint)
    assert len(open_positions) == 3

    # Close one
    await position_tracker.close_position(
        position_id=positions[0].position_id,
        exit_price_sol=0.6,
        exit_slot=12350
    )

    # Should have 2 open, 1 closed
    open_positions = await position_tracker.get_open_positions(mint=example_mint)
    closed_positions = await position_tracker.get_closed_positions()

    assert len(open_positions) == 2
    assert len(closed_positions) == 1


# =============================================================================
# DATABASE ERROR HANDLING
# =============================================================================

@pytest.mark.asyncio
async def test_database_not_started():
    """Test operations fail gracefully if database not started"""
    storage = PositionStorage(db_path=":memory:")
    tracker = PositionTracker(storage)
    # Don't call start()

    with pytest.raises(Exception):  # Should fail because storage not initialized
        await tracker.open_position(
            wallet=Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T"),
            mint=Pubkey.from_string("So11111111111111111111111111111111111111112"),
            amount_tokens=1_000_000_000,
            entry_price_sol=0.5,
            entry_slot=12345,
            strategy="frontrun"
        )
