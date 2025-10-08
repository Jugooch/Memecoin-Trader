"""
Unit tests for PnL Calculator
Tests profit/loss calculations, fee accounting, and edge cases
"""

import pytest
from datetime import datetime, timedelta, timezone
from core.pnl import (
    PnLCalculator,
    TradeFees,
    RealizedPnL,
    UnrealizedPnL,
    LAMPORTS_PER_SOL
)


@pytest.fixture
def calculator():
    """Create PnL calculator instance"""
    return PnLCalculator()


@pytest.fixture
def entry_fees():
    """Standard entry fees"""
    return TradeFees(
        platform_fee_lamports=10_000_000,  # 0.01 SOL
        priority_fee_lamports=500_000  # 0.0005 SOL
    )


@pytest.fixture
def exit_fees():
    """Standard exit fees"""
    return TradeFees(
        platform_fee_lamports=15_000_000,  # 0.015 SOL
        priority_fee_lamports=500_000  # 0.0005 SOL
    )


@pytest.fixture
def entry_time():
    """Standard entry time"""
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def exit_time():
    """Standard exit time (10 seconds after entry)"""
    return datetime(2025, 1, 1, 12, 0, 10, tzinfo=timezone.utc)


# =============================================================================
# TRADE FEES
# =============================================================================

def test_trade_fees_total_lamports():
    """Test total fees calculation in lamports"""
    fees = TradeFees(
        platform_fee_lamports=10_000_000,
        priority_fee_lamports=500_000,
        network_fee_lamports=5_000
    )

    assert fees.total_lamports == 10_505_000


def test_trade_fees_total_sol():
    """Test total fees calculation in SOL"""
    fees = TradeFees(
        platform_fee_lamports=10_000_000,
        priority_fee_lamports=500_000,
        network_fee_lamports=5_000
    )

    expected_sol = 10_505_000 / LAMPORTS_PER_SOL
    assert abs(fees.total_sol - expected_sol) < 1e-9


def test_trade_fees_default_network_fee():
    """Test default network fee"""
    fees = TradeFees(
        platform_fee_lamports=10_000_000,
        priority_fee_lamports=500_000
    )

    assert fees.network_fee_lamports == 5_000


# =============================================================================
# REALIZED PNL - PROFITABLE TRADES
# =============================================================================

def test_calculate_realized_pnl_profitable(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test realized PnL for profitable trade"""
    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        exit_price_lamports=1_500_000_000,  # 1.5 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    assert isinstance(pnl, RealizedPnL)
    assert pnl.entry_price_lamports == 1_000_000_000
    assert pnl.exit_price_lamports == 1_500_000_000
    assert pnl.position_size_tokens == 1_000_000

    # Gross PnL: 1.5 - 1.0 = 0.5 SOL
    assert pnl.gross_pnl_lamports == 500_000_000

    # Total fees: entry + exit
    expected_total_fees = entry_fees.total_lamports + exit_fees.total_lamports
    assert pnl.total_fees_lamports == expected_total_fees

    # Net PnL: gross - fees
    assert pnl.net_pnl_lamports == 500_000_000 - expected_total_fees
    assert pnl.is_profitable is True
    assert pnl.holding_time_seconds == 10


def test_calculate_realized_pnl_large_profit(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test realized PnL with large profit (10x) - verify exact percentage calculation"""
    entry_price = 1_000_000_000  # 1 SOL
    exit_price = 10_000_000_000  # 10 SOL

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Verify gross PnL
    assert pnl.gross_pnl_lamports == 9_000_000_000  # 9 SOL gross

    # Verify net PnL = gross - fees
    total_fees = entry_fees.total_lamports + exit_fees.total_lamports
    expected_net = 9_000_000_000 - total_fees
    assert pnl.net_pnl_lamports == expected_net

    # Verify exact percentage: (net_pnl / entry_price) * 100
    expected_pct = (expected_net / entry_price) * 100
    assert abs(pnl.pnl_percentage - expected_pct) < 0.01, \
        f"Expected {expected_pct:.2f}%, got {pnl.pnl_percentage:.2f}%"

    assert pnl.is_profitable is True


def test_calculate_realized_pnl_small_profit(calculator, entry_time, exit_time):
    """Test realized PnL with small profit margin - verify exact calculation"""
    entry_fees = TradeFees(platform_fee_lamports=5_000_000, priority_fee_lamports=500_000)  # 5.5M total
    exit_fees = TradeFees(platform_fee_lamports=5_500_000, priority_fee_lamports=500_000)  # 6M total

    entry_price = 1_000_000_000  # 1 SOL
    exit_price = 1_020_000_000  # 1.02 SOL

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Calculate expected values
    gross_pnl = exit_price - entry_price  # 20M lamports
    total_fees = entry_fees.total_lamports + exit_fees.total_lamports  # 11.5M
    expected_net = gross_pnl - total_fees  # 8.5M lamports

    # Verify exact calculations
    assert pnl.gross_pnl_lamports == gross_pnl
    assert pnl.net_pnl_lamports == expected_net
    assert pnl.net_pnl_sol == expected_net / 1e9  # 0.0085 SOL

    # Verify percentage
    expected_pct = (expected_net / entry_price) * 100  # 0.85%
    assert abs(pnl.pnl_percentage - expected_pct) < 0.01

    assert pnl.is_profitable is True


# =============================================================================
# REALIZED PNL - LOSING TRADES
# =============================================================================

def test_calculate_realized_pnl_losing(calculator, entry_fees, entry_time, exit_time):
    """Test realized PnL for losing trade"""
    exit_fees = TradeFees(platform_fee_lamports=8_000_000, priority_fee_lamports=500_000)

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        exit_price_lamports=800_000_000,  # 0.8 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Gross PnL: 0.8 - 1.0 = -0.2 SOL
    assert pnl.gross_pnl_lamports == -200_000_000

    # Net PnL: gross - fees (more negative)
    assert pnl.net_pnl_lamports < pnl.gross_pnl_lamports
    assert pnl.is_profitable is False
    assert pnl.pnl_percentage < 0


def test_calculate_realized_pnl_total_loss(calculator, entry_fees, entry_time, exit_time):
    """Test realized PnL for total loss (rug pull) - verify exact calculation"""
    exit_fees = TradeFees(platform_fee_lamports=0, priority_fee_lamports=500_000)

    entry_price = 1_000_000_000  # 1 SOL
    exit_price = 0  # Total loss (rug pull)

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Calculate expected values
    gross_pnl = exit_price - entry_price  # -1 SOL
    total_fees = entry_fees.total_lamports + exit_fees.total_lamports
    expected_net = gross_pnl - total_fees

    # Verify exact calculations
    assert pnl.gross_pnl_lamports == gross_pnl
    assert pnl.net_pnl_lamports == expected_net

    # Verify exact percentage
    expected_pct = (expected_net / entry_price) * 100
    assert abs(pnl.pnl_percentage - expected_pct) < 0.01

    assert pnl.is_profitable is False
    assert pnl.pnl_percentage < -100  # More than 100% loss (including fees)


# =============================================================================
# REALIZED PNL - BREAKEVEN & EDGE CASES
# =============================================================================

def test_calculate_realized_pnl_exact_breakeven(calculator, entry_time, exit_time):
    """Test realized PnL at exact breakeven (net = 0) - verify exact calculation"""
    entry_fees = TradeFees(platform_fee_lamports=10_000_000, priority_fee_lamports=0, network_fee_lamports=5_000)
    exit_fees = TradeFees(platform_fee_lamports=10_000_000, priority_fee_lamports=0, network_fee_lamports=5_000)

    entry_price = 1_000_000_000  # 1 SOL

    # Total fees: 10M + 5k + 10M + 5k = 20,010,000 lamports
    total_fees = entry_fees.total_lamports + exit_fees.total_lamports
    assert total_fees == 20_010_000  # Verify our math

    # Exit price = entry + fees for exact breakeven
    exit_price = entry_price + total_fees

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,  # 1.02001 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Should be EXACTLY breakeven (0)
    assert pnl.gross_pnl_lamports == total_fees
    assert pnl.net_pnl_lamports == 0, f"Breakeven should be exactly 0, got {pnl.net_pnl_lamports}"
    assert pnl.pnl_percentage == 0.0
    assert pnl.is_profitable is False  # Breakeven is not profitable


def test_calculate_realized_pnl_zero_entry_price(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test realized PnL with zero entry price (edge case)"""
    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=0,
        exit_price_lamports=1_000_000_000,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Should handle gracefully
    assert pnl.pnl_percentage == 0.0  # Avoid division by zero


def test_calculate_realized_pnl_long_holding_time(calculator, entry_fees, exit_fees):
    """Test realized PnL with long holding time"""
    entry_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    exit_time = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)  # 1 day later

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,
        exit_price_lamports=1_500_000_000,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    assert pnl.holding_time_seconds == 86400  # 1 day = 86400 seconds


# =============================================================================
# UNREALIZED PNL
# =============================================================================

def test_calculate_unrealized_pnl_profitable(calculator, entry_fees, entry_time):
    """Test unrealized PnL for profitable open position"""
    pnl = calculator.calculate_unrealized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        current_price_lamports=1_300_000_000,  # 1.3 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        entry_time=entry_time
    )

    assert isinstance(pnl, UnrealizedPnL)
    assert pnl.entry_price_lamports == 1_000_000_000
    assert pnl.current_price_lamports == 1_300_000_000
    assert pnl.current_value_lamports == 1_300_000_000

    # Unrealized PnL: current - entry - entry_fees
    expected_pnl = 1_300_000_000 - 1_000_000_000 - entry_fees.total_lamports
    assert pnl.unrealized_pnl_lamports == expected_pnl
    assert pnl.is_profitable is True


def test_calculate_unrealized_pnl_losing(calculator, entry_fees, entry_time):
    """Test unrealized PnL for losing open position"""
    pnl = calculator.calculate_unrealized_pnl(
        entry_price_lamports=1_000_000_000,  # 1 SOL
        current_price_lamports=700_000_000,  # 0.7 SOL
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        entry_time=entry_time
    )

    # Unrealized PnL: negative
    assert pnl.unrealized_pnl_lamports < 0
    assert pnl.is_profitable is False
    assert pnl.pnl_percentage < 0


def test_calculate_unrealized_pnl_zero_gain(calculator, entry_fees, entry_time):
    """Test unrealized PnL with price unchanged (still losing due to fees)"""
    pnl = calculator.calculate_unrealized_pnl(
        entry_price_lamports=1_000_000_000,
        current_price_lamports=1_000_000_000,  # Same price
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        entry_time=entry_time
    )

    # Even at same price, losing due to entry fees
    assert pnl.unrealized_pnl_lamports == -entry_fees.total_lamports
    assert pnl.is_profitable is False


# =============================================================================
# BREAKEVEN CALCULATIONS
# =============================================================================

def test_calculate_breakeven_price(calculator, entry_fees, exit_fees):
    """Test breakeven price calculation"""
    entry_price = 1_000_000_000  # 1 SOL

    breakeven = calculator.calculate_breakeven_price(
        entry_price_lamports=entry_price,
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees
    )

    # Breakeven = entry + all fees
    expected = entry_price + entry_fees.total_lamports + exit_fees.total_lamports
    assert breakeven == expected


def test_calculate_breakeven_price_high_fees(calculator):
    """Test breakeven with high fees"""
    entry_fees = TradeFees(platform_fee_lamports=50_000_000, priority_fee_lamports=1_000_000)
    exit_fees = TradeFees(platform_fee_lamports=50_000_000, priority_fee_lamports=1_000_000)

    breakeven = calculator.calculate_breakeven_price(
        entry_price_lamports=1_000_000_000,
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees
    )

    # Breakeven should be significantly higher
    assert breakeven > 1_100_000_000  # More than 1.1 SOL


def test_calculate_required_profit_percentage(calculator, entry_fees, exit_fees):
    """Test required profit percentage calculation"""
    entry_price = 1_000_000_000  # 1 SOL

    required = calculator.calculate_required_profit_percentage(
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees,
        entry_price_lamports=entry_price
    )

    # Calculate expected percentage
    total_fees = entry_fees.total_lamports + exit_fees.total_lamports
    expected_pct = (total_fees / entry_price) * 100

    assert abs(required - expected_pct) < 0.01


def test_calculate_required_profit_percentage_zero_entry(calculator, entry_fees, exit_fees):
    """Test required profit with zero entry price (edge case)"""
    required = calculator.calculate_required_profit_percentage(
        entry_fees=entry_fees,
        estimated_exit_fees=exit_fees,
        entry_price_lamports=0
    )

    assert required == 0.0  # Avoid division by zero


# =============================================================================
# ROI CALCULATIONS
# =============================================================================

def test_calculate_roi_profitable(calculator):
    """Test ROI calculation for profitable trade"""
    roi = calculator.calculate_roi(
        net_pnl_lamports=500_000_000,  # 0.5 SOL profit
        entry_price_lamports=1_000_000_000  # 1 SOL investment
    )

    assert roi == 50.0  # 50% ROI


def test_calculate_roi_losing(calculator):
    """Test ROI calculation for losing trade"""
    roi = calculator.calculate_roi(
        net_pnl_lamports=-200_000_000,  # 0.2 SOL loss
        entry_price_lamports=1_000_000_000  # 1 SOL investment
    )

    assert roi == -20.0  # -20% ROI


def test_calculate_roi_zero_entry(calculator):
    """Test ROI with zero entry (edge case)"""
    roi = calculator.calculate_roi(
        net_pnl_lamports=500_000_000,
        entry_price_lamports=0
    )

    assert roi == 0.0


def test_calculate_roi_breakeven(calculator):
    """Test ROI at breakeven"""
    roi = calculator.calculate_roi(
        net_pnl_lamports=0,
        entry_price_lamports=1_000_000_000
    )

    assert roi == 0.0


# =============================================================================
# FEE ESTIMATION
# =============================================================================

def test_estimate_exit_fees_default(calculator):
    """Test exit fee estimation with defaults"""
    exit_price = 1_500_000_000  # 1.5 SOL

    fees = calculator.estimate_exit_fees(exit_price_lamports=exit_price)

    # Platform fee: 1% of 1.5 SOL = 0.015 SOL = 15M lamports
    expected_platform_fee = (exit_price * 100) // 10_000
    assert fees.platform_fee_lamports == expected_platform_fee
    assert fees.priority_fee_lamports == 500_000  # Default
    assert fees.network_fee_lamports == 5_000  # Default


def test_estimate_exit_fees_custom(calculator):
    """Test exit fee estimation with custom values"""
    exit_price = 2_000_000_000  # 2 SOL

    fees = calculator.estimate_exit_fees(
        exit_price_lamports=exit_price,
        platform_fee_bps=200,  # 2%
        estimated_priority_fee_lamports=1_000_000  # 0.001 SOL
    )

    # Platform fee: 2% of 2 SOL = 0.04 SOL = 40M lamports
    expected_platform_fee = (exit_price * 200) // 10_000
    assert fees.platform_fee_lamports == expected_platform_fee
    assert fees.priority_fee_lamports == 1_000_000


# =============================================================================
# CRITICAL EDGE CASES (ADDED FROM AUDIT)
# =============================================================================

def test_negative_holding_time(calculator, entry_fees, exit_fees):
    """Test that negative holding time (exit before entry) raises ValueError"""
    from datetime import datetime, timezone

    entry_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    exit_time = datetime(2025, 1, 1, 11, 0, 0, tzinfo=timezone.utc)  # 1 hour BEFORE entry

    with pytest.raises(ValueError, match="Exit time must be after entry time"):
        calculator.calculate_realized_pnl(
            entry_price_lamports=1_000_000_000,
            exit_price_lamports=1_500_000_000,
            position_size_tokens=1_000_000,
            entry_fees=entry_fees,
            exit_fees=exit_fees,
            entry_time=entry_time,
            exit_time=exit_time
        )


def test_position_size_does_not_affect_percentage(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test that PnL percentage is independent of position size"""
    entry_price = 1_000_000_000
    exit_price = 1_500_000_000

    # Small position
    pnl_small = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=100,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Large position
    pnl_large = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=10_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Percentage should be identical (position size doesn't affect it)
    assert pnl_small.pnl_percentage == pnl_large.pnl_percentage, \
        f"Position size should not affect percentage: {pnl_small.pnl_percentage} != {pnl_large.pnl_percentage}"

    # But absolute PnL should be the same since it's based on entry/exit price, not position size
    assert pnl_small.net_pnl_lamports == pnl_large.net_pnl_lamports


def test_breakeven_roi(calculator):
    """Test ROI at exact breakeven is 0%"""
    roi = calculator.calculate_roi(
        net_pnl_lamports=0,
        entry_price_lamports=1_000_000_000
    )

    assert roi == 0.0


def test_zero_fees_scenario(calculator, entry_time, exit_time):
    """Test PnL calculation with zero fees (theoretical)"""
    zero_fees = TradeFees(platform_fee_lamports=0, priority_fee_lamports=0, network_fee_lamports=0)

    entry_price = 1_000_000_000
    exit_price = 1_500_000_000

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=entry_price,
        exit_price_lamports=exit_price,
        position_size_tokens=1_000_000,
        entry_fees=zero_fees,
        exit_fees=zero_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # With zero fees, gross = net
    assert pnl.gross_pnl_lamports == pnl.net_pnl_lamports
    assert pnl.total_fees_lamports == 0

    # Verify exact percentage: 50% gain
    expected_pct = ((exit_price - entry_price) / entry_price) * 100
    assert abs(pnl.pnl_percentage - expected_pct) < 0.01


# =============================================================================
# SERIALIZATION
# =============================================================================

def test_realized_pnl_to_dict(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test RealizedPnL serialization to dict"""
    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000,
        exit_price_lamports=1_500_000_000,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    data = pnl.to_dict()

    assert isinstance(data, dict)
    assert "entry_price_sol" in data
    assert "exit_price_sol" in data
    assert "net_pnl_sol" in data
    assert "pnl_percentage" in data
    assert "is_profitable" in data
    assert data["is_profitable"] is True


def test_unrealized_pnl_to_dict(calculator, entry_fees, entry_time):
    """Test UnrealizedPnL serialization to dict"""
    pnl = calculator.calculate_unrealized_pnl(
        entry_price_lamports=1_000_000_000,
        current_price_lamports=1_300_000_000,
        position_size_tokens=1_000_000,
        entry_fees=entry_fees,
        entry_time=entry_time
    )

    data = pnl.to_dict()

    assert isinstance(data, dict)
    assert "entry_price_sol" in data
    assert "current_price_sol" in data
    assert "unrealized_pnl_sol" in data
    assert "pnl_percentage" in data
    assert "is_profitable" in data


# =============================================================================
# EDGE CASES
# =============================================================================

def test_very_small_amounts(calculator, entry_time, exit_time):
    """Test PnL with very small amounts"""
    entry_fees = TradeFees(platform_fee_lamports=1000, priority_fee_lamports=1000, network_fee_lamports=1000)
    exit_fees = TradeFees(platform_fee_lamports=1500, priority_fee_lamports=1000, network_fee_lamports=1000)

    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=10_000,
        exit_price_lamports=20_000,  # Increased to make profitable after fees
        position_size_tokens=100,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    # Should handle small amounts gracefully
    assert pnl.gross_pnl_lamports == 10_000
    assert pnl.net_pnl_lamports > 0  # Profitable after fees (10,000 - 6,500 = 3,500)


def test_very_large_amounts(calculator, entry_fees, exit_fees, entry_time, exit_time):
    """Test PnL with very large amounts"""
    pnl = calculator.calculate_realized_pnl(
        entry_price_lamports=1_000_000_000_000,  # 1000 SOL
        exit_price_lamports=2_000_000_000_000,  # 2000 SOL
        position_size_tokens=100_000_000_000,
        entry_fees=entry_fees,
        exit_fees=exit_fees,
        entry_time=entry_time,
        exit_time=exit_time
    )

    assert pnl.gross_pnl_lamports == 1_000_000_000_000  # 1000 SOL gross
    assert pnl.is_profitable is True
