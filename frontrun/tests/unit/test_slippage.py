"""
Unit tests for Slippage Manager
Tests slippage calculations, validation, and edge cases
"""

import pytest
from core.slippage import (
    SlippageManager,
    SlippageConfig,
    SlippageCheck,
    TradeUrgency,
    SLIPPAGE_TOLERANCE_BPS,
    BPS_DENOMINATOR
)


@pytest.fixture
def manager():
    """Create slippage manager with default config"""
    return SlippageManager()


@pytest.fixture
def manager_low_urgency():
    """Create slippage manager with LOW default urgency"""
    config = SlippageConfig(default_urgency=TradeUrgency.LOW)
    return SlippageManager(config)


@pytest.fixture
def manager_custom_slippage():
    """Create slippage manager with custom slippage"""
    config = SlippageConfig(custom_slippage_bps=300)  # 3%
    return SlippageManager(config)


# =============================================================================
# MINIMUM AMOUNT CALCULATIONS
# =============================================================================

def test_calculate_min_amount_out_normal_urgency(manager):
    """Test min amount calculation with normal urgency (5%)"""
    expected = 1_000_000
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # 5% slippage: 1,000,000 * 0.95 = 950,000
    assert min_out == 950_000


def test_calculate_min_amount_out_low_urgency(manager):
    """Test min amount calculation with low urgency (1%)"""
    expected = 1_000_000
    urgency = TradeUrgency.LOW

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # 1% slippage: 1,000,000 * 0.99 = 990,000
    assert min_out == 990_000


def test_calculate_min_amount_out_high_urgency(manager):
    """Test min amount calculation with high urgency (10%)"""
    expected = 1_000_000
    urgency = TradeUrgency.HIGH

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # 10% slippage: 1,000,000 * 0.90 = 900,000
    assert min_out == 900_000


def test_calculate_min_amount_out_critical_urgency(manager):
    """Test min amount calculation with critical urgency (20%)"""
    expected = 1_000_000
    urgency = TradeUrgency.CRITICAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # 20% slippage: 1,000,000 * 0.80 = 800,000
    assert min_out == 800_000


def test_calculate_min_amount_out_custom_slippage(manager):
    """Test min amount calculation with custom slippage"""
    expected = 1_000_000
    custom_slippage_bps = 750  # 7.5%

    min_out = manager.calculate_min_amount_out(expected, custom_slippage_bps=custom_slippage_bps)

    # 7.5% slippage: 1,000,000 * 0.925 = 925,000
    assert min_out == 925_000


def test_calculate_min_amount_out_default_urgency(manager):
    """Test min amount uses default urgency when not specified"""
    expected = 1_000_000

    # Manager has NORMAL as default (5%)
    min_out = manager.calculate_min_amount_out(expected)

    assert min_out == 950_000


def test_calculate_min_amount_out_integer_precision(manager):
    """Test min amount calculation uses integer math correctly"""
    expected = 1_234_567
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # Manual calculation: 1,234,567 * 9500 / 10000 = 1,172,838.65 -> 1,172,838 (floor)
    expected_min = (expected * (BPS_DENOMINATOR - SLIPPAGE_TOLERANCE_BPS[urgency])) // BPS_DENOMINATOR
    assert min_out == expected_min


def test_calculate_min_amount_out_zero_amount(manager):
    """Test min amount with zero expected raises ValueError"""
    with pytest.raises(ValueError, match="Expected amount must be positive"):
        manager.calculate_min_amount_out(0, TradeUrgency.NORMAL)


def test_calculate_min_amount_out_negative_amount(manager):
    """Test min amount with negative expected raises ValueError"""
    with pytest.raises(ValueError, match="Expected amount must be positive"):
        manager.calculate_min_amount_out(-1000, TradeUrgency.NORMAL)


def test_calculate_min_amount_out_very_large_amount(manager):
    """Test min amount with very large amount"""
    expected = 10_000_000_000_000  # 10 trillion
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    assert min_out == 9_500_000_000_000  # 5% slippage


# =============================================================================
# SLIPPAGE VALIDATION
# =============================================================================

def test_validate_slippage_within_tolerance(manager):
    """Test validation passes when within tolerance"""
    expected = 1_000_000
    actual = 980_000  # 2% slippage
    urgency = TradeUrgency.NORMAL  # 5% tolerance

    check = manager.validate_slippage(expected, actual, urgency)

    assert isinstance(check, SlippageCheck)
    assert check.is_valid is True
    assert check.expected_amount == expected
    assert check.actual_amount == actual
    assert check.slippage_pct == 2.0
    assert check.tolerance_pct == 5.0


def test_validate_slippage_at_exact_tolerance(manager):
    """Test validation passes at exact tolerance limit"""
    expected = 1_000_000
    actual = 950_000  # Exactly 5% slippage
    urgency = TradeUrgency.NORMAL  # 5% tolerance

    check = manager.validate_slippage(expected, actual, urgency)

    assert check.is_valid is True
    assert check.slippage_pct == 5.0


def test_validate_slippage_exceeds_tolerance(manager):
    """Test validation fails when exceeds tolerance"""
    expected = 1_000_000
    actual = 900_000  # 10% slippage
    urgency = TradeUrgency.NORMAL  # 5% tolerance

    check = manager.validate_slippage(expected, actual, urgency)

    assert check.is_valid is False
    assert check.slippage_pct == 10.0
    assert "exceeds tolerance" in check.message


def test_validate_slippage_better_than_expected(manager):
    """Test validation passes when better than expected"""
    expected = 1_000_000
    actual = 1_050_000  # Got more than expected

    check = manager.validate_slippage(expected, actual, TradeUrgency.NORMAL)

    assert check.is_valid is True
    assert check.slippage_pct == 0.0  # No negative slippage


def test_validate_slippage_exact_expected(manager):
    """Test validation passes when exactly as expected"""
    expected = 1_000_000
    actual = 1_000_000

    check = manager.validate_slippage(expected, actual, TradeUrgency.NORMAL)

    assert check.is_valid is True
    assert check.slippage_pct == 0.0


def test_validate_slippage_zero_expected(manager):
    """Test validation with zero expected returns invalid"""
    check = manager.validate_slippage(0, 100, TradeUrgency.NORMAL)

    assert check.is_valid is False
    assert "must be positive" in check.message


def test_validate_slippage_different_urgencies(manager):
    """Test validation with different urgency levels"""
    expected = 1_000_000
    actual = 920_000  # 8% slippage

    # Should fail with NORMAL (5% tolerance)
    check_normal = manager.validate_slippage(expected, actual, TradeUrgency.NORMAL)
    assert check_normal.is_valid is False

    # Should pass with HIGH (10% tolerance)
    check_high = manager.validate_slippage(expected, actual, TradeUrgency.HIGH)
    assert check_high.is_valid is True


def test_validate_slippage_custom_tolerance(manager):
    """Test validation with custom slippage tolerance"""
    expected = 1_000_000
    actual = 970_000  # 3% slippage
    custom_slippage_bps = 200  # 2% tolerance

    check = manager.validate_slippage(expected, actual, custom_slippage_bps=custom_slippage_bps)

    assert check.is_valid is False  # 3% exceeds 2% tolerance
    assert check.tolerance_pct == 2.0


# =============================================================================
# SLIPPAGE TOLERANCE QUERIES
# =============================================================================

def test_get_slippage_tolerance_bps_all_urgencies(manager):
    """Test getting tolerance BPS for all urgency levels"""
    assert manager.get_slippage_tolerance_bps(TradeUrgency.LOW) == 100
    assert manager.get_slippage_tolerance_bps(TradeUrgency.NORMAL) == 500
    assert manager.get_slippage_tolerance_bps(TradeUrgency.HIGH) == 1000
    assert manager.get_slippage_tolerance_bps(TradeUrgency.CRITICAL) == 2000


def test_get_slippage_tolerance_pct_all_urgencies(manager):
    """Test getting tolerance percentage for all urgency levels"""
    assert manager.get_slippage_tolerance_pct(TradeUrgency.LOW) == 1.0
    assert manager.get_slippage_tolerance_pct(TradeUrgency.NORMAL) == 5.0
    assert manager.get_slippage_tolerance_pct(TradeUrgency.HIGH) == 10.0
    assert manager.get_slippage_tolerance_pct(TradeUrgency.CRITICAL) == 20.0


def test_get_slippage_tolerance_default_urgency(manager_low_urgency):
    """Test getting tolerance uses default urgency"""
    # Manager has LOW as default
    bps = manager_low_urgency.get_slippage_tolerance_bps()
    pct = manager_low_urgency.get_slippage_tolerance_pct()

    assert bps == 100
    assert pct == 1.0


def test_get_slippage_tolerance_custom_config(manager_custom_slippage):
    """Test getting tolerance with custom slippage config"""
    # Manager has custom 3% slippage
    bps = manager_custom_slippage.get_slippage_tolerance_bps()
    pct = manager_custom_slippage.get_slippage_tolerance_pct()

    assert bps == 300
    assert pct == 3.0


# =============================================================================
# MAX INPUT AMOUNT CALCULATIONS
# =============================================================================

def test_calculate_max_input_amount_normal_urgency(manager):
    """Test max input calculation with normal urgency"""
    expected_output = 1_000_000
    urgency = TradeUrgency.NORMAL  # 5% slippage

    max_input = manager.calculate_max_input_amount(expected_output, urgency)

    # With 5% slippage, need to input more to get expected output
    # expected / 0.95 = 1,052,631.57... -> 1,052,632 (rounded up)
    assert max_input >= 1_052_631
    assert max_input <= 1_052_633  # Allow small rounding variance


def test_calculate_max_input_amount_low_urgency(manager):
    """Test max input calculation with low urgency"""
    expected_output = 1_000_000
    urgency = TradeUrgency.LOW  # 1% slippage

    max_input = manager.calculate_max_input_amount(expected_output, urgency)

    # With 1% slippage: expected / 0.99 = 1,010,101.01... -> 1,010,102
    assert max_input >= 1_010_101
    assert max_input <= 1_010_103


def test_calculate_max_input_amount_zero_output(manager):
    """Test max input with zero output raises ValueError"""
    with pytest.raises(ValueError, match="Expected output must be positive"):
        manager.calculate_max_input_amount(0, TradeUrgency.NORMAL)


def test_calculate_max_input_amount_negative_output(manager):
    """Test max input with negative output raises ValueError"""
    with pytest.raises(ValueError, match="Expected output must be positive"):
        manager.calculate_max_input_amount(-1000, TradeUrgency.NORMAL)


# =============================================================================
# CONFIGURATION
# =============================================================================

def test_manager_default_config():
    """Test manager initialization with default config"""
    manager = SlippageManager()

    assert manager.config.default_urgency == TradeUrgency.NORMAL
    assert manager.config.custom_slippage_bps is None


def test_manager_custom_urgency_config():
    """Test manager initialization with custom urgency"""
    config = SlippageConfig(default_urgency=TradeUrgency.HIGH)
    manager = SlippageManager(config)

    assert manager.config.default_urgency == TradeUrgency.HIGH

    # Should use HIGH urgency by default
    min_out = manager.calculate_min_amount_out(1_000_000)
    assert min_out == 900_000  # 10% slippage


def test_manager_custom_slippage_config():
    """Test manager initialization with custom slippage"""
    config = SlippageConfig(custom_slippage_bps=250)  # 2.5%
    manager = SlippageManager(config)

    # Should use custom slippage instead of urgency-based
    min_out = manager.calculate_min_amount_out(1_000_000, TradeUrgency.NORMAL)
    assert min_out == 975_000  # 2.5% slippage, not 5%


# =============================================================================
# EDGE CASES
# =============================================================================

def test_very_small_expected_amount(manager):
    """Test with very small expected amount"""
    expected = 100
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # Should still work with small amounts
    assert min_out == 95


def test_slippage_calculation_precision(manager):
    """Test slippage calculation precision with odd numbers"""
    expected = 123_456_789
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # Verify integer division is correct
    expected_min = (expected * 9500) // 10_000
    assert min_out == expected_min


def test_validate_slippage_edge_case_one_lamport_under(manager):
    """Test validation at edge case (one lamport under min)"""
    expected = 1_000_000
    min_out = manager.calculate_min_amount_out(expected, TradeUrgency.NORMAL)  # 950,000
    actual = min_out - 1  # 949,999

    check = manager.validate_slippage(expected, actual, TradeUrgency.NORMAL)

    assert check.is_valid is False


def test_validate_slippage_edge_case_exact_min(manager):
    """Test validation at edge case (exactly at min)"""
    expected = 1_000_000
    min_out = manager.calculate_min_amount_out(expected, TradeUrgency.NORMAL)  # 950,000
    actual = min_out  # Exactly at minimum

    check = manager.validate_slippage(expected, actual, TradeUrgency.NORMAL)

    assert check.is_valid is True


# =============================================================================
# CRITICAL EDGE CASES (ADDED FROM AUDIT)
# =============================================================================

def test_zero_slippage_tolerance(manager):
    """Test zero slippage (no tolerance allowed)"""
    expected = 1_000_000

    # Zero slippage means min_out = expected
    min_out = manager.calculate_min_amount_out(expected, custom_slippage_bps=0)
    assert min_out == expected  # No reduction

    # Any slippage should fail validation
    check = manager.validate_slippage(expected, expected - 1, custom_slippage_bps=0)
    assert check.is_valid is False

    # Exact match should pass
    check = manager.validate_slippage(expected, expected, custom_slippage_bps=0)
    assert check.is_valid is True


def test_maximum_slippage_tolerance(manager):
    """Test 100% slippage (always accepts)"""
    expected = 1_000_000

    # 100% slippage (10000 BPS) means min_out = 0
    min_out = manager.calculate_min_amount_out(expected, custom_slippage_bps=10000)
    assert min_out == 0

    # Even zero should pass validation
    check = manager.validate_slippage(expected, 0, custom_slippage_bps=10000)
    assert check.is_valid is True

    # Any amount should pass
    check = manager.validate_slippage(expected, 1, custom_slippage_bps=10000)
    assert check.is_valid is True


def test_slippage_with_very_large_amounts(manager):
    """Test slippage calculation with very large amounts"""
    # Use large but safe value
    expected = 10_000_000_000_000_000  # 10 million SOL (unlikely but safe)
    urgency = TradeUrgency.NORMAL

    min_out = manager.calculate_min_amount_out(expected, urgency)

    # Should not overflow or error
    assert min_out > 0
    assert min_out < expected

    # Verify calculation is correct
    expected_min = (expected * 9500) // 10000  # 5% slippage
    assert min_out == expected_min


def test_slippage_bps_boundary_values(manager):
    """Test slippage with boundary BPS values (0, 1, 9999, 10000)"""
    expected = 1_000_000

    # BPS = 1 (0.01% slippage)
    min_out = manager.calculate_min_amount_out(expected, custom_slippage_bps=1)
    expected_min = (expected * 9999) // 10000
    assert min_out == expected_min

    # BPS = 9999 (99.99% slippage)
    min_out = manager.calculate_min_amount_out(expected, custom_slippage_bps=9999)
    expected_min = (expected * 1) // 10000
    assert min_out == expected_min


# =============================================================================
# STATISTICS
# =============================================================================

def test_get_stats(manager):
    """Test getting manager statistics"""
    stats = manager.get_stats()

    assert stats["default_urgency"] == "normal"
    assert stats["default_slippage_pct"] == 5.0
    assert "urgency_mappings" in stats
    assert stats["urgency_mappings"]["low"] == 1.0
    assert stats["urgency_mappings"]["normal"] == 5.0
    assert stats["urgency_mappings"]["high"] == 10.0
    assert stats["urgency_mappings"]["critical"] == 20.0


def test_get_stats_custom_config(manager_custom_slippage):
    """Test getting stats with custom config"""
    stats = manager_custom_slippage.get_stats()

    assert stats["default_slippage_pct"] == 3.0
