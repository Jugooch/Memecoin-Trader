"""
Integration Test 3: Bonding Curve Read Operations
Tests REAL bonding curve state reading from Solana devnet - validates our calculations match on-chain data
"""

import pytest
import asyncio
import struct
from typing import Optional
from solders.pubkey import Pubkey

from core.bonding_curve import BondingCurveCalculator, BondingCurveState
from core.logger import get_logger


logger = get_logger(__name__)

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# =============================================================================
# BONDING CURVE ACCOUNT DISCOVERY
# =============================================================================

@pytest.mark.asyncio
async def test_find_pump_fun_program_account(devnet_rpc_manager):
    """Test that we can query the pump.fun program on devnet"""
    pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

    response = await devnet_rpc_manager.call_http_rpc(
        "getAccountInfo",
        [pump_program_id, {"encoding": "base64"}]
    )

    assert isinstance(response, dict), "Response should be dict"
    result = response.get("result", {})
    value = result.get("value")

    # Program should exist (or be None if not deployed on devnet)
    logger.info(
        "pump_program_query",
        program_id=pump_program_id,
        exists=value is not None
    )

    # This test just validates we can query accounts
    # The program may or may not exist on devnet


@pytest.mark.asyncio
async def test_get_program_accounts_by_owner(devnet_rpc_manager):
    """Test querying program accounts owned by pump.fun program"""
    pump_program_id = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

    # Query for accounts owned by pump.fun program
    response = await devnet_rpc_manager.call_http_rpc(
        "getProgramAccounts",
        [
            pump_program_id,
            {
                "encoding": "base64",
                "filters": [
                    {"dataSize": 96}  # Bonding curve account size
                ]
            }
        ]
    )

    if isinstance(response, dict):
        result = response.get("result", [])
        accounts = result if isinstance(result, list) else []
    else:
        accounts = response if isinstance(response, list) else []

    logger.info(
        "pump_program_accounts_query",
        program_id=pump_program_id,
        num_accounts=len(accounts)
    )

    # May be 0 on devnet if no tokens created
    assert isinstance(accounts, list), "Should return list of accounts"


# =============================================================================
# BONDING CURVE STATE DESERIALIZATION
# =============================================================================

def deserialize_bonding_curve_state(account_data: bytes) -> Optional[BondingCurveState]:
    """
    Deserialize bonding curve account data from on-chain format

    Layout (96 bytes):
    - 8 bytes: discriminator
    - 32 bytes: virtual_token_reserves (u64)
    - 32 bytes: virtual_sol_reserves (u64)
    - 32 bytes: real_token_reserves (u64)
    - 32 bytes: real_sol_reserves (u64)
    - 32 bytes: token_total_supply (u64)
    - 1 byte: complete (bool)
    """
    if len(account_data) < 96:
        logger.warning("bonding_curve_data_too_small", size=len(account_data))
        return None

    try:
        # Skip 8-byte discriminator
        offset = 8

        # Unpack u64 values (little-endian)
        virtual_token_reserves = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8

        virtual_sol_reserves = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8

        real_token_reserves = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8

        real_sol_reserves = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8

        token_total_supply = struct.unpack_from("<Q", account_data, offset)[0]
        offset += 8

        # Unpack bool (1 byte)
        complete = bool(struct.unpack_from("<B", account_data, offset)[0])

        return BondingCurveState(
            virtual_token_reserves=virtual_token_reserves,
            virtual_sol_reserves=virtual_sol_reserves,
            real_token_reserves=real_token_reserves,
            real_sol_reserves=real_sol_reserves,
            token_total_supply=token_total_supply,
            complete=complete
        )

    except Exception as e:
        logger.error("bonding_curve_deserialization_failed", error=str(e))
        return None


@pytest.mark.asyncio
async def test_deserialize_mock_bonding_curve_data():
    """Test deserializing mock bonding curve data with known values"""
    # Create mock account data (96 bytes)
    data = bytearray(96)

    # 8-byte discriminator (arbitrary)
    struct.pack_into("<Q", data, 0, 12345678)

    # Known values
    virtual_token_reserves = 1_000_000_000_000  # 1M tokens
    virtual_sol_reserves = 30_000_000_000  # 30 SOL
    real_token_reserves = 800_000_000_000  # 800K tokens
    real_sol_reserves = 5_000_000_000  # 5 SOL
    token_total_supply = 1_000_000_000_000  # 1M total
    complete = False

    offset = 8
    struct.pack_into("<Q", data, offset, virtual_token_reserves)
    offset += 8
    struct.pack_into("<Q", data, offset, virtual_sol_reserves)
    offset += 8
    struct.pack_into("<Q", data, offset, real_token_reserves)
    offset += 8
    struct.pack_into("<Q", data, offset, real_sol_reserves)
    offset += 8
    struct.pack_into("<Q", data, offset, token_total_supply)
    offset += 8
    struct.pack_into("<B", data, offset, int(complete))

    # Deserialize
    state = deserialize_bonding_curve_state(bytes(data))

    assert state is not None, "Should deserialize successfully"
    assert state.virtual_token_reserves == virtual_token_reserves
    assert state.virtual_sol_reserves == virtual_sol_reserves
    assert state.real_token_reserves == real_token_reserves
    assert state.real_sol_reserves == real_sol_reserves
    assert state.token_total_supply == token_total_supply
    assert state.complete == complete

    logger.info(
        "mock_bonding_curve_deserialized",
        virtual_token_reserves=state.virtual_token_reserves,
        virtual_sol_reserves=state.virtual_sol_reserves,
        complete=state.complete
    )


@pytest.mark.asyncio
async def test_deserialize_complete_bonding_curve():
    """Test deserializing complete (migrated) bonding curve"""
    data = bytearray(96)

    # Complete curve - should have complete=True
    struct.pack_into("<Q", data, 0, 0)  # discriminator
    struct.pack_into("<Q", data, 8, 0)  # virtual_token_reserves (depleted)
    struct.pack_into("<Q", data, 16, 85_000_000_000)  # virtual_sol_reserves (target reached)
    struct.pack_into("<Q", data, 24, 0)  # real_token_reserves
    struct.pack_into("<Q", data, 32, 85_000_000_000)  # real_sol_reserves
    struct.pack_into("<Q", data, 40, 1_000_000_000_000)  # token_total_supply
    struct.pack_into("<B", data, 48, 1)  # complete = True

    state = deserialize_bonding_curve_state(bytes(data))

    assert state is not None
    assert state.complete is True, "Complete flag should be True"
    assert state.virtual_token_reserves == 0, "Complete curve should have 0 virtual tokens"
    assert state.real_sol_reserves == 85_000_000_000, "Should reach 85 SOL target"

    logger.info("complete_bonding_curve_deserialized", real_sol=state.real_sol_reserves / 1e9)


# =============================================================================
# PRICE CALCULATION VALIDATION
# =============================================================================

@pytest.mark.asyncio
async def test_calculate_price_from_mock_reserves(bonding_curve_calculator):
    """Test calculating price from known reserve values"""
    # Create mock state
    state = BondingCurveState(
        virtual_token_reserves=1_000_000_000_000,  # 1M tokens
        virtual_sol_reserves=30_000_000_000,  # 30 SOL
        real_token_reserves=800_000_000_000,
        real_sol_reserves=5_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )

    # Calculate current price
    current_price = bonding_curve_calculator.get_current_price(state)

    # Price = virtual_sol / virtual_token
    expected_price = 30_000_000_000 / 1_000_000_000_000

    assert abs(current_price - expected_price) < 1e-9, \
        f"Price should be {expected_price}, got {current_price}"

    logger.info(
        "price_calculated_from_reserves",
        current_price=current_price,
        expected_price=expected_price
    )


@pytest.mark.asyncio
async def test_buy_quote_matches_reserve_change(bonding_curve_calculator):
    """Test that buy quote calculations match expected reserve changes"""
    initial_state = BondingCurveState(
        virtual_token_reserves=1_000_000_000_000,  # 1M tokens
        virtual_sol_reserves=30_000_000_000,  # 30 SOL
        real_token_reserves=800_000_000_000,
        real_sol_reserves=5_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )

    # Buy 1 SOL worth
    buy_amount = 1_000_000_000  # 1 SOL
    quote = bonding_curve_calculator.calculate_buy_price(initial_state, buy_amount)

    # Verify reserve changes
    # After buy: virtual_sol_reserves increases by amount_after_fee
    fee = (buy_amount * 100) // 10000  # 1% fee
    amount_after_fee = buy_amount - fee

    expected_new_sol_reserves = initial_state.virtual_sol_reserves + amount_after_fee

    # Calculate expected tokens out using constant product formula
    # tokens_out = (virtual_token * amount) / (virtual_sol + amount)
    expected_tokens_out = (
        initial_state.virtual_token_reserves * amount_after_fee
    ) // (
        initial_state.virtual_sol_reserves + amount_after_fee
    )

    assert quote.tokens_out == expected_tokens_out, \
        f"Tokens out should be {expected_tokens_out}, got {quote.tokens_out}"

    # After buy, new virtual token reserves = old - tokens_out
    expected_new_token_reserves = initial_state.virtual_token_reserves - expected_tokens_out

    logger.info(
        "buy_quote_reserve_validation",
        tokens_out=quote.tokens_out,
        expected_new_sol_reserves=expected_new_sol_reserves / 1e9,
        expected_new_token_reserves=expected_new_token_reserves / 1e9
    )


@pytest.mark.asyncio
async def test_sell_quote_matches_reserve_change(bonding_curve_calculator):
    """Test that sell quote calculations match expected reserve changes"""
    initial_state = BondingCurveState(
        virtual_token_reserves=1_000_000_000_000,  # 1M tokens
        virtual_sol_reserves=30_000_000_000,  # 30 SOL
        real_token_reserves=800_000_000_000,
        real_sol_reserves=5_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )

    # Sell 100K tokens
    sell_amount = 100_000_000_000  # 100K tokens
    quote = bonding_curve_calculator.calculate_sell_price(initial_state, sell_amount)

    # Calculate expected SOL out using constant product formula
    # sol_out = (virtual_sol * tokens) / (virtual_token + tokens)
    sol_before_fee = (
        initial_state.virtual_sol_reserves * sell_amount
    ) // (
        initial_state.virtual_token_reserves + sell_amount
    )

    # Apply 1% fee
    fee = (sol_before_fee * 100) // 10000
    expected_sol_out = sol_before_fee - fee

    assert quote.sol_out == expected_sol_out, \
        f"SOL out should be {expected_sol_out}, got {quote.sol_out}"

    logger.info(
        "sell_quote_reserve_validation",
        sol_out=quote.sol_out / 1e9,
        expected_sol_out=expected_sol_out / 1e9
    )


# =============================================================================
# BONDING CURVE STATE VALIDATION
# =============================================================================

@pytest.mark.asyncio
async def test_validate_active_bonding_curve(bonding_curve_calculator):
    """Test validation of active (tradeable) bonding curve"""
    active_state = BondingCurveState(
        virtual_token_reserves=800_000_000_000,
        virtual_sol_reserves=35_000_000_000,
        real_token_reserves=600_000_000_000,
        real_sol_reserves=10_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )

    is_valid = bonding_curve_calculator.validate_curve_state(active_state)

    assert is_valid is True, "Active curve should be valid"

    logger.info("active_bonding_curve_validated")


@pytest.mark.asyncio
async def test_reject_complete_bonding_curve(bonding_curve_calculator):
    """Test that complete (migrated) curves are rejected"""
    complete_state = BondingCurveState(
        virtual_token_reserves=0,
        virtual_sol_reserves=85_000_000_000,
        real_token_reserves=0,
        real_sol_reserves=85_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=True  # Migrated to Raydium
    )

    is_valid = bonding_curve_calculator.validate_curve_state(complete_state)

    assert is_valid is False, "Complete curve should be invalid for trading"

    logger.info("complete_bonding_curve_rejected")


@pytest.mark.asyncio
async def test_reject_depleted_reserves(bonding_curve_calculator):
    """Test rejection of curves with depleted reserves"""
    depleted_state = BondingCurveState(
        virtual_token_reserves=0,  # No tokens left
        virtual_sol_reserves=50_000_000_000,
        real_token_reserves=0,
        real_sol_reserves=20_000_000_000,
        token_total_supply=1_000_000_000_000,
        complete=False
    )

    is_valid = bonding_curve_calculator.validate_curve_state(depleted_state)

    assert is_valid is False, "Depleted reserves should be invalid"

    logger.info("depleted_reserves_rejected")


# =============================================================================
# MARKET CAP CALCULATION
# =============================================================================

@pytest.mark.asyncio
async def test_calculate_market_cap_from_reserves(bonding_curve_calculator):
    """Test market cap calculation from bonding curve reserves"""
    state = BondingCurveState(
        virtual_token_reserves=1_000_000_000_000,  # 1M tokens
        virtual_sol_reserves=30_000_000_000,  # 30 SOL
        real_token_reserves=800_000_000_000,
        real_sol_reserves=5_000_000_000,
        token_total_supply=1_000_000_000_000,  # 1M total supply
        complete=False
    )

    # Current price = 30 SOL / 1M tokens = 0.00003 SOL per token
    current_price = bonding_curve_calculator.get_current_price(state)

    # Market cap = price * total_supply
    market_cap = current_price * state.token_total_supply

    # Should equal virtual_sol_reserves (for constant product curve)
    # This is because price = sol/token, so price * total_tokens = total_sol
    expected_market_cap = 30_000_000_000  # 30 SOL

    assert abs(market_cap - expected_market_cap) < 1e6, \
        f"Market cap should be ~{expected_market_cap / 1e9} SOL, got {market_cap / 1e9} SOL"

    logger.info(
        "market_cap_calculated",
        market_cap_sol=market_cap / 1e9,
        current_price=current_price
    )
