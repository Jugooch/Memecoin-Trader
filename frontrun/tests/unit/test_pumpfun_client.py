"""
Unit tests for Pump.fun Program Client
Tests instruction encoding, PDA derivation, and account deserialization
"""

import pytest
import struct
import base64
from unittest.mock import Mock, AsyncMock, patch
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.sysvar import RENT

from clients.pumpfun_client import (
    PumpFunClient,
    PumpFunConfig,
    PUMP_FUN_PROGRAM_ID,
    TOKEN_PROGRAM_ID,
    ASSOCIATED_TOKEN_PROGRAM_ID,
    EVENT_AUTHORITY,
    BUY_DISCRIMINATOR,
    SELL_DISCRIMINATOR,
    BONDING_CURVE_SEED,
)
from core.rpc_manager import RPCManager
from core.bonding_curve import BondingCurveState


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_rpc_manager():
    """Mock RPC manager for testing"""
    manager = Mock(spec=RPCManager)
    manager.call_http_rpc = AsyncMock()
    return manager


@pytest.fixture
def pumpfun_client(mock_rpc_manager):
    """Create PumpFunClient with mocked RPC manager"""
    return PumpFunClient(mock_rpc_manager)


@pytest.fixture
def example_mint():
    """Example mint pubkey for testing"""
    return Pubkey.from_string("So11111111111111111111111111111111111111112")


@pytest.fixture
def example_user():
    """Example user pubkey for testing"""
    return Pubkey.from_string("4Nd1mBQtrMJVYVfKf2PJy9NZUZdTAsp7D4xWLs4gDB4T")


# =============================================================================
# PDA DERIVATION TESTS
# =============================================================================

def test_derive_bonding_curve_pda(pumpfun_client, example_mint):
    """Test bonding curve PDA derivation"""
    pda = pumpfun_client.derive_bonding_curve_pda(example_mint)

    # Should return a valid pubkey
    assert isinstance(pda, Pubkey)

    # Verify derivation is correct
    expected_pda, _ = Pubkey.find_program_address(
        [BONDING_CURVE_SEED, bytes(example_mint)],
        PUMP_FUN_PROGRAM_ID
    )

    assert pda == expected_pda, "PDA should match expected derivation"


def test_bonding_curve_pda_caching(pumpfun_client, example_mint):
    """Test that PDA derivation results are cached"""
    # First call should derive
    pda1 = pumpfun_client.derive_bonding_curve_pda(example_mint)

    # Second call should return cached result
    pda2 = pumpfun_client.derive_bonding_curve_pda(example_mint)

    assert pda1 == pda2, "Cached PDA should match"

    # Verify it's actually cached (same object)
    assert example_mint in pumpfun_client._pda_cache


def test_clear_pda_cache(pumpfun_client, example_mint):
    """Test clearing PDA cache"""
    # Derive a PDA to populate cache
    pumpfun_client.derive_bonding_curve_pda(example_mint)
    assert len(pumpfun_client._pda_cache) > 0

    # Clear cache
    pumpfun_client.clear_pda_cache()
    assert len(pumpfun_client._pda_cache) == 0


def test_derive_associated_bonding_curve(pumpfun_client, example_mint):
    """Test associated token account derivation for bonding curve"""
    bonding_curve = pumpfun_client.derive_bonding_curve_pda(example_mint)
    ata = pumpfun_client.derive_associated_bonding_curve(bonding_curve, example_mint)

    # Should return a valid pubkey
    assert isinstance(ata, Pubkey)

    # Verify derivation is correct
    expected_ata, _ = Pubkey.find_program_address(
        [bytes(bonding_curve), bytes(TOKEN_PROGRAM_ID), bytes(example_mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID
    )

    assert ata == expected_ata, "ATA should match expected derivation"


def test_derive_associated_token_account(pumpfun_client, example_user, example_mint):
    """Test associated token account derivation for user"""
    ata = pumpfun_client.derive_associated_token_account(example_user, example_mint)

    # Should return a valid pubkey
    assert isinstance(ata, Pubkey)

    # Verify derivation is correct
    expected_ata, _ = Pubkey.find_program_address(
        [bytes(example_user), bytes(TOKEN_PROGRAM_ID), bytes(example_mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID
    )

    assert ata == expected_ata, "User ATA should match expected derivation"


# =============================================================================
# BUY INSTRUCTION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_build_buy_instruction(pumpfun_client, example_mint, example_user):
    """Test building a buy instruction"""
    amount_sol = 500_000_000  # 0.5 SOL
    max_slippage_bps = 500  # 5%

    instruction = await pumpfun_client.build_buy_instruction(
        mint=example_mint,
        user=example_user,
        amount_sol=amount_sol,
        max_slippage_bps=max_slippage_bps
    )

    # Should return an Instruction
    assert isinstance(instruction, Instruction)

    # Verify program ID
    assert instruction.program_id == PUMP_FUN_PROGRAM_ID

    # Verify instruction data (discriminator + amount)
    assert instruction.data[:8] == BUY_DISCRIMINATOR, "Should start with buy discriminator"

    # Verify amount is encoded correctly (u64 little-endian)
    encoded_amount = struct.unpack_from("<Q", instruction.data, 8)[0]
    assert encoded_amount == amount_sol, f"Amount should be {amount_sol}, got {encoded_amount}"

    # Verify total data length
    assert len(instruction.data) == 16, "Data should be 8 bytes discriminator + 8 bytes amount"


@pytest.mark.asyncio
async def test_buy_instruction_account_ordering(pumpfun_client, example_mint, example_user):
    """Test that buy instruction has correct account ordering"""
    instruction = await pumpfun_client.build_buy_instruction(
        mint=example_mint,
        user=example_user,
        amount_sol=1_000_000_000
    )

    # Derive expected accounts
    bonding_curve = pumpfun_client.derive_bonding_curve_pda(example_mint)
    associated_bonding_curve = pumpfun_client.derive_associated_bonding_curve(bonding_curve, example_mint)
    user_token_account = pumpfun_client.derive_associated_token_account(example_user, example_mint)

    # Should have exactly 9 accounts
    assert len(instruction.accounts) == 9, "Buy instruction should have 9 accounts"

    # Verify account ordering and properties
    expected_accounts = [
        (bonding_curve, False, True),  # bonding_curve: writable
        (associated_bonding_curve, False, True),  # associated_bonding_curve: writable
        (example_user, True, True),  # user: signer, writable
        (user_token_account, False, True),  # user_token_account: writable
        (SYSTEM_PROGRAM_ID, False, False),  # system_program
        (TOKEN_PROGRAM_ID, False, False),  # token_program
        (RENT, False, False),  # rent
        (EVENT_AUTHORITY, False, False),  # event_authority
        (PUMP_FUN_PROGRAM_ID, False, False),  # program
    ]

    for i, (expected_pubkey, expected_signer, expected_writable) in enumerate(expected_accounts):
        account = instruction.accounts[i]
        assert account.pubkey == expected_pubkey, f"Account {i} pubkey mismatch"
        assert account.is_signer == expected_signer, f"Account {i} signer flag mismatch"
        assert account.is_writable == expected_writable, f"Account {i} writable flag mismatch"


@pytest.mark.asyncio
async def test_buy_instruction_different_amounts(pumpfun_client, example_mint, example_user):
    """Test buy instruction with different SOL amounts"""
    test_amounts = [
        100_000_000,  # 0.1 SOL
        1_000_000_000,  # 1 SOL
        5_000_000_000,  # 5 SOL
    ]

    for amount in test_amounts:
        instruction = await pumpfun_client.build_buy_instruction(
            mint=example_mint,
            user=example_user,
            amount_sol=amount
        )

        # Verify encoded amount
        encoded_amount = struct.unpack_from("<Q", instruction.data, 8)[0]
        assert encoded_amount == amount, f"Should encode {amount} lamports correctly"


# =============================================================================
# SELL INSTRUCTION TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_build_sell_instruction(pumpfun_client, example_mint, example_user):
    """Test building a sell instruction"""
    amount_tokens = 1_000_000_000  # 1B tokens
    min_sol_output = 450_000_000  # Min 0.45 SOL

    instruction = await pumpfun_client.build_sell_instruction(
        mint=example_mint,
        user=example_user,
        amount_tokens=amount_tokens,
        min_sol_output=min_sol_output
    )

    # Should return an Instruction
    assert isinstance(instruction, Instruction)

    # Verify program ID
    assert instruction.program_id == PUMP_FUN_PROGRAM_ID

    # Verify instruction data (discriminator + amount)
    assert instruction.data[:8] == SELL_DISCRIMINATOR, "Should start with sell discriminator"

    # Verify amount is encoded correctly (u64 little-endian)
    encoded_amount = struct.unpack_from("<Q", instruction.data, 8)[0]
    assert encoded_amount == amount_tokens, f"Amount should be {amount_tokens}, got {encoded_amount}"

    # Verify total data length
    assert len(instruction.data) == 16, "Data should be 8 bytes discriminator + 8 bytes amount"


@pytest.mark.asyncio
async def test_sell_instruction_account_ordering(pumpfun_client, example_mint, example_user):
    """Test that sell instruction has correct account ordering"""
    instruction = await pumpfun_client.build_sell_instruction(
        mint=example_mint,
        user=example_user,
        amount_tokens=1_000_000_000,
        min_sol_output=400_000_000
    )

    # Derive expected accounts
    bonding_curve = pumpfun_client.derive_bonding_curve_pda(example_mint)
    associated_bonding_curve = pumpfun_client.derive_associated_bonding_curve(bonding_curve, example_mint)
    user_token_account = pumpfun_client.derive_associated_token_account(example_user, example_mint)

    # Should have exactly 8 accounts (one less than buy - no RENT)
    assert len(instruction.accounts) == 8, "Sell instruction should have 8 accounts"

    # Verify account ordering and properties
    expected_accounts = [
        (bonding_curve, False, True),  # bonding_curve: writable
        (associated_bonding_curve, False, True),  # associated_bonding_curve: writable
        (example_user, True, True),  # user: signer, writable
        (user_token_account, False, True),  # user_token_account: writable
        (SYSTEM_PROGRAM_ID, False, False),  # system_program
        (TOKEN_PROGRAM_ID, False, False),  # token_program
        (EVENT_AUTHORITY, False, False),  # event_authority
        (PUMP_FUN_PROGRAM_ID, False, False),  # program
    ]

    for i, (expected_pubkey, expected_signer, expected_writable) in enumerate(expected_accounts):
        account = instruction.accounts[i]
        assert account.pubkey == expected_pubkey, f"Account {i} pubkey mismatch"
        assert account.is_signer == expected_signer, f"Account {i} signer flag mismatch"
        assert account.is_writable == expected_writable, f"Account {i} writable flag mismatch"


# =============================================================================
# BONDING CURVE ACCOUNT FETCHING
# =============================================================================

@pytest.mark.asyncio
async def test_get_bonding_curve_state_success(pumpfun_client, example_mint):
    """Test fetching bonding curve state data"""
    # Mock bonding curve data
    virtual_token_reserves = 1_000_000_000_000
    virtual_sol_reserves = 30_000_000_000  # 30 SOL
    real_token_reserves = 800_000_000_000
    real_sol_reserves = 5_000_000_000  # 5 SOL
    token_total_supply = 1_000_000_000_000
    complete = False

    # Build mock account data
    data = bytearray(49)
    data[0:8] = b'\x00' * 8  # Discriminator
    struct.pack_into("<Q", data, 8, virtual_token_reserves)
    struct.pack_into("<Q", data, 16, virtual_sol_reserves)
    struct.pack_into("<Q", data, 24, real_token_reserves)
    struct.pack_into("<Q", data, 32, real_sol_reserves)
    struct.pack_into("<Q", data, 40, token_total_supply)
    data[48] = 0  # complete = false

    # Encode as base64
    data_base64 = base64.b64encode(bytes(data)).decode('utf-8')

    # Mock RPC response
    pumpfun_client.rpc_manager.call_http_rpc.return_value = {
        "result": {
            "value": {
                "data": [data_base64, "base64"]
            }
        }
    }

    # Fetch bonding curve state
    curve = await pumpfun_client.get_bonding_curve_state(example_mint)

    # Verify result
    assert curve is not None, "Should return bonding curve state"
    assert isinstance(curve, BondingCurveState)

    assert curve.virtual_token_reserves == virtual_token_reserves
    assert curve.virtual_sol_reserves == virtual_sol_reserves
    assert curve.real_token_reserves == real_token_reserves
    assert curve.real_sol_reserves == real_sol_reserves
    assert curve.token_total_supply == token_total_supply
    assert curve.complete == complete


@pytest.mark.asyncio
async def test_get_bonding_curve_state_not_found(pumpfun_client, example_mint):
    """Test fetching bonding curve state when it doesn't exist"""
    # Mock RPC response (account not found)
    pumpfun_client.rpc_manager.call_http_rpc.return_value = {
        "result": {
            "value": None
        }
    }

    # Fetch bonding curve state
    curve = await pumpfun_client.get_bonding_curve_state(example_mint)

    # Should return None
    assert curve is None, "Should return None for non-existent account"


@pytest.mark.asyncio
async def test_get_bonding_curve_state_invalid_data(pumpfun_client, example_mint):
    """Test fetching bonding curve state with invalid data"""
    # Mock RPC response with data that's too short
    short_data = base64.b64encode(b'\x00' * 10).decode('utf-8')

    pumpfun_client.rpc_manager.call_http_rpc.return_value = {
        "result": {
            "value": {
                "data": [short_data, "base64"]
            }
        }
    }

    # Fetch bonding curve state
    curve = await pumpfun_client.get_bonding_curve_state(example_mint)

    # Should return None on error
    assert curve is None, "Should return None for invalid data"


@pytest.mark.asyncio
async def test_get_bonding_curve_state_rpc_error(pumpfun_client, example_mint):
    """Test handling RPC errors when fetching bonding curve"""
    # Mock RPC error
    pumpfun_client.rpc_manager.call_http_rpc.side_effect = Exception("RPC connection failed")

    # Fetch bonding curve state
    curve = await pumpfun_client.get_bonding_curve_state(example_mint)

    # Should return None on error
    assert curve is None, "Should return None on RPC error"


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

def test_pumpfun_client_default_config(mock_rpc_manager):
    """Test PumpFunClient with default configuration"""
    client = PumpFunClient(mock_rpc_manager)

    assert client.config.program_id == PUMP_FUN_PROGRAM_ID
    assert client.config.fee_bps == 100  # 1%
    assert client.config.bonding_curve_seed == BONDING_CURVE_SEED


def test_pumpfun_client_custom_config(mock_rpc_manager):
    """Test PumpFunClient with custom configuration"""
    custom_config = PumpFunConfig(
        fee_bps=200  # 2%
    )

    client = PumpFunClient(mock_rpc_manager, custom_config)

    assert client.config.fee_bps == 200
    assert client.config.program_id == PUMP_FUN_PROGRAM_ID  # Should still use default


# =============================================================================
# DISCRIMINATOR TESTS
# =============================================================================

def test_buy_discriminator():
    """Test buy discriminator value"""
    # Should be first 8 bytes of SHA256("global:buy")
    expected = bytes.fromhex("66063d1201daebea")
    assert BUY_DISCRIMINATOR == expected, "Buy discriminator should match"


def test_sell_discriminator():
    """Test sell discriminator value"""
    # Should be first 8 bytes of SHA256("global:sell")
    expected = bytes.fromhex("33e685a4017f83ad")
    assert SELL_DISCRIMINATOR == expected, "Sell discriminator should match"


# =============================================================================
# EDGE CASES
# =============================================================================

@pytest.mark.asyncio
async def test_buy_instruction_zero_amount(pumpfun_client, example_mint, example_user):
    """Test buy instruction with zero amount (edge case)"""
    instruction = await pumpfun_client.build_buy_instruction(
        mint=example_mint,
        user=example_user,
        amount_sol=0
    )

    # Should still create instruction (validation happens on-chain)
    assert isinstance(instruction, Instruction)
    encoded_amount = struct.unpack_from("<Q", instruction.data, 8)[0]
    assert encoded_amount == 0


@pytest.mark.asyncio
async def test_sell_instruction_max_tokens(pumpfun_client, example_mint, example_user):
    """Test sell instruction with maximum token amount"""
    max_u64 = 2**64 - 1

    instruction = await pumpfun_client.build_sell_instruction(
        mint=example_mint,
        user=example_user,
        amount_tokens=max_u64,
        min_sol_output=0
    )

    # Should handle max u64 value
    assert isinstance(instruction, Instruction)
    encoded_amount = struct.unpack_from("<Q", instruction.data, 8)[0]
    assert encoded_amount == max_u64


def test_pda_derivation_deterministic(pumpfun_client, example_mint):
    """Test that PDA derivation is deterministic"""
    # Derive PDA multiple times
    pda1 = pumpfun_client.derive_bonding_curve_pda(example_mint)
    pumpfun_client.clear_pda_cache()  # Clear cache between calls
    pda2 = pumpfun_client.derive_bonding_curve_pda(example_mint)

    # Should always derive the same PDA for the same mint
    assert pda1 == pda2, "PDA derivation should be deterministic"
