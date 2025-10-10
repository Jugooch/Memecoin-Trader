"""
Pump.fun Program Client V2
CORRECTED version based on real Solscan transactions
"""

import struct
from typing import Optional
from dataclasses import dataclass

from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYSTEM_PROGRAM_ID

from core.logger import get_logger

logger = get_logger(__name__)

# Program IDs (verified from Solscan)
PUMP_FUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")

# Global accounts (verified from Solscan transaction)
GLOBAL_ACCOUNT = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
FEE_RECIPIENT = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")

# Seeds for PDA derivation
BONDING_CURVE_SEED = b"bonding-curve"

# Instruction discriminator (verified from Solscan)
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")


@dataclass
class SimpleBuyParams:
    """
    Simplified buy parameters using SOL amount

    The client will estimate token amount based on bonding curve math
    """
    mint: Pubkey
    user: Pubkey
    sol_amount: int  # Lamports
    slippage_bps: int = 500  # 5% default


def derive_bonding_curve_pda(mint: Pubkey) -> Pubkey:
    """Derive bonding curve PDA from mint"""
    pda, _bump = Pubkey.find_program_address(
        [BONDING_CURVE_SEED, bytes(mint)],
        PUMP_FUN_PROGRAM_ID
    )
    return pda


def derive_associated_token_account(owner: Pubkey, mint: Pubkey) -> Pubkey:
    """Derive ATA for owner"""
    pda, _bump = Pubkey.find_program_address(
        [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID
    )
    return pda


def build_pump_buy_instruction_simple(
    mint: Pubkey,
    user: Pubkey,
    sol_amount_lamports: int,
    slippage_bps: int = 500
) -> Instruction:
    """
    Build Pump.fun buy instruction (SIMPLIFIED - uses SOL amount)

    Args:
        mint: Token mint to buy
        user: Buyer wallet
        sol_amount_lamports: Amount of SOL to spend (lamports)
        slippage_bps: Slippage tolerance (default 500 = 5%)

    Returns:
        Pump.fun buy instruction

    Note:
        This uses a simplified bonding curve estimation.
        For production, fetch real curve state and calculate exact token amount.
    """
    # Derive PDAs
    bonding_curve = derive_bonding_curve_pda(mint)
    associated_bonding_curve = derive_associated_token_account(bonding_curve, mint)
    user_token_account = derive_associated_token_account(user, mint)

    # TODO: These PDAs need to be derived per-mint and per-user
    # For now, using placeholder addresses - THIS WILL FAIL!
    # We need to derive these properly or fetch them from chain

    # CRITICAL: We need to calculate token amount from bonding curve
    # For now using a rough estimate: ~2.9M tokens per 0.1 SOL (from example)
    # This is WRONG and needs bonding curve math!
    estimated_tokens = int((sol_amount_lamports / 1e9) * 29_000_000 * 1e6)  # 6 decimals

    # Add slippage to SOL cost
    max_sol_cost = int(sol_amount_lamports * (1 + slippage_bps / 10000))

    # Build instruction data: discriminator + token_amount + max_sol_cost
    data = BUY_DISCRIMINATOR + struct.pack("<QQ", estimated_tokens, max_sol_cost)

    # Build accounts list - ORDER MATTERS! (verified from Solscan)
    accounts = [
        # #1 - Global
        AccountMeta(pubkey=GLOBAL_ACCOUNT, is_signer=False, is_writable=False),
        # #2 - Fee Recipient (writable)
        AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
        # #3 - Mint
        AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
        # #4 - Bonding Curve (writable)
        AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
        # #5 - Associated Bonding Curve (writable)
        AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
        # #6 - Associated User (writable)
        AccountMeta(pubkey=user_token_account, is_signer=False, is_writable=True),
        # #7 - User (writable, signer)
        AccountMeta(pubkey=user, is_signer=True, is_writable=True),
        # #8 - System Program
        AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        # #9 - Token Program
        AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
        # #10-15: Creator vault, event authority, program, volume accumulators, fee config
        # These need to be derived/fetched properly
        # For now, this instruction will fail without them!
    ]

    logger.warning(
        "using_incomplete_pump_buy_instruction",
        msg="Missing accounts #10-15! This will fail. Need to add: creator_vault, event_authority, program, volume_accumulators, fee_config"
    )

    return Instruction(
        program_id=PUMP_FUN_PROGRAM_ID,
        accounts=accounts,
        data=data
    )


# For testing/reference
if __name__ == "__main__":
    # Example from Solscan
    example_mint = Pubkey.from_string("GASneQtATBSWVNVCixbmAHuRRTGrbF3XBhTMKBtypump")
    example_user = Pubkey.from_string("76ejXhsjdNPqLfnn2ii5NmsXQ2NinEULcYURufsJZjyb")

    instruction = build_pump_buy_instruction_simple(
        mint=example_mint,
        user=example_user,
        sol_amount_lamports=486_700_000,  # 0.4867 SOL
        slippage_bps=200  # 2%
    )

    print(f"Instruction program: {instruction.program_id}")
    print(f"Accounts: {len(instruction.accounts)}")
    print(f"Data length: {len(instruction.data)} bytes")
