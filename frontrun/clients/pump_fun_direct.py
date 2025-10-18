"""
Direct Pump.fun Transaction Builder

Builds buy/sell transactions directly without Pump Portal API.
This is 10x faster (50ms vs 500ms) and critical for sniping.

Based on: https://github.com/example (community implementation)
"""

import struct
from typing import Optional

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.system_program import transfer, TransferParams
from spl.token.instructions import create_associated_token_account, get_associated_token_address
import random

from core.logger import get_logger
from core.bonding_curve import (
    BondingCurveState,
    BondingCurveCalculator,
    fetch_bonding_curve_state,
    derive_bonding_curve_pda,
    get_initial_bonding_curve_state
)

logger = get_logger(__name__)

# Pump.fun program IDs and constants
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
GLOBAL = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
FEE_RECIPIENT = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
TOKEN_PROGRAM = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
FEE_PROGRAM = Pubkey.from_string("pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ")

# Buy instruction discriminator
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
# Sell instruction discriminator
SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")

# Helius Sender Jito tip accounts (for ultra-low latency via Jito validators)
JITO_TIP_ACCOUNTS = [
    "4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE",
    "D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ",
    "9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta",
    "5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn",
    "2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD",
    "2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ",
    "wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF",
    "3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT",
    "4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey",
    "4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or",
]


class PumpFunDirectClient:
    """
    Direct Pump.fun transaction builder (NO API)

    This builds transactions ~10x faster than Pump Portal API:
    - Pump Portal: 200-500ms
    - Direct building: 20-50ms

    Critical for sniping new token launches.
    """

    def __init__(self, rpc_manager, enable_jito_tips: bool = True, jito_tip_lamports: int = 1_000_000):
        """
        Initialize direct client

        Args:
            rpc_manager: RPC manager for chain calls
            enable_jito_tips: Enable Jito tips for faster inclusion (default: True)
            jito_tip_lamports: Tip amount in lamports (default: 1_000_000 = 0.001 SOL)
        """
        self.rpc_manager = rpc_manager
        self.calculator = BondingCurveCalculator()
        self.enable_jito_tips = enable_jito_tips
        self.jito_tip_lamports = jito_tip_lamports

        logger.info(
            "pump_fun_direct_client_initialized",
            jito_tips_enabled=enable_jito_tips,
            jito_tip_lamports=jito_tip_lamports
        )

    async def create_buy_transaction(
        self,
        keypair: Keypair,
        mint_str: str,
        sol_amount: float,
        slippage_bps: int = 500,  # 5% default
        priority_fee_lamports: int = 5_000_000,  # 0.005 SOL for high priority
        compute_units: int = 120_000,
        curve_state: Optional[BondingCurveState] = None,
        bonding_curve_address: Optional[str] = None,
        associated_bonding_curve_address: Optional[str] = None,
        creator_address: Optional[str] = None
    ) -> Optional[VersionedTransaction]:
        """
        Build buy transaction directly (NO API)

        Args:
            keypair: Payer keypair
            mint_str: Token mint address
            sol_amount: SOL to spend
            slippage_bps: Slippage tolerance (500 = 5%)
            priority_fee_lamports: Priority fee for faster inclusion
            compute_units: Compute units budget
            curve_state: Pre-known bonding curve state (optional, skips RPC fetch)
            bonding_curve_address: Pre-known bonding curve address (optional)
            associated_bonding_curve_address: Pre-known associated bonding curve (optional)
            creator_address: Pre-known creator address (optional)

        Returns:
            Signed VersionedTransaction ready to submit, or None if failed
        """
        try:
            start_time = logger._get_timestamp() if hasattr(logger, '_get_timestamp') else None

            # Use pre-known curve state or fetch from RPC
            if curve_state and bonding_curve_address and associated_bonding_curve_address and creator_address:
                # Fast path: Use pre-known state (for frontrunning new tokens)
                bonding_curve = Pubkey.from_string(bonding_curve_address)
                associated_bonding_curve = Pubkey.from_string(associated_bonding_curve_address)
                creator = Pubkey.from_string(creator_address)

                logger.debug(
                    "using_preknown_curve_state",
                    mint=mint_str[:16],
                    virtual_sol=curve_state.virtual_sol_reserves / 1e9
                )
            else:
                # Slow path: Fetch bonding curve state from RPC
                curve_data = await fetch_bonding_curve_state(self.rpc_manager, mint_str)
                if not curve_data:
                    logger.error("bonding_curve_fetch_failed", mint=mint_str[:16])
                    return None

                curve_state, bonding_curve, associated_bonding_curve, creator = curve_data

            # Validate curve
            if not self.calculator.validate_curve_state(curve_state):
                logger.error("invalid_curve_state", mint=mint_str[:16])
                return None

            # Get user and mint pubkeys
            user = keypair.pubkey()
            mint = Pubkey.from_string(mint_str)

            # Calculate tokens out
            sol_lamports = int(sol_amount * 1e9)
            quote = self.calculator.calculate_buy_price(curve_state, sol_lamports)

            # CRITICAL: Ensure tokens_out is an integer (avoid float precision issues)
            tokens_out = int(quote.tokens_out)

            # Apply slippage to max SOL cost with extra buffer for rounding differences
            # Use a 2% extra buffer on top of user slippage to account for calculation rounding
            slippage_multiplier = 1 + ((slippage_bps + 200) / 10000)  # Add 2% extra buffer
            max_sol_cost = int(sol_lamports * slippage_multiplier)

            logger.info(
                "buy_tx_building",
                mint=mint_str[:16],
                sol_in=sol_amount,
                tokens_out=tokens_out,
                max_sol_cost=max_sol_cost,
                slippage_bps=slippage_bps,
                effective_slippage_bps=slippage_bps + 200
            )

            # Check if user has associated token account
            associated_user = get_associated_token_address(user, mint)

            # Check if token account exists
            token_account_instruction = None
            try:
                response = await self.rpc_manager.call_http_rpc(
                    "getAccountInfo",
                    [str(associated_user), {"encoding": "base64"}]
                )
                account_info = response.get("result", {}).get("value")

                if not account_info:
                    # Need to create token account
                    token_account_instruction = create_associated_token_account(user, user, mint)
                    logger.debug("creating_token_account", account=str(associated_user))

            except Exception as e:
                logger.warning("token_account_check_failed", error=str(e))
                # Assume we need to create it
                token_account_instruction = create_associated_token_account(user, user, mint)

            # Derive all required PDAs
            creator_vault = Pubkey.find_program_address(
                [b'creator-vault', bytes(creator)],
                PUMP_FUN_PROGRAM
            )[0]

            global_volume_accumulator = Pubkey.find_program_address(
                [b"global_volume_accumulator"],
                PUMP_FUN_PROGRAM
            )[0]

            user_volume_accumulator = Pubkey.find_program_address(
                [b"user_volume_accumulator", bytes(user)],
                PUMP_FUN_PROGRAM
            )[0]

            fee_config = Pubkey.find_program_address(
                [b"fee_config", bytes(PUMP_FUN_PROGRAM)],
                FEE_PROGRAM
            )[0]

            # Build swap instruction accounts (16 accounts for buy)
            # Based on working example from pump-fun-sniper
            accounts = [
                AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
                AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
                AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
                AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
                AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
                AccountMeta(pubkey=associated_user, is_signer=False, is_writable=True),
                AccountMeta(pubkey=user, is_signer=True, is_writable=True),
                AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=creator_vault, is_signer=False, is_writable=True),
                AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
                AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=global_volume_accumulator, is_signer=False, is_writable=True),
                AccountMeta(pubkey=user_volume_accumulator, is_signer=False, is_writable=True),
                AccountMeta(pubkey=fee_config, is_signer=False, is_writable=False),
                AccountMeta(pubkey=FEE_PROGRAM, is_signer=False, is_writable=False),
            ]

            # Build instruction data
            data = bytearray()
            data.extend(BUY_DISCRIMINATOR)
            data.extend(struct.pack('<Q', tokens_out))  # amount (tokens expected, as integer)
            data.extend(struct.pack('<Q', max_sol_cost))  # max_sol_cost (with slippage)

            swap_instruction = Instruction(PUMP_FUN_PROGRAM, bytes(data), accounts)

            # Build full instruction list
            instructions = [
                set_compute_unit_limit(compute_units),
                set_compute_unit_price(priority_fee_lamports // compute_units),  # Micro-lamports per CU
            ]

            # Add Jito tip for ultra-low latency routing via Jito validators
            if self.enable_jito_tips:
                tip_account = Pubkey.from_string(random.choice(JITO_TIP_ACCOUNTS))
                tip_instruction = transfer(
                    TransferParams(
                        from_pubkey=user,
                        to_pubkey=tip_account,
                        lamports=self.jito_tip_lamports
                    )
                )
                instructions.append(tip_instruction)
                logger.debug(
                    "jito_tip_added",
                    tip_lamports=self.jito_tip_lamports,
                    tip_account=str(tip_account)[:16] + "..."
                )

            if token_account_instruction:
                instructions.append(token_account_instruction)

            instructions.append(swap_instruction)

            # Get recent blockhash
            blockhash_response = await self.rpc_manager.call_http_rpc("getLatestBlockhash", [])
            blockhash = blockhash_response.get("result", {}).get("value", {}).get("blockhash")
            if not blockhash:
                logger.error("failed_to_get_blockhash")
                return None

            # Compile message
            from solders.hash import Hash
            compiled_message = MessageV0.try_compile(
                user,
                instructions,
                [],  # No lookup tables
                Hash.from_string(blockhash)
            )

            # Sign transaction
            signed_tx = VersionedTransaction(compiled_message, [keypair])

            if start_time:
                duration_ms = (logger._get_timestamp() - start_time) if hasattr(logger, '_get_timestamp') else 0
                logger.info("buy_tx_built", duration_ms=duration_ms, mint=mint_str[:16])

            return signed_tx

        except Exception as e:
            logger.error("buy_tx_build_failed", mint=mint_str[:16], error=str(e), exc_info=True)
            return None

    async def create_sell_transaction(
        self,
        keypair: Keypair,
        mint_str: str,
        token_amount: float,  # UI amount (e.g., 1000000.5)
        slippage_bps: int = 500,  # 5% default
        priority_fee_lamports: int = 5_000_000,  # 0.005 SOL
        compute_units: int = 120_000,
        curve_state: Optional[BondingCurveState] = None,
        bonding_curve_address: Optional[str] = None,
        associated_bonding_curve_address: Optional[str] = None,
        creator_address: Optional[str] = None
    ) -> Optional[VersionedTransaction]:
        """
        Build sell transaction directly (NO API)

        Args:
            keypair: Payer keypair
            mint_str: Token mint address
            token_amount: Tokens to sell (UI amount, e.g. 1000000.5)
            slippage_bps: Slippage tolerance (500 = 5%)
            priority_fee_lamports: Priority fee
            compute_units: Compute units budget
            curve_state: Pre-known bonding curve state (optional, skips RPC fetch)
            bonding_curve_address: Pre-known bonding curve address (optional)
            associated_bonding_curve_address: Pre-known associated bonding curve (optional)
            creator_address: Pre-known creator address (optional)

        Returns:
            Signed VersionedTransaction ready to submit, or None if failed
        """
        try:
            # Use pre-known curve state or fetch from RPC
            if curve_state and bonding_curve_address and associated_bonding_curve_address and creator_address:
                # Fast path: Use pre-known state (for newly created tokens)
                bonding_curve = Pubkey.from_string(bonding_curve_address)
                associated_bonding_curve = Pubkey.from_string(associated_bonding_curve_address)
                creator = Pubkey.from_string(creator_address)

                logger.debug(
                    "using_preknown_curve_state_for_sell",
                    mint=mint_str[:16],
                    virtual_sol=curve_state.virtual_sol_reserves / 1e9
                )
            else:
                # Slow path: Fetch bonding curve state from RPC
                curve_data = await fetch_bonding_curve_state(self.rpc_manager, mint_str)
                if not curve_data:
                    logger.error("bonding_curve_fetch_failed", mint=mint_str[:16])
                    return None

                curve_state, bonding_curve, associated_bonding_curve, creator = curve_data

            # Validate curve
            if not self.calculator.validate_curve_state(curve_state):
                logger.error("invalid_curve_state", mint=mint_str[:16])
                return None

            # Get user and mint pubkeys
            user = keypair.pubkey()
            mint = Pubkey.from_string(mint_str)
            associated_user = get_associated_token_address(user, mint)

            # Convert UI amount to raw units (Pump.fun uses 6 decimals)
            token_amount_raw = int(token_amount * 1e6)

            # Calculate SOL out
            quote = self.calculator.calculate_sell_price(curve_state, token_amount_raw)

            # Apply slippage to min SOL output
            slippage_multiplier = 1 - (slippage_bps / 10000)
            min_sol_output = int(quote.sol_out * slippage_multiplier)

            logger.info(
                "sell_tx_building",
                mint=mint_str[:16],
                tokens_in=token_amount,
                sol_out=quote.sol_out / 1e9,
                min_sol_output=min_sol_output,
                slippage_bps=slippage_bps
            )

            # Derive creator vault and fee config
            creator_vault = Pubkey.find_program_address(
                [b'creator-vault', bytes(creator)],
                PUMP_FUN_PROGRAM
            )[0]

            fee_config = Pubkey.find_program_address(
                [b"fee_config", bytes(PUMP_FUN_PROGRAM)],
                FEE_PROGRAM
            )[0]

            # Build swap instruction accounts (14 accounts for sell)
            # Based on working example from pump-fun-sniper
            accounts = [
                AccountMeta(pubkey=GLOBAL, is_signer=False, is_writable=False),
                AccountMeta(pubkey=FEE_RECIPIENT, is_signer=False, is_writable=True),
                AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
                AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
                AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
                AccountMeta(pubkey=associated_user, is_signer=False, is_writable=True),
                AccountMeta(pubkey=user, is_signer=True, is_writable=True),
                AccountMeta(pubkey=SYSTEM_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=creator_vault, is_signer=False, is_writable=True),
                AccountMeta(pubkey=TOKEN_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
                AccountMeta(pubkey=PUMP_FUN_PROGRAM, is_signer=False, is_writable=False),
                AccountMeta(pubkey=fee_config, is_signer=False, is_writable=False),
                AccountMeta(pubkey=FEE_PROGRAM, is_signer=False, is_writable=False),
            ]

            # Build instruction data
            data = bytearray()
            data.extend(SELL_DISCRIMINATOR)
            data.extend(struct.pack('<Q', token_amount_raw))  # amount (tokens to sell)
            data.extend(struct.pack('<Q', min_sol_output))  # min_sol_output (with slippage)

            swap_instruction = Instruction(PUMP_FUN_PROGRAM, bytes(data), accounts)

            # Build instruction list
            instructions = [
                set_compute_unit_limit(compute_units),
                set_compute_unit_price(priority_fee_lamports // compute_units),
            ]

            # Add Jito tip for ultra-low latency routing via Jito validators
            if self.enable_jito_tips:
                tip_account = Pubkey.from_string(random.choice(JITO_TIP_ACCOUNTS))
                tip_instruction = transfer(
                    TransferParams(
                        from_pubkey=user,
                        to_pubkey=tip_account,
                        lamports=self.jito_tip_lamports
                    )
                )
                instructions.append(tip_instruction)
                logger.debug(
                    "jito_tip_added_to_sell",
                    tip_lamports=self.jito_tip_lamports,
                    tip_account=str(tip_account)[:16] + "..."
                )

            instructions.append(swap_instruction)

            # Get recent blockhash
            blockhash_response = await self.rpc_manager.call_http_rpc("getLatestBlockhash", [])
            blockhash = blockhash_response.get("result", {}).get("value", {}).get("blockhash")
            if not blockhash:
                logger.error("failed_to_get_blockhash")
                return None

            # Compile message
            from solders.hash import Hash
            compiled_message = MessageV0.try_compile(
                user,
                instructions,
                [],  # No lookup tables
                Hash.from_string(blockhash)
            )

            # Sign transaction
            signed_tx = VersionedTransaction(compiled_message, [keypair])

            logger.info("sell_tx_built", mint=mint_str[:16])

            return signed_tx

        except Exception as e:
            logger.error("sell_tx_build_failed", mint=mint_str[:16], error=str(e), exc_info=True)
            return None

    async def close(self):
        """Close client (no-op, included for compatibility)"""
        logger.info("pump_fun_direct_client_closed")
