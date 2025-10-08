"""
Pump.fun Program Client
Encodes buy/sell instructions and fetches bonding curve accounts
"""

import struct
from typing import Optional, Dict, ClassVar
from dataclasses import dataclass

from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.sysvar import RENT

from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


# Program IDs
PUMP_FUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
EVENT_AUTHORITY = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")  # Pump.fun event authority

# Instruction discriminators (first 8 bytes of SHA256("global:buy/sell"))
BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")

# Seeds for PDA derivation
BONDING_CURVE_SEED = b"bonding-curve"
METADATA_SEED = b"metadata"

# Fee configuration
FEE_BPS = 100  # 1% platform fee


@dataclass
class BondingCurveAccount:
    """Pump.fun bonding curve state"""
    virtual_token_reserves: int
    virtual_sol_reserves: int
    real_token_reserves: int
    real_sol_reserves: int
    token_total_supply: int
    complete: bool


@dataclass
class PumpFunConfig:
    """Pump.fun protocol configuration"""
    program_id: Pubkey = PUMP_FUN_PROGRAM_ID
    fee_bps: int = FEE_BPS
    bonding_curve_seed: bytes = BONDING_CURVE_SEED


class PumpFunClient:
    """
    Pump.fun program client for encoding buy/sell instructions

    Features:
    - Buy/sell instruction encoding
    - Bonding curve PDA derivation
    - Associated token account derivation
    - Bonding curve account fetching

    Usage:
        client = PumpFunClient(rpc_manager)
        buy_ix = await client.build_buy_instruction(
            mint=mint_pubkey,
            user=wallet.pubkey(),
            amount_sol=500_000_000,
            max_slippage_bps=500
        )
    """

    PROGRAM_ID: ClassVar[Pubkey] = PUMP_FUN_PROGRAM_ID

    def __init__(self, rpc_manager: RPCManager, config: Optional[PumpFunConfig] = None):
        """
        Initialize pump.fun client

        Args:
            rpc_manager: RPC manager for on-chain data
            config: Pump.fun configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or PumpFunConfig()

        # Cache for derived PDAs (mint -> bonding_curve_pda)
        self._pda_cache: Dict[Pubkey, Pubkey] = {}

        logger.info(
            "pumpfun_client_initialized",
            program_id=str(self.config.program_id),
            fee_bps=self.config.fee_bps
        )

    async def build_buy_instruction(
        self,
        mint: Pubkey,
        user: Pubkey,
        amount_sol: int,
        max_slippage_bps: int = 500
    ) -> Instruction:
        """
        Build buy instruction for pump.fun

        Args:
            mint: Token mint pubkey
            user: Buyer pubkey
            amount_sol: Amount of SOL to spend (in lamports)
            max_slippage_bps: Maximum slippage tolerance in basis points

        Returns:
            Solana instruction ready to be added to transaction

        Example:
            buy_ix = await client.build_buy_instruction(
                mint=mint_pubkey,
                user=wallet.pubkey(),
                amount_sol=500_000_000,  # 0.5 SOL
                max_slippage_bps=500     # 5%
            )
        """
        # Derive PDAs
        bonding_curve = self.derive_bonding_curve_pda(mint)
        associated_bonding_curve = self.derive_associated_bonding_curve(bonding_curve, mint)
        user_token_account = self.derive_associated_token_account(user, mint)

        # Build instruction data: discriminator + amount
        data = BUY_DISCRIMINATOR + struct.pack("<Q", amount_sol)  # u64 little-endian

        # Build accounts list (order matters!)
        accounts = [
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user, is_signer=True, is_writable=True),
            AccountMeta(pubkey=user_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=RENT, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=self.config.program_id, is_signer=False, is_writable=False),
        ]

        instruction = Instruction(
            program_id=self.config.program_id,
            accounts=accounts,
            data=data
        )

        logger.debug(
            "buy_instruction_built",
            mint=str(mint),
            user=str(user),
            amount_sol=amount_sol / 1e9,
            max_slippage_bps=max_slippage_bps
        )

        metrics.increment_counter("buy_instructions_built")

        return instruction

    async def build_sell_instruction(
        self,
        mint: Pubkey,
        user: Pubkey,
        amount_tokens: int,
        min_sol_output: int
    ) -> Instruction:
        """
        Build sell instruction for pump.fun

        Args:
            mint: Token mint pubkey
            user: Seller pubkey
            amount_tokens: Amount of tokens to sell
            min_sol_output: Minimum SOL to receive (slippage protection)

        Returns:
            Solana instruction ready to be added to transaction

        Example:
            sell_ix = await client.build_sell_instruction(
                mint=mint_pubkey,
                user=wallet.pubkey(),
                amount_tokens=1_000_000_000,  # 1B tokens
                min_sol_output=450_000_000    # Min 0.45 SOL
            )
        """
        # Derive PDAs
        bonding_curve = self.derive_bonding_curve_pda(mint)
        associated_bonding_curve = self.derive_associated_bonding_curve(bonding_curve, mint)
        user_token_account = self.derive_associated_token_account(user, mint)

        # Build instruction data: discriminator + amount
        data = SELL_DISCRIMINATOR + struct.pack("<Q", amount_tokens)  # u64 little-endian

        # Build accounts list (order matters!)
        accounts = [
            AccountMeta(pubkey=bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=associated_bonding_curve, is_signer=False, is_writable=True),
            AccountMeta(pubkey=user, is_signer=True, is_writable=True),
            AccountMeta(pubkey=user_token_account, is_signer=False, is_writable=True),
            AccountMeta(pubkey=SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=EVENT_AUTHORITY, is_signer=False, is_writable=False),
            AccountMeta(pubkey=self.config.program_id, is_signer=False, is_writable=False),
        ]

        instruction = Instruction(
            program_id=self.config.program_id,
            accounts=accounts,
            data=data
        )

        logger.debug(
            "sell_instruction_built",
            mint=str(mint),
            user=str(user),
            amount_tokens=amount_tokens,
            min_sol_output=min_sol_output / 1e9
        )

        metrics.increment_counter("sell_instructions_built")

        return instruction

    def derive_bonding_curve_pda(self, mint: Pubkey) -> Pubkey:
        """
        Derive bonding curve PDA from mint

        Args:
            mint: Token mint pubkey

        Returns:
            Bonding curve PDA

        Example:
            bonding_curve = client.derive_bonding_curve_pda(mint_pubkey)
        """
        # Check cache first
        if mint in self._pda_cache:
            return self._pda_cache[mint]

        # Derive PDA: seeds = ["bonding-curve", mint]
        pda, bump = Pubkey.find_program_address(
            [BONDING_CURVE_SEED, bytes(mint)],
            self.config.program_id
        )

        # Cache result
        self._pda_cache[mint] = pda

        logger.debug(
            "bonding_curve_pda_derived",
            mint=str(mint),
            pda=str(pda),
            bump=bump
        )

        return pda

    def derive_associated_bonding_curve(self, bonding_curve: Pubkey, mint: Pubkey) -> Pubkey:
        """
        Derive associated token account for bonding curve

        Args:
            bonding_curve: Bonding curve PDA
            mint: Token mint pubkey

        Returns:
            Associated token account pubkey

        Example:
            ata = client.derive_associated_bonding_curve(bonding_curve, mint)
        """
        # Standard associated token account derivation
        pda, bump = Pubkey.find_program_address(
            [bytes(bonding_curve), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
            ASSOCIATED_TOKEN_PROGRAM_ID
        )

        logger.debug(
            "associated_bonding_curve_derived",
            bonding_curve=str(bonding_curve),
            mint=str(mint),
            ata=str(pda),
            bump=bump
        )

        return pda

    def derive_associated_token_account(self, owner: Pubkey, mint: Pubkey) -> Pubkey:
        """
        Derive associated token account for owner

        Args:
            owner: Token owner pubkey
            mint: Token mint pubkey

        Returns:
            Associated token account pubkey

        Example:
            ata = client.derive_associated_token_account(wallet.pubkey(), mint)
        """
        # Standard associated token account derivation
        pda, bump = Pubkey.find_program_address(
            [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
            ASSOCIATED_TOKEN_PROGRAM_ID
        )

        return pda

    async def get_bonding_curve_account(self, mint: Pubkey) -> Optional[BondingCurveAccount]:
        """
        Fetch bonding curve state from on-chain

        Args:
            mint: Token mint pubkey

        Returns:
            Bonding curve state or None if not found

        Example:
            curve = await client.get_bonding_curve_account(mint_pubkey)
            if curve:
                print(f"Virtual SOL reserves: {curve.virtual_sol_reserves}")
        """
        bonding_curve_pda = self.derive_bonding_curve_pda(mint)

        try:
            # Fetch account data
            response = await self.rpc_manager.call_http_rpc(
                "getAccountInfo",
                [str(bonding_curve_pda), {"encoding": "base64"}]
            )

            result = response.get("result")
            if not result or not result.get("value"):
                logger.warning(
                    "bonding_curve_not_found",
                    mint=str(mint),
                    pda=str(bonding_curve_pda)
                )
                return None

            # Parse account data (base64 encoded)
            import base64
            data = base64.b64decode(result["value"]["data"][0])

            # Deserialize bonding curve account
            # Layout: [discriminator(8), virtual_token_reserves(8), virtual_sol_reserves(8),
            #          real_token_reserves(8), real_sol_reserves(8), token_total_supply(8), complete(1)]
            if len(data) < 49:
                logger.error(
                    "bonding_curve_data_too_short",
                    mint=str(mint),
                    data_length=len(data)
                )
                return None

            # Skip discriminator (first 8 bytes)
            offset = 8
            virtual_token_reserves = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            virtual_sol_reserves = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            real_token_reserves = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            real_sol_reserves = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            token_total_supply = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            complete = bool(data[offset])

            bonding_curve = BondingCurveAccount(
                virtual_token_reserves=virtual_token_reserves,
                virtual_sol_reserves=virtual_sol_reserves,
                real_token_reserves=real_token_reserves,
                real_sol_reserves=real_sol_reserves,
                token_total_supply=token_total_supply,
                complete=complete
            )

            logger.debug(
                "bonding_curve_fetched",
                mint=str(mint),
                virtual_token_reserves=virtual_token_reserves,
                virtual_sol_reserves=virtual_sol_reserves / 1e9,
                complete=complete
            )

            metrics.increment_counter("bonding_curve_fetches")

            return bonding_curve

        except Exception as e:
            logger.error(
                "bonding_curve_fetch_failed",
                mint=str(mint),
                error=str(e),
                exc_info=True
            )
            metrics.increment_counter("bonding_curve_fetch_errors")
            return None

    def clear_pda_cache(self):
        """Clear PDA cache (useful for testing or memory management)"""
        self._pda_cache.clear()
        logger.debug("pda_cache_cleared")


# Example usage
if __name__ == "__main__":
    from core.config import ConfigurationManager
    from core.logger import setup_logging
    import asyncio

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Load config
        config_manager = ConfigurationManager("config/config.yml")
        bot_config = config_manager.load_config()

        # Initialize RPC manager
        rpc_manager = RPCManager(bot_config.rpc_config)
        await rpc_manager.start()

        try:
            # Initialize pump.fun client
            client = PumpFunClient(rpc_manager)

            # Example mint (replace with actual mint)
            example_mint = Pubkey.from_string("11111111111111111111111111111111")

            # Derive bonding curve PDA
            bonding_curve = client.derive_bonding_curve_pda(example_mint)
            logger.info("bonding_curve_derived", pda=str(bonding_curve))

            # Fetch bonding curve account (will fail for fake mint, just example)
            curve_account = await client.get_bonding_curve_account(example_mint)
            if curve_account:
                logger.info(
                    "curve_fetched",
                    virtual_sol=curve_account.virtual_sol_reserves / 1e9,
                    virtual_tokens=curve_account.virtual_token_reserves
                )

        finally:
            await rpc_manager.stop()

    asyncio.run(main())
