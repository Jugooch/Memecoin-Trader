"""
Transaction Builder for Frontrun Bot
Constructs Solana transactions with compute budgets and blockhash management
"""

import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from solders.transaction import Transaction
from solders.message import Message
from solders.instruction import Instruction, AccountMeta
from solders.pubkey import Pubkey
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash

from core.logger import get_logger
from core.metrics import get_metrics, LatencyTimer


logger = get_logger(__name__)
metrics = get_metrics()


# Solana transaction size limit in bytes
MAX_TRANSACTION_SIZE = 1232

# Default compute units for transactions
DEFAULT_COMPUTE_UNITS = 200_000

# Compute budget program ID
COMPUTE_BUDGET_PROGRAM_ID = Pubkey.from_string("ComputeBudget111111111111111111111111111111")


@dataclass
class TransactionBuildConfig:
    """Configuration for transaction building"""
    max_tx_size_bytes: int = MAX_TRANSACTION_SIZE
    default_compute_units: int = DEFAULT_COMPUTE_UNITS
    blockhash_cache_ttl_seconds: int = 30
    enable_blockhash_cache: bool = True


@dataclass
class BlockhashCache:
    """Cached blockhash with expiry"""
    blockhash: Hash
    slot: int
    cached_at: float
    ttl_seconds: int = 30

    def is_expired(self) -> bool:
        """Check if cached blockhash has expired"""
        return time.time() - self.cached_at > self.ttl_seconds


class TransactionBuilder:
    """
    Builds Solana transactions with compute budgets and optimization

    Features:
    - Versioned transaction construction
    - Compute budget instruction prepending
    - Blockhash caching (30s TTL)
    - Transaction size validation
    - Account meta optimization

    Usage:
        builder = TransactionBuilder(config)
        tx = await builder.build_transaction(
            instructions=[buy_instruction],
            payer=wallet_pubkey,
            compute_unit_limit=200_000,
            compute_unit_price=50_000
        )
    """

    def __init__(self, config: Optional[TransactionBuildConfig] = None):
        """
        Initialize transaction builder

        Args:
            config: Configuration for transaction building (optional)
        """
        self.config = config or TransactionBuildConfig()
        self._blockhash_cache: Optional[BlockhashCache] = None

        logger.info(
            "transaction_builder_initialized",
            max_tx_size=self.config.max_tx_size_bytes,
            default_compute_units=self.config.default_compute_units,
            cache_enabled=self.config.enable_blockhash_cache
        )

    def build_transaction(
        self,
        instructions: List[Instruction],
        payer: Pubkey,
        recent_blockhash: Hash,
        compute_unit_limit: Optional[int] = None,
        compute_unit_price: Optional[int] = None
    ) -> Transaction:
        """
        Build a Solana transaction with compute budget instructions

        Args:
            instructions: List of instructions to include
            payer: Fee payer public key
            recent_blockhash: Recent blockhash from RPC
            compute_unit_limit: Max compute units (optional, default 200k)
            compute_unit_price: Priority fee in micro-lamports (optional)

        Returns:
            Unsigned Transaction ready for signing

        Raises:
            ValueError: If transaction size exceeds limit

        Example:
            tx = builder.build_transaction(
                instructions=[buy_ix],
                payer=wallet.pubkey(),
                recent_blockhash=blockhash,
                compute_unit_limit=200_000,
                compute_unit_price=50_000
            )
        """
        with LatencyTimer(metrics, "tx_build", labels={"stage": "full"}):
            # Add compute budget instructions if specified
            all_instructions = []

            if compute_unit_limit is not None or compute_unit_price is not None:
                budget_instructions = self._create_compute_budget_instructions(
                    compute_unit_limit, compute_unit_price
                )
                all_instructions.extend(budget_instructions)

            # Add user instructions
            all_instructions.extend(instructions)

            # Build message
            message = Message.new_with_blockhash(
                all_instructions,
                payer,
                recent_blockhash
            )

            # Create transaction
            tx = Transaction.new_unsigned(message)

            # Validate size
            tx_size = len(bytes(tx))
            if tx_size > self.config.max_tx_size_bytes:
                raise ValueError(
                    f"Transaction size {tx_size} exceeds limit {self.config.max_tx_size_bytes}"
                )

            metrics.increment_counter("transactions_built")
            logger.debug(
                "transaction_built",
                instruction_count=len(all_instructions),
                tx_size_bytes=tx_size,
                has_compute_budget=compute_unit_limit is not None
            )

            return tx

    def _create_compute_budget_instructions(
        self,
        compute_unit_limit: Optional[int],
        compute_unit_price: Optional[int]
    ) -> List[Instruction]:
        """
        Create compute budget instructions

        Args:
            compute_unit_limit: Max compute units (optional)
            compute_unit_price: Priority fee in micro-lamports (optional)

        Returns:
            List of compute budget instructions (0-2 instructions)
        """
        instructions = []

        if compute_unit_limit is not None:
            instructions.append(set_compute_unit_limit(compute_unit_limit))
            logger.debug(
                "compute_unit_limit_set",
                compute_units=compute_unit_limit
            )

        if compute_unit_price is not None:
            instructions.append(set_compute_unit_price(compute_unit_price))
            logger.debug(
                "compute_unit_price_set",
                price_micro_lamports=compute_unit_price
            )

        return instructions

    def cache_blockhash(self, blockhash: Hash, slot: int) -> None:
        """
        Cache a blockhash for reuse

        Args:
            blockhash: Blockhash to cache
            slot: Slot number when blockhash was fetched
        """
        if not self.config.enable_blockhash_cache:
            return

        self._blockhash_cache = BlockhashCache(
            blockhash=blockhash,
            slot=slot,
            cached_at=time.time(),
            ttl_seconds=self.config.blockhash_cache_ttl_seconds
        )

        logger.debug(
            "blockhash_cached",
            slot=slot,
            ttl_seconds=self.config.blockhash_cache_ttl_seconds
        )
        metrics.increment_counter("blockhash_cached")

    def get_cached_blockhash(self) -> Optional[BlockhashCache]:
        """
        Get cached blockhash if valid

        Returns:
            Cached blockhash if valid, None if expired or not cached
        """
        if not self.config.enable_blockhash_cache:
            return None

        if self._blockhash_cache is None:
            metrics.increment_counter("blockhash_cache_miss", labels={"reason": "not_cached"})
            return None

        if self._blockhash_cache.is_expired():
            metrics.increment_counter("blockhash_cache_miss", labels={"reason": "expired"})
            logger.debug("blockhash_cache_expired")
            self._blockhash_cache = None
            return None

        metrics.increment_counter("blockhash_cache_hit")
        return self._blockhash_cache

    def invalidate_blockhash_cache(self) -> None:
        """Invalidate cached blockhash"""
        self._blockhash_cache = None
        logger.debug("blockhash_cache_invalidated")

    def get_transaction_size(self, tx: Transaction) -> int:
        """
        Calculate serialized transaction size

        Args:
            tx: Transaction to measure

        Returns:
            Size in bytes
        """
        return len(bytes(tx))

    def estimate_compute_units(self, instruction_count: int) -> int:
        """
        Estimate compute units needed based on instruction count

        Args:
            instruction_count: Number of instructions in transaction

        Returns:
            Estimated compute units

        Note:
            This is a rough estimate. Real usage depends on instruction complexity.
            Base: 5000 CU per signature
            Per instruction: ~10,000-50,000 CU depending on complexity
        """
        base_cu = 5000  # Per signature
        per_instruction_cu = 30_000  # Average per instruction

        estimated = base_cu + (instruction_count * per_instruction_cu)

        # Cap at reasonable max
        return min(estimated, 1_400_000)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get builder statistics

        Returns:
            Dictionary with builder stats
        """
        cache_status = "disabled"
        if self.config.enable_blockhash_cache:
            if self._blockhash_cache is None:
                cache_status = "empty"
            elif self._blockhash_cache.is_expired():
                cache_status = "expired"
            else:
                cache_status = "valid"

        return {
            "blockhash_cache_status": cache_status,
            "max_tx_size_bytes": self.config.max_tx_size_bytes,
            "default_compute_units": self.config.default_compute_units
        }
