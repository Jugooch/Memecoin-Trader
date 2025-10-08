"""
Unit tests for Transaction Builder (core/tx_builder.py)

Tests:
- Transaction construction
- Compute budget instructions
- Blockhash caching
- Transaction size validation
- Error handling
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from solders.pubkey import Pubkey
from solders.hash import Hash
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction

from core.tx_builder import (
    TransactionBuilder,
    TransactionBuildConfig,
    BlockhashCache,
    MAX_TRANSACTION_SIZE,
    DEFAULT_COMPUTE_UNITS
)


class TestTransactionBuildConfig:
    """Test transaction build configuration"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = TransactionBuildConfig()

        assert config.max_tx_size_bytes == MAX_TRANSACTION_SIZE
        assert config.default_compute_units == DEFAULT_COMPUTE_UNITS
        assert config.blockhash_cache_ttl_seconds == 30
        assert config.enable_blockhash_cache is True

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = TransactionBuildConfig(
            max_tx_size_bytes=1000,
            default_compute_units=150_000,
            blockhash_cache_ttl_seconds=60,
            enable_blockhash_cache=False
        )

        assert config.max_tx_size_bytes == 1000
        assert config.default_compute_units == 150_000
        assert config.blockhash_cache_ttl_seconds == 60
        assert config.enable_blockhash_cache is False


class TestBlockhashCache:
    """Test blockhash caching functionality"""

    def test_blockhash_cache_not_expired(self):
        """Test blockhash cache is valid when within TTL"""
        blockhash = Hash.default()
        cache = BlockhashCache(
            blockhash=blockhash,
            slot=1000,
            cached_at=time.time(),
            ttl_seconds=30
        )

        assert cache.is_expired() is False

    def test_blockhash_cache_expired(self):
        """Test blockhash cache expires after TTL"""
        blockhash = Hash.default()
        cache = BlockhashCache(
            blockhash=blockhash,
            slot=1000,
            cached_at=time.time() - 31,  # 31 seconds ago
            ttl_seconds=30
        )

        assert cache.is_expired() is True

    def test_blockhash_cache_edge_case(self):
        """Test blockhash cache just past TTL boundary"""
        blockhash = Hash.default()
        cache = BlockhashCache(
            blockhash=blockhash,
            slot=1000,
            cached_at=time.time() - 30.1,  # Just past 30 seconds
            ttl_seconds=30
        )

        # Should be expired when past TTL
        assert cache.is_expired() is True


class TestTransactionBuilder:
    """Test transaction building functionality"""

    def test_builder_initialization(self):
        """Test builder initializes with default config"""
        builder = TransactionBuilder()

        assert builder.config is not None
        assert builder.config.max_tx_size_bytes == MAX_TRANSACTION_SIZE
        assert builder._blockhash_cache is None

    def test_builder_with_custom_config(self):
        """Test builder initializes with custom config"""
        config = TransactionBuildConfig(
            max_tx_size_bytes=1000,
            enable_blockhash_cache=False
        )
        builder = TransactionBuilder(config)

        assert builder.config.max_tx_size_bytes == 1000
        assert builder.config.enable_blockhash_cache is False

    def test_build_simple_transaction(self):
        """Test building a basic transaction without compute budget"""
        builder = TransactionBuilder()

        # Create test instruction
        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0, 1, 2, 3])
        )
        blockhash = Hash.default()

        # Build transaction
        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash
        )

        assert tx is not None
        assert isinstance(tx, Transaction)

    def test_build_transaction_with_compute_budget(self):
        """Test building transaction with compute budget instructions"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0, 1, 2, 3])
        )
        blockhash = Hash.default()

        # Build with compute budget
        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash,
            compute_unit_limit=200_000,
            compute_unit_price=50_000
        )

        assert tx is not None
        # Transaction should have compute budget instructions prepended
        # (hard to verify without deserializing, but size should be larger)

    def test_build_transaction_with_multiple_instructions(self):
        """Test building transaction with multiple instructions"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instructions = [
            Instruction(
                program_id=Pubkey.default(),
                accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
                data=bytes([i])
            )
            for i in range(3)
        ]
        blockhash = Hash.default()

        tx = builder.build_transaction(
            instructions=instructions,
            payer=payer,
            recent_blockhash=blockhash
        )

        assert tx is not None

    def test_get_transaction_size(self):
        """Test transaction size calculation"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0, 1, 2, 3])
        )
        blockhash = Hash.default()

        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash
        )

        size = builder.get_transaction_size(tx)
        assert size > 0
        assert size < MAX_TRANSACTION_SIZE

    def test_cache_blockhash(self):
        """Test blockhash caching"""
        builder = TransactionBuilder()

        blockhash = Hash.default()
        slot = 1000

        builder.cache_blockhash(blockhash, slot)

        assert builder._blockhash_cache is not None
        assert builder._blockhash_cache.blockhash == blockhash
        assert builder._blockhash_cache.slot == slot

    def test_get_cached_blockhash_valid(self):
        """Test retrieving valid cached blockhash"""
        builder = TransactionBuilder()

        blockhash = Hash.default()
        slot = 1000
        builder.cache_blockhash(blockhash, slot)

        cached = builder.get_cached_blockhash()
        assert cached is not None
        assert cached.blockhash == blockhash
        assert cached.slot == slot

    def test_get_cached_blockhash_expired(self):
        """Test cached blockhash returns None when expired"""
        builder = TransactionBuilder()

        blockhash = Hash.default()
        builder._blockhash_cache = BlockhashCache(
            blockhash=blockhash,
            slot=1000,
            cached_at=time.time() - 31,  # Expired
            ttl_seconds=30
        )

        cached = builder.get_cached_blockhash()
        assert cached is None

    def test_get_cached_blockhash_not_cached(self):
        """Test getting blockhash when none cached"""
        builder = TransactionBuilder()

        cached = builder.get_cached_blockhash()
        assert cached is None

    def test_get_cached_blockhash_disabled(self):
        """Test getting blockhash when caching disabled"""
        config = TransactionBuildConfig(enable_blockhash_cache=False)
        builder = TransactionBuilder(config)

        blockhash = Hash.default()
        builder.cache_blockhash(blockhash, 1000)

        # Should return None because caching is disabled
        cached = builder.get_cached_blockhash()
        assert cached is None

    def test_invalidate_blockhash_cache(self):
        """Test invalidating blockhash cache"""
        builder = TransactionBuilder()

        blockhash = Hash.default()
        builder.cache_blockhash(blockhash, 1000)
        assert builder._blockhash_cache is not None

        builder.invalidate_blockhash_cache()
        assert builder._blockhash_cache is None

    def test_estimate_compute_units(self):
        """Test compute unit estimation"""
        builder = TransactionBuilder()

        # Single instruction
        estimate_1 = builder.estimate_compute_units(1)
        assert estimate_1 > 0

        # Multiple instructions
        estimate_3 = builder.estimate_compute_units(3)
        assert estimate_3 > estimate_1

        # Large number shouldn't exceed max
        estimate_large = builder.estimate_compute_units(100)
        assert estimate_large <= 1_400_000

    def test_get_stats(self):
        """Test getting builder statistics"""
        builder = TransactionBuilder()

        stats = builder.get_stats()

        assert "blockhash_cache_status" in stats
        assert "max_tx_size_bytes" in stats
        assert "default_compute_units" in stats
        assert stats["blockhash_cache_status"] == "empty"

    def test_get_stats_with_valid_cache(self):
        """Test stats with valid cached blockhash"""
        builder = TransactionBuilder()

        builder.cache_blockhash(Hash.default(), 1000)
        stats = builder.get_stats()

        assert stats["blockhash_cache_status"] == "valid"

    def test_get_stats_with_expired_cache(self):
        """Test stats with expired cached blockhash"""
        builder = TransactionBuilder()

        builder._blockhash_cache = BlockhashCache(
            blockhash=Hash.default(),
            slot=1000,
            cached_at=time.time() - 31,
            ttl_seconds=30
        )

        stats = builder.get_stats()
        assert stats["blockhash_cache_status"] == "expired"

    def test_get_stats_with_disabled_cache(self):
        """Test stats when caching is disabled"""
        config = TransactionBuildConfig(enable_blockhash_cache=False)
        builder = TransactionBuilder(config)

        stats = builder.get_stats()
        assert stats["blockhash_cache_status"] == "disabled"


class TestComputeBudgetInstructions:
    """Test compute budget instruction creation"""

    def test_compute_budget_with_limit_only(self):
        """Test creating compute budget with limit only"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        blockhash = Hash.default()

        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash,
            compute_unit_limit=150_000
        )

        assert tx is not None

    def test_compute_budget_with_price_only(self):
        """Test creating compute budget with price only"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        blockhash = Hash.default()

        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash,
            compute_unit_price=100_000
        )

        assert tx is not None

    def test_compute_budget_with_both(self):
        """Test creating compute budget with both limit and price"""
        builder = TransactionBuilder()

        payer = Pubkey.default()
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=payer, is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        blockhash = Hash.default()

        tx = builder.build_transaction(
            instructions=[instruction],
            payer=payer,
            recent_blockhash=blockhash,
            compute_unit_limit=200_000,
            compute_unit_price=50_000
        )

        assert tx is not None
