"""
Unit tests for Transaction Signer (core/tx_signer.py)

Tests:
- Keypair management
- Transaction signing
- Message signing
- Signature statistics
- Key rotation
- Error handling
"""

import pytest
from datetime import datetime, timedelta, timezone

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.hash import Hash
from solders.transaction import Transaction
from solders.message import Message
from solders.instruction import Instruction, AccountMeta

from core.tx_signer import (
    TransactionSigner,
    SignerConfig,
    SignatureStats
)


class TestSignerConfig:
    """Test signer configuration"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = SignerConfig()

        assert config.enable_key_rotation is True
        assert config.max_signatures_per_key == 10_000
        assert config.key_rotation_interval_minutes == 60
        assert config.track_signature_count is True

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = SignerConfig(
            enable_key_rotation=False,
            max_signatures_per_key=5_000,
            key_rotation_interval_minutes=30,
            track_signature_count=False
        )

        assert config.enable_key_rotation is False
        assert config.max_signatures_per_key == 5_000
        assert config.key_rotation_interval_minutes == 30
        assert config.track_signature_count is False


class TestSignatureStats:
    """Test signature statistics"""

    def test_stats_creation(self):
        """Test creating signature stats"""
        pubkey = Keypair().pubkey()
        stats = SignatureStats(
            pubkey=pubkey,
            total_signatures=10,
            last_used=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        )

        assert stats.pubkey == pubkey
        assert stats.total_signatures == 10

    def test_stats_to_dict(self):
        """Test converting stats to dictionary"""
        pubkey = Keypair().pubkey()
        now = datetime.now(timezone.utc)
        stats = SignatureStats(
            pubkey=pubkey,
            total_signatures=10,
            last_used=now,
            created_at=now
        )

        stats_dict = stats.to_dict()

        assert "pubkey" in stats_dict
        assert "total_signatures" in stats_dict
        assert "last_used" in stats_dict
        assert "created_at" in stats_dict
        assert stats_dict["total_signatures"] == 10


class TestTransactionSigner:
    """Test transaction signer functionality"""

    def test_signer_initialization_empty(self):
        """Test signer initializes with no keypairs"""
        signer = TransactionSigner()

        assert len(signer.get_all_pubkeys()) == 0
        assert signer.config is not None

    def test_signer_initialization_with_keypairs(self):
        """Test signer initializes with keypairs"""
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs)

        assert len(signer.get_all_pubkeys()) == 3

    def test_signer_initialization_with_config(self):
        """Test signer initializes with custom config"""
        config = SignerConfig(enable_key_rotation=False)
        signer = TransactionSigner(config=config)

        assert signer.config.enable_key_rotation is False

    def test_add_keypair(self):
        """Test adding a keypair"""
        signer = TransactionSigner()
        keypair = Keypair()

        signer.add_keypair(keypair)

        assert signer.has_keypair(keypair.pubkey())
        assert len(signer.get_all_pubkeys()) == 1

    def test_add_duplicate_keypair(self):
        """Test adding duplicate keypair is handled"""
        signer = TransactionSigner()
        keypair = Keypair()

        signer.add_keypair(keypair)
        signer.add_keypair(keypair)  # Add again

        # Should only have one
        assert len(signer.get_all_pubkeys()) == 1

    def test_remove_keypair(self):
        """Test removing a keypair"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        assert signer.has_keypair(keypair.pubkey())

        signer.remove_keypair(keypair.pubkey())

        assert not signer.has_keypair(keypair.pubkey())
        assert len(signer.get_all_pubkeys()) == 0

    def test_remove_nonexistent_keypair(self):
        """Test removing keypair that doesn't exist"""
        signer = TransactionSigner()
        keypair = Keypair()

        # Should not raise error
        signer.remove_keypair(keypair.pubkey())

    def test_get_keypair(self):
        """Test retrieving a keypair"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        retrieved = signer.get_keypair(keypair.pubkey())

        assert retrieved is not None
        assert retrieved.pubkey() == keypair.pubkey()

    def test_get_nonexistent_keypair(self):
        """Test retrieving keypair that doesn't exist"""
        signer = TransactionSigner()
        keypair = Keypair()

        retrieved = signer.get_keypair(keypair.pubkey())

        assert retrieved is None

    def test_has_keypair(self):
        """Test checking if keypair exists"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        assert signer.has_keypair(keypair.pubkey()) is True
        assert signer.has_keypair(Keypair().pubkey()) is False

    def test_get_all_pubkeys(self):
        """Test getting all public keys"""
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs)

        pubkeys = signer.get_all_pubkeys()

        assert len(pubkeys) == 3
        for keypair in keypairs:
            assert keypair.pubkey() in pubkeys

    def test_sign_transaction(self):
        """Test signing a transaction"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        # Create unsigned transaction
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=bytes([0, 1, 2, 3])
        )
        message = Message.new_with_blockhash(
            [instruction],
            keypair.pubkey(),
            Hash.default()
        )
        unsigned_tx = Transaction.new_unsigned(message)

        # Sign it
        signed_tx = signer.sign_transaction(unsigned_tx, [keypair.pubkey()])

        assert signed_tx is not None

    def test_sign_transaction_with_missing_keypair(self):
        """Test signing transaction with missing keypair raises error"""
        signer = TransactionSigner()
        keypair = Keypair()

        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        message = Message.new_with_blockhash(
            [instruction],
            keypair.pubkey(),
            Hash.default()
        )
        unsigned_tx = Transaction.new_unsigned(message)

        with pytest.raises(KeyError):
            signer.sign_transaction(unsigned_tx, [keypair.pubkey()])

    def test_sign_message(self):
        """Test signing arbitrary message"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        message = b"Hello, Solana!"
        signature = signer.sign_message(message, keypair.pubkey())

        assert signature is not None

    def test_sign_message_with_missing_keypair(self):
        """Test signing message with missing keypair raises error"""
        signer = TransactionSigner()
        keypair = Keypair()

        with pytest.raises(KeyError):
            signer.sign_message(b"test", keypair.pubkey())

    def test_signature_stats_tracking(self):
        """Test signature statistics are tracked"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        # Initial stats should be zero
        stats = signer.get_signature_stats(keypair.pubkey())
        assert stats is not None
        assert stats.total_signatures == 0

        # Sign a transaction
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        message = Message.new_with_blockhash(
            [instruction],
            keypair.pubkey(),
            Hash.default()
        )
        unsigned_tx = Transaction.new_unsigned(message)
        signer.sign_transaction(unsigned_tx, [keypair.pubkey()])

        # Stats should be incremented
        stats = signer.get_signature_stats(keypair.pubkey())
        assert stats.total_signatures == 1

    def test_signature_stats_disabled(self):
        """Test signature tracking can be disabled"""
        config = SignerConfig(track_signature_count=False)
        keypair = Keypair()
        signer = TransactionSigner([keypair], config)

        # Sign a transaction
        instruction = Instruction(
            program_id=Pubkey.default(),
            accounts=[AccountMeta(pubkey=keypair.pubkey(), is_signer=True, is_writable=True)],
            data=bytes([0])
        )
        message = Message.new_with_blockhash(
            [instruction],
            keypair.pubkey(),
            Hash.default()
        )
        unsigned_tx = Transaction.new_unsigned(message)
        signer.sign_transaction(unsigned_tx, [keypair.pubkey()])

        # Stats should still be zero because tracking disabled
        stats = signer.get_signature_stats(keypair.pubkey())
        assert stats.total_signatures == 0

    def test_get_all_signature_stats(self):
        """Test getting all signature stats"""
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs)

        all_stats = signer.get_all_signature_stats()

        assert len(all_stats) == 3
        for keypair in keypairs:
            assert keypair.pubkey() in all_stats

    def test_needs_rotation_by_signature_count(self):
        """Test key rotation needed by signature count"""
        config = SignerConfig(max_signatures_per_key=10)
        keypair = Keypair()
        signer = TransactionSigner([keypair], config)

        # Manually set high signature count
        stats = signer.get_signature_stats(keypair.pubkey())
        stats.total_signatures = 10

        assert signer.needs_rotation(keypair.pubkey()) is True

    def test_needs_rotation_by_age(self):
        """Test key rotation needed by age"""
        config = SignerConfig(key_rotation_interval_minutes=60)
        keypair = Keypair()
        signer = TransactionSigner([keypair], config)

        # Manually set old creation time
        stats = signer.get_signature_stats(keypair.pubkey())
        stats.created_at = datetime.now(timezone.utc) - timedelta(minutes=61)

        assert signer.needs_rotation(keypair.pubkey()) is True

    def test_needs_rotation_disabled(self):
        """Test rotation check when disabled"""
        config = SignerConfig(enable_key_rotation=False)
        keypair = Keypair()
        signer = TransactionSigner([keypair], config)

        # Even with high signature count
        stats = signer.get_signature_stats(keypair.pubkey())
        stats.total_signatures = 10_000

        assert signer.needs_rotation(keypair.pubkey()) is False

    def test_get_keypairs_needing_rotation(self):
        """Test getting all keypairs needing rotation"""
        config = SignerConfig(max_signatures_per_key=10)
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs, config)

        # Set high signature count on first two
        for i in range(2):
            stats = signer.get_signature_stats(keypairs[i].pubkey())
            stats.total_signatures = 10

        needing_rotation = signer.get_keypairs_needing_rotation()

        assert len(needing_rotation) == 2

    def test_clear_all_keypairs(self):
        """Test clearing all keypairs"""
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs)

        assert len(signer.get_all_pubkeys()) == 3

        signer.clear_all_keypairs()

        assert len(signer.get_all_pubkeys()) == 0
        assert len(signer.get_all_signature_stats()) == 0

    def test_get_stats(self):
        """Test getting signer statistics"""
        keypairs = [Keypair() for _ in range(3)]
        signer = TransactionSigner(keypairs)

        stats = signer.get_stats()

        assert "total_keypairs" in stats
        assert "total_signatures" in stats
        assert "keypairs_needing_rotation" in stats
        assert stats["total_keypairs"] == 3
        assert stats["total_signatures"] == 0

    def test_get_stats_with_signatures(self):
        """Test stats reflect signature counts"""
        keypair = Keypair()
        signer = TransactionSigner([keypair])

        # Manually increment signature count
        sig_stats = signer.get_signature_stats(keypair.pubkey())
        sig_stats.total_signatures = 5

        stats = signer.get_stats()

        assert stats["total_signatures"] == 5
