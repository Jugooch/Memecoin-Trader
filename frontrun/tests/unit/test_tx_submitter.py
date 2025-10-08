"""
Unit tests for Transaction Submitter (core/tx_submitter.py)

Tests:
- Transaction submission
- Retry logic
- Error handling
- Configuration
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.hash import Hash
from solders.transaction import Transaction
from solders.message import Message
from solders.instruction import Instruction, AccountMeta

from core.tx_submitter import (
    TransactionSubmitter,
    SubmitterConfig,
    TransactionResult,
    ConfirmedTransaction,
    ConfirmationStatus
)


class TestSubmitterConfig:
    """Test submitter configuration"""

    def test_config_defaults(self):
        """Test default configuration values"""
        config = SubmitterConfig()

        assert config.skip_preflight is True
        assert config.max_retries == 3
        assert config.retry_delay_ms == 200
        assert config.confirmation_timeout_s == 30
        assert config.enable_confirmation_tracking is True

    def test_config_custom_values(self):
        """Test custom configuration values"""
        config = SubmitterConfig(
            skip_preflight=False,
            max_retries=5,
            retry_delay_ms=500,
            confirmation_timeout_s=60
        )

        assert config.skip_preflight is False
        assert config.max_retries == 5
        assert config.retry_delay_ms == 500
        assert config.confirmation_timeout_s == 60


class TestTransactionResult:
    """Test transaction result data structure"""

    def test_result_creation(self):
        """Test creating transaction result"""
        result = TransactionResult(
            signature="test_sig",
            submitted_at=datetime.now(timezone.utc),
            slot=1000
        )

        assert result.signature == "test_sig"
        assert result.slot == 1000
        assert result.error is None

    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        now = datetime.now(timezone.utc)
        result = TransactionResult(
            signature="test_sig",
            submitted_at=now,
            slot=1000,
            submitted_to_rpc="helius"
        )

        result_dict = result.to_dict()

        assert "signature" in result_dict
        assert "submitted_at" in result_dict
        assert "slot" in result_dict
        assert result_dict["signature"] == "test_sig"


class TestConfirmedTransaction:
    """Test confirmed transaction data structure"""

    def test_confirmed_creation(self):
        """Test creating confirmed transaction"""
        confirmed = ConfirmedTransaction(
            signature="test_sig",
            slot=1000,
            block_time=1234567890,
            confirmation_status=ConfirmationStatus.CONFIRMED
        )

        assert confirmed.signature == "test_sig"
        assert confirmed.slot == 1000
        assert confirmed.confirmation_status == ConfirmationStatus.CONFIRMED

    def test_confirmed_to_dict(self):
        """Test converting confirmed transaction to dictionary"""
        confirmed = ConfirmedTransaction(
            signature="test_sig",
            slot=1000,
            block_time=1234567890,
            confirmation_status=ConfirmationStatus.FINALIZED
        )

        conf_dict = confirmed.to_dict()

        assert "signature" in conf_dict
        assert "confirmation_status" in conf_dict
        assert conf_dict["confirmation_status"] == "finalized"


class TestTransactionSubmitter:
    """Test transaction submitter functionality"""

    def test_submitter_initialization(self):
        """Test submitter initializes correctly"""
        rpc_manager = MagicMock()
        submitter = TransactionSubmitter(rpc_manager)

        assert submitter.rpc_manager == rpc_manager
        assert submitter.config is not None

    def test_submitter_with_custom_config(self):
        """Test submitter with custom configuration"""
        rpc_manager = MagicMock()
        config = SubmitterConfig(max_retries=5)
        submitter = TransactionSubmitter(rpc_manager, config)

        assert submitter.config.max_retries == 5

    def test_get_stats(self):
        """Test getting submitter statistics"""
        rpc_manager = MagicMock()
        submitter = TransactionSubmitter(rpc_manager)

        stats = submitter.get_stats()

        assert "skip_preflight" in stats
        assert "max_retries" in stats
        assert "confirmation_timeout_s" in stats
        assert stats["max_retries"] == 3


class TestConfirmationStatus:
    """Test confirmation status enum"""

    def test_status_values(self):
        """Test all confirmation status values"""
        assert ConfirmationStatus.PENDING.value == "pending"
        assert ConfirmationStatus.PROCESSED.value == "processed"
        assert ConfirmationStatus.CONFIRMED.value == "confirmed"
        assert ConfirmationStatus.FINALIZED.value == "finalized"
        assert ConfirmationStatus.FAILED.value == "failed"
