"""
Transaction Signer for Frontrun Bot
Signs transactions with Ed25519 keypairs and manages hot wallet keys
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.transaction import Transaction
from solders.signature import Signature
import base58

from core.logger import get_logger
from core.metrics import get_metrics, LatencyTimer


logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class SignerConfig:
    """Configuration for transaction signer"""
    enable_key_rotation: bool = True
    max_signatures_per_key: int = 10_000
    key_rotation_interval_minutes: int = 60
    track_signature_count: bool = True


@dataclass
class SignatureStats:
    """Statistics for a keypair"""
    pubkey: Pubkey
    total_signatures: int
    last_used: datetime
    created_at: datetime

    def to_dict(self) -> Dict:
        """Convert stats to dictionary"""
        return {
            "pubkey": str(self.pubkey),
            "total_signatures": self.total_signatures,
            "last_used": self.last_used.isoformat(),
            "created_at": self.created_at.isoformat()
        }


class TransactionSigner:
    """
    Signs Solana transactions with Ed25519 keypairs

    Features:
    - Hot wallet key management (in-memory only)
    - Transaction signing
    - Signature counting for key rotation
    - Multi-keypair support

    Security Notes:
    - Keypairs stored in memory only, never persisted to disk
    - Use key rotation to limit exposure
    - Track signature counts to trigger rotation

    Usage:
        signer = TransactionSigner([keypair1, keypair2])
        signed_tx = signer.sign_transaction(tx, [keypair1.pubkey()])
    """

    def __init__(
        self,
        keypairs: Optional[List[Keypair]] = None,
        config: Optional[SignerConfig] = None
    ):
        """
        Initialize transaction signer

        Args:
            keypairs: List of keypairs to manage (optional)
            config: Signer configuration (optional)
        """
        self.config = config or SignerConfig()
        self._keypairs: Dict[Pubkey, Keypair] = {}
        self._signature_stats: Dict[Pubkey, SignatureStats] = {}

        # Add initial keypairs
        if keypairs:
            for keypair in keypairs:
                self.add_keypair(keypair)

        logger.info(
            "transaction_signer_initialized",
            keypair_count=len(self._keypairs),
            key_rotation_enabled=self.config.enable_key_rotation,
            track_signatures=self.config.track_signature_count
        )

    def sign_transaction(
        self,
        transaction: Transaction,
        signers: List[Pubkey]
    ) -> Transaction:
        """
        Sign a transaction with specified keypairs

        Args:
            transaction: Unsigned transaction
            signers: List of public keys to sign with

        Returns:
            Signed transaction

        Raises:
            KeyError: If signer keypair not found
            ValueError: If transaction already signed

        Example:
            signed_tx = signer.sign_transaction(tx, [wallet.pubkey()])
        """
        with LatencyTimer(metrics, "tx_sign", labels={"signer_count": str(len(signers))}):
            # Verify all signers are available
            for signer_pubkey in signers:
                if signer_pubkey not in self._keypairs:
                    raise KeyError(f"Keypair not found for {signer_pubkey}")

            # Sign with each keypair
            signed_tx = transaction
            for signer_pubkey in signers:
                keypair = self._keypairs[signer_pubkey]
                signed_tx = self._sign_with_keypair(signed_tx, keypair)

                # Update signature stats
                if self.config.track_signature_count:
                    self._increment_signature_count(signer_pubkey)

            metrics.increment_counter("transactions_signed")
            logger.debug(
                "transaction_signed",
                signer_count=len(signers),
                signers=[str(s) for s in signers]
            )

            return signed_tx

    def _sign_with_keypair(
        self,
        transaction: Transaction,
        keypair: Keypair
    ) -> Transaction:
        """
        Sign transaction with a single keypair

        Args:
            transaction: Transaction to sign
            keypair: Keypair to sign with

        Returns:
            Signed transaction
        """
        # Extract recent blockhash from message
        # In solders, we need to use partial_sign to sign with one keypair at a time
        try:
            # Try to use partial_sign if available
            transaction.partial_sign([keypair], transaction.message.recent_blockhash)
            return transaction
        except AttributeError:
            # Fallback: create new signed transaction
            # This creates a fully signed transaction from the message
            from solders.transaction import Transaction as SoldersTransaction
            signed = SoldersTransaction([keypair], transaction.message, transaction.message.recent_blockhash)
            return signed

    def sign_message(self, message: bytes, signer: Pubkey) -> Signature:
        """
        Sign arbitrary message with keypair

        Args:
            message: Message bytes to sign
            signer: Public key of signer

        Returns:
            Ed25519 signature

        Raises:
            KeyError: If signer keypair not found
        """
        if signer not in self._keypairs:
            raise KeyError(f"Keypair not found for {signer}")

        keypair = self._keypairs[signer]
        signature = keypair.sign_message(message)

        logger.debug(
            "message_signed",
            signer=str(signer),
            message_length=len(message)
        )

        return signature

    def add_keypair(self, keypair: Keypair) -> None:
        """
        Add keypair to signer

        Args:
            keypair: Keypair to add
        """
        pubkey = keypair.pubkey()

        if pubkey in self._keypairs:
            logger.warning(
                "keypair_already_exists",
                pubkey=str(pubkey)
            )
            return

        self._keypairs[pubkey] = keypair

        # Initialize stats
        self._signature_stats[pubkey] = SignatureStats(
            pubkey=pubkey,
            total_signatures=0,
            last_used=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        )

        metrics.increment_counter("keypairs_added")
        logger.info(
            "keypair_added",
            pubkey=str(pubkey),
            total_keypairs=len(self._keypairs)
        )

    def remove_keypair(self, pubkey: Pubkey) -> None:
        """
        Remove keypair from signer (for rotation)

        Args:
            pubkey: Public key of keypair to remove

        Security Note:
            This only removes from memory. Keypair is not securely erased.
        """
        if pubkey not in self._keypairs:
            logger.warning(
                "keypair_not_found_for_removal",
                pubkey=str(pubkey)
            )
            return

        # Remove keypair and stats
        del self._keypairs[pubkey]
        if pubkey in self._signature_stats:
            del self._signature_stats[pubkey]

        metrics.increment_counter("keypairs_removed")
        logger.info(
            "keypair_removed",
            pubkey=str(pubkey),
            remaining_keypairs=len(self._keypairs)
        )

    def get_keypair(self, pubkey: Pubkey) -> Optional[Keypair]:
        """
        Retrieve keypair by public key

        Args:
            pubkey: Public key to lookup

        Returns:
            Keypair if found, None otherwise
        """
        return self._keypairs.get(pubkey)

    def has_keypair(self, pubkey: Pubkey) -> bool:
        """
        Check if keypair exists

        Args:
            pubkey: Public key to check

        Returns:
            True if keypair exists
        """
        return pubkey in self._keypairs

    def get_all_pubkeys(self) -> List[Pubkey]:
        """
        Get all managed public keys

        Returns:
            List of public keys
        """
        return list(self._keypairs.keys())

    def get_signature_stats(self, pubkey: Pubkey) -> Optional[SignatureStats]:
        """
        Get signature statistics for a keypair

        Args:
            pubkey: Public key to get stats for

        Returns:
            SignatureStats if found, None otherwise
        """
        return self._signature_stats.get(pubkey)

    def get_all_signature_stats(self) -> Dict[Pubkey, SignatureStats]:
        """
        Get signature statistics for all keypairs

        Returns:
            Dictionary of public key to stats
        """
        return self._signature_stats.copy()

    def _increment_signature_count(self, pubkey: Pubkey) -> None:
        """
        Increment signature count for a keypair

        Args:
            pubkey: Public key to update
        """
        if pubkey in self._signature_stats:
            self._signature_stats[pubkey].total_signatures += 1
            self._signature_stats[pubkey].last_used = datetime.now(timezone.utc)

            # Check rotation threshold
            if self.config.enable_key_rotation:
                stats = self._signature_stats[pubkey]
                if stats.total_signatures >= self.config.max_signatures_per_key:
                    logger.warning(
                        "keypair_rotation_threshold_reached",
                        pubkey=str(pubkey),
                        total_signatures=stats.total_signatures,
                        threshold=self.config.max_signatures_per_key
                    )
                    metrics.increment_counter("keypair_rotation_needed")

    def needs_rotation(self, pubkey: Pubkey) -> bool:
        """
        Check if keypair needs rotation

        Args:
            pubkey: Public key to check

        Returns:
            True if keypair should be rotated
        """
        if not self.config.enable_key_rotation:
            return False

        stats = self._signature_stats.get(pubkey)
        if not stats:
            return False

        # Check signature count
        if stats.total_signatures >= self.config.max_signatures_per_key:
            return True

        # Check age
        age_minutes = (datetime.now(timezone.utc) - stats.created_at).total_seconds() / 60
        if age_minutes >= self.config.key_rotation_interval_minutes:
            return True

        return False

    def get_keypairs_needing_rotation(self) -> List[Pubkey]:
        """
        Get all keypairs that need rotation

        Returns:
            List of public keys needing rotation
        """
        return [
            pubkey for pubkey in self._keypairs.keys()
            if self.needs_rotation(pubkey)
        ]

    def clear_all_keypairs(self) -> None:
        """
        Remove all keypairs (for shutdown/cleanup)

        Security Note:
            This does not securely erase keypairs from memory
        """
        count = len(self._keypairs)
        self._keypairs.clear()
        self._signature_stats.clear()

        logger.info(
            "all_keypairs_cleared",
            keypairs_removed=count
        )
        metrics.set_gauge("active_keypairs", 0)

    def get_stats(self) -> Dict:
        """
        Get signer statistics

        Returns:
            Dictionary with signer stats
        """
        total_signatures = sum(
            stats.total_signatures
            for stats in self._signature_stats.values()
        )

        rotation_needed = len(self.get_keypairs_needing_rotation())

        return {
            "total_keypairs": len(self._keypairs),
            "total_signatures": total_signatures,
            "keypairs_needing_rotation": rotation_needed,
            "key_rotation_enabled": self.config.enable_key_rotation,
            "signature_tracking_enabled": self.config.track_signature_count
        }
