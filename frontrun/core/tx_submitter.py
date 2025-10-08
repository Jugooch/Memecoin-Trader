"""
Transaction Submitter for Frontrun Bot
Submits signed transactions to Solana network with retry logic and confirmation tracking
"""

import asyncio
import base64
import aiohttp
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from solders.transaction import Transaction
from solders.signature import Signature
from solders.rpc.responses import SendTransactionResp

from core.rpc_manager import RPCManager
from core.logger import get_logger
from core.metrics import get_metrics, LatencyTimer


logger = get_logger(__name__)
metrics = get_metrics()


class ConfirmationStatus(Enum):
    """Transaction confirmation status"""
    PENDING = "pending"
    PROCESSED = "processed"
    CONFIRMED = "confirmed"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass
class SubmitterConfig:
    """Configuration for transaction submitter"""
    skip_preflight: bool = True  # Skip simulation for speed
    max_retries: int = 3
    retry_delay_ms: int = 200
    confirmation_timeout_s: int = 30
    confirmation_poll_interval_s: float = 0.5
    enable_confirmation_tracking: bool = True


@dataclass
class TransactionResult:
    """Result of transaction submission"""
    signature: str
    submitted_at: datetime
    slot: Optional[int] = None
    error: Optional[str] = None
    submitted_to_rpc: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "signature": self.signature,
            "submitted_at": self.submitted_at.isoformat(),
            "slot": self.slot,
            "error": self.error,
            "submitted_to_rpc": self.submitted_to_rpc
        }


@dataclass
class ConfirmedTransaction:
    """Confirmed transaction details"""
    signature: str
    slot: int
    block_time: Optional[int]
    confirmation_status: ConfirmationStatus
    compute_units_consumed: Optional[int] = None
    error: Optional[str] = None
    confirmed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signature": self.signature,
            "slot": self.slot,
            "block_time": self.block_time,
            "confirmation_status": self.confirmation_status.value,
            "compute_units_consumed": self.compute_units_consumed,
            "error": self.error,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None
        }


class TransactionSubmitter:
    """
    Submits transactions to Solana network with reliability features

    Features:
    - Multi-RPC submission with automatic failover
    - Retry logic with exponential backoff
    - Confirmation tracking
    - Error classification and handling
    - Metrics and logging

    Usage:
        submitter = TransactionSubmitter(rpc_manager)
        result = await submitter.submit_transaction(signed_tx)
        confirmed = await submitter.submit_and_confirm(signed_tx, timeout_seconds=30)
    """

    def __init__(
        self,
        rpc_manager: RPCManager,
        config: Optional[SubmitterConfig] = None
    ):
        """
        Initialize transaction submitter

        Args:
            rpc_manager: RPC manager for network communication
            config: Submitter configuration (optional)
        """
        self.rpc_manager = rpc_manager
        self.config = config or SubmitterConfig()

        logger.info(
            "transaction_submitter_initialized",
            skip_preflight=self.config.skip_preflight,
            max_retries=self.config.max_retries,
            confirmation_timeout_s=self.config.confirmation_timeout_s
        )

    async def submit_transaction(
        self,
        signed_tx: Transaction,
        max_retries: Optional[int] = None
    ) -> TransactionResult:
        """
        Submit a signed transaction to the network

        Args:
            signed_tx: Signed transaction to submit
            max_retries: Maximum retry attempts (optional, uses config default)

        Returns:
            TransactionResult with signature and status

        Raises:
            Exception: If all retries fail

        Example:
            result = await submitter.submit_transaction(signed_tx)
            print(f"Transaction signature: {result.signature}")
        """
        max_retries = max_retries or self.config.max_retries

        with LatencyTimer(metrics, "tx_submit", labels={"stage": "full"}):
            for attempt in range(max_retries):
                try:
                    result = await self._submit_once(signed_tx)

                    if result.error is None:
                        metrics.increment_counter("transactions_submitted_success")
                        logger.info(
                            "transaction_submitted",
                            signature=result.signature,
                            slot=result.slot,
                            rpc=result.submitted_to_rpc,
                            attempt=attempt + 1
                        )
                        return result

                    # Log error but continue retrying
                    logger.warning(
                        "transaction_submission_error",
                        error=result.error,
                        attempt=attempt + 1,
                        max_retries=max_retries
                    )

                except Exception as e:
                    logger.warning(
                        "transaction_submission_exception",
                        error=str(e),
                        attempt=attempt + 1,
                        max_retries=max_retries
                    )

                    if attempt == max_retries - 1:
                        metrics.increment_counter("transactions_submitted_failed")
                        raise

                # Exponential backoff
                if attempt < max_retries - 1:
                    delay = self.config.retry_delay_ms * (2 ** attempt) / 1000
                    await asyncio.sleep(delay)

            # All retries failed
            metrics.increment_counter("transactions_submitted_failed")
            raise Exception(f"Transaction submission failed after {max_retries} attempts")

    async def _submit_once(self, signed_tx: Transaction) -> TransactionResult:
        """
        Submit transaction once to RPC

        Args:
            signed_tx: Signed transaction

        Returns:
            TransactionResult
        """
        # Get healthy RPC connection
        connection = await self.rpc_manager.get_healthy_connection()
        if connection is None:
            return TransactionResult(
                signature="",
                submitted_at=datetime.now(timezone.utc),
                error="No healthy RPC connection available"
            )

        try:
            # Serialize transaction
            tx_bytes = bytes(signed_tx)

            # Submit via RPC
            response = await self._send_transaction_rpc(
                connection.endpoint.url,
                tx_bytes
            )

            # Extract signature
            signature = str(signed_tx.signatures[0]) if signed_tx.signatures else ""

            return TransactionResult(
                signature=signature,
                submitted_at=datetime.now(timezone.utc),
                submitted_to_rpc=connection.endpoint.label
            )

        except Exception as e:
            logger.error(
                "rpc_submit_error",
                rpc=connection.endpoint.label,
                error=str(e)
            )
            return TransactionResult(
                signature="",
                submitted_at=datetime.now(timezone.utc),
                error=str(e),
                submitted_to_rpc=connection.endpoint.label
            )

    async def _send_transaction_rpc(self, rpc_url: str, tx_bytes: bytes) -> Dict:
        """
        Send transaction via HTTP RPC using sendTransaction method

        Args:
            rpc_url: RPC endpoint URL
            tx_bytes: Serialized transaction bytes

        Returns:
            RPC response dictionary

        Raises:
            Exception: If RPC call fails
        """
        # Encode transaction as base64
        tx_base64 = base64.b64encode(tx_bytes).decode('utf-8')

        # Build RPC request
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendTransaction",
            "params": [
                tx_base64,
                {
                    "skipPreflight": self.config.skip_preflight,
                    "encoding": "base64",
                    "maxRetries": 0  # We handle retries ourselves
                }
            ]
        }

        # Send HTTP POST request
        timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(rpc_url, json=payload) as response:
                result = await response.json()

                # Check for RPC errors
                if "error" in result:
                    error_msg = result["error"].get("message", str(result["error"]))
                    raise Exception(f"RPC error: {error_msg}")

                return result

    async def submit_and_confirm(
        self,
        signed_tx: Transaction,
        timeout_seconds: Optional[int] = None
    ) -> ConfirmedTransaction:
        """
        Submit transaction and wait for confirmation

        Args:
            signed_tx: Signed transaction to submit
            timeout_seconds: Confirmation timeout (optional, uses config default)

        Returns:
            ConfirmedTransaction with confirmation details

        Raises:
            TimeoutError: If confirmation times out
            Exception: If submission fails

        Example:
            confirmed = await submitter.submit_and_confirm(signed_tx, timeout_seconds=30)
            print(f"Confirmed in slot {confirmed.slot}")
        """
        timeout = timeout_seconds or self.config.confirmation_timeout_s

        with LatencyTimer(metrics, "tx_submit_and_confirm"):
            # Submit transaction
            result = await self.submit_transaction(signed_tx)

            if result.error:
                return ConfirmedTransaction(
                    signature=result.signature,
                    slot=0,
                    block_time=None,
                    confirmation_status=ConfirmationStatus.FAILED,
                    error=result.error
                )

            # Wait for confirmation
            if self.config.enable_confirmation_tracking:
                confirmed = await self._wait_for_confirmation(
                    result.signature,
                    timeout
                )
                return confirmed
            else:
                # Return pending status if confirmation tracking disabled
                return ConfirmedTransaction(
                    signature=result.signature,
                    slot=0,
                    block_time=None,
                    confirmation_status=ConfirmationStatus.PENDING
                )

    async def _wait_for_confirmation(
        self,
        signature: str,
        timeout_seconds: int
    ) -> ConfirmedTransaction:
        """
        Wait for transaction confirmation

        Args:
            signature: Transaction signature to track
            timeout_seconds: Maximum wait time

        Returns:
            ConfirmedTransaction

        Raises:
            TimeoutError: If confirmation times out
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time

            if elapsed > timeout_seconds:
                metrics.increment_counter("transaction_confirmations_timeout")
                raise TimeoutError(
                    f"Transaction confirmation timed out after {timeout_seconds}s"
                )

            # Check transaction status
            status = await self._get_transaction_status(signature)

            if status:
                if status.confirmation_status in [
                    ConfirmationStatus.CONFIRMED,
                    ConfirmationStatus.FINALIZED,
                    ConfirmationStatus.FAILED
                ]:
                    # Final status reached
                    metrics.increment_counter(
                        "transaction_confirmations",
                        labels={"status": status.confirmation_status.value}
                    )
                    return status

            # Wait before next poll
            await asyncio.sleep(self.config.confirmation_poll_interval_s)

    async def _get_transaction_status(
        self,
        signature: str
    ) -> Optional[ConfirmedTransaction]:
        """
        Get current transaction status via getSignatureStatuses RPC

        Args:
            signature: Transaction signature

        Returns:
            ConfirmedTransaction if found, None otherwise
        """
        try:
            # Get healthy RPC connection
            connection = await self.rpc_manager.get_healthy_connection()
            if connection is None:
                return None

            # Build RPC request
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignatureStatuses",
                "params": [
                    [signature],  # Array of signatures
                    {"searchTransactionHistory": True}
                ]
            }

            # Send HTTP POST request
            timeout = aiohttp.ClientTimeout(total=5)  # 5 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(connection.endpoint.url, json=payload) as response:
                    result = await response.json()

                    # Check for RPC errors
                    if "error" in result:
                        logger.warning(
                            "get_signature_status_error",
                            signature=signature,
                            error=result["error"]
                        )
                        return None

                    # Parse response
                    value = result.get("result", {}).get("value", [])
                    if not value or len(value) == 0 or value[0] is None:
                        return None

                    status_data = value[0]

                    # Map Solana confirmation status to our enum
                    confirmation_level = status_data.get("confirmationStatus", "processed")
                    if confirmation_level == "processed":
                        status = ConfirmationStatus.PROCESSED
                    elif confirmation_level == "confirmed":
                        status = ConfirmationStatus.CONFIRMED
                    elif confirmation_level == "finalized":
                        status = ConfirmationStatus.FINALIZED
                    else:
                        status = ConfirmationStatus.PENDING

                    # Check for error
                    error = None
                    if status_data.get("err"):
                        status = ConfirmationStatus.FAILED
                        error = str(status_data["err"])

                    return ConfirmedTransaction(
                        signature=signature,
                        slot=status_data.get("slot", 0),
                        block_time=None,  # Would need separate getBlock call
                        confirmation_status=status,
                        compute_units_consumed=status_data.get("computeUnitsConsumed"),
                        error=error,
                        confirmed_at=datetime.now(timezone.utc)
                    )

        except Exception as e:
            logger.warning(
                "get_transaction_status_exception",
                signature=signature,
                error=str(e)
            )
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get submitter statistics

        Returns:
            Dictionary with submitter stats
        """
        return {
            "skip_preflight": self.config.skip_preflight,
            "max_retries": self.config.max_retries,
            "confirmation_timeout_s": self.config.confirmation_timeout_s,
            "confirmation_tracking_enabled": self.config.enable_confirmation_tracking
        }
