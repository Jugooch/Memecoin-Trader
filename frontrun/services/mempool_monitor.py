"""
Production LaserStream Mempool Monitor
Ultra-low latency transaction streaming via Helius LaserStream (Yellowstone gRPC)
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Callable, Awaitable, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time
import base58
import base64
import struct

import grpc

# Add proto directory to path for imports
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))

import geyser_pb2
import geyser_pb2_grpc

from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from core.logger import get_logger
from core.metrics import get_metrics


logger = get_logger(__name__)
metrics = get_metrics()


# Pump.fun program IDs
PUMP_FUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_AMM_PROGRAM_ID = Pubkey.from_string("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA")

# TradeEvent discriminator (emitted on every buy/sell)
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])
EVENT_DISCRIMINATOR_SIZE = 8


def parse_trade_event_from_logs(logs: List[str]) -> Optional[Dict]:
    """
    Parse TradeEvent data from transaction logs

    TradeEvent is emitted by Pump.fun on every buy/sell with complete transaction details.
    This is much more reliable than parsing post_token_balances!

    Args:
        logs: Transaction log messages

    Returns:
        Dictionary with trade event data or None if not found
    """
    for log in logs:
        if "Program data:" in log:
            try:
                # Extract base64 encoded data
                encoded_data = log.split("Program data: ")[1].strip()
                decoded_data = base64.b64decode(encoded_data)

                # Check discriminator
                if len(decoded_data) >= EVENT_DISCRIMINATOR_SIZE:
                    discriminator = decoded_data[:EVENT_DISCRIMINATOR_SIZE]
                    if discriminator == TRADE_EVENT_DISCRIMINATOR:
                        # Parse the TradeEvent structure
                        return decode_trade_event(decoded_data[EVENT_DISCRIMINATOR_SIZE:])
            except (ValueError, Exception) as e:
                logger.debug("trade_event_parse_error", error=str(e))
                continue
    return None


def decode_trade_event(data: bytes) -> Optional[Dict]:
    """
    Decode TradeEvent structure from raw bytes

    TradeEvent structure (from Pump.fun program):
    - mint: [u8; 32] (32 bytes)
    - sol_amount: u64 (8 bytes)
    - token_amount: u64 (8 bytes)
    - is_buy: bool (1 byte)
    - user: [u8; 32] (32 bytes)
    - timestamp: i64 (8 bytes)
    - virtual_sol_reserves: u64 (8 bytes)
    - virtual_token_reserves: u64 (8 bytes)

    Args:
        data: Raw event data (after discriminator)

    Returns:
        Dictionary with parsed trade event or None if invalid
    """
    # Minimum size check (32 + 8 + 8 + 1 + 32 + 8 + 8 + 8 = 105 bytes)
    if len(data) < 105:
        return None

    offset = 0

    # Parse mint (32 bytes)
    mint_bytes = data[offset:offset + 32]
    mint = base58.b58encode(mint_bytes).decode('utf-8')
    offset += 32

    # Parse sol_amount (u64, little endian)
    sol_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse token_amount (u64, little endian)
    token_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse is_buy (bool, 1 byte)
    is_buy = bool(data[offset])
    offset += 1

    # Parse user (32 bytes)
    user_bytes = data[offset:offset + 32]
    user = base58.b58encode(user_bytes).decode('utf-8')
    offset += 32

    # Parse timestamp (i64, little endian)
    timestamp = struct.unpack('<q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_sol_reserves (u64, little endian)
    virtual_sol_reserves = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_token_reserves (u64, little endian)
    virtual_token_reserves = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    return {
        'mint': mint,
        'sol_amount': sol_amount,
        'token_amount': token_amount,
        'is_buy': is_buy,
        'user': user,
        'timestamp': timestamp,
        'virtual_sol_reserves': virtual_sol_reserves,
        'virtual_token_reserves': virtual_token_reserves,
        'sol_amount_ui': sol_amount / 1e9,  # Convert to SOL
        'token_amount_ui': token_amount / 1e6,  # Convert to UI amount (6 decimals)
    }


def is_pump_bonding_curve_transaction(logs: List[str]) -> bool:
    """
    Check if transaction is a Pump.fun bonding curve transaction

    Args:
        logs: Transaction log messages

    Returns:
        True if this is a bonding curve transaction (not AMM/graduated)
    """
    # Must mention Pump.fun bonding curve program
    pump_program_str = str(PUMP_FUN_PROGRAM_ID)
    if not any(pump_program_str in log for log in logs):
        return False

    # Must NOT mention Pump AMM program (exclude graduated tokens)
    amm_program_str = str(PUMP_AMM_PROGRAM_ID)
    if any(amm_program_str in log for log in logs):
        return False

    return True


class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
    """Authentication plugin for Helius LaserStream"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, context, callback):
        metadata = (('x-token', self.api_key),)
        callback(metadata, None)


@dataclass
class LaserStreamConfig:
    """
    Configuration for Helius LaserStream/Yellowstone connection

    Attributes:
        endpoint: LaserStream gRPC endpoint (e.g., laserstream-mainnet-ewr.helius-rpc.com:443)
        api_key: Helius API key
        program_filters: List of program IDs to monitor (default: Pump.fun only)
        commitment: Commitment level (processed=fastest, confirmed=safer)
        reconnect_delay_s: Delay between reconnection attempts
        max_reconnect_attempts: Maximum reconnection attempts (None = infinite)
    """
    endpoint: str
    api_key: str
    program_filters: List[str] = None
    commitment: str = "processed"  # processed, confirmed, finalized
    reconnect_delay_s: float = 5.0
    max_reconnect_attempts: Optional[int] = None

    def __post_init__(self):
        if self.program_filters is None:
            # Default to Pump.fun only
            self.program_filters = [str(PUMP_FUN_PROGRAM_ID)]

        # Add :443 port if not specified
        if ':' not in self.endpoint:
            self.endpoint = f"{self.endpoint}:443"


@dataclass
class PendingTransaction:
    """
    Pending/unconfirmed transaction from mempool

    Attributes:
        signature: Transaction signature
        slot: Slot number
        transaction: Versioned transaction (if available)
        received_at: When we saw this transaction
        fee: Priority fee in lamports
        accounts: List of accounts involved
        program_id: Program ID (if detected)
        buy_amount_sol: Amount of SOL in buy (parsed from balance changes)
        mint: Token mint address (extracted from Pump.fun transactions)
        virtual_sol_reserves: Virtual SOL reserves from TradeEvent (lamports)
        virtual_token_reserves: Virtual token reserves from TradeEvent (raw units)
    """
    signature: str
    slot: int
    transaction: Optional[VersionedTransaction]
    received_at: datetime
    fee: int
    accounts: List[Pubkey]
    program_id: Optional[Pubkey]
    buy_amount_sol: float = 0.0
    mint: Optional[str] = None
    virtual_sol_reserves: Optional[int] = None
    virtual_token_reserves: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "signature": self.signature,
            "slot": self.slot,
            "received_at": self.received_at.isoformat(),
            "fee": self.fee,
            "accounts": [str(a) for a in self.accounts],
            "program_id": str(self.program_id) if self.program_id else None,
            "buy_amount_sol": self.buy_amount_sol,
            "mint": self.mint,
            "virtual_sol_reserves": self.virtual_sol_reserves,
            "virtual_token_reserves": self.virtual_token_reserves
        }


@dataclass
class MempoolStats:
    """
    Mempool monitoring statistics

    Attributes:
        transactions_seen: Total transactions observed
        pump_fun_txs_seen: Pump.fun transactions seen
        reconnections: Number of reconnections
        errors: Number of errors
        avg_latency_ms: Average detection latency
    """
    transactions_seen: int
    pump_fun_txs_seen: int
    reconnections: int
    errors: int
    avg_latency_ms: float


class LaserStreamMonitor:
    """
    Production-grade LaserStream mempool monitor using Yellowstone gRPC

    Features:
    - Ultra-low latency (<10ms) transaction detection
    - Automatic reconnection with exponential backoff
    - Program-filtered subscriptions (Pump.fun only)
    - Balance change parsing for buy amount detection
    - Production error handling and metrics

    Usage:
        config = LaserStreamConfig(
            endpoint="laserstream-mainnet-ewr.helius-rpc.com",
            api_key="your_api_key"
        )
        monitor = LaserStreamMonitor(config)

        async def on_pending_tx(tx: PendingTransaction):
            if tx.buy_amount_sol >= 0.5:
                logger.info("Large buy detected", **tx.to_dict())

        await monitor.start_monitoring(on_pending_tx)
    """

    def __init__(self, config: LaserStreamConfig):
        """
        Initialize LaserStream mempool monitor

        Args:
            config: LaserStream configuration
        """
        self.config = config
        self._monitoring = False
        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[geyser_pb2_grpc.GeyserStub] = None

        # Stats
        self._transactions_seen = 0
        self._pump_fun_txs_seen = 0
        self._reconnections = 0
        self._errors = 0
        self._latencies: List[float] = []

        logger.info(
            "laserstream_monitor_initialized",
            endpoint=config.endpoint,
            program_filters=config.program_filters,
            commitment=config.commitment
        )

    async def start_monitoring(
        self,
        callback: Callable[[PendingTransaction], Awaitable[None]]
    ):
        """
        Start monitoring mempool with callback

        Args:
            callback: Async function to call for each pending transaction

        Example:
            async def on_pending_tx(tx: PendingTransaction):
                if tx.buy_amount_sol >= 0.5:
                    await execute_frontrun(tx)

            await monitor.start_monitoring(on_pending_tx)
        """
        self._monitoring = True
        reconnect_count = 0

        logger.info("laserstream_monitoring_started")

        while self._monitoring:
            try:
                await self._connect_and_stream(callback)

            except Exception as e:
                self._errors += 1
                logger.error(
                    "laserstream_connection_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    reconnect_count=reconnect_count
                )

                # Check if should reconnect
                if (self.config.max_reconnect_attempts is not None and
                    reconnect_count >= self.config.max_reconnect_attempts):
                    logger.error("max_reconnect_attempts_reached")
                    break

                # Exponential backoff
                delay = min(
                    self.config.reconnect_delay_s * (2 ** reconnect_count),
                    60.0  # Max 60 seconds
                )
                logger.info(
                    "reconnecting_to_laserstream",
                    delay_s=delay,
                    attempt=reconnect_count + 1
                )
                await asyncio.sleep(delay)

                reconnect_count += 1
                self._reconnections += 1

    async def _connect_and_stream(
        self,
        callback: Callable[[PendingTransaction], Awaitable[None]]
    ):
        """
        Connect to LaserStream and process transactions

        Args:
            callback: Callback for each transaction
        """
        logger.info(
            "connecting_to_laserstream",
            endpoint=self.config.endpoint
        )

        # Setup authentication
        auth = HeliusAuthMetadata(self.config.api_key)
        call_creds = grpc.metadata_call_credentials(auth)

        # Setup SSL
        ssl_creds = grpc.ssl_channel_credentials()

        # Combine credentials
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        # Create secure channel
        self._channel = grpc.aio.secure_channel(
            self.config.endpoint,
            credentials=combined_creds
        )

        # Create stub
        self._stub = geyser_pb2_grpc.GeyserStub(self._channel)

        # Build subscription request
        subscription_request = self._build_subscription_request()

        logger.info("subscribing_to_transactions", filters=self.config.program_filters)

        try:
            # Create request iterator (gRPC streaming requires this)
            async def request_generator():
                # Send initial subscription request
                yield subscription_request
                # Keep stream open (no more requests needed)
                while self._monitoring:
                    await asyncio.sleep(30)  # Keep-alive

            # Subscribe to transaction stream (bidirectional streaming)
            async for message in self._stub.Subscribe(request_generator()):
                try:
                    await self._process_message(message, callback)
                except Exception as e:
                    logger.error(
                        "message_processing_error",
                        error=str(e),
                        error_type=type(e).__name__
                    )

        finally:
            if self._channel:
                await self._channel.close()
                self._channel = None
                self._stub = None

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """
        Build Yellowstone subscription request for Pump.fun transactions

        Returns:
            Subscription request
        """
        # Map commitment string to enum
        commitment_map = {
            "processed": geyser_pb2.CommitmentLevel.PROCESSED,
            "confirmed": geyser_pb2.CommitmentLevel.CONFIRMED,
            "finalized": geyser_pb2.CommitmentLevel.FINALIZED
        }
        commitment = commitment_map.get(
            self.config.commitment,
            geyser_pb2.CommitmentLevel.PROCESSED
        )

        # Create transaction subscription filter
        # If no filters, subscribe to ALL (don't set account_include)
        if self.config.program_filters:
            tx_filter = geyser_pb2.SubscribeRequestFilterTransactions(
                account_include=self.config.program_filters,
                vote=False,  # Exclude vote transactions
                failed=False,  # Exclude failed transactions
            )
        else:
            # Empty filter = all transactions
            tx_filter = geyser_pb2.SubscribeRequestFilterTransactions(
                vote=False,  # Exclude vote transactions
                failed=False,  # Exclude failed transactions
            )

        # Build subscription request
        request = geyser_pb2.SubscribeRequest(
            transactions={
                "client": tx_filter  # Subscription name
            },
            commitment=commitment
        )

        return request

    async def _process_message(
        self,
        message: geyser_pb2.SubscribeUpdate,
        callback: Callable[[PendingTransaction], Awaitable[None]]
    ):
        """
        Process incoming LaserStream message

        Args:
            message: Message from Yellowstone stream
            callback: Callback for each transaction
        """
        # Check if this is a transaction update
        if not message.HasField('transaction'):
            return

        tx_update = message.transaction

        # Calculate latency
        detection_time = time.time()

        # Parse transaction
        pending_tx = self._parse_transaction(tx_update, detection_time)

        if pending_tx:
            self._transactions_seen += 1

            # Track Pump.fun specific
            if pending_tx.program_id == PUMP_FUN_PROGRAM_ID:
                self._pump_fun_txs_seen += 1

            # Track latency
            self._latencies.append(time.time() - detection_time)
            if len(self._latencies) > 1000:
                self._latencies.pop(0)  # Keep last 1000

            # Call callback
            try:
                await callback(pending_tx)

                metrics.increment_counter(
                    "laserstream_transactions_seen",
                    labels={"program": "pump_fun"}
                )

            except Exception as e:
                logger.error(
                    "callback_error",
                    error=str(e),
                    signature=pending_tx.signature
                )

    def _parse_transaction(
        self,
        tx_update: geyser_pb2.SubscribeUpdateTransaction,
        detection_time: float
    ) -> Optional[PendingTransaction]:
        """
        Parse Yellowstone transaction update using TradeEvent

        Args:
            tx_update: Transaction update from Yellowstone
            detection_time: Time when we detected this transaction

        Returns:
            PendingTransaction or None if not a valid BUY transaction
        """
        try:
            transaction = tx_update.transaction

            # Extract signature
            signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')

            # Extract slot
            slot = tx_update.slot

            # Extract logs
            logs = []
            if transaction.meta:
                # Try different possible attribute names
                if hasattr(transaction.meta, 'log_messages'):
                    logs = list(transaction.meta.log_messages)
                elif hasattr(transaction.meta, 'logs'):
                    logs = list(transaction.meta.logs)
                elif hasattr(transaction.meta, 'logMessages'):
                    logs = list(transaction.meta.logMessages)
            else:
                logger.debug(
                    "no_meta",
                    signature=signature[:16] + "..."
                )

            # Check if this is a bonding curve transaction (not AMM/graduated)
            if not is_pump_bonding_curve_transaction(logs):
                logger.debug(
                    "not_bonding_curve_tx",
                    signature=signature[:16] + "..."
                )
                return None

            # Parse TradeEvent
            trade_event = parse_trade_event_from_logs(logs)
            if not trade_event:
                # No TradeEvent found - likely token creation or other non-trade instruction
                logger.debug(
                    "no_trade_event_found",
                    signature=signature[:16] + "..."
                )
                return None

            # CRITICAL: Only process BUY transactions
            if not trade_event['is_buy']:
                logger.debug(
                    "skipping_sell_transaction",
                    signature=signature[:16] + "...",
                    mint=trade_event['mint'][:8] + "..."
                )
                return None

            # Extract accounts
            accounts = []
            if transaction.transaction and transaction.transaction.message:
                account_keys = transaction.transaction.message.account_keys
                for key_bytes in account_keys:
                    try:
                        # key_bytes is already bytes from protobuf
                        if isinstance(key_bytes, bytes) and len(key_bytes) == 32:
                            pubkey = Pubkey(key_bytes)
                            accounts.append(pubkey)
                        elif isinstance(key_bytes, str):
                            # If it's a string, it should be base58
                            pubkey = Pubkey.from_string(key_bytes)
                            accounts.append(pubkey)
                    except Exception:
                        pass

            # Extract fee
            fee = 0
            if transaction.meta:
                fee = transaction.meta.fee

            # Use TradeEvent data (much more reliable!)
            mint = trade_event['mint']
            buy_amount_sol = trade_event['sol_amount_ui']  # Already in SOL
            program_id = PUMP_FUN_PROGRAM_ID

            pending_tx = PendingTransaction(
                signature=signature,
                slot=slot,
                transaction=None,  # Don't store full tx to save memory
                received_at=datetime.fromtimestamp(detection_time),
                fee=fee,
                accounts=accounts,
                program_id=program_id,
                buy_amount_sol=buy_amount_sol,
                mint=mint,
                virtual_sol_reserves=trade_event['virtual_sol_reserves'],
                virtual_token_reserves=trade_event['virtual_token_reserves']
            )

            logger.debug(
                "buy_transaction_parsed",
                signature=signature[:16] + "...",
                slot=slot,
                buy_amount_sol=buy_amount_sol,
                mint=mint[:16] + "..."
            )

            return pending_tx

        except Exception as e:
            logger.error(
                "transaction_parsing_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return None

    async def stop_monitoring(self):
        """Stop monitoring mempool"""
        self._monitoring = False

        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

        logger.info("laserstream_monitoring_stopped")

    def get_mempool_stats(self) -> MempoolStats:
        """
        Get mempool monitoring statistics

        Returns:
            MempoolStats with current statistics

        Example:
            stats = monitor.get_mempool_stats()
            logger.info("Stats", **stats.__dict__)
        """
        avg_latency = 0.0
        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies) * 1000  # ms

        return MempoolStats(
            transactions_seen=self._transactions_seen,
            pump_fun_txs_seen=self._pump_fun_txs_seen,
            reconnections=self._reconnections,
            errors=self._errors,
            avg_latency_ms=avg_latency
        )


# Example usage
if __name__ == "__main__":
    from core.logger import setup_logging
    import os

    async def main():
        # Setup logging
        setup_logging(level="INFO", format="console")

        # Get credentials from environment
        api_key = os.getenv("HELIUS_API_KEY")
        if not api_key:
            logger.error("HELIUS_API_KEY environment variable required")
            return

        # Create config
        config = LaserStreamConfig(
            endpoint="laserstream-mainnet-ewr.helius-rpc.com",
            api_key=api_key,
            commitment="processed"  # Fastest
        )

        # Create monitor
        monitor = LaserStreamMonitor(config)

        # Define callback
        async def on_pending_tx(tx: PendingTransaction):
            logger.info(
                "pending_transaction_detected",
                **tx.to_dict()
            )

            # Example: Check if worth frontrunning
            if tx.buy_amount_sol >= 0.5:
                logger.warning(
                    "large_buy_detected",
                    buy_amount_sol=tx.buy_amount_sol,
                    signature=tx.signature
                )

        # Start monitoring
        logger.info("Starting LaserStream mempool monitoring...")
        logger.info("Press Ctrl+C to stop")

        try:
            await monitor.start_monitoring(on_pending_tx)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            await monitor.stop_monitoring()

            # Print stats
            stats = monitor.get_mempool_stats()
            logger.info(
                "mempool_stats",
                transactions_seen=stats.transactions_seen,
                pump_fun_txs=stats.pump_fun_txs_seen,
                reconnections=stats.reconnections,
                errors=stats.errors,
                avg_latency_ms=round(stats.avg_latency_ms, 2)
            )

    asyncio.run(main())
