"""
LaserStream Wallet Monitor - Fast wallet activity detection using Geyser/LaserStream

This module provides real-time monitoring of alpha wallet buys using Helius LaserStream.
Replaces slow BitQuery polling with instant transaction detection.

Based on working frontrun implementation in frontrun/services/mempool_monitor.py
"""

import asyncio
import logging
import grpc
import sys
from pathlib import Path
from typing import Dict, List, Callable, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import base58
import base64
import struct

# Import proto files from frontrun directory (working implementation)
proto_path = Path(__file__).parent.parent.parent / "frontrun" / "proto"
sys.path.insert(0, str(proto_path))

try:
    import geyser_pb2
    import geyser_pb2_grpc
    GEYSER_AVAILABLE = True
except ImportError:
    GEYSER_AVAILABLE = False
    logging.warning("Geyser proto files not available - cannot use LaserStream")


@dataclass
class WalletBuyEvent:
    """Event emitted when an alpha wallet buys a token"""
    wallet: str
    mint: str
    sol_amount: float
    timestamp: datetime
    slot: int
    signature: str
    token_amount: Optional[float] = None
    virtual_sol_reserves: Optional[int] = None
    virtual_token_reserves: Optional[int] = None


# Pump.fun program ID
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

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
            except (ValueError, Exception):
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


class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
    """Authentication plugin for Helius LaserStream (from working frontrun implementation)"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, context, callback):
        metadata = (('x-token', self.api_key),)
        callback(metadata, None)


class LaserStreamWalletMonitor:
    """
    Fast wallet activity monitor using Helius LaserStream

    Based on working implementation in frontrun/services/mempool_monitor.py
    Monitors specific wallets for Pump.fun buy transactions in real-time.
    """

    def __init__(self, config: Dict):
        """
        Initialize LaserStream wallet monitor

        Args:
            config: Configuration dict with:
                - helius_grpc_endpoint: LaserStream gRPC endpoint
                - helius_grpc_token: Authentication token
                - watched_wallets: List of wallet addresses to monitor
                - pump_fun_program: Pump.fun program address
        """
        self.logger = logging.getLogger(__name__)

        if not GEYSER_AVAILABLE:
            raise RuntimeError("Geyser proto files not available - cannot use LaserStream")

        # Configuration
        self.grpc_endpoint = config.get('helius_grpc_endpoint')
        self.grpc_token = config.get('helius_grpc_token')
        self.watched_wallets = set(config.get('watched_wallets', []))
        self.pump_fun_program = config.get('pump_fun_program', PUMP_FUN_PROGRAM_ID)

        if not self.grpc_endpoint or not self.grpc_token:
            raise ValueError("helius_grpc_endpoint and helius_grpc_token required for LaserStream")

        # Strip https:// prefix if present (Helius dashboard shows it, but gRPC doesn't accept it)
        if self.grpc_endpoint.startswith('https://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('https://', '')
        elif self.grpc_endpoint.startswith('http://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('http://', '')

        # Add :443 port if not specified
        if ':' not in self.grpc_endpoint:
            self.grpc_endpoint = f"{self.grpc_endpoint}:443"

        # Connection state
        self.channel = None
        self.stub = None
        self.running = False

        # Event callbacks
        self.on_wallet_buy: Optional[Callable[[WalletBuyEvent], None]] = None

        # Statistics
        self.transactions_processed = 0
        self.wallet_buys_detected = 0
        self.connection_time = None

        self.logger.info(f"LaserStreamWalletMonitor initialized")
        self.logger.info(f"  Endpoint: {self.grpc_endpoint}")
        self.logger.info(f"  Watching {len(self.watched_wallets)} wallets")
        self.logger.info(f"  Pump.fun program: {self.pump_fun_program}")

    def add_wallet(self, wallet: str):
        """Add a wallet to the watch list"""
        self.watched_wallets.add(wallet)
        self.logger.info(f"Added wallet to watch list: {wallet[:8]}... (total: {len(self.watched_wallets)})")

    def remove_wallet(self, wallet: str):
        """Remove a wallet from the watch list"""
        if wallet in self.watched_wallets:
            self.watched_wallets.remove(wallet)
            self.logger.info(f"Removed wallet from watch list: {wallet[:8]}... (total: {len(self.watched_wallets)})")

    def update_wallets(self, wallets: List[str]):
        """Update the entire wallet watch list"""
        old_count = len(self.watched_wallets)
        self.watched_wallets = set(wallets)
        self.logger.info(f"Updated wallet watch list: {old_count} -> {len(self.watched_wallets)} wallets")

    async def start(self):
        """Start monitoring wallet activity"""
        if self.running:
            self.logger.warning("LaserStream monitor already running")
            return

        self.running = True
        self.connection_time = datetime.now()

        try:
            await self._connect_and_stream()
        except Exception as e:
            self.logger.error(f"LaserStream monitor error: {e}")
            self.running = False
            raise

    async def _connect_and_stream(self):
        """Connect to LaserStream and process transactions (from working frontrun implementation)"""
        self.logger.info(f"Connecting to LaserStream at {self.grpc_endpoint}...")

        # Setup authentication (using working frontrun method)
        auth = HeliusAuthMetadata(self.grpc_token)
        call_creds = grpc.metadata_call_credentials(auth)

        # Setup SSL
        ssl_creds = grpc.ssl_channel_credentials()

        # Combine credentials
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        # Create secure channel
        self.channel = grpc.aio.secure_channel(
            self.grpc_endpoint,
            credentials=combined_creds,
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
            ]
        )

        # Create stub
        self.stub = geyser_pb2_grpc.GeyserStub(self.channel)

        # Build subscription request (using working frontrun format)
        subscription_request = self._build_subscription_request()

        self.logger.info("✅ LaserStream connection established")
        self.logger.info(f"Subscribing to Pump.fun transactions...")

        try:
            # Create request iterator (gRPC streaming requires this)
            async def request_generator():
                # Send initial subscription request
                yield subscription_request
                # Keep stream open (no more requests needed)
                while self.running:
                    await asyncio.sleep(30)  # Keep-alive

            # Subscribe to transaction stream (bidirectional streaming)
            self.logger.info("✅ LaserStream subscription active")
            self.logger.info(f"   Monitoring {len(self.watched_wallets)} wallets via LaserStream...")

            async for message in self.stub.Subscribe(request_generator()):
                if not self.running:
                    break

                try:
                    await self._process_message(message)
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")

        finally:
            if self.channel:
                await self.channel.close()
                self.channel = None
                self.stub = None

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """
        Build Yellowstone subscription request for Pump.fun transactions
        (Using working frontrun implementation)

        Returns:
            Subscription request
        """
        # Map commitment to enum
        commitment = geyser_pb2.CommitmentLevel.PROCESSED

        # Create transaction subscription filter
        tx_filter = geyser_pb2.SubscribeRequestFilterTransactions(
            account_include=[self.pump_fun_program],
            vote=False,  # Exclude vote transactions
            failed=False,  # Exclude failed transactions
        )

        # Build subscription request
        request = geyser_pb2.SubscribeRequest(
            transactions={
                "pump_fun_wallet_txs": tx_filter  # Subscription name
            },
            commitment=commitment
        )

        return request

    async def _process_message(self, message: geyser_pb2.SubscribeUpdate):
        """
        Process incoming LaserStream message
        (Using working frontrun parsing logic)

        Args:
            message: Message from Yellowstone stream
        """
        # Check if this is a transaction update
        if not message.HasField('transaction'):
            return

        tx_update = message.transaction
        transaction = tx_update.transaction

        # Extract signature
        signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
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

        # Parse TradeEvent (using working frontrun method)
        trade_event = parse_trade_event_from_logs(logs)
        if not trade_event:
            return

        # CRITICAL: Only process BUY transactions
        if not trade_event['is_buy']:
            return

        # Check if this is one of our watched wallets
        wallet = trade_event['user']
        if wallet not in self.watched_wallets:
            return

        # Create buy event
        buy_event = WalletBuyEvent(
            wallet=wallet,
            mint=trade_event['mint'],
            sol_amount=trade_event['sol_amount_ui'],
            timestamp=datetime.now(),
            slot=slot,
            signature=signature,
            token_amount=trade_event['token_amount_ui'],
            virtual_sol_reserves=trade_event['virtual_sol_reserves'],
            virtual_token_reserves=trade_event['virtual_token_reserves']
        )

        self.wallet_buys_detected += 1
        self.transactions_processed += 1

        self.logger.info(
            f"🎯 Alpha wallet buy detected",
            extra={
                'wallet': buy_event.wallet[:8] + '...',
                'mint': buy_event.mint[:8] + '...',
                'sol_amount': round(buy_event.sol_amount, 3),
                'latency_ms': (datetime.now() - buy_event.timestamp).total_seconds() * 1000
            }
        )

        # Emit event
        if self.on_wallet_buy:
            await self._safe_callback(self.on_wallet_buy, buy_event)

    async def _safe_callback(self, callback: Callable, event: WalletBuyEvent):
        """Safely execute callback without blocking stream processing"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            self.logger.error(f"Error in wallet buy callback: {e}")

    async def stop(self):
        """Stop monitoring"""
        self.running = False

        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

        uptime = (datetime.now() - self.connection_time).total_seconds() if self.connection_time else 0

        self.logger.info("LaserStream monitor stopped")
        self.logger.info(f"  Uptime: {uptime:.1f}s")
        self.logger.info(f"  Transactions processed: {self.transactions_processed:,}")
        self.logger.info(f"  Wallet buys detected: {self.wallet_buys_detected:,}")

    def get_stats(self) -> Dict:
        """Get monitor statistics"""
        uptime = (datetime.now() - self.connection_time).total_seconds() if self.connection_time else 0

        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'watched_wallets': len(self.watched_wallets),
            'transactions_processed': self.transactions_processed,
            'wallet_buys_detected': self.wallet_buys_detected,
            'buys_per_minute': (self.wallet_buys_detected / (uptime / 60)) if uptime > 0 else 0
        }
