"""
Swarm Trigger Sniper

Waits for early signs of life, then buys to trigger bot swarms.

Strategy based on alpha analysis:
- Successful 1+ SOL alphas are at position 11+, age 30-60s
- Position #1 has 70% failure rate
- Need initial activity before buying

Usage:
    python frontrun/scripts/snipe_swarm_trigger.py \
        --wallet frontrun/data/phase2-wallet.json \
        --buy-amount 1.0 \
        --hold-seconds 10 \
        --priority-fee 1000000 \
        --max-trades 5 \
        --real-trades
"""

import asyncio
import sys
import json
import struct
import base58
import base64
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager
from core.wallet_manager import WalletManager, WalletManagerConfig
from core.bonding_curve import (
    get_initial_bonding_curve_state,
    BondingCurveCalculator,
    BondingCurveState,
    calculate_curve_state_after_buy,
    fetch_bonding_curve_state
)
from clients.pump_fun_direct import PumpFunDirectClient
from core.tx_submitter import TransactionSubmitter, SubmitterConfig

# Import Geyser
import grpc
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))
import geyser_pb2
import geyser_pb2_grpc

logger = get_logger(__name__)

# Constants
PUMP_CREATE_DISCRIMINATOR = struct.pack("<Q", 8576854823835016728)
PUMP_BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")
PUMP_SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")

# TRIGGER THRESHOLDS (based on alpha analysis)
MIN_BUYERS_FOR_TRIGGER = 5  # Need at least 5 buyers before we enter (moderately aggressive)
MIN_VOLUME_SOL = 2.5  # Need at least 2.5 SOL volume (moderately aggressive)
MAX_AVERAGE_BUY_SOL = 20.0  # Reject if average buy > 20 SOL (likely wash trading)
TARGET_AGE_MIN = 20  # Don't buy before 20 seconds (avoid position #1 failure)
TARGET_AGE_MAX = 60  # Buy by 60 seconds if criteria met
MAX_MONITOR_TIME = 90  # Stop monitoring after 90 seconds (token is stale)
STALE_CUTOFF = 30  # If no activity for 30s, abandon

# DUMP PROTECTION THRESHOLDS
MAX_RECENT_SELL_SOL = 0.5  # Reject if any sell > 0.5 SOL in last 15s
MIN_RECENT_TRADE_LOOKBACK = 15  # Check last 15 seconds for sells
MIN_BUY_RATIO = 0.8  # Recent activity must be 80%+ buys (by count)
MIN_PEAK_PROXIMITY = 0.95  # Must be within 5% of peak SOL reserves
MAX_SECONDS_SINCE_LAST_BUY = 5  # Last buy must be < 5s ago (momentum check)

# ENTRY PRICE PROTECTION (CRITICAL - prevents buying late-stage pumps)
MAX_ENTRY_RESERVES_SOL = 38.0  # NEVER enter above 38 SOL reserves (45% to graduation at 85 SOL)
# Rationale:
# - At 30 SOL (start): Position #1, early entry (baseline price)
# - At 35 SOL: 17% markup - acceptable for strong momentum
# - At 38 SOL: 27% markup - maximum acceptable
# - At 44.85 SOL (TIVO): 49% markup - SUICIDE, early buyers exit here
# - Above 40 SOL = late stage, extreme whale exit risk


@dataclass
class TokenCreation:
    """Token creation event"""
    mint: str
    bonding_curve: str
    associated_bonding_curve: str
    creator: str
    name: str
    symbol: str
    signature: str
    slot: int
    timestamp: datetime


@dataclass
class TokenActivity:
    """Track activity for a token"""
    mint: str
    symbol: str
    creation_time: datetime
    buy_count: int = 0
    sell_count: int = 0
    volume_sol: float = 0.0
    last_activity: datetime = None
    triggered: bool = False
    # Track bonding curve state from Geyser (updated on each trade)
    virtual_sol_reserves: int = 30000000000  # Initial: 30 SOL
    virtual_token_reserves: int = 1073000000000000  # Initial: 1.073B tokens
    peak_sol_reserves: int = 30000000000  # Track all-time high
    # Recent activity tracking (for momentum detection)
    recent_trades: list = None  # List of (timestamp, is_buy, sol_amount, reserves_after)

    def __post_init__(self):
        if self.recent_trades is None:
            self.recent_trades = []


@dataclass
class TradeResult:
    """Result of a single trade"""
    mint: str
    symbol: str
    buy_signature: str
    buy_slot: int
    buy_sol: float
    sell_signature: str
    sell_slot: int
    trigger_age_seconds: float  # Age when we bought
    trigger_buyers: int  # Buyers when we bought
    trigger_volume: float  # Volume when we bought
    hold_seconds: float
    success: bool
    error: Optional[str] = None


class TokenActivityMonitor:
    """Monitor token creations and buy activity via Geyser"""

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._monitoring = False
        self.on_creation_callback = None
        self.on_buy_callback = None

    def decode_create_instruction(self, ix_data: bytes, account_keys, accounts):
        """Decode token creation instruction"""
        try:
            offset = 8  # Skip discriminator

            def get_account_key(index: int) -> str:
                if index >= len(accounts):
                    return None
                account_index = accounts[index]
                if account_index >= len(account_keys):
                    return None
                key_bytes = account_keys[account_index]
                return base58.b58encode(bytes(key_bytes)).decode()

            def read_string() -> str:
                nonlocal offset
                length = struct.unpack_from("<I", ix_data, offset)[0]
                offset += 4
                value = ix_data[offset:offset + length].decode('utf-8')
                offset += length
                return value

            def read_pubkey() -> str:
                nonlocal offset
                value = base58.b58encode(ix_data[offset:offset + 32]).decode('utf-8')
                offset += 32
                return value

            name = read_string()
            symbol = read_string()
            uri = read_string()
            creator = read_pubkey()

            return {
                'name': name,
                'symbol': symbol,
                'uri': uri,
                'creator': creator,
                'mint': get_account_key(0),
                'bonding_curve': get_account_key(2),
                'associated_bonding_curve': get_account_key(3),
            }

        except Exception as e:
            logger.debug("create_instruction_parse_failed", error=str(e))
            return None

    def decode_buy_instruction(self, ix_data: bytes, account_keys, accounts):
        """Decode buy instruction to get mint and amount"""
        try:
            offset = 8  # Skip discriminator

            def get_account_key(index: int) -> str:
                if index >= len(accounts):
                    return None
                account_index = accounts[index]
                if account_index >= len(account_keys):
                    return None
                key_bytes = account_keys[account_index]
                return base58.b58encode(bytes(key_bytes)).decode()

            # Buy instruction structure:
            # 8 bytes: discriminator
            # 8 bytes: token_amount
            # 8 bytes: max_sol_cost
            token_amount = struct.unpack('<Q', ix_data[8:16])[0]
            max_sol_cost = struct.unpack('<Q', ix_data[16:24])[0]

            # Mint is account #2
            mint = get_account_key(2)

            return {
                'mint': mint,
                'token_amount': token_amount,
                'max_sol_cost': max_sol_cost,
                'sol_estimate': max_sol_cost / 1e9  # Rough estimate
            }

        except Exception as e:
            logger.debug("buy_instruction_parse_failed", error=str(e))
            return None

    def parse_trade_event_from_logs(self, logs):
        """
        Parse TradeEvent from transaction logs using BINARY parsing (from copy trading flow)

        TradeEvent is emitted as base64-encoded binary data in "Program data:" logs
        """
        TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])
        EVENT_DISCRIMINATOR_SIZE = 8

        program_data_logs = 0
        for log in logs:
            if "Program data:" in log:
                program_data_logs += 1
                try:
                    # Extract base64 encoded data
                    encoded_data = log.split("Program data: ")[1].strip()
                    decoded_data = base64.b64decode(encoded_data)

                    # Check discriminator
                    if len(decoded_data) >= EVENT_DISCRIMINATOR_SIZE:
                        discriminator = decoded_data[:EVENT_DISCRIMINATOR_SIZE]
                        if discriminator == TRADE_EVENT_DISCRIMINATOR:
                            # Parse the TradeEvent structure
                            logger.debug(f"✅ Found TradeEvent discriminator in log #{program_data_logs}")
                            return self._decode_trade_event(decoded_data[EVENT_DISCRIMINATOR_SIZE:])
                        else:
                            logger.debug(f"❌ Wrong discriminator in log #{program_data_logs}: {discriminator[:8].hex()}")
                except (ValueError, Exception) as e:
                    logger.debug(f"trade_event_decode_failed log #{program_data_logs}", error=str(e))
                    continue

        if program_data_logs > 0:
            logger.debug(f"⚠️ Checked {program_data_logs} Program data logs, no TradeEvent found")
        return None

    def _decode_trade_event(self, data: bytes) -> Optional[Dict]:
        """
        Decode TradeEvent structure from raw bytes (from copy trading flow)

        TradeEvent structure:
        - mint: [u8; 32] (32 bytes)
        - sol_amount: u64 (8 bytes)
        - token_amount: u64 (8 bytes)
        - is_buy: bool (1 byte)
        - user: [u8; 32] (32 bytes)
        - timestamp: i64 (8 bytes)
        - virtual_sol_reserves: u64 (8 bytes)
        - virtual_token_reserves: u64 (8 bytes)
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

    async def start_monitoring(self, on_creation, on_buy):
        """Start monitoring for token creations and buys"""
        self._monitoring = True
        self.on_creation_callback = on_creation
        self.on_buy_callback = on_buy

        class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
            def __init__(self, api_key: str):
                self.api_key = api_key

            def __call__(self, context, callback_fn):
                metadata = (('x-token', self.api_key),)
                callback_fn(metadata, None)

        auth = HeliusAuthMetadata(self.api_key)
        call_creds = grpc.metadata_call_credentials(auth)
        ssl_creds = grpc.ssl_channel_credentials()
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        channel = grpc.aio.secure_channel(
            self.endpoint,
            credentials=combined_creds,
            options=[('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        )
        stub = geyser_pb2_grpc.GeyserStub(channel)

        subscription_request = geyser_pb2.SubscribeRequest()
        subscription_request.transactions["pump_activity"].account_include.append(str(PUMP_FUN_PROGRAM))
        subscription_request.transactions["pump_activity"].failed = False
        subscription_request.commitment = geyser_pb2.CommitmentLevel.PROCESSED

        logger.info("subscribing_to_pump_activity")

        try:
            async def request_generator():
                yield subscription_request
                while self._monitoring:
                    await asyncio.sleep(30)

            async for message in stub.Subscribe(request_generator()):
                if not message.HasField('transaction'):
                    continue

                try:
                    tx_update = message.transaction
                    transaction = tx_update.transaction

                    signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
                    slot = tx_update.slot

                    tx_inner = transaction.transaction
                    msg = getattr(tx_inner, "message", None)

                    if msg is None:
                        continue

                    # Extract logs (contains TradeEvent with bonding curve state!)
                    logs = []
                    if transaction.meta:
                        if hasattr(transaction.meta, 'log_messages'):
                            logs = list(transaction.meta.log_messages)
                        elif hasattr(transaction.meta, 'logs'):
                            logs = list(transaction.meta.logs)

                    # Debug: Log extraction
                    if logs:
                        logger.debug(f"📝 Extracted {len(logs)} logs from transaction {signature[:16]}...")
                    else:
                        logger.debug(f"⚠️ No logs found in transaction {signature[:16]}...")

                    # Parse TradeEvent (includes bonding curve reserves)
                    trade_event = self.parse_trade_event_from_logs(logs) if logs else None

                    for ix in msg.instructions:
                        ix_data = bytes(ix.data)

                        # Check for token creation
                        if ix_data.startswith(PUMP_CREATE_DISCRIMINATOR):
                            creation_data = self.decode_create_instruction(
                                ix_data,
                                msg.account_keys,
                                list(ix.accounts)
                            )

                            if creation_data:
                                token_creation = TokenCreation(
                                    mint=creation_data['mint'],
                                    bonding_curve=creation_data['bonding_curve'],
                                    associated_bonding_curve=creation_data['associated_bonding_curve'],
                                    creator=creation_data['creator'],
                                    name=creation_data['name'],
                                    symbol=creation_data['symbol'],
                                    signature=signature,
                                    slot=slot,
                                    timestamp=datetime.now()
                                )

                                try:
                                    await self.on_creation_callback(token_creation)
                                except Exception as e:
                                    logger.error("creation_callback_error", error=str(e))

                        # Check for buys
                        elif ix_data.startswith(PUMP_BUY_DISCRIMINATOR):
                            buy_data = self.decode_buy_instruction(
                                ix_data,
                                msg.account_keys,
                                list(ix.accounts)
                            )

                            if buy_data:
                                # Add TradeEvent data if available (bonding curve state!)
                                if trade_event and trade_event['mint'] == buy_data['mint']:
                                    buy_data['trade_event'] = trade_event

                                try:
                                    await self.on_buy_callback(buy_data)
                                except Exception as e:
                                    logger.error("buy_callback_error", error=str(e))

                        # Check for sells (same callback, distinguished by is_buy in TradeEvent)
                        elif ix_data.startswith(PUMP_SELL_DISCRIMINATOR):
                            sell_data = self.decode_buy_instruction(  # Use same decoder, mint is still account #2
                                ix_data,
                                msg.account_keys,
                                list(ix.accounts)
                            )

                            if sell_data:
                                # Add TradeEvent data if available (bonding curve state!)
                                if trade_event and trade_event['mint'] == sell_data['mint']:
                                    sell_data['trade_event'] = trade_event

                                try:
                                    await self.on_buy_callback(sell_data)  # Same callback handles both
                                except Exception as e:
                                    logger.error("sell_callback_error", error=str(e))

                except Exception as e:
                    logger.error("message_processing_error", error=str(e))

        finally:
            await channel.close()

    async def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False


class SwarmTriggerSniper:
    """Smart sniper that waits for signs of life then triggers bot swarms"""

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        buy_amount_sol: float,
        hold_seconds: int,
        priority_fee_lamports: int,
        slippage_bps: int,
        max_trades: int,
        enable_real_trades: bool
    ):
        self.buy_amount_sol = buy_amount_sol
        self.hold_seconds = hold_seconds
        self.priority_fee_lamports = priority_fee_lamports
        self.slippage_bps = slippage_bps
        self.max_trades = max_trades
        self.enable_real_trades = enable_real_trades

        self.trades_completed = 0
        self.trades_attempted = 0
        self.trades_results = []
        self.should_stop = False

        # Track tokens being monitored
        self.monitored_tokens: Dict[str, TokenActivity] = {}
        self.active_monitors = 0
        self.max_concurrent_monitors = 10  # Monitor max 10 tokens at once

        # Only allow 1 concurrent trade
        self.trade_in_progress = False

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        logger.info("wallet_loaded", pubkey=str(self.keypair.pubkey()))

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        self.geyser_endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")
        self.geyser_api_key = helius_endpoint.laserstream_api_key

        # Initialize trading components
        if self.enable_real_trades:
            self.rpc_manager = RPCManager(self.bot_config.rpc_config)
            self.wallet_manager = WalletManager(
                rpc_manager=self.rpc_manager,
                keypairs=[self.keypair],
                config=WalletManagerConfig(cooldown_seconds=0, balance_cache_ttl_seconds=0)  # No cache
            )
            self.direct_client = PumpFunDirectClient(self.rpc_manager)
            self.tx_submitter = TransactionSubmitter(
                rpc_manager=self.rpc_manager,
                config=SubmitterConfig(skip_preflight=True, max_retries=3, confirmation_timeout_s=30)
            )

    def _load_wallet(self, wallet_path: str) -> Keypair:
        """Load wallet keypair"""
        wallet_file = Path(wallet_path)
        if not wallet_file.exists():
            raise FileNotFoundError(f"Wallet not found: {wallet_path}")

        with open(wallet_file, 'r') as f:
            key_data = json.load(f)

        return Keypair.from_bytes(bytes(key_data))

    async def _get_tokens_from_transaction(self, tx_signature: str, max_wait_seconds: float = 8.0) -> Optional[float]:
        """
        Get tokens received from transaction using postTokenBalances (EXACT copy bot method)

        This is the "blob" parsing method used in the main copy trading flow.
        Retries aggressively for up to max_wait_seconds to allow transaction to be indexed.
        """
        start_time = asyncio.get_event_loop().time()
        attempt = 0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= max_wait_seconds:
                logger.warning(f"⚠️ Timeout after {elapsed:.1f}s waiting for transaction to be indexed")
                return None

            attempt += 1
            try:
                # Exponential backoff for first few tries, then poll every 0.5s
                if attempt > 1:
                    if attempt <= 4:
                        delay = 0.3 * (1.5 ** (attempt - 2))  # 0.3s, 0.45s, 0.68s, 1.0s
                    else:
                        delay = 0.5  # Poll every 0.5s after initial attempts

                    remaining = max_wait_seconds - elapsed
                    if delay > remaining:
                        delay = remaining

                    if delay > 0:
                        logger.debug(f"      ⏳ Retry {attempt} after {delay:.1f}s (elapsed: {elapsed:.1f}s)...")
                        await asyncio.sleep(delay)

                # Fetch transaction details via RPC
                tx_details = await self.rpc_manager.call_http_rpc(
                    method="getTransaction",
                    params=[
                        tx_signature,
                        {
                            "encoding": "jsonParsed",
                            "maxSupportedTransactionVersion": 0
                        }
                    ],
                    timeout=5.0
                )

                if not tx_details:
                    continue  # Retry

                # Check if transaction failed
                result = tx_details.get("result")
                if not result:
                    continue  # Retry

                meta = result.get("meta", {})
                if meta.get("err"):
                    logger.warning(f"⚠️ Transaction failed: {meta.get('err')}")
                    return None  # Don't retry on failed transaction

                # Get wallet address
                wallet_address = str(self.keypair.pubkey())

                # Parse postTokenBalances and preTokenBalances (THE BLOB!)
                post_balances = meta.get("postTokenBalances", [])
                pre_balances = meta.get("preTokenBalances", [])

                if not post_balances:
                    logger.debug(f"      📊 No post balances yet on attempt {attempt}, retrying...")
                    continue  # Retry

                logger.info(f"      📊 Found {len(post_balances)} post balances, {len(pre_balances)} pre balances")

                # Calculate token balance change
                post_amount = 0.0
                pre_amount = 0.0

                for balance in post_balances:
                    if balance.get("owner") == wallet_address:
                        post_amount = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                        logger.info(f"      📊 Post balance: {post_amount:,.2f}")
                        break

                for balance in pre_balances:
                    if balance.get("owner") == wallet_address:
                        pre_amount = float(balance.get("uiTokenAmount", {}).get("uiAmount", 0))
                        logger.info(f"      📊 Pre balance: {pre_amount:,.2f}")
                        break

                tokens_received = post_amount - pre_amount

                if tokens_received > 0:
                    logger.info(f"      ✅ Tokens received (from postTokenBalances blob): {tokens_received:,.2f}")
                    return tokens_received
                else:
                    logger.debug(f"⚠️ No token increase on attempt {attempt}, retrying...")
                    continue  # Retry

            except Exception as e:
                logger.debug(f"⚠️ Error on attempt {attempt}: {e}")
                continue  # Retry

    async def run(self):
        """Run the sniper"""
        logger.info("=== SWARM TRIGGER SNIPER STARTED ===")
        logger.info(f"Strategy: Wait for {MIN_BUYERS_FOR_TRIGGER}+ buyers, {MIN_VOLUME_SOL}+ SOL volume, avg buy < {MAX_AVERAGE_BUY_SOL} SOL")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL")
        logger.info(f"Slippage: {self.slippage_bps / 100:.1f}%")
        logger.info(f"Hold time: {self.hold_seconds} seconds")
        logger.info(f"Priority fee: {self.priority_fee_lamports / 1e9:.6f} SOL")
        logger.info(f"Max trades: {self.max_trades}")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Waiting for tokens with early signs of life...\n")

        if self.enable_real_trades:
            await self.rpc_manager.start()
            await self.wallet_manager.start()

            balance = self.wallet_manager.get_total_balance_sol()
            logger.info(f"Wallet balance: {balance:.4f} SOL")

        # Start activity monitor
        monitor = TokenActivityMonitor(self.geyser_endpoint, self.geyser_api_key)
        monitoring_task = asyncio.create_task(
            monitor.start_monitoring(
                on_creation=self._on_token_creation,
                on_buy=self._on_buy_activity
            )
        )

        try:
            while not self.should_stop:
                await asyncio.sleep(1)

            logger.info(f"\n✅ Sniper complete!")
            logger.info(f"   Successful trades: {self.trades_completed}/{self.max_trades}")
            logger.info(f"   Total attempts: {self.trades_attempted}")

            # Print summary
            if self.trades_results:
                successful = sum(1 for r in self.trades_results if r.success)
                logger.info(f"\n📊 SUMMARY:")
                logger.info(f"   Success rate: {successful}/{self.trades_attempted} ({successful/self.trades_attempted*100:.1f}%)")

                for i, result in enumerate(self.trades_results, 1):
                    logger.info(f"\n   Trade {i}: {result.symbol}")
                    logger.info(f"      Trigger: {result.trigger_age_seconds:.1f}s old, {result.trigger_buyers} buyers, {result.trigger_volume:.2f} SOL vol")
                    logger.info(f"      Buy: {result.buy_signature[:16]}...")
                    if result.success:
                        logger.info(f"      Sell: {result.sell_signature[:16]}...")
                        logger.info(f"      Status: ✅ SUCCESS")
                    else:
                        logger.info(f"      Status: ❌ FAILED - {result.error}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

    async def _on_token_creation(self, creation: TokenCreation):
        """Handle new token creation - start monitoring"""
        if self.trades_completed >= self.max_trades or self.should_stop:
            return

        # Don't monitor if too many concurrent
        if self.active_monitors >= self.max_concurrent_monitors:
            logger.debug("max_concurrent_monitors_reached", mint=creation.mint[:16])
            return

        # Start monitoring this token
        self.monitored_tokens[creation.mint] = TokenActivity(
            mint=creation.mint,
            symbol=creation.symbol,
            creation_time=creation.timestamp,
            last_activity=datetime.now()
        )
        self.active_monitors += 1

        logger.info(f"👁️ Monitoring: {creation.symbol} ({creation.mint[:16]}...)")

        # Start monitoring task
        asyncio.create_task(self._monitor_token(creation))

    async def _on_buy_activity(self, buy_data: dict):
        """Handle buy/sell activity - update token stats AND bonding curve state"""
        mint = buy_data['mint']

        if mint not in self.monitored_tokens:
            return

        activity = self.monitored_tokens[mint]
        now = datetime.now()

        # Check if this is a buy or sell (need to parse from trade_event)
        is_buy = True  # Default assumption for buy instruction
        if 'trade_event' in buy_data:
            is_buy = buy_data['trade_event'].get('is_buy', True)

        # Update activity counts
        if is_buy:
            activity.buy_count += 1
        else:
            activity.sell_count += 1

        activity.volume_sol += buy_data['sol_estimate']
        activity.last_activity = now

        # Update bonding curve state from TradeEvent (if available)
        if 'trade_event' in buy_data:
            trade_event = buy_data['trade_event']
            old_sol = activity.virtual_sol_reserves / 1e9
            old_tokens = activity.virtual_token_reserves / 1e6
            activity.virtual_sol_reserves = trade_event['virtual_sol_reserves']
            activity.virtual_token_reserves = trade_event['virtual_token_reserves']
            new_sol = activity.virtual_sol_reserves / 1e9
            new_tokens = activity.virtual_token_reserves / 1e6

            # Update peak
            if activity.virtual_sol_reserves > activity.peak_sol_reserves:
                activity.peak_sol_reserves = activity.virtual_sol_reserves

            # Track recent trades (keep last 30 seconds)
            activity.recent_trades.append({
                'timestamp': now,
                'is_buy': is_buy,
                'sol_amount': buy_data['sol_estimate'],
                'reserves_after': activity.virtual_sol_reserves
            })

            # Clean old trades (keep only last 30s)
            activity.recent_trades = [
                t for t in activity.recent_trades
                if (now - t['timestamp']).total_seconds() <= 30
            ]

            direction = "BUY" if is_buy else "SELL"
            logger.info(f"📊 {activity.symbol} {direction}: SOL {old_sol:.2f}→{new_sol:.2f}, Tokens {old_tokens:,.0f}→{new_tokens:,.0f}")
        else:
            logger.debug(f"⚠️ No TradeEvent for {activity.symbol} trade - curve at {activity.virtual_sol_reserves / 1e9:.2f} SOL")

    async def _monitor_token(self, creation: TokenCreation):
        """Monitor a token for trigger conditions"""
        mint = creation.mint
        symbol = creation.symbol

        try:
            while mint in self.monitored_tokens:
                activity = self.monitored_tokens[mint]
                now = datetime.now()

                # Check if stale (no activity for STALE_CUTOFF seconds)
                if activity.last_activity:
                    time_since_activity = (now - activity.last_activity).total_seconds()
                    if time_since_activity > STALE_CUTOFF:
                        logger.debug("token_stale_abandoned",
                                   mint=mint[:16],
                                   symbol=symbol,
                                   buyers=activity.buy_count,
                                   volume=activity.volume_sol)
                        break

                # Check if exceeded max monitor time
                token_age = (now - activity.creation_time).total_seconds()
                if token_age > MAX_MONITOR_TIME:
                    logger.debug("token_monitoring_timeout",
                               mint=mint[:16],
                               symbol=symbol,
                               buyers=activity.buy_count,
                               volume=activity.volume_sol)
                    break

                # Check trigger conditions
                if (activity.buy_count >= MIN_BUYERS_FOR_TRIGGER and
                    activity.volume_sol >= MIN_VOLUME_SOL and
                    token_age >= TARGET_AGE_MIN and
                    not activity.triggered):

                    # Check for wash trading (average buy too high)
                    avg_buy = activity.volume_sol / activity.buy_count
                    if avg_buy > MAX_AVERAGE_BUY_SOL:
                        logger.info(f"🚫 REJECTED: {symbol} - Avg buy {avg_buy:.1f} SOL (likely wash trading)")
                        activity.triggered = True  # Mark as triggered so we don't check again
                        break

                    # === DUMP PROTECTION CHECKS ===

                    # Check 1: Recent large sells (avoid buying after dumps)
                    recent_trades = [t for t in activity.recent_trades
                                   if (now - t['timestamp']).total_seconds() <= MIN_RECENT_TRADE_LOOKBACK]

                    if recent_trades:
                        # Check for large sells
                        large_sells = [t for t in recent_trades if not t['is_buy'] and t['sol_amount'] > MAX_RECENT_SELL_SOL]
                        if large_sells:
                            logger.info(f"🚫 REJECTED: {symbol} - Large sell detected in last {MIN_RECENT_TRADE_LOOKBACK}s ({large_sells[-1]['sol_amount']:.2f} SOL)")
                            activity.triggered = True
                            break

                        # Check 2: Buy/sell ratio (need 80%+ buys)
                        recent_buys = sum(1 for t in recent_trades if t['is_buy'])
                        recent_sells = sum(1 for t in recent_trades if not t['is_buy'])
                        buy_ratio = recent_buys / (recent_buys + recent_sells) if (recent_buys + recent_sells) > 0 else 0

                        if buy_ratio < MIN_BUY_RATIO:
                            logger.info(f"🚫 REJECTED: {symbol} - Buy ratio too low ({buy_ratio:.1%}, need {MIN_BUY_RATIO:.0%}+)")
                            activity.triggered = True
                            break

                        # Check 3: Time since last buy (momentum check)
                        last_buy_time = max((t['timestamp'] for t in recent_trades if t['is_buy']), default=None)
                        if last_buy_time:
                            seconds_since_last_buy = (now - last_buy_time).total_seconds()
                            if seconds_since_last_buy > MAX_SECONDS_SINCE_LAST_BUY:
                                logger.info(f"🚫 REJECTED: {symbol} - No recent buys ({seconds_since_last_buy:.1f}s since last buy)")
                                activity.triggered = True
                                break

                    # Check 4: Peak proximity (avoid buying retracements)
                    peak_proximity = activity.virtual_sol_reserves / activity.peak_sol_reserves if activity.peak_sol_reserves > 0 else 1.0
                    if peak_proximity < MIN_PEAK_PROXIMITY:
                        logger.info(f"🚫 REJECTED: {symbol} - Retracing from peak ({peak_proximity:.1%} of ATH, need {MIN_PEAK_PROXIMITY:.0%}+)")
                        activity.triggered = True
                        break

                    # Check 5: Entry price protection (CRITICAL - avoid late-stage pumps)
                    current_reserves_sol = activity.virtual_sol_reserves / 1e9
                    if current_reserves_sol > MAX_ENTRY_RESERVES_SOL:
                        logger.info(f"🚫 REJECTED: {symbol} - Entry price too high ({current_reserves_sol:.2f} SOL reserves, max {MAX_ENTRY_RESERVES_SOL:.1f} SOL)")
                        activity.triggered = True
                        break

                    # Check if already trading
                    if self.trade_in_progress:
                        logger.debug("trade_in_progress_skipping",
                                   mint=mint[:16],
                                   symbol=symbol)
                        await asyncio.sleep(0.5)
                        continue

                    # TRIGGER!
                    activity.triggered = True
                    logger.info(f"🎯 TRIGGER: {symbol} - {activity.buy_count} buyers, {activity.volume_sol:.2f} SOL (avg: {avg_buy:.1f}), {token_age:.1f}s old")
                    logger.info(f"   ✅ Momentum checks passed: {len(recent_trades)} recent trades, {buy_ratio:.0%} buys, {peak_proximity:.1%} of peak")

                    # Execute trade (sets trade_in_progress flag)
                    await self._execute_trade(creation, activity, token_age)
                    break

                # Sleep and check again
                await asyncio.sleep(0.5)

        finally:
            # Cleanup
            if mint in self.monitored_tokens:
                del self.monitored_tokens[mint]
                self.active_monitors -= 1

    async def _execute_trade(self, creation: TokenCreation, activity: TokenActivity, trigger_age: float):
        """Execute buy-hold-sell cycle"""
        if not self.enable_real_trades:
            logger.info("   ⏭️ Simulation mode - skipping trade")
            return

        if self.trades_completed >= self.max_trades:
            self.should_stop = True
            return

        # Set trade in progress flag
        self.trade_in_progress = True

        start_time = datetime.now()
        mint = creation.mint
        symbol = creation.symbol

        try:
            # Step 1: BUY
            logger.info(f"   1️⃣ BUYING {self.buy_amount_sol} SOL...")

            # Use TRACKED bonding curve state from Geyser (accurate AND fast!)
            curve_state = BondingCurveState(
                virtual_token_reserves=activity.virtual_token_reserves,
                virtual_sol_reserves=activity.virtual_sol_reserves,
                real_token_reserves=0,
                real_sol_reserves=0,
                token_total_supply=1_000_000_000_000_000,  # 1B tokens
                complete=False
            )
            bonding_curve = creation.bonding_curve
            associated_bonding_curve = creation.associated_bonding_curve
            creator = creation.creator

            logger.info(f"   Using tracked curve state: {activity.virtual_sol_reserves / 1e9:.2f} SOL reserves")

            # Calculate expected tokens (for slippage protection)
            calculator = BondingCurveCalculator()
            sol_lamports = int(self.buy_amount_sol * 1e9)
            buy_quote = calculator.calculate_buy_price(curve_state, sol_lamports)
            tokens_expected_raw = int(buy_quote.tokens_out)
            tokens_expected_ui = tokens_expected_raw / 1e6

            logger.info(f"   Expected tokens: {tokens_expected_ui:,.2f} (from tracked curve state)")

            async with self.wallet_manager.get_wallet(min_sol=self.buy_amount_sol + 0.01) as wallet:
                buy_signed = await self.direct_client.create_buy_transaction(
                    keypair=self.keypair,
                    mint_str=mint,
                    sol_amount=self.buy_amount_sol,
                    slippage_bps=self.slippage_bps,
                    priority_fee_lamports=self.priority_fee_lamports,
                    compute_units=150_000,
                    curve_state=curve_state,
                    bonding_curve_address=bonding_curve,
                    associated_bonding_curve_address=associated_bonding_curve,
                    creator_address=creator
                )

                if not buy_signed:
                    raise RuntimeError("Buy transaction build failed")

                buy_confirmed = await self.tx_submitter.submit_and_confirm(
                    buy_signed,
                    timeout_seconds=30
                )

                if buy_confirmed.error:
                    raise RuntimeError(f"Buy failed: {buy_confirmed.error}")

                buy_slot = buy_confirmed.slot
                buy_signature = buy_confirmed.signature

                logger.info(f"      ✅ Buy confirmed: {buy_signature[:16]}...")

                # Get ACTUAL tokens from transaction using postTokenBalances (like copy bot!)
                # NOTE: We try for almost the full hold duration since we're waiting anyway!
                fetch_start = asyncio.get_event_loop().time()
                max_fetch_time = self.hold_seconds - 0.5  # Leave 0.5s for the actual hold
                logger.info(f"      🔍 Fetching actual tokens from transaction ({max_fetch_time:.1f}s timeout)...")
                actual_tokens_ui = await self._get_tokens_from_transaction(buy_signature, max_wait_seconds=max_fetch_time)
                fetch_elapsed = asyncio.get_event_loop().time() - fetch_start

                if not actual_tokens_ui:
                    # Fall back to estimate if RPC fetch fails
                    logger.warning(f"      ⚠️ Could not fetch tokens from RPC, using estimate")
                    actual_tokens_ui = tokens_expected_ui

                logger.info(f"      📊 Tokens to sell: {actual_tokens_ui:,.2f} (from postTokenBalances blob)")

            # Step 2: HOLD
            # Adjust hold time based on how long we spent fetching tokens
            remaining_hold = max(0, self.hold_seconds - fetch_elapsed)
            if remaining_hold > 0:
                logger.info(f"   2️⃣ HOLDING for {remaining_hold:.1f} more seconds (spent {fetch_elapsed:.1f}s fetching tokens)...")
                await asyncio.sleep(remaining_hold)
            else:
                logger.info(f"   2️⃣ Skipping hold (already waited {fetch_elapsed:.1f}s fetching tokens)...")

            # Step 3: SELL
            logger.info(f"   3️⃣ SELLING {actual_tokens_ui:,.2f} tokens...")

            # Fetch FRESH bonding curve state from RPC (don't use stale tracked/calculated state!)
            logger.info(f"      🔍 Fetching current bonding curve state from RPC...")
            fetch_result = await fetch_bonding_curve_state(
                self.rpc_manager,
                mint  # Pass MINT address, not bonding curve address!
            )

            if not fetch_result:
                raise RuntimeError("Could not fetch current bonding curve state for sell")

            # Unpack result (returns tuple of state + addresses)
            fresh_curve_state, fresh_bonding_curve, fresh_assoc_curve, fresh_creator = fetch_result

            logger.info(f"      📊 Fresh curve state: {fresh_curve_state.virtual_sol_reserves / 1e9:.2f} SOL reserves")

            async with self.wallet_manager.get_wallet(min_sol=0.01) as wallet:
                sell_signed = await self.direct_client.create_sell_transaction(
                    keypair=self.keypair,
                    mint_str=mint,
                    token_amount=actual_tokens_ui,  # Use ACTUAL tokens from postTokenBalances!
                    slippage_bps=self.slippage_bps,
                    priority_fee_lamports=self.priority_fee_lamports,
                    compute_units=120_000,
                    curve_state=fresh_curve_state,  # Use FRESH curve state from RPC!
                    bonding_curve_address=str(fresh_bonding_curve),  # Use FRESH addresses from RPC!
                    associated_bonding_curve_address=str(fresh_assoc_curve),
                    creator_address=str(fresh_creator)
                )

                if not sell_signed:
                    raise RuntimeError("Sell transaction build failed")

                sell_confirmed = await self.tx_submitter.submit_and_confirm(
                    sell_signed,
                    timeout_seconds=30
                )

                if sell_confirmed.error:
                    raise RuntimeError(f"Sell failed: {sell_confirmed.error}")

                sell_slot = sell_confirmed.slot
                sell_signature = sell_confirmed.signature

                logger.info(f"      ✅ Sell confirmed: {sell_signature[:16]}...")

            end_time = datetime.now()
            hold_seconds = (end_time - start_time).total_seconds()

            result = TradeResult(
                mint=mint,
                symbol=symbol,
                buy_signature=buy_signature,
                buy_slot=buy_slot,
                buy_sol=self.buy_amount_sol,
                sell_signature=sell_signature,
                sell_slot=sell_slot,
                trigger_age_seconds=trigger_age,
                trigger_buyers=activity.buy_count,
                trigger_volume=activity.volume_sol,
                hold_seconds=hold_seconds,
                success=True
            )

            self.trades_results.append(result)
            self.trades_completed += 1
            self.trades_attempted += 1
            logger.info(f"   ✅ Trade {self.trades_completed} complete ({self.trades_attempted} attempted)")

            if self.trades_completed >= self.max_trades:
                self.should_stop = True

        except Exception as e:
            logger.error(f"   ❌ Trade failed: {e}")
            import traceback
            traceback.print_exc()

            # Record failed trade
            self.trades_results.append(TradeResult(
                mint=mint,
                symbol=symbol,
                buy_signature="",
                buy_slot=0,
                buy_sol=self.buy_amount_sol,
                sell_signature="",
                sell_slot=0,
                trigger_age_seconds=trigger_age,
                trigger_buyers=activity.buy_count,
                trigger_volume=activity.volume_sol,
                hold_seconds=0,
                success=False,
                error=str(e)
            ))
            self.trades_attempted += 1
            logger.info(f"   Failed trade doesn't count toward limit ({self.trades_completed}/{self.max_trades} completed, {self.trades_attempted} attempted)")

            # Don't increment trades_completed on failure - try again!

        finally:
            # Always clear the trade in progress flag
            self.trade_in_progress = False


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Swarm Trigger Sniper - Wait for signs of life, trigger bot swarms")
    parser.add_argument("--config", default="frontrun/config/config.yml", help="Config path")
    parser.add_argument("--wallet", required=True, help="Wallet keypair JSON")
    parser.add_argument("--buy-amount", type=float, default=1.0, help="SOL to buy per trade")
    parser.add_argument("--hold-seconds", type=int, default=10, help="Seconds to hold each position")
    parser.add_argument("--priority-fee", type=int, default=1000000, help="Priority fee lamports")
    parser.add_argument("--slippage-bps", type=int, default=2000, help="Slippage tolerance in basis points (default: 2000 = 20%%)")
    parser.add_argument("--max-trades", type=int, default=5, help="Maximum SUCCESSFUL trades before stopping")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades")

    args = parser.parse_args()

    setup_logging(level="INFO", format="json", output_file="logs/snipe_swarm_trigger.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE ⚠️")
        logger.warning(f"Strategy: Wait for {MIN_BUYERS_FOR_TRIGGER}+ buyers, {MIN_VOLUME_SOL}+ SOL volume, avg < {MAX_AVERAGE_BUY_SOL} SOL, age {TARGET_AGE_MIN}-{TARGET_AGE_MAX}s")
        logger.warning(f"Will execute {args.max_trades} trades with {args.buy_amount} SOL each")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    sniper = SwarmTriggerSniper(
        config_path=args.config,
        wallet_path=args.wallet,
        buy_amount_sol=args.buy_amount,
        hold_seconds=args.hold_seconds,
        priority_fee_lamports=args.priority_fee,
        slippage_bps=args.slippage_bps,
        max_trades=args.max_trades,
        enable_real_trades=args.real_trades
    )

    await sniper.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Sniper interrupted")
    except Exception as e:
        logger.error("sniper_failed", error=str(e))
        raise
