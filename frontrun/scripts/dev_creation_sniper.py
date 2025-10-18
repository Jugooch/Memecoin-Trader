"""
Dev Token Creation Sniper

Strategy: Frontrun the DEV's initial buy by detecting token creation!

Instead of frontrunning retail (after dev buys 5 SOL), we frontrun the DEV ITSELF:
1. Dev creates token → LaserStream detects CREATION
2. Check if creator is on our watchlist (high buy-in rate devs)
3. BUY IMMEDIATELY (before dev's first buy!)
4. Dev buys 5 SOL → Price pumps
5. Retail wave → Price pumps more
6. SELL into the waves

This is a MUCH better edge than phase 2!

Usage:
    python scripts/dev_creation_sniper.py --wallet data/phase2-wallet.json --real-trades --max-trades 5 --watchlist data/dev_watchlist.json
"""

import asyncio
import sys
import json
import struct
import base58
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.rpc_manager import RPCManager
from core.wallet_manager import WalletManager, WalletManagerConfig
from core.dev_watchlist_loader import DevWatchlistLoader
from core.bonding_curve import get_initial_bonding_curve_state, BondingCurveCalculator, calculate_curve_state_after_buy
from clients.pump_fun_direct import PumpFunDirectClient
from core.tx_submitter import TransactionSubmitter, SubmitterConfig

# Import Geyser/LaserStream (we'll create a token creation monitor)
import grpc
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))
import geyser_pb2
import geyser_pb2_grpc

logger = get_logger(__name__)


# Token creation discriminator (from GitHub example)
PUMP_CREATE_DISCRIMINATOR = struct.pack("<Q", 8576854823835016728)
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")


@dataclass
class TokenCreation:
    """Token creation event data"""
    mint: str
    bonding_curve: str
    associated_bonding_curve: str
    creator: str
    name: str
    symbol: str
    uri: str
    signature: str
    slot: int
    timestamp: datetime


@dataclass
class SniperTrade:
    """Record of a sniper trade"""
    timestamp: datetime
    mint: str
    creator: str
    signature_buy: str
    signature_sell: str
    buy_amount_sol: float
    sell_amount_sol: float
    profit_sol: float
    success: bool
    reason: str
    creation_slot: int
    buy_slot: int
    sell_slot: int
    time_to_buy_ms: float  # How fast we bought after creation

    def to_dict(self):
        return {
            **asdict(self),
            "timestamp": self.timestamp.isoformat()
        }


class TokenCreationMonitor:
    """
    Monitor for Pump.fun token CREATION events via LaserStream/Geyser

    Unlike mempool_monitor which detects BUYS, this detects token CREATIONS
    """

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self._monitoring = False
        logger.info("token_creation_monitor_initialized", endpoint=endpoint)

    def decode_create_instruction(self, ix_data: bytes, account_keys: List, accounts: List[int]) -> Optional[Dict]:
        """
        Decode token creation instruction

        Args:
            ix_data: Instruction data bytes
            account_keys: List of account keys (as bytes)
            accounts: List of account indices used in this instruction

        Returns:
            Dict with creation data or None if invalid
        """
        try:
            # Skip discriminator (first 8 bytes)
            offset = 8

            # Helper to read account key
            def get_account_key(index: int) -> str:
                if index >= len(accounts):
                    return None
                account_index = accounts[index]
                if account_index >= len(account_keys):
                    return None
                key_bytes = account_keys[account_index]
                return base58.b58encode(bytes(key_bytes)).decode()

            # Helper to read string (4-byte length prefix + string)
            def read_string() -> str:
                nonlocal offset
                length = struct.unpack_from("<I", ix_data, offset)[0]
                offset += 4
                value = ix_data[offset:offset + length].decode('utf-8')
                offset += length
                return value

            # Helper to read pubkey (32 bytes)
            def read_pubkey() -> str:
                nonlocal offset
                value = base58.b58encode(ix_data[offset:offset + 32]).decode('utf-8')
                offset += 32
                return value

            # Parse fields
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

    async def start_monitoring(self, callback):
        """
        Start monitoring for token creation events

        Args:
            callback: Async function to call for each token creation
        """
        self._monitoring = True

        # Setup authentication
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

        # Create channel
        channel = grpc.aio.secure_channel(self.endpoint, credentials=combined_creds)
        stub = geyser_pb2_grpc.GeyserStub(channel)

        # Build subscription request
        subscription_request = geyser_pb2.SubscribeRequest()
        subscription_request.transactions["pump_creates"].account_include.append(str(PUMP_FUN_PROGRAM))
        subscription_request.transactions["pump_creates"].failed = False
        subscription_request.commitment = geyser_pb2.CommitmentLevel.PROCESSED

        logger.info("subscribing_to_token_creations")

        try:
            # Create request iterator
            async def request_generator():
                yield subscription_request
                while self._monitoring:
                    await asyncio.sleep(30)

            # Subscribe to stream
            async for message in stub.Subscribe(request_generator()):
                if not message.HasField('transaction'):
                    continue

                try:
                    tx_update = message.transaction
                    transaction = tx_update.transaction

                    # Extract signature (from the transaction wrapper, not the inner transaction)
                    signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
                    slot = tx_update.slot

                    # Get the actual transaction message
                    tx_inner = transaction.transaction
                    msg = getattr(tx_inner, "message", None)

                    if msg is None:
                        continue

                    # Check each instruction for token creation
                    for ix in msg.instructions:
                        ix_data = bytes(ix.data)

                        # Check for creation discriminator
                        if not ix_data.startswith(PUMP_CREATE_DISCRIMINATOR):
                            continue

                        # Found a token creation!
                        creation_data = self.decode_create_instruction(
                            ix_data,
                            msg.account_keys,
                            list(ix.accounts)
                        )

                        if not creation_data:
                            continue

                        # Build TokenCreation object
                        token_creation = TokenCreation(
                            mint=creation_data['mint'],
                            bonding_curve=creation_data['bonding_curve'],
                            associated_bonding_curve=creation_data['associated_bonding_curve'],
                            creator=creation_data['creator'],
                            name=creation_data['name'],
                            symbol=creation_data['symbol'],
                            uri=creation_data['uri'],
                            signature=signature,
                            slot=slot,
                            timestamp=datetime.now()
                        )

                        # Call callback
                        try:
                            await callback(token_creation)
                        except Exception as e:
                            logger.error("callback_error", error=str(e), signature=signature)

                except Exception as e:
                    logger.error("message_processing_error", error=str(e))

        finally:
            await channel.close()

    async def stop_monitoring(self):
        """Stop monitoring"""
        self._monitoring = False
        logger.info("token_creation_monitoring_stopped")


class DevCreationSniper:
    """
    Snipe token creations from watchlisted devs

    This frontrun's the DEV's initial buy, not retail!
    """

    def __init__(
        self,
        config_path: str,
        wallet_path: str,
        watchlist_path: str,
        max_trades: int = 5,
        enable_real_trades: bool = False,
        buy_amount_sol: float = 0.01,
        hold_time_seconds: float = 3.0,
        min_buy_rate: float = 80.0
    ):
        """
        Initialize dev creation sniper

        Args:
            config_path: Path to config.yml
            wallet_path: Path to wallet keypair JSON
            watchlist_path: Path to dev watchlist JSON
            max_trades: Maximum number of trades (default: 5)
            enable_real_trades: If True, execute real trades (default: False)
            buy_amount_sol: Amount of SOL to spend per trade (default: 0.01)
            hold_time_seconds: Time to hold before selling (default: 3.0)
            min_buy_rate: Minimum buy rate % to frontrun (default: 80.0)
        """
        self.max_trades = max_trades
        self.trades: List[SniperTrade] = []
        self.enable_real_trades = enable_real_trades
        self.buy_amount_sol = buy_amount_sol
        self.hold_time_seconds = hold_time_seconds
        self.min_buy_rate = min_buy_rate

        # Track tokens we've already traded
        self._traded_mints: Set[str] = set()

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Load wallet
        self.keypair = self._load_wallet(wallet_path)
        logger.info("wallet_loaded", pubkey=str(self.keypair.pubkey()))

        # Load dev watchlist with O(1) lookup
        self.dev_watchlist = DevWatchlistLoader(watchlist_path)
        stats = self.dev_watchlist.get_stats()
        logger.info(
            "watchlist_loaded",
            dev_count=stats['total_devs'],
            avg_buy_rate=round(stats['avg_buy_rate'], 1),
            min_buy_rate_filter=min_buy_rate
        )

        # Get Helius config
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")

        # Initialize token creation monitor
        self.creation_monitor = TokenCreationMonitor(
            endpoint=endpoint,
            api_key=helius_endpoint.laserstream_api_key
        )

        # Initialize transaction components (if real trades enabled)
        self.rpc_manager: Optional[RPCManager] = None
        self.wallet_manager: Optional[WalletManager] = None
        self.direct_client: Optional[PumpFunDirectClient] = None
        self.tx_submitter: Optional[TransactionSubmitter] = None

        if self.enable_real_trades:
            self.rpc_manager = RPCManager(self.bot_config.rpc_config)
            self.wallet_manager = WalletManager(
                rpc_manager=self.rpc_manager,
                keypairs=[self.keypair],
                config=WalletManagerConfig(
                    cooldown_seconds=0,
                    balance_cache_ttl_seconds=30
                )
            )
            self.direct_client = PumpFunDirectClient(self.rpc_manager)
            self.tx_submitter = TransactionSubmitter(
                rpc_manager=self.rpc_manager,
                config=SubmitterConfig(
                    skip_preflight=True,
                    max_retries=3,
                    confirmation_timeout_s=30
                )
            )

        logger.info(
            "dev_creation_sniper_initialized",
            max_trades=max_trades,
            real_trades=enable_real_trades,
            buy_amount_sol=buy_amount_sol,
            hold_time=hold_time_seconds
        )

    def _load_wallet(self, wallet_path: str) -> Keypair:
        """Load wallet keypair"""
        wallet_file = Path(wallet_path)
        if not wallet_file.exists():
            raise FileNotFoundError(f"Wallet file not found: {wallet_path}")

        with open(wallet_file, 'r') as f:
            key_data = json.load(f)

        return Keypair.from_bytes(bytes(key_data))

    async def run(self):
        """Run dev creation sniper"""
        stats = self.dev_watchlist.get_stats()
        logger.info("=== DEV CREATION SNIPER STARTED ===")
        logger.info(f"Strategy: Frontrun dev's initial buy by detecting token creation")
        logger.info(f"Watchlist: {stats['total_devs']} devs (avg buy rate: {stats['avg_buy_rate']:.1f}%)")
        logger.info(f"Quality filter: Only frontrun devs with {self.min_buy_rate}%+ buy rate")
        logger.info(f"Will execute {self.max_trades} trades")
        logger.info(f"Buy amount: {self.buy_amount_sol} SOL per trade")
        logger.info(f"Hold time: {self.hold_time_seconds}s (to catch dev buy + retail wave)")
        logger.info(f"Mode: {'REAL TRADES' if self.enable_real_trades else 'SIMULATION'}")
        logger.info("Press Ctrl+C to stop early")

        # Start RPC manager if real trades enabled
        if self.enable_real_trades:
            await self.rpc_manager.start()
            await self.wallet_manager.start()

            balance = self.wallet_manager.get_total_balance_sol()
            min_balance = (self.max_trades * self.buy_amount_sol) + 0.1
            logger.info(
                "wallet_balance_check",
                balance_sol=balance,
                min_required=min_balance,
                sufficient=balance >= min_balance
            )

            if balance < min_balance:
                raise RuntimeError(
                    f"Insufficient balance: {balance:.4f} SOL. Need {min_balance:.1f} SOL"
                )

        # Start monitoring
        monitoring_task = asyncio.create_task(
            self.creation_monitor.start_monitoring(self._on_token_creation)
        )

        try:
            # Wait until we've executed max_trades
            while len(self.trades) < self.max_trades:
                await asyncio.sleep(1)

            logger.info(f"Completed {self.max_trades} trades!")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            await self.creation_monitor.stop_monitoring()
            monitoring_task.cancel()

            if self.enable_real_trades:
                await self.wallet_manager.stop()
                await self.rpc_manager.stop()

            await self._generate_report()

    async def _on_token_creation(self, creation: TokenCreation):
        """
        Handle token creation event

        Args:
            creation: TokenCreation event
        """
        # Check if we're done
        if len(self.trades) >= self.max_trades:
            return

        # Skip if already traded this token
        if creation.mint in self._traded_mints:
            return

        # CRITICAL: Check if creator is on watchlist (O(1) lookup!)
        if not self.dev_watchlist.is_tracked_dev(creation.creator):
            logger.debug(
                "creator_not_on_watchlist",
                creator=creation.creator[:16] + "...",
                mint=creation.mint[:16] + "..."
            )
            return

        # Get dev profile to check quality
        profile = self.dev_watchlist.get_dev_profile(creation.creator)

        # Filter by buy rate (only frontrun high-quality devs)
        if profile['buy_rate'] < self.min_buy_rate:
            logger.info(
                "dev_buy_rate_too_low",
                creator=creation.creator[:16] + "...",
                buy_rate=profile['buy_rate'],
                min_required=self.min_buy_rate,
                verdict="SKIP"
            )
            return

        # Found a high-quality dev creating a token!
        logger.info(
            "🎯 HIGH-QUALITY DEV CREATED TOKEN!",
            creator=creation.creator[:16] + "...",
            buy_rate=profile['buy_rate'],
            tokens_created=profile['tokens_created'],
            mint=creation.mint[:16] + "...",
            symbol=creation.symbol,
            name=creation.name
        )

        # Mark as traded immediately
        self._traded_mints.add(creation.mint)

        # Execute trade
        await self._execute_snipe(creation)

    async def _execute_snipe(self, creation: TokenCreation):
        """
        Execute a snipe trade

        Args:
            creation: Token creation event
        """
        trade_num = len(self.trades) + 1
        creation_time = creation.timestamp

        logger.info(
            f"=== SNIPE #{trade_num} ===",
            mint=creation.mint,
            creator=creation.creator[:16] + "...",
            symbol=creation.symbol
        )

        if self.enable_real_trades:
            await self._execute_real_snipe(trade_num, creation, creation_time)
        else:
            await self._execute_simulated_snipe(trade_num, creation, creation_time)

    async def _execute_real_snipe(self, trade_num: int, creation: TokenCreation, creation_time: datetime):
        """Execute real snipe trade"""
        import time

        buy_signature = ""
        sell_signature = ""
        buy_slot = 0
        sell_slot = 0
        sell_sol_received = 0.0

        try:
            # Get wallet
            async with self.wallet_manager.get_wallet(min_sol=self.buy_amount_sol + 0.01) as wallet:
                # BUY IMMEDIATELY using initial bonding curve state (no RPC fetch!)
                buy_start = time.time()

                logger.info("🚀 BUYING (frontrunning dev!)", mint=creation.mint[:16] + "...")

                # Use initial bonding curve state (before any buys)
                initial_curve_state = get_initial_bonding_curve_state()

                # Calculate buy quote to get tokens_out (we'll use this for selling later)
                calculator = BondingCurveCalculator()
                sol_lamports = int(self.buy_amount_sol * 1e9)
                buy_quote = calculator.calculate_buy_price(initial_curve_state, sol_lamports)
                tokens_out_raw = buy_quote.tokens_out  # Raw token units (with 6 decimals)

                logger.debug(
                    "buy_quote_calculated",
                    tokens_out_raw=tokens_out_raw,
                    tokens_out_ui=tokens_out_raw / 1e6,
                    sol_in=self.buy_amount_sol
                )

                buy_signed = await self.direct_client.create_buy_transaction(
                    keypair=self.keypair,
                    mint_str=creation.mint,
                    sol_amount=self.buy_amount_sol,
                    slippage_bps=1000,  # 10% slippage for new tokens
                    priority_fee_lamports=10_000_000,  # 0.01 SOL priority (ULTRA HIGH for speed)
                    compute_units=150_000,
                    # Pass pre-known addresses and state (no RPC fetch needed!)
                    curve_state=initial_curve_state,
                    bonding_curve_address=creation.bonding_curve,
                    associated_bonding_curve_address=creation.associated_bonding_curve,
                    creator_address=creation.creator
                )

                if not buy_signed:
                    raise RuntimeError("Buy transaction build failed")

                buy_confirmed = await self.tx_submitter.submit_and_confirm(
                    buy_signed,
                    timeout_seconds=30
                )

                if buy_confirmed.error:
                    raise RuntimeError(f"Buy failed: {buy_confirmed.error}")

                buy_signature = buy_confirmed.signature
                buy_slot = buy_confirmed.slot
                time_to_buy_ms = (time.time() - buy_start) * 1000

                logger.info(
                    "✅ BUY COMPLETE",
                    signature=buy_signature[:16] + "...",
                    buy_slot=buy_slot,
                    creation_slot=creation.slot,
                    time_to_buy_ms=round(time_to_buy_ms, 2)
                )

                # HOLD (wait for dev buy + retail wave)
                logger.info(
                    "⏳ HOLDING",
                    hold_seconds=self.hold_time_seconds,
                    strategy="Waiting for dev buy + retail wave"
                )

                await asyncio.sleep(self.hold_time_seconds)

                # SELL
                logger.info("💰 SELLING", mint=creation.mint[:16] + "...")

                # Use the tokens_out from our buy quote (no RPC query needed!)
                # This avoids RPC indexing delays
                token_ui_amount = tokens_out_raw / 1e6

                # Calculate the bonding curve state AFTER our buy
                # This allows us to build the sell transaction without fetching from RPC
                post_buy_curve_state = calculate_curve_state_after_buy(
                    initial_curve_state,
                    sol_lamports,
                    tokens_out_raw
                )

                logger.debug(
                    "post_buy_curve_state_calculated",
                    virtual_sol_reserves=post_buy_curve_state.virtual_sol_reserves / 1e9,
                    virtual_token_reserves=post_buy_curve_state.virtual_token_reserves / 1e6,
                    tokens_to_sell=token_ui_amount
                )

                sell_signed = await self.direct_client.create_sell_transaction(
                    keypair=self.keypair,
                    mint_str=creation.mint,
                    token_amount=token_ui_amount,
                    slippage_bps=1000,  # 10% slippage
                    priority_fee_lamports=10_000_000,
                    compute_units=150_000,
                    # Pass pre-known post-buy curve state and addresses (no RPC fetch!)
                    curve_state=post_buy_curve_state,
                    bonding_curve_address=creation.bonding_curve,
                    associated_bonding_curve_address=creation.associated_bonding_curve,
                    creator_address=creation.creator
                )

                if not sell_signed:
                    raise RuntimeError("Sell transaction build failed")

                sell_confirmed = await self.tx_submitter.submit_and_confirm(
                    sell_signed,
                    timeout_seconds=30
                )

                if sell_confirmed.error:
                    raise RuntimeError(f"Sell failed: {sell_confirmed.error}")

                sell_signature = sell_confirmed.signature
                sell_slot = sell_confirmed.slot

                # Extract SOL received
                sell_sol_received = await self._get_sol_from_transaction(sell_signature, str(wallet.pubkey))

                logger.info(
                    "✅ SELL COMPLETE",
                    signature=sell_signature[:16] + "...",
                    sol_received=sell_sol_received
                )

                # Calculate profit
                profit_sol = sell_sol_received - self.buy_amount_sol
                profit_pct = (profit_sol / self.buy_amount_sol) * 100

                logger.info(
                    "trade_pnl",
                    profit_sol=profit_sol,
                    profit_pct=round(profit_pct, 2),
                    verdict="✅ PROFIT" if profit_sol > 0 else "❌ LOSS"
                )

                # Record trade
                trade = SniperTrade(
                    timestamp=datetime.now(),
                    mint=creation.mint,
                    creator=creation.creator,
                    signature_buy=buy_signature,
                    signature_sell=sell_signature,
                    buy_amount_sol=self.buy_amount_sol,
                    sell_amount_sol=sell_sol_received,
                    profit_sol=profit_sol,
                    success=True,
                    reason=f"Complete: {profit_pct:.2f}% {'profit' if profit_sol > 0 else 'loss'}",
                    creation_slot=creation.slot,
                    buy_slot=buy_slot,
                    sell_slot=sell_slot,
                    time_to_buy_ms=time_to_buy_ms
                )

                self.trades.append(trade)

        except Exception as e:
            logger.error("trade_failed", error=str(e), trade_num=trade_num)

            trade = SniperTrade(
                timestamp=datetime.now(),
                mint=creation.mint,
                creator=creation.creator,
                signature_buy=buy_signature,
                signature_sell=sell_signature,
                buy_amount_sol=self.buy_amount_sol,
                sell_amount_sol=sell_sol_received,
                profit_sol=sell_sol_received - self.buy_amount_sol if sell_sol_received > 0 else -self.buy_amount_sol,
                success=False,
                reason=f"Failed: {str(e)}",
                creation_slot=creation.slot,
                buy_slot=buy_slot if buy_slot > 0 else 0,
                sell_slot=sell_slot if sell_slot > 0 else 0,
                time_to_buy_ms=0
            )

            self.trades.append(trade)

    async def _execute_simulated_snipe(self, trade_num: int, creation: TokenCreation, creation_time: datetime):
        """Execute simulated snipe trade"""
        simulated_profit = self.buy_amount_sol * 0.10  # Assume 10% profit

        trade = SniperTrade(
            timestamp=datetime.now(),
            mint=creation.mint,
            creator=creation.creator,
            signature_buy=f"sim_buy_{creation.signature[:16]}",
            signature_sell=f"sim_sell_{creation.signature[:16]}",
            buy_amount_sol=self.buy_amount_sol,
            sell_amount_sol=self.buy_amount_sol + simulated_profit,
            profit_sol=simulated_profit,
            success=True,
            reason="Simulated trade",
            creation_slot=creation.slot,
            buy_slot=creation.slot,
            sell_slot=creation.slot + 2,
            time_to_buy_ms=50.0
        )

        self.trades.append(trade)

        logger.info(
            "trade_simulated",
            trade_num=trade_num,
            profit_sol=simulated_profit
        )

    async def _get_token_balance(self, owner: Pubkey, mint: str) -> float:
        """Get token balance (reuse from phase2)"""
        try:
            response = await self.rpc_manager.call_http_rpc(
                "getTokenAccountsByOwner",
                [
                    str(owner),
                    {"mint": mint},
                    {"encoding": "jsonParsed"}
                ]
            )

            accounts = response.get("result", {}).get("value", [])
            if not accounts:
                return 0

            token_data = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
            token_amount = token_data.get("tokenAmount", {})
            balance = float(token_amount.get("amount", 0))

            return balance

        except Exception as e:
            logger.error("token_balance_fetch_failed", error=str(e))
            return 0

    async def _get_sol_from_transaction(self, signature: str, wallet_pubkey: str) -> float:
        """Extract SOL received from transaction (reuse from phase2)"""
        try:
            response = await self.rpc_manager.call_http_rpc(
                "getTransaction",
                [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0,
                        "commitment": "confirmed"
                    }
                ]
            )

            result = response.get("result")
            if not result:
                return 0.0

            transaction = result.get("transaction", {})
            message = transaction.get("message", {})
            account_keys = message.get("accountKeys", [])

            meta = result.get("meta", {})
            pre_balances = meta.get("preBalances", [])
            post_balances = meta.get("postBalances", [])

            # Find wallet index
            wallet_index = None
            for i, account in enumerate(account_keys):
                pubkey = account if isinstance(account, str) else account.get("pubkey")
                if pubkey == wallet_pubkey:
                    wallet_index = i
                    break

            if wallet_index is None:
                return 0.0

            pre_balance = pre_balances[wallet_index] if wallet_index < len(pre_balances) else 0
            post_balance = post_balances[wallet_index] if wallet_index < len(post_balances) else 0

            sol_diff = (post_balance - pre_balance) / 1e9

            return sol_diff

        except Exception as e:
            logger.error("sol_extraction_failed", error=str(e))
            return 0.0

    async def _generate_report(self):
        """Generate final report"""
        logger.info("\n=== DEV CREATION SNIPER COMPLETE ===\n")

        total_trades = len(self.trades)
        successful = sum(1 for t in self.trades if t.success)
        total_profit = sum(t.profit_sol for t in self.trades)

        winning_trades = sum(1 for t in self.trades if t.profit_sol > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_time_to_buy = sum(t.time_to_buy_ms for t in self.trades) / total_trades if total_trades > 0 else 0

        logger.info(
            "sniper_results",
            total_trades=total_trades,
            successful_trades=successful,
            winning_trades=winning_trades,
            win_rate=round(win_rate, 2),
            total_profit_sol=round(total_profit, 6),
            avg_time_to_buy_ms=round(avg_time_to_buy, 2)
        )

        # Export trades
        output_dir = Path("data/sniper")
        output_dir.mkdir(parents=True, exist_ok=True)

        trades_file = output_dir / f"snipes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trades_file, 'w') as f:
            json.dump([t.to_dict() for t in self.trades], f, indent=2)

        logger.info(f"✅ Trades exported to: {trades_file}")

        logger.info("\n=== STRATEGY VERDICT ===")
        if total_profit > 0 and win_rate >= 0.4:
            logger.info("✅ STRATEGY IS PROFITABLE!")
            logger.info(f"Frontrunning the DEV works! Avg speed: {avg_time_to_buy:.0f}ms")
        else:
            logger.warning("⚠️ Strategy needs tuning")
            logger.warning(f"Win rate: {win_rate*100:.1f}% | Profit: {total_profit:.6f} SOL")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dev Token Creation Sniper - Frontrun the dev's initial buy!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy:
  1. Monitor for token CREATION events (not buys)
  2. Check if creator is on watchlist (high buy-in rate devs)
  3. BUY IMMEDIATELY (frontrunning the dev's first buy!)
  4. Dev buys 5 SOL → Price pumps
  5. Retail wave → Price pumps more
  6. SELL into the waves

Examples:
  # Conservative: Only 95%+ buy rate devs, 5 trades, 0.01 SOL
  python scripts/dev_creation_sniper.py --wallet data/phase2-wallet.json --real-trades --max-trades 5 --buy-amount 0.01 --min-buy-rate 95

  # Balanced: 90%+ buy rate, 10 trades, 0.05 SOL
  python scripts/dev_creation_sniper.py --wallet data/phase2-wallet.json --real-trades --max-trades 10 --buy-amount 0.05 --min-buy-rate 90

  # Wide Net: 80%+ buy rate, more opportunities
  python scripts/dev_creation_sniper.py --wallet data/phase2-wallet.json --real-trades --max-trades 20 --buy-amount 0.01 --min-buy-rate 80

Prerequisites:
  1. Run build_dev_watchlist.py first to create data/dev_watchlist.json
  2. Fund your wallet with enough SOL
        """
    )

    parser.add_argument("--config", default="config/config.yml", help="Path to config.yml")
    parser.add_argument("--wallet", required=True, help="Path to wallet keypair JSON")
    parser.add_argument("--watchlist", default="data/dev_watchlist.json", help="Path to dev watchlist JSON")
    parser.add_argument("--max-trades", type=int, default=5, help="Maximum trades (default: 5)")
    parser.add_argument("--real-trades", action="store_true", help="Execute REAL trades (WARNING: Uses real capital!)")
    parser.add_argument("--buy-amount", type=float, default=0.01, help="SOL to spend per trade (default: 0.01)")
    parser.add_argument("--hold-time", type=float, default=3.0, help="Hold time in seconds (default: 3.0)")
    parser.add_argument("--min-buy-rate", type=float, default=90.0, help="Minimum dev buy rate %% to frontrun (default: 90%%)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="json", output_file="logs/dev_creation_sniper.log")

    if args.real_trades:
        logger.warning("⚠️ REAL TRADES MODE ENABLED - WILL USE REAL CAPITAL ⚠️")
        logger.warning(f"This will execute {args.max_trades} trades with {args.buy_amount} SOL each")
        logger.warning("Press Ctrl+C within 5 seconds to cancel...")
        await asyncio.sleep(5)

    # Create and run sniper
    sniper = DevCreationSniper(
        config_path=args.config,
        wallet_path=args.wallet,
        watchlist_path=args.watchlist,
        max_trades=args.max_trades,
        enable_real_trades=args.real_trades,
        buy_amount_sol=args.buy_amount,
        hold_time_seconds=args.hold_time,
        min_buy_rate=args.min_buy_rate
    )

    await sniper.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dev creation sniper interrupted by user")
    except Exception as e:
        logger.error("sniper_failed", error=str(e), error_type=type(e).__name__)
        raise
