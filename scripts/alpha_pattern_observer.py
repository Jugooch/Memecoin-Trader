#!/usr/bin/env python3
"""
Alpha Pattern Observer - Zero-Risk Data Collection

This analyzer OBSERVES naturally occurring "alpha" buy patterns on Pump.fun
and measures what happens after them. NO CAPITAL RISK - pure observation.

We test two hypotheses:
1. Large buys (0.5-2 SOL) trigger bot swarms → Can we replicate this?
2. Large early snipes (slot 1-3) trigger swarms → Should we snipe large?

By watching thousands of examples, we learn:
- Which patterns trigger pumps
- What size works best
- What token age works best
- When to exit
- What the success rate is

Then we can implement the winning pattern with confidence.

Usage:
    python scripts/alpha_pattern_observer.py --hours 6
"""

import asyncio
import json
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import grpc
import base58
import base64
import struct
import aiohttp

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "frontrun"))

# Import proto files
proto_path = Path(__file__).parent.parent / "frontrun" / "proto"
sys.path.insert(0, str(proto_path))

try:
    import geyser_pb2
    import geyser_pb2_grpc
    GEYSER_AVAILABLE = True
except ImportError:
    GEYSER_AVAILABLE = False
    print("❌ Geyser proto files not available")
    sys.exit(1)

from src.utils.logger_setup import setup_logging
from src.utils.config_loader import load_config


# Pump.fun program ID
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# ComputeBudgetProgram ID
COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"

# Known Jito tip accounts
JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]

# TradeEvent discriminator
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])
EVENT_DISCRIMINATOR_SIZE = 8


@dataclass
class AlphaBuyEvent:
    """A potential 'alpha' buy that might trigger a swarm"""
    token: str
    buyer_wallet: str
    buy_size_sol: float
    buy_price: float
    buy_slot: int
    buy_timestamp: datetime
    buy_position: int  # Was this buy #1, #2, #3, etc.
    token_age_seconds: float  # How old was token when this buy happened

    # Context
    holder_count_before: int = 0
    volume_before_sol: float = 0
    mcap_before: float = 0

    # Transaction identifier (for fee lookup)
    signature: str = ""

    # Fee tracking (filled in batch after observation)
    total_fee_sol: float = 0
    priority_fee_sol: float = 0
    compute_units: int = 0
    used_jito_bundle: bool = False
    jito_tip_sol: float = 0

    # Aftermath (filled in after 60s observation)
    followers: List[Dict] = field(default_factory=list)
    volume_added_sol: float = 0
    unique_followers: int = 0
    price_history: List[Dict] = field(default_factory=list)
    peak_price: float = 0
    peak_gain_pct: float = 0
    peak_time_seconds: float = 0

    # Analysis
    did_trigger_swarm: bool = False
    swarm_confidence: float = 0.0
    pattern_type: str = ""  # "early_snipe" or "established_trigger"

    # Outcome simulation (if we had copied this)
    profitable_exit_windows: List[Dict] = field(default_factory=list)
    best_exit_timing: float = 0
    best_exit_gain: float = 0


@dataclass
class TokenTracker:
    """Tracks a token's complete history"""
    mint: str
    first_seen: datetime
    creation_slot: Optional[int] = None

    buys: List[Dict] = field(default_factory=list)
    unique_wallets: Set[str] = field(default_factory=set)
    total_volume: float = 0

    # Identified alpha buys on this token
    potential_alphas: List[AlphaBuyEvent] = field(default_factory=list)


def parse_trade_event_from_logs(logs: List[str]) -> Optional[Dict]:
    """Parse TradeEvent data from transaction logs"""
    for log in logs:
        if "Program data:" in log:
            try:
                encoded_data = log.split("Program data: ")[1].strip()
                decoded_data = base64.b64decode(encoded_data)

                if len(decoded_data) >= EVENT_DISCRIMINATOR_SIZE:
                    discriminator = decoded_data[:EVENT_DISCRIMINATOR_SIZE]
                    if discriminator == TRADE_EVENT_DISCRIMINATOR:
                        return decode_trade_event(decoded_data[EVENT_DISCRIMINATOR_SIZE:])
            except Exception:
                continue
    return None


def decode_trade_event(data: bytes) -> Optional[Dict]:
    """Decode TradeEvent structure from raw bytes"""
    if len(data) < 105:
        return None

    offset = 0

    # Parse mint (32 bytes)
    mint_bytes = data[offset:offset + 32]
    mint = base58.b58encode(mint_bytes).decode('utf-8')
    offset += 32

    # Parse sol_amount (u64)
    sol_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse token_amount (u64)
    token_amount = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse is_buy (bool)
    is_buy = bool(data[offset])
    offset += 1

    # Parse user (32 bytes)
    user_bytes = data[offset:offset + 32]
    user = base58.b58encode(user_bytes).decode('utf-8')
    offset += 32

    # Parse timestamp (i64)
    timestamp = struct.unpack('<q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_sol_reserves (u64)
    virtual_sol_reserves = struct.unpack('<Q', data[offset:offset + 8])[0]
    offset += 8

    # Parse virtual_token_reserves (u64)
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
        'sol_amount_ui': sol_amount / 1e9,
        'token_amount_ui': token_amount / 1e6,
    }


async def batch_fetch_transaction_fees(signatures: List[str], rpc_url: str, logger) -> Dict[str, Dict]:
    """
    Fetch transaction fee details in batches using RPC.
    Returns dict mapping signature -> fee_info
    """
    if not signatures:
        return {}

    logger.info(f"\n📊 Fetching fee data for {len(signatures)} transactions...")
    logger.info(f"   RPC: {rpc_url}")
    logger.info(f"   ⏳ Waiting 30s for RPC indexing...")
    await asyncio.sleep(30)  # Wait for RPC indexing

    fee_map = {}
    batch_size = 100  # Process 100 signatures at a time

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(signatures), batch_size):
            batch = signatures[i:i + batch_size]
            logger.info(f"   Fetching batch {i//batch_size + 1}/{(len(signatures) + batch_size - 1)//batch_size} ({len(batch)} txs)...")

            try:
                # Build batch RPC request
                requests = []
                for idx, sig in enumerate(batch):
                    requests.append({
                        "jsonrpc": "2.0",
                        "id": idx,
                        "method": "getTransaction",
                        "params": [
                            sig,
                            {
                                "encoding": "jsonParsed",
                                "maxSupportedTransactionVersion": 0
                            }
                        ]
                    })

                async with session.post(rpc_url, json=requests) as resp:
                    if resp.status != 200:
                        logger.warning(f"   ⚠️ Batch failed: HTTP {resp.status}")
                        continue

                    responses = await resp.json()

                    # Parse responses
                    for idx, response in enumerate(responses):
                        if "result" not in response or not response["result"]:
                            continue

                        result = response["result"]
                        sig = batch[idx]

                        # Extract fee info
                        fee_info = extract_fee_from_rpc_response(result)
                        fee_map[sig] = fee_info

            except Exception as e:
                logger.error(f"   ❌ Batch error: {e}")
                continue

    logger.info(f"   ✅ Successfully fetched fee data for {len(fee_map)}/{len(signatures)} transactions\n")
    return fee_map


def extract_fee_from_rpc_response(tx_result: Dict) -> Dict:
    """Extract fee information from RPC transaction response"""
    fee_info = {
        'total_fee': 0.0,
        'priority_fee': 0.0,
        'compute_units': 0,
        'used_jito': False,
        'jito_tip': 0.0,
    }

    # Total fee from meta
    if "meta" in tx_result and "fee" in tx_result["meta"]:
        fee_info['total_fee'] = tx_result["meta"]["fee"] / 1e9

    # Parse instructions for ComputeBudget and Jito
    if "transaction" not in tx_result:
        return fee_info

    tx = tx_result["transaction"]
    if "message" not in tx:
        return fee_info

    message = tx["message"]
    instructions = message.get("instructions", [])

    # Track compute unit limit for priority fee calculation
    compute_unit_limit = 0
    compute_unit_price = 0

    for ix in instructions:
        # Check for parsed instructions
        if "parsed" in ix:
            parsed = ix["parsed"]
            if parsed.get("type") == "setComputeUnitLimit":
                compute_unit_limit = parsed.get("info", {}).get("units", 0)
            elif parsed.get("type") == "setComputeUnitPrice":
                compute_unit_price = parsed.get("info", {}).get("microLamports", 0)

        # Check for ComputeBudgetProgram (non-parsed)
        elif ix.get("programId") == COMPUTE_BUDGET_PROGRAM_ID:
            data = ix.get("data", "")
            try:
                # Decode base58 instruction data
                ix_bytes = base58.b58decode(data)
                if len(ix_bytes) >= 1:
                    discriminator = ix_bytes[0]
                    # SetComputeUnitLimit (0x02)
                    if discriminator == 2 and len(ix_bytes) >= 5:
                        compute_unit_limit = struct.unpack('<I', ix_bytes[1:5])[0]
                    # SetComputeUnitPrice (0x03)
                    elif discriminator == 3 and len(ix_bytes) >= 9:
                        compute_unit_price = struct.unpack('<Q', ix_bytes[1:9])[0]
            except:
                pass

        # Check for Jito tips (transfers to known Jito accounts)
        if "parsed" in ix:
            parsed = ix["parsed"]
            if parsed.get("type") == "transfer":
                info = parsed.get("info", {})
                destination = info.get("destination")
                if destination in JITO_TIP_ACCOUNTS:
                    fee_info['used_jito'] = True
                    lamports = info.get("lamports", 0)
                    fee_info['jito_tip'] = lamports / 1e9

    # Calculate priority fee
    if compute_unit_limit > 0 and compute_unit_price > 0:
        fee_info['compute_units'] = compute_unit_limit
        priority_fee_lamports = (compute_unit_price * compute_unit_limit) // 1_000_000
        fee_info['priority_fee'] = priority_fee_lamports / 1e9

    return fee_info


class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
    """Authentication plugin for Helius LaserStream"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, context, callback):
        metadata = (('x-token', self.api_key),)
        callback(metadata, None)


class AlphaPatternObserver:
    """
    Observes naturally occurring alpha buy patterns on Pump.fun
    Collects data on what triggers bot swarms - NO CAPITAL RISK
    """

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)

        if not GEYSER_AVAILABLE:
            raise RuntimeError("Geyser proto files not available")

        # Configuration
        self.grpc_endpoint = config.get('helius_grpc_endpoint')
        self.grpc_token = config.get('helius_grpc_token')
        self.rpc_url = config.get('rpc_url')  # For batch fee fetching

        if not self.grpc_endpoint or not self.grpc_token:
            raise ValueError("helius_grpc_endpoint and helius_grpc_token required")

        if not self.rpc_url:
            raise ValueError("rpc_url required for fee fetching")

        # Strip https:// prefix if present
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

        # Tracking state
        self.tokens: Dict[str, TokenTracker] = {}  # mint -> TokenTracker
        self.alpha_buys: List[AlphaBuyEvent] = []
        self.completed_alphas: List[AlphaBuyEvent] = []
        self.tracking_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self.total_buys_seen = 0
        self.large_buys_seen = 0
        self.early_snipes_seen = 0
        self.start_time = None

        # Alpha detection thresholds
        self.MIN_ALPHA_SIZE = 0.5  # 0.5 SOL minimum
        self.MAX_ALPHA_SIZE = 2.0  # 2.0 SOL maximum
        self.EARLY_SNIPE_WINDOW = 10  # seconds
        self.MAX_EARLY_POSITION = 5  # Consider positions 1-5

        self.logger.info("AlphaPatternObserver initialized")
        self.logger.info(f"  Endpoint: {self.grpc_endpoint}")
        self.logger.info(f"  Alpha size range: {self.MIN_ALPHA_SIZE}-{self.MAX_ALPHA_SIZE} SOL")

    async def start(self, run_hours: float):
        """Start observing alpha patterns"""
        if self.running:
            self.logger.warning("Observer already running")
            return

        self.running = True
        self.start_time = datetime.now()

        self.logger.info(f"\n{'='*80}")
        self.logger.info("ALPHA PATTERN OBSERVER STARTING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Run duration: {run_hours} hours")
        self.logger.info(f"Observing: Large buys (0.5-2 SOL) and their aftermath")
        self.logger.info(f"Hypothesis 1: Large buys on any-age tokens trigger swarms")
        self.logger.info(f"Hypothesis 2: Large early snipes (position 1-5) trigger swarms")
        self.logger.info(f"{'='*80}\n")

        # Start LaserStream monitoring
        monitor_task = asyncio.create_task(self._connect_and_stream())

        # Wait for specified time
        try:
            await asyncio.sleep(run_hours * 3600)
        except KeyboardInterrupt:
            self.logger.info("\n\n⚠️ Interrupted by user")

        # Stop monitoring
        self.logger.info("\n\n⏱️ Time limit reached, stopping...")
        await self.stop()

        # Wait for tracking tasks to complete
        if self.tracking_tasks:
            self.logger.info(f"Waiting for {len(self.tracking_tasks)} tracking tasks to complete...")
            await asyncio.gather(*self.tracking_tasks.values(), return_exceptions=True)

        # Cancel monitor task
        monitor_task.cancel()

        # Batch fetch fee data for all completed alphas
        await self._fetch_fee_data_batch()

        # Print final statistics
        self._print_statistics()

    async def _connect_and_stream(self):
        """Connect to LaserStream and observe Pump.fun transactions"""
        self.logger.info(f"Connecting to LaserStream at {self.grpc_endpoint}...")

        # Setup authentication
        auth = HeliusAuthMetadata(self.grpc_token)
        call_creds = grpc.metadata_call_credentials(auth)
        ssl_creds = grpc.ssl_channel_credentials()
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        # Create secure channel
        self.channel = grpc.aio.secure_channel(
            self.grpc_endpoint,
            credentials=combined_creds,
            options=[
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
            ]
        )

        self.stub = geyser_pb2_grpc.GeyserStub(self.channel)

        # Build subscription request
        subscription_request = self._build_subscription_request()

        self.logger.info("✅ LaserStream connection established")
        self.logger.info("📡 Observing ALL Pump.fun transactions...")

        try:
            async def request_generator():
                yield subscription_request
                while self.running:
                    await asyncio.sleep(30)

            self.logger.info("✅ LaserStream subscription active - observing alpha patterns\n")

            async for message in self.stub.Subscribe(request_generator()):
                if not self.running:
                    break

                try:
                    await self._process_message(message)
                except Exception as e:
                    self.logger.error(f"Message processing error: {e}")

        except asyncio.CancelledError:
            self.logger.info("LaserStream monitoring cancelled")
        except Exception as e:
            self.logger.error(f"LaserStream error: {e}")
        finally:
            if self.channel:
                await self.channel.close()

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """Build subscription request for ALL Pump.fun transactions"""
        commitment = geyser_pb2.CommitmentLevel.PROCESSED

        tx_filter = geyser_pb2.SubscribeRequestFilterTransactions(
            account_include=[PUMP_FUN_PROGRAM_ID],
            vote=False,
            failed=False,
        )

        request = geyser_pb2.SubscribeRequest(
            transactions={
                "pump_fun_all_txs": tx_filter
            },
            commitment=commitment
        )

        return request

    async def _process_message(self, message: geyser_pb2.SubscribeUpdate):
        """Process incoming LaserStream message"""
        if not message.HasField('transaction'):
            return

        tx_update = message.transaction
        transaction = tx_update.transaction

        # Extract signature and slot
        signature = base58.b58encode(bytes(transaction.signature)).decode('utf-8')
        slot = tx_update.slot

        # Extract logs
        logs = []
        if transaction.meta:
            if hasattr(transaction.meta, 'log_messages'):
                logs = list(transaction.meta.log_messages)
            elif hasattr(transaction.meta, 'logs'):
                logs = list(transaction.meta.logs)

        # Parse TradeEvent
        trade_event = parse_trade_event_from_logs(logs)
        if not trade_event:
            return

        # Only process BUY transactions
        if not trade_event['is_buy']:
            return

        self.total_buys_seen += 1

        # Extract data
        mint = trade_event['mint']
        wallet = trade_event['user']
        sol_amount = trade_event['sol_amount_ui']
        virtual_sol = trade_event['virtual_sol_reserves']
        virtual_token = trade_event['virtual_token_reserves']

        # Calculate price
        price_sol = 0.0
        if virtual_token > 0:
            price_sol = (virtual_sol / virtual_token) * 1e6  # SOL per million tokens

        # Get or create token tracker
        if mint not in self.tokens:
            self.tokens[mint] = TokenTracker(
                mint=mint,
                first_seen=datetime.now(),
                creation_slot=slot
            )

        tracker = self.tokens[mint]

        # Record this buy
        buy_record = {
            'slot': slot,
            'wallet': wallet,
            'sol_amount': sol_amount,
            'price': price_sol,
            'timestamp': datetime.now(),
            'position': len(tracker.buys) + 1
        }
        tracker.buys.append(buy_record)
        tracker.unique_wallets.add(wallet)
        tracker.total_volume += sol_amount

        # Calculate token age
        token_age = (datetime.now() - tracker.first_seen).total_seconds()

        # CHECK IF THIS IS A POTENTIAL ALPHA BUY
        is_alpha = self._is_alpha_buy(sol_amount, token_age, buy_record['position'])

        if is_alpha:
            # ALPHA BUY DETECTED!
            pattern_type = self._classify_pattern(token_age, buy_record['position'])

            alpha_event = AlphaBuyEvent(
                token=mint,
                buyer_wallet=wallet,
                buy_size_sol=sol_amount,
                buy_price=price_sol,
                buy_slot=slot,
                buy_timestamp=datetime.now(),
                buy_position=buy_record['position'],
                token_age_seconds=token_age,
                holder_count_before=len(tracker.unique_wallets) - 1,
                volume_before_sol=tracker.total_volume - sol_amount,
                pattern_type=pattern_type,
                signature=signature  # Store for batch fee lookup later
            )

            self.alpha_buys.append(alpha_event)

            if pattern_type == "early_snipe":
                self.early_snipes_seen += 1
            self.large_buys_seen += 1

            self.logger.info(f"\n🎯 ALPHA BUY DETECTED: {mint[:8]}...")
            self.logger.info(f"   Pattern: {pattern_type}")
            self.logger.info(f"   Buyer: {wallet[:8]}...")
            self.logger.info(f"   Size: {sol_amount:.3f} SOL")
            self.logger.info(f"   Position: #{buy_record['position']}")
            self.logger.info(f"   Token age: {token_age:.1f}s")
            self.logger.info(f"   Signature: {signature[:20]}...")
            self.logger.info(f"   ⏱️ Tracking for 60 seconds...")

            # Start tracking task for this alpha
            task_key = f"{mint}_{slot}"
            task = asyncio.create_task(self._track_alpha_aftermath(alpha_event, tracker))
            self.tracking_tasks[task_key] = task

    def _is_alpha_buy(self, sol_amount: float, token_age: float, position: int) -> bool:
        """Determine if this buy qualifies as a potential alpha trigger"""
        # Must be in our size range
        if sol_amount < self.MIN_ALPHA_SIZE or sol_amount > self.MAX_ALPHA_SIZE:
            return False

        # Either: early snipe (position 1-5, age < 10s)
        # Or: any position on any age token
        # We want to observe both patterns

        return True  # Track all large buys

    def _classify_pattern(self, token_age: float, position: int) -> str:
        """Classify what type of alpha pattern this is"""
        if token_age < self.EARLY_SNIPE_WINDOW and position <= self.MAX_EARLY_POSITION:
            return "early_snipe"  # Option 4: Large snipe at creation
        else:
            return "established_trigger"  # Option 1: Large buy on existing token

    async def _track_alpha_aftermath(self, alpha: AlphaBuyEvent, tracker: TokenTracker):
        """Track what happens in the 60 seconds after an alpha buy"""
        try:
            start_buys = len(tracker.buys)
            start_wallets = len(tracker.unique_wallets)
            start_volume = tracker.total_volume
            start_time = datetime.now()

            # Sample price every 5 seconds
            for i in range(12):  # 12 samples * 5s = 60s
                await asyncio.sleep(5)

                if not self.running:
                    break

                # Get current state
                current_price = self._get_current_price(tracker)
                elapsed = (datetime.now() - start_time).total_seconds()

                alpha.price_history.append({
                    'time': elapsed,
                    'price': current_price,
                    'buys': len(tracker.buys),
                    'volume': tracker.total_volume
                })

            # Calculate aftermath metrics
            buys_after = len(tracker.buys) - start_buys
            wallets_after = len(tracker.unique_wallets) - start_wallets
            volume_after = tracker.total_volume - start_volume

            alpha.unique_followers = wallets_after
            alpha.volume_added_sol = volume_after

            # Get followers (buys that came after the alpha)
            for buy in tracker.buys[start_buys:]:
                delay = (buy['timestamp'] - alpha.buy_timestamp).total_seconds()
                alpha.followers.append({
                    'delay_seconds': delay,
                    'wallet': buy['wallet'],
                    'size': buy['sol_amount'],
                    'position': buy['position']
                })

            # Analyze price action
            if alpha.price_history:
                prices = [p['price'] for p in alpha.price_history]
                alpha.peak_price = max(prices) if prices else 0

                if alpha.buy_price > 0:
                    alpha.peak_gain_pct = ((alpha.peak_price - alpha.buy_price) / alpha.buy_price) * 100

                    # Find when peak occurred
                    for record in alpha.price_history:
                        if record['price'] == alpha.peak_price:
                            alpha.peak_time_seconds = record['time']
                            break

                # Calculate profitable exit windows
                alpha.profitable_exit_windows = self._calculate_exit_windows(alpha)

            # Determine if this triggered a swarm
            alpha.did_trigger_swarm = self._analyze_if_triggered_swarm(alpha)
            alpha.swarm_confidence = self._calculate_swarm_confidence(alpha)

            # Log results
            self.logger.info(f"\n✅ TRACKING COMPLETE: {alpha.token[:8]}...")
            self.logger.info(f"   Pattern: {alpha.pattern_type}")
            self.logger.info(f"   Alpha size: {alpha.buy_size_sol:.3f} SOL")
            self.logger.info(f"   Followers: {alpha.unique_followers}")
            self.logger.info(f"   Volume added: {alpha.volume_added_sol:.3f} SOL")
            self.logger.info(f"   Peak gain: {alpha.peak_gain_pct:.1f}% at {alpha.peak_time_seconds:.1f}s")
            self.logger.info(f"   Triggered swarm: {'YES' if alpha.did_trigger_swarm else 'NO'} (confidence: {alpha.swarm_confidence:.2f})")

            # Move to completed
            self.completed_alphas.append(alpha)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error tracking alpha: {e}")

    def _get_current_price(self, tracker: TokenTracker) -> float:
        """Get most recent price from tracker"""
        if tracker.buys:
            return tracker.buys[-1]['price']
        return 0.0

    def _analyze_if_triggered_swarm(self, alpha: AlphaBuyEvent) -> bool:
        """Determine if this alpha buy triggered a bot swarm"""
        # Criteria for swarm:
        # - At least 5 unique followers
        # - At least 2x the alpha size in follow volume
        # - Most followers came within 30 seconds

        if alpha.unique_followers < 5:
            return False

        if alpha.volume_added_sol < (alpha.buy_size_sol * 2):
            return False

        # Check timing concentration
        early_followers = sum(1 for f in alpha.followers if f['delay_seconds'] <= 30)
        if early_followers < 4:
            return False

        return True

    def _calculate_swarm_confidence(self, alpha: AlphaBuyEvent) -> float:
        """Calculate confidence that WE triggered the swarm (not something else)"""
        confidence = 0.0

        # More followers = higher confidence
        if alpha.unique_followers >= 10:
            confidence += 0.4
        elif alpha.unique_followers >= 5:
            confidence += 0.2

        # High volume ratio = higher confidence
        if alpha.volume_added_sol >= (alpha.buy_size_sol * 4):
            confidence += 0.3
        elif alpha.volume_added_sol >= (alpha.buy_size_sol * 2):
            confidence += 0.15

        # Fast followers = higher confidence
        avg_delay = sum(f['delay_seconds'] for f in alpha.followers) / len(alpha.followers) if alpha.followers else 999
        if avg_delay < 15:
            confidence += 0.3
        elif avg_delay < 30:
            confidence += 0.15

        return min(confidence, 1.0)

    def _calculate_exit_windows(self, alpha: AlphaBuyEvent) -> List[Dict]:
        """Calculate when exits would have been profitable"""
        windows = []

        if not alpha.price_history or alpha.buy_price == 0:
            return windows

        # Test different exit timings
        exit_times = [10, 20, 30, 40, 60]

        for exit_time in exit_times:
            # Find closest price sample
            closest = min(alpha.price_history,
                         key=lambda x: abs(x['time'] - exit_time),
                         default=None)

            if closest:
                exit_price = closest['price']
                gain_pct = ((exit_price - alpha.buy_price) / alpha.buy_price) * 100

                windows.append({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'gain_pct': gain_pct,
                    'profitable': gain_pct > 5  # At least 5% to cover fees
                })

        return windows

    async def _fetch_fee_data_batch(self):
        """Fetch fee data in batch for all completed alphas"""
        if not self.completed_alphas:
            self.logger.info("\n⏭️ No completed alphas to fetch fees for")
            return

        # Collect signatures
        signatures = [alpha.signature for alpha in self.completed_alphas if alpha.signature]

        if not signatures:
            self.logger.info("\n⏭️ No signatures to fetch")
            return

        # Batch fetch fees
        fee_map = await batch_fetch_transaction_fees(signatures, self.rpc_url, self.logger)

        # Associate fees with alphas
        updated_count = 0
        for alpha in self.completed_alphas:
            if alpha.signature in fee_map:
                fee_data = fee_map[alpha.signature]
                alpha.total_fee_sol = fee_data['total_fee']
                alpha.priority_fee_sol = fee_data['priority_fee']
                alpha.compute_units = fee_data['compute_units']
                alpha.used_jito_bundle = fee_data['used_jito']
                alpha.jito_tip_sol = fee_data['jito_tip']
                updated_count += 1

        self.logger.info(f"✅ Updated fee data for {updated_count}/{len(self.completed_alphas)} alphas\n")

    def _print_statistics(self):
        """Print final statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600

        self.logger.info(f"\n\n{'='*80}")
        self.logger.info("ALPHA PATTERN OBSERVER - FINAL STATISTICS")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Runtime: {runtime:.2f} hours")
        self.logger.info(f"Total buys observed: {self.total_buys_seen:,}")
        self.logger.info(f"Large buys observed: {self.large_buys_seen}")
        self.logger.info(f"Early snipes observed: {self.early_snipes_seen}")
        self.logger.info(f"Completed alpha observations: {len(self.completed_alphas)}")

        if self.completed_alphas:
            self._analyze_patterns()

    def _analyze_patterns(self):
        """Analyze all observed alpha patterns"""
        alphas = self.completed_alphas

        self.logger.info(f"\n{'='*80}")
        self.logger.info("PATTERN ANALYSIS")
        self.logger.info(f"{'='*80}")

        # Split by pattern type
        early_snipes = [a for a in alphas if a.pattern_type == "early_snipe"]
        established = [a for a in alphas if a.pattern_type == "established_trigger"]

        self.logger.info(f"\nPattern Distribution:")
        self.logger.info(f"  Early snipes (0-10s, pos 1-5): {len(early_snipes)}")
        self.logger.info(f"  Established triggers (any age): {len(established)}")

        # Analyze each pattern type
        if early_snipes:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("EARLY SNIPE PATTERN (Option 4)")
            self.logger.info(f"{'='*80}")
            self._analyze_pattern_group(early_snipes)

        if established:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("ESTABLISHED TRIGGER PATTERN (Option 1)")
            self.logger.info(f"{'='*80}")
            self._analyze_pattern_group(established)

    def _analyze_pattern_group(self, alphas: List[AlphaBuyEvent]):
        """Analyze a specific group of alphas"""
        # Success rate
        successful = [a for a in alphas if a.did_trigger_swarm]
        success_rate = (len(successful) / len(alphas)) * 100 if alphas else 0

        self.logger.info(f"\nSuccess Rate: {len(successful)}/{len(alphas)} ({success_rate:.1f}%)")

        if not successful:
            self.logger.info("  ❌ No successful patterns observed")
            return

        # Average metrics
        avg_followers = sum(a.unique_followers for a in successful) / len(successful)
        avg_volume = sum(a.volume_added_sol for a in successful) / len(successful)
        avg_peak = sum(a.peak_gain_pct for a in successful) / len(successful)
        avg_peak_time = sum(a.peak_time_seconds for a in successful) / len(successful)
        avg_confidence = sum(a.swarm_confidence for a in successful) / len(successful)

        self.logger.info(f"\nSuccessful Pattern Averages:")
        self.logger.info(f"  Followers: {avg_followers:.1f}")
        self.logger.info(f"  Volume added: {avg_volume:.3f} SOL")
        self.logger.info(f"  Peak gain: {avg_peak:.1f}%")
        self.logger.info(f"  Peak timing: {avg_peak_time:.1f}s")
        self.logger.info(f"  Confidence: {avg_confidence:.2f}")

        # Fee analysis
        self._analyze_fees(alphas, successful)

        # Position size analysis
        self._analyze_position_sizes(successful)

        # Token age analysis
        self._analyze_token_ages(successful)

        # Exit timing analysis
        self._analyze_exit_timings(successful)

    def _analyze_fees(self, all_alphas: List[AlphaBuyEvent], successful: List[AlphaBuyEvent]):
        """Analyze fee patterns in successful vs unsuccessful alphas"""
        self.logger.info(f"\nFee Analysis:")

        # Compare successful vs unsuccessful
        unsuccessful = [a for a in all_alphas if not a.did_trigger_swarm]

        if successful:
            avg_total_fee_success = sum(a.total_fee_sol for a in successful) / len(successful)
            avg_priority_fee_success = sum(a.priority_fee_sol for a in successful) / len(successful)
            jito_count_success = sum(1 for a in successful if a.used_jito_bundle)
            jito_pct_success = (jito_count_success / len(successful)) * 100

            self.logger.info(f"  Successful alphas:")
            self.logger.info(f"    Avg total fee: {avg_total_fee_success:.6f} SOL")
            self.logger.info(f"    Avg priority fee: {avg_priority_fee_success:.6f} SOL")
            self.logger.info(f"    Jito usage: {jito_count_success}/{len(successful)} ({jito_pct_success:.1f}%)")

            if jito_count_success > 0:
                jito_alphas = [a for a in successful if a.used_jito_bundle]
                avg_jito_tip = sum(a.jito_tip_sol for a in jito_alphas) / len(jito_alphas)
                self.logger.info(f"    Avg Jito tip: {avg_jito_tip:.6f} SOL")

        if unsuccessful:
            avg_total_fee_fail = sum(a.total_fee_sol for a in unsuccessful) / len(unsuccessful)
            avg_priority_fee_fail = sum(a.priority_fee_sol for a in unsuccessful) / len(unsuccessful)
            jito_count_fail = sum(1 for a in unsuccessful if a.used_jito_bundle)
            jito_pct_fail = (jito_count_fail / len(unsuccessful)) * 100

            self.logger.info(f"  Unsuccessful alphas:")
            self.logger.info(f"    Avg total fee: {avg_total_fee_fail:.6f} SOL")
            self.logger.info(f"    Avg priority fee: {avg_priority_fee_fail:.6f} SOL")
            self.logger.info(f"    Jito usage: {jito_count_fail}/{len(unsuccessful)} ({jito_pct_fail:.1f}%)")

        # Fee range analysis
        if successful:
            self.logger.info(f"\n  Fee Ranges (Successful alphas):")

            # Group by priority fee ranges
            no_priority = [a for a in successful if a.priority_fee_sol == 0]
            low_priority = [a for a in successful if 0 < a.priority_fee_sol < 0.001]
            med_priority = [a for a in successful if 0.001 <= a.priority_fee_sol < 0.01]
            high_priority = [a for a in successful if a.priority_fee_sol >= 0.01]

            for name, group in [("No priority fee", no_priority),
                                ("Low (< 0.001 SOL)", low_priority),
                                ("Medium (0.001-0.01 SOL)", med_priority),
                                ("High (>= 0.01 SOL)", high_priority)]:
                if group:
                    avg_peak = sum(a.peak_gain_pct for a in group) / len(group)
                    self.logger.info(f"    {name}: {len(group)} samples, {avg_peak:.1f}% avg peak")

    def _analyze_position_sizes(self, alphas: List[AlphaBuyEvent]):
        """Analyze which position sizes work best"""
        # Group by size
        small = [a for a in alphas if a.buy_size_sol < 0.75]
        medium = [a for a in alphas if 0.75 <= a.buy_size_sol < 1.25]
        large = [a for a in alphas if a.buy_size_sol >= 1.25]

        self.logger.info(f"\nPosition Size Analysis:")

        for name, group in [("Small (0.5-0.75)", small),
                            ("Medium (0.75-1.25)", medium),
                            ("Large (1.25-2.0)", large)]:
            if group:
                avg_peak = sum(a.peak_gain_pct for a in group) / len(group)
                avg_followers = sum(a.unique_followers for a in group) / len(group)
                self.logger.info(f"  {name}: {len(group)} samples, {avg_peak:.1f}% peak, {avg_followers:.1f} followers")

    def _analyze_token_ages(self, alphas: List[AlphaBuyEvent]):
        """Analyze which token ages work best"""
        # Group by age
        very_new = [a for a in alphas if a.token_age_seconds < 30]
        new = [a for a in alphas if 30 <= a.token_age_seconds < 300]
        established = [a for a in alphas if 300 <= a.token_age_seconds < 1800]
        old = [a for a in alphas if a.token_age_seconds >= 1800]

        self.logger.info(f"\nToken Age Analysis:")

        for name, group in [("Very new (0-30s)", very_new),
                            ("New (30s-5min)", new),
                            ("Established (5-30min)", established),
                            ("Old (30min+)", old)]:
            if group:
                avg_peak = sum(a.peak_gain_pct for a in group) / len(group)
                success_rate = (len([a for a in group if a.did_trigger_swarm]) / len(group)) * 100
                self.logger.info(f"  {name}: {len(group)} samples, {success_rate:.1f}% success, {avg_peak:.1f}% avg peak")

    def _analyze_exit_timings(self, alphas: List[AlphaBuyEvent]):
        """Analyze optimal exit timings"""
        self.logger.info(f"\nExit Timing Analysis:")

        exit_times = [10, 20, 30, 40, 60]

        for exit_time in exit_times:
            gains = []
            profitable_count = 0

            for alpha in alphas:
                for window in alpha.profitable_exit_windows:
                    if window['exit_time'] == exit_time:
                        gains.append(window['gain_pct'])
                        if window['profitable']:
                            profitable_count += 1

            if gains:
                avg_gain = sum(gains) / len(gains)
                win_rate = (profitable_count / len(gains)) * 100
                self.logger.info(f"  Exit at {exit_time}s: {avg_gain:.1f}% avg gain, {win_rate:.1f}% win rate")

    def save_results(self, output_dir: Path):
        """Save observation results to JSON"""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"alpha_pattern_analysis_{timestamp}.json"

        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_buys_observed': self.total_buys_seen,
                'large_buys_observed': self.large_buys_seen,
                'early_snipes_observed': self.early_snipes_seen,
                'completed_observations': len(self.completed_alphas),
            },
            'observations': [self._alpha_to_dict(a) for a in self.completed_alphas]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"\n✅ Results saved to: {output_file}")
        self.logger.info(f"   Observations analyzed: {len(self.completed_alphas)}")

    def _alpha_to_dict(self, alpha: AlphaBuyEvent) -> Dict:
        """Convert AlphaBuyEvent to dict"""
        return {
            'token': alpha.token,
            'signature': alpha.signature,
            'buyer_wallet': alpha.buyer_wallet[:8] + '...',
            'buy_size_sol': alpha.buy_size_sol,
            'buy_position': alpha.buy_position,
            'token_age_seconds': alpha.token_age_seconds,
            'pattern_type': alpha.pattern_type,
            'holder_count_before': alpha.holder_count_before,
            'volume_before_sol': alpha.volume_before_sol,
            # Fee data
            'total_fee_sol': alpha.total_fee_sol,
            'priority_fee_sol': alpha.priority_fee_sol,
            'compute_units': alpha.compute_units,
            'used_jito_bundle': alpha.used_jito_bundle,
            'jito_tip_sol': alpha.jito_tip_sol,
            # Outcome data
            'unique_followers': alpha.unique_followers,
            'volume_added_sol': alpha.volume_added_sol,
            'peak_gain_pct': alpha.peak_gain_pct,
            'peak_time_seconds': alpha.peak_time_seconds,
            'did_trigger_swarm': alpha.did_trigger_swarm,
            'swarm_confidence': alpha.swarm_confidence,
            'profitable_exit_windows': alpha.profitable_exit_windows,
            'price_history': alpha.price_history,
            'followers': [
                {
                    'delay_seconds': f['delay_seconds'],
                    'size': f['size'],
                    'position': f['position']
                }
                for f in alpha.followers
            ]
        }

    async def stop(self):
        """Stop observer"""
        self.running = False

        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Alpha Pattern Observer")
    parser.add_argument('--hours', type=float, default=6.0, help='How many hours to observe (default: 6)')
    parser.add_argument('--output', type=str, default='data/alpha_pattern_analysis', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Config file to use')

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO", "logs/alpha_pattern_observer.log")
    logger = logging.getLogger(__name__)

    # Load config
    config = None
    config_file = args.config

    if config_file:
        try:
            config = load_config(config_file)
            logger.info(f"Loaded config from: {config_file}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return
    else:
        # Auto-detect: try config_fast.yml first, then config.yml
        for config_name in ['config_fast.yml', 'config.yml']:
            try:
                config = load_config(config_name)
                logger.info(f"Loaded config from: {config_name}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error loading {config_name}: {e}")
                continue

        if not config:
            logger.error("Could not find config.yml or config_fast.yml")
            return

    # Get LaserStream config
    fast_config = config.get('fast_execution', {})
    observer_config = {
        'helius_grpc_endpoint': fast_config.get('helius_grpc_endpoint'),
        'helius_grpc_token': fast_config.get('helius_grpc_token'),
        'rpc_url': fast_config.get('helius_rpc_endpoint'),  # For batch fee fetching
    }

    if not observer_config['helius_grpc_endpoint'] or not observer_config['helius_grpc_token']:
        logger.error("Missing helius_grpc_endpoint or helius_grpc_token in config")
        return

    if not observer_config['rpc_url']:
        logger.error("Missing helius_rpc_endpoint in config for fee fetching")
        return

    # Create observer
    try:
        observer = AlphaPatternObserver(observer_config)
    except Exception as e:
        logger.error(f"Failed to initialize observer: {e}")
        return

    # Run observation
    try:
        await observer.start(run_hours=args.hours)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user")
        await observer.stop()
    except Exception as e:
        logger.error(f"Error during observation: {e}")
        import traceback
        traceback.print_exc()

    # Save results
    try:
        output_dir = Path(args.output)
        observer.save_results(output_dir)

        logger.info(f"\n{'='*80}")
        logger.info("OBSERVATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\nData collected with ZERO CAPITAL RISK")
        logger.info(f"Now you can implement the winning pattern with confidence!")
        logger.info(f"\n{'='*80}\n")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
