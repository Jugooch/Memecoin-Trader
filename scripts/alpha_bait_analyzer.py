#!/usr/bin/env python3
"""
Alpha Bait Strategy Analyzer - Data Collection Script

Monitors ALL Pump.fun token launches via LaserStream for hours to collect data on:
1. WHEN to buy - what timeframe attracts most copy trading bots
2. HOW MUCH to buy - what position size attracts most attention
3. WHAT coins to buy - what token characteristics attract bot swarms

This script tracks each new token for 2 minutes after first significant buy,
collecting comprehensive data for pattern analysis.

Usage:
    python scripts/alpha_bait_analyzer.py --hours 3 --min-buy 0.3

Arguments:
    --hours: How many hours to run (default: 3)
    --min-buy: Minimum SOL for "significant" first buy (default: 0.3)
    --output: Output directory (default: data/alpha_bait_analysis)
"""

import asyncio
import json
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict
import grpc
import base58
import base64
import struct

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
    print("❌ Geyser proto files not available - cannot use LaserStream")
    sys.exit(1)

from src.utils.logger_setup import setup_logging
from src.utils.config_loader import load_config


# Pump.fun program ID
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# TradeEvent discriminator
TRADE_EVENT_DISCRIMINATOR = bytes([189, 219, 127, 211, 78, 230, 97, 238])
EVENT_DISCRIMINATOR_SIZE = 8


@dataclass
class BuyEvent:
    """Single buy transaction"""
    time: float  # Seconds since token tracking started
    timestamp: datetime
    wallet: str
    sol_amount: float
    token_amount: float
    signature: str
    slot: int
    virtual_sol_reserves: int
    virtual_token_reserves: int
    price_usd: float = 0.0


@dataclass
class TokenAnalysis:
    """Complete analysis data for one token"""
    mint: str
    first_detected: datetime
    first_significant_buy: Optional[BuyEvent] = None

    # Track all buys for 2 minutes
    buys: List[BuyEvent] = field(default_factory=list)
    unique_wallets: Set[str] = field(default_factory=set)

    # Statistics (calculated after 2 minutes)
    tracking_complete: bool = False
    total_buys: int = 0
    total_volume_sol: float = 0.0
    peak_price_usd: float = 0.0
    peak_time_seconds: float = 0.0
    peak_gain_pct: float = 0.0
    bot_swarm_detected: bool = False
    bot_swarm_size: int = 0

    # Timing analysis
    buys_0_10s: int = 0  # Buys in first 10 seconds
    buys_10_30s: int = 0  # Buys in 10-30 seconds
    buys_30_60s: int = 0  # Buys in 30-60 seconds
    buys_60_120s: int = 0  # Buys in 60-120 seconds

    # Position size analysis
    small_buys: int = 0  # < 0.1 SOL
    medium_buys: int = 0  # 0.1-0.5 SOL
    large_buys: int = 0  # 0.5-2 SOL
    whale_buys: int = 0  # > 2 SOL

    # Developer analysis
    dev_wallet: Optional[str] = None
    dev_previous_tokens: int = 0  # TODO: Research this

    # Atomic dev buy detection
    first_buy_slot: Optional[int] = None  # Slot of very first buy (any size)
    first_significant_buy_slot: Optional[int] = None  # Slot of first significant buy
    is_atomic_dev_buy: bool = False  # True if first significant buy is in same slot as first buy
    seconds_to_significant_buy: float = 0.0  # Time from first buy to first significant buy

    # Detection latency tracking
    detected_at: Optional[datetime] = None  # When WE first detected this token
    first_buy_at: Optional[datetime] = None  # Timestamp of first buy we saw
    detection_delay_seconds: float = 0.0  # How long after first buy did we detect it
    slots_behind_first_buy: int = 0  # How many slots behind first buy were we

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        result = {
            'mint': self.mint,
            'first_detected': self.first_detected.isoformat(),
            'tracking_complete': self.tracking_complete,
            'total_buys': self.total_buys,
            'unique_wallets': len(self.unique_wallets),
            'total_volume_sol': round(self.total_volume_sol, 3),
            'peak_price_usd': self.peak_price_usd,
            'peak_time_seconds': self.peak_time_seconds,
            'peak_gain_pct': self.peak_gain_pct,
            'bot_swarm_detected': self.bot_swarm_detected,
            'bot_swarm_size': self.bot_swarm_size,
            'timing_analysis': {
                'buys_0_10s': self.buys_0_10s,
                'buys_10_30s': self.buys_10_30s,
                'buys_30_60s': self.buys_30_60s,
                'buys_60_120s': self.buys_60_120s,
            },
            'position_size_analysis': {
                'small_buys': self.small_buys,
                'medium_buys': self.medium_buys,
                'large_buys': self.large_buys,
                'whale_buys': self.whale_buys,
            },
            'dev_wallet': self.dev_wallet,
            'dev_previous_tokens': self.dev_previous_tokens,
            'atomic_dev_buy_analysis': {
                'is_atomic_dev_buy': self.is_atomic_dev_buy,
                'first_buy_slot': self.first_buy_slot,
                'first_significant_buy_slot': self.first_significant_buy_slot,
                'seconds_to_significant_buy': round(self.seconds_to_significant_buy, 2),
            },
            'detection_latency_analysis': {
                'detected_at': self.detected_at.isoformat() if self.detected_at else None,
                'first_buy_at': self.first_buy_at.isoformat() if self.first_buy_at else None,
                'detection_delay_seconds': round(self.detection_delay_seconds, 3),
                'slots_behind_first_buy': self.slots_behind_first_buy,
                'can_beat_first_sniper': self.slots_behind_first_buy <= 1,  # Within 1 slot = maybe
            },
        }

        # Add first significant buy details
        if self.first_significant_buy:
            result['first_significant_buy'] = {
                'time_seconds': self.first_significant_buy.time,
                'wallet': self.first_significant_buy.wallet[:8] + '...',
                'sol_amount': self.first_significant_buy.sol_amount,
                'timestamp': self.first_significant_buy.timestamp.isoformat(),
            }

        # Add buy history
        result['buy_history'] = [
            {
                'time': b.time,
                'wallet': b.wallet[:8] + '...',
                'sol': b.sol_amount,
                'tokens': b.token_amount,
                'price_usd': b.price_usd,
            }
            for b in self.buys
        ]

        return result


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


class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
    """Authentication plugin for Helius LaserStream"""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def __call__(self, context, callback):
        metadata = (('x-token', self.api_key),)
        callback(metadata, None)


class AlphaBaitAnalyzer:
    """
    Analyzes Pump.fun token launches to determine optimal alpha bait strategy
    """

    def __init__(self, config: Dict, min_significant_buy: float = 0.3):
        """
        Initialize analyzer

        Args:
            config: Configuration dict
            min_significant_buy: Minimum SOL for "significant" first buy
        """
        self.logger = logging.getLogger(__name__)

        if not GEYSER_AVAILABLE:
            raise RuntimeError("Geyser proto files not available")

        # Configuration
        self.grpc_endpoint = config.get('helius_grpc_endpoint')
        self.grpc_token = config.get('helius_grpc_token')
        self.min_significant_buy = min_significant_buy

        if not self.grpc_endpoint or not self.grpc_token:
            raise ValueError("helius_grpc_endpoint and helius_grpc_token required")

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
        self.tracked_tokens: Dict[str, TokenAnalysis] = {}  # mint -> TokenAnalysis
        self.completed_tokens: List[TokenAnalysis] = []
        self.tracking_tasks: Dict[str, asyncio.Task] = {}  # mint -> tracking task

        # Statistics
        self.total_buys_seen = 0
        self.total_tokens_tracked = 0
        self.start_time = None

        self.logger.info("AlphaBaitAnalyzer initialized")
        self.logger.info(f"  Endpoint: {self.grpc_endpoint}")
        self.logger.info(f"  Min significant buy: {min_significant_buy} SOL")

    async def start(self, run_hours: float):
        """
        Start analysis for specified hours

        Args:
            run_hours: How many hours to run
        """
        if self.running:
            self.logger.warning("Analyzer already running")
            return

        self.running = True
        self.start_time = datetime.now()

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"ALPHA BAIT ANALYZER STARTING")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Run duration: {run_hours} hours")
        self.logger.info(f"Minimum significant buy: {self.min_significant_buy} SOL")
        self.logger.info(f"Tracking window: 120 seconds per token")
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

        # Wait for all tracking tasks to complete
        if self.tracking_tasks:
            self.logger.info(f"Waiting for {len(self.tracking_tasks)} tracking tasks to complete...")
            await asyncio.gather(*self.tracking_tasks.values(), return_exceptions=True)

        # Cancel monitor task
        monitor_task.cancel()

        # Print final statistics
        self._print_statistics()

    async def _connect_and_stream(self):
        """Connect to LaserStream and process ALL Pump.fun transactions"""
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

        # Build subscription request (subscribe to ALL Pump.fun transactions)
        subscription_request = self._build_subscription_request()

        self.logger.info("✅ LaserStream connection established")
        self.logger.info("📡 Subscribing to ALL Pump.fun transactions...")

        try:
            async def request_generator():
                yield subscription_request
                while self.running:
                    await asyncio.sleep(30)

            self.logger.info("✅ LaserStream subscription active - monitoring all tokens\n")

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
        token_amount = trade_event['token_amount_ui']
        virtual_sol = trade_event['virtual_sol_reserves']
        virtual_token = trade_event['virtual_token_reserves']

        # Calculate price (SOL per million tokens)
        price_usd = 0.0
        if virtual_token > 0:
            sol_per_token = virtual_sol / virtual_token
            price_usd = sol_per_token * 1e6 * 150  # Assuming $150/SOL

        # Check if this is a new token OR already being tracked
        if mint not in self.tracked_tokens and mint not in [t.mint for t in self.completed_tokens]:
            # New token detected!
            now = datetime.now()
            analysis = TokenAnalysis(
                mint=mint,
                first_detected=now,
            )
            self.tracked_tokens[mint] = analysis

            # Record the FIRST buy slot (any size) for atomic dev buy detection
            analysis.first_buy_slot = slot

            # Record detection latency
            analysis.detected_at = now
            # Convert Unix timestamp to datetime for comparison
            on_chain_time = datetime.fromtimestamp(trade_event['timestamp'])
            analysis.first_buy_at = on_chain_time
            analysis.detection_delay_seconds = (now - on_chain_time).total_seconds()

            # Check if this is a "significant" buy to start tracking
            if sol_amount >= self.min_significant_buy:
                self.logger.info(f"\n🆕 NEW TOKEN DETECTED: {mint[:8]}...")
                self.logger.info(f"   First significant buy: {sol_amount:.3f} SOL from {wallet[:8]}...")
                self.logger.info(f"   ⏱️ Detection latency: {analysis.detection_delay_seconds:.3f}s")

                # Record first significant buy
                buy_event = BuyEvent(
                    time=0.0,
                    timestamp=datetime.now(),
                    wallet=wallet,
                    sol_amount=sol_amount,
                    token_amount=token_amount,
                    signature=signature,
                    slot=slot,
                    virtual_sol_reserves=virtual_sol,
                    virtual_token_reserves=virtual_token,
                    price_usd=price_usd,
                )

                analysis.first_significant_buy = buy_event
                analysis.first_significant_buy_slot = slot
                analysis.buys.append(buy_event)
                analysis.unique_wallets.add(wallet)
                analysis.dev_wallet = wallet  # Assume first buyer is dev

                # Check if this is an atomic dev buy (first buy IS the significant buy)
                if analysis.first_buy_slot == analysis.first_significant_buy_slot:
                    analysis.is_atomic_dev_buy = True
                    analysis.seconds_to_significant_buy = 0.0
                    self.logger.info(f"   🔥 ATOMIC DEV BUY detected (slot {slot})")
                else:
                    analysis.is_atomic_dev_buy = False
                    self.logger.info(f"   ⏰ Non-atomic buy (first slot: {analysis.first_buy_slot}, significant slot: {slot})")

                # Start tracking task
                task = asyncio.create_task(self._track_token(mint))
                self.tracking_tasks[mint] = task

                self.total_tokens_tracked += 1
                self.logger.info(f"   ⏱️ Tracking for 120 seconds... (Token #{self.total_tokens_tracked})\n")
            else:
                # Small buy detected, wait for significant buy
                # This allows us to detect when significant buy comes later (non-atomic)
                pass

        # If token is being tracked, record this buy
        elif mint in self.tracked_tokens:
            analysis = self.tracked_tokens[mint]

            # Check if we haven't started tracking yet (waiting for significant buy)
            if not analysis.first_significant_buy and sol_amount >= self.min_significant_buy:
                # This is the first significant buy, but token had smaller buys before
                time_since_first = (datetime.now() - analysis.first_detected).total_seconds()

                self.logger.info(f"\n🆕 DELAYED SIGNIFICANT BUY: {mint[:8]}...")
                self.logger.info(f"   {time_since_first:.1f}s after first buy: {sol_amount:.3f} SOL from {wallet[:8]}...")

                # Record first significant buy
                buy_event = BuyEvent(
                    time=0.0,  # Reset timer from this point
                    timestamp=datetime.now(),
                    wallet=wallet,
                    sol_amount=sol_amount,
                    token_amount=token_amount,
                    signature=signature,
                    slot=slot,
                    virtual_sol_reserves=virtual_sol,
                    virtual_token_reserves=virtual_token,
                    price_usd=price_usd,
                )

                analysis.first_significant_buy = buy_event
                analysis.first_significant_buy_slot = slot
                analysis.first_detected = datetime.now()  # Reset tracking start time
                analysis.buys.append(buy_event)
                analysis.unique_wallets.add(wallet)

                # Check if atomic (same slot as first buy)
                if analysis.first_buy_slot == analysis.first_significant_buy_slot:
                    analysis.is_atomic_dev_buy = True
                    analysis.seconds_to_significant_buy = 0.0
                    self.logger.info(f"   🔥 ATOMIC DEV BUY (slot {slot})")
                else:
                    analysis.is_atomic_dev_buy = False
                    analysis.seconds_to_significant_buy = time_since_first
                    self.logger.info(f"   ⏰ Non-atomic: {time_since_first:.1f}s after first buy (slots: {analysis.first_buy_slot} → {slot})")

                # Now start tracking
                task = asyncio.create_task(self._track_token(mint))
                self.tracking_tasks[mint] = task

                self.total_tokens_tracked += 1
                self.logger.info(f"   ⏱️ Tracking for 120 seconds... (Token #{self.total_tokens_tracked})\n")
                return

            # Don't record buys if tracking is complete
            if analysis.tracking_complete:
                return

            # If we're not tracking yet (no significant buy), don't record
            if not analysis.first_significant_buy:
                return

            # Calculate time since tracking started
            time_since_start = (datetime.now() - analysis.first_detected).total_seconds()

            # Only track for 120 seconds
            if time_since_start > 120:
                return

            # Record buy
            buy_event = BuyEvent(
                time=time_since_start,
                timestamp=datetime.now(),
                wallet=wallet,
                sol_amount=sol_amount,
                token_amount=token_amount,
                signature=signature,
                slot=slot,
                virtual_sol_reserves=virtual_sol,
                virtual_token_reserves=virtual_token,
                price_usd=price_usd,
            )

            analysis.buys.append(buy_event)
            analysis.unique_wallets.add(wallet)

    async def _track_token(self, mint: str):
        """
        Track a token for 120 seconds, then analyze results

        Args:
            mint: Token mint address
        """
        try:
            # Wait 120 seconds
            await asyncio.sleep(120)

            # Get analysis
            analysis = self.tracked_tokens.get(mint)
            if not analysis:
                return

            # Mark tracking complete
            analysis.tracking_complete = True

            # Calculate statistics
            self._calculate_statistics(analysis)

            # Log results
            self.logger.info(f"\n✅ TRACKING COMPLETE: {mint[:8]}...")
            self.logger.info(f"   Total buys: {analysis.total_buys}")
            self.logger.info(f"   Unique wallets: {len(analysis.unique_wallets)}")
            self.logger.info(f"   Total volume: {analysis.total_volume_sol:.3f} SOL")
            self.logger.info(f"   Peak gain: {analysis.peak_gain_pct:.1f}% at {analysis.peak_time_seconds:.1f}s")
            self.logger.info(f"   Bot swarm: {'YES' if analysis.bot_swarm_detected else 'NO'} ({analysis.bot_swarm_size} bots)")
            self.logger.info(f"   Atomic dev buy: {'YES' if analysis.is_atomic_dev_buy else f'NO ({analysis.seconds_to_significant_buy:.1f}s delay)'}")
            self.logger.info(f"   Timing: {analysis.buys_0_10s} (0-10s), {analysis.buys_10_30s} (10-30s), {analysis.buys_30_60s} (30-60s), {analysis.buys_60_120s} (60-120s)")

            # Move to completed
            self.completed_tokens.append(analysis)
            del self.tracked_tokens[mint]
            del self.tracking_tasks[mint]

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error tracking token {mint[:8]}: {e}")

    def _calculate_statistics(self, analysis: TokenAnalysis):
        """Calculate statistics for a tracked token"""
        if not analysis.buys:
            return

        # Basic counts
        analysis.total_buys = len(analysis.buys)
        analysis.total_volume_sol = sum(b.sol_amount for b in analysis.buys)

        # Peak price
        if analysis.first_significant_buy:
            first_price = analysis.first_significant_buy.price_usd
            peak_price = max(b.price_usd for b in analysis.buys)

            analysis.peak_price_usd = peak_price

            if first_price > 0:
                analysis.peak_gain_pct = ((peak_price - first_price) / first_price) * 100

            # Find when peak occurred
            for buy in analysis.buys:
                if buy.price_usd == peak_price:
                    analysis.peak_time_seconds = buy.time
                    break

        # Timing analysis
        for buy in analysis.buys:
            if buy.time <= 10:
                analysis.buys_0_10s += 1
            elif buy.time <= 30:
                analysis.buys_10_30s += 1
            elif buy.time <= 60:
                analysis.buys_30_60s += 1
            else:
                analysis.buys_60_120s += 1

        # Position size analysis
        for buy in analysis.buys:
            if buy.sol_amount < 0.1:
                analysis.small_buys += 1
            elif buy.sol_amount < 0.5:
                analysis.medium_buys += 1
            elif buy.sol_amount < 2.0:
                analysis.large_buys += 1
            else:
                analysis.whale_buys += 1

        # Bot swarm detection (10+ unique wallets in 60 seconds)
        early_wallets = {b.wallet for b in analysis.buys if b.time <= 60}
        if len(early_wallets) >= 10:
            analysis.bot_swarm_detected = True
            analysis.bot_swarm_size = len(early_wallets)

    def _print_statistics(self):
        """Print final statistics"""
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600

        self.logger.info(f"\n\n{'='*80}")
        self.logger.info("ALPHA BAIT ANALYZER - FINAL STATISTICS")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Runtime: {runtime:.2f} hours")
        self.logger.info(f"Total buys seen: {self.total_buys_seen:,}")
        self.logger.info(f"Total tokens tracked: {self.total_tokens_tracked}")
        self.logger.info(f"Completed tokens: {len(self.completed_tokens)}")
        self.logger.info(f"Still tracking: {len(self.tracked_tokens)}")

        # Analyze completed tokens
        if self.completed_tokens:
            self.logger.info(f"\n{'='*80}")
            self.logger.info("AGGREGATE ANALYSIS")
            self.logger.info(f"{'='*80}")

            # Bot swarm frequency
            swarm_tokens = [t for t in self.completed_tokens if t.bot_swarm_detected]
            swarm_pct = (len(swarm_tokens) / len(self.completed_tokens)) * 100

            self.logger.info(f"\nBot Swarm Frequency: {len(swarm_tokens)}/{len(self.completed_tokens)} ({swarm_pct:.1f}%)")

            # Atomic dev buy frequency
            atomic_tokens = [t for t in self.completed_tokens if t.is_atomic_dev_buy]
            atomic_pct = (len(atomic_tokens) / len(self.completed_tokens)) * 100

            self.logger.info(f"Atomic Dev Buys: {len(atomic_tokens)}/{len(self.completed_tokens)} ({atomic_pct:.1f}%)")

            # Detection latency analysis
            detection_latencies = [t.detection_delay_seconds for t in self.completed_tokens if t.detection_delay_seconds > 0]
            if detection_latencies:
                avg_latency = sum(detection_latencies) / len(detection_latencies)
                min_latency = min(detection_latencies)
                max_latency = max(detection_latencies)

                self.logger.info(f"\nDetection Latency:")
                self.logger.info(f"  Average: {avg_latency:.3f}s")
                self.logger.info(f"  Best:    {min_latency:.3f}s")
                self.logger.info(f"  Worst:   {max_latency:.3f}s")

                # Estimate if we can beat snipers
                if avg_latency < 0.5:
                    self.logger.info(f"  ✅ Very fast - might beat some snipers!")
                elif avg_latency < 1.0:
                    self.logger.info(f"  ⚠️ Fast - can be early but not first")
                elif avg_latency < 2.0:
                    self.logger.info(f"  ⚠️ Moderate - will be in early swarm")
                else:
                    self.logger.info(f"  ❌ Slow - will miss initial snipers")

            # Average statistics
            avg_buys = sum(t.total_buys for t in self.completed_tokens) / len(self.completed_tokens)
            avg_volume = sum(t.total_volume_sol for t in self.completed_tokens) / len(self.completed_tokens)
            avg_peak = sum(t.peak_gain_pct for t in self.completed_tokens) / len(self.completed_tokens)

            self.logger.info(f"\nAverage Statistics:")
            self.logger.info(f"  Buys per token: {avg_buys:.1f}")
            self.logger.info(f"  Volume per token: {avg_volume:.3f} SOL")
            self.logger.info(f"  Peak gain: {avg_peak:.1f}%")

            # Timing breakdown
            total_buys_0_10s = sum(t.buys_0_10s for t in self.completed_tokens)
            total_buys_10_30s = sum(t.buys_10_30s for t in self.completed_tokens)
            total_buys_30_60s = sum(t.buys_30_60s for t in self.completed_tokens)
            total_buys_60_120s = sum(t.buys_60_120s for t in self.completed_tokens)
            total_buys = total_buys_0_10s + total_buys_10_30s + total_buys_30_60s + total_buys_60_120s

            if total_buys > 0:
                self.logger.info(f"\nTiming Breakdown:")
                self.logger.info(f"  0-10s:   {total_buys_0_10s:4d} ({total_buys_0_10s/total_buys*100:.1f}%)")
                self.logger.info(f"  10-30s:  {total_buys_10_30s:4d} ({total_buys_10_30s/total_buys*100:.1f}%)")
                self.logger.info(f"  30-60s:  {total_buys_30_60s:4d} ({total_buys_30_60s/total_buys*100:.1f}%)")
                self.logger.info(f"  60-120s: {total_buys_60_120s:4d} ({total_buys_60_120s/total_buys*100:.1f}%)")

            # Position size breakdown
            total_small = sum(t.small_buys for t in self.completed_tokens)
            total_medium = sum(t.medium_buys for t in self.completed_tokens)
            total_large = sum(t.large_buys for t in self.completed_tokens)
            total_whale = sum(t.whale_buys for t in self.completed_tokens)
            total_buys = total_small + total_medium + total_large + total_whale

            if total_buys > 0:
                self.logger.info(f"\nPosition Size Breakdown:")
                self.logger.info(f"  Small (<0.1):    {total_small:4d} ({total_small/total_buys*100:.1f}%)")
                self.logger.info(f"  Medium (0.1-0.5): {total_medium:4d} ({total_medium/total_buys*100:.1f}%)")
                self.logger.info(f"  Large (0.5-2):   {total_large:4d} ({total_large/total_buys*100:.1f}%)")
                self.logger.info(f"  Whale (>2):      {total_whale:4d} ({total_whale/total_buys*100:.1f}%)")

        self.logger.info(f"\n{'='*80}\n")

    def save_results(self, output_dir: Path):
        """
        Save analysis results to JSON

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save completed tokens
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"alpha_bait_analysis_{timestamp}.json"

        data = {
            'metadata': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'total_buys_seen': self.total_buys_seen,
                'total_tokens_tracked': self.total_tokens_tracked,
                'completed_tokens': len(self.completed_tokens),
                'min_significant_buy': self.min_significant_buy,
            },
            'tokens': [t.to_dict() for t in self.completed_tokens]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"✅ Results saved to: {output_file}")
        self.logger.info(f"   Tokens analyzed: {len(self.completed_tokens)}")

    async def stop(self):
        """Stop analyzer"""
        self.running = False

        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Alpha Bait Strategy Analyzer")
    parser.add_argument('--hours', type=float, default=3.0, help='How many hours to run (default: 3)')
    parser.add_argument('--min-buy', type=float, default=0.3, help='Minimum SOL for significant buy (default: 0.3)')
    parser.add_argument('--output', type=str, default='data/alpha_bait_analysis', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Config file to use (auto-detects if not specified)')

    args = parser.parse_args()

    # Setup logging
    setup_logging("INFO", "logs/alpha_bait_analyzer.log")
    logger = logging.getLogger(__name__)

    # Load config - try config_fast.yml first (where fast_execution settings are), then config.yml
    config = None
    config_file = args.config

    if config_file:
        # User specified a config file
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
            logger.error("Could not find config.yml or config_fast.yml in config directory")
            return

    # Get LaserStream config from fast_execution section
    fast_config = config.get('fast_execution', {})
    analyzer_config = {
        'helius_grpc_endpoint': fast_config.get('helius_grpc_endpoint'),
        'helius_grpc_token': fast_config.get('helius_grpc_token'),
    }

    if not analyzer_config['helius_grpc_endpoint'] or not analyzer_config['helius_grpc_token']:
        logger.error("Missing helius_grpc_endpoint or helius_grpc_token in config")
        logger.error("Make sure fast_execution section exists with these fields:")
        logger.error("  fast_execution:")
        logger.error("    helius_grpc_endpoint: https://laserstream-mainnet-xxx.helius-rpc.com")
        logger.error("    helius_grpc_token: your-token-here")
        return

    # Create analyzer
    try:
        analyzer = AlphaBaitAnalyzer(analyzer_config, min_significant_buy=args.min_buy)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    # Run analysis
    try:
        await analyzer.start(run_hours=args.hours)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrupted by user - saving data collected so far...")
        await analyzer.stop()
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    # Save results (works even if stopped early)
    try:
        output_dir = Path(args.output)
        analyzer.save_results(output_dir)

        # Print how to analyze results
        logger.info(f"\n{'='*80}")
        logger.info("RESULTS SAVED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"\nTo analyze these results, run:")
        logger.info(f"  python scripts/analyze_alpha_bait_results.py {output_dir}/alpha_bait_analysis_*.json")
        logger.info(f"\nOr find the latest file with:")
        logger.info(f"  ls -lt {output_dir}/")
        logger.info(f"\n{'='*80}\n")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
