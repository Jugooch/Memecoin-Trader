"""
Token Creation Activity Analyzer

Captures ALL activity in the first 5 seconds after token creation.

Purpose: Understand the exact sequence of events to find profitable patterns.

Tracks:
- Token creation event
- ALL buys in first 5 seconds (with buyer addresses, amounts, timing)
- Exact slot and timestamp for each event
- Which buys are from dev vs others

Usage:
    python scripts/analyze_dev_buy_patterns.py --duration 30 --limit 20
"""

import asyncio
import sys
import json
import struct
import base58
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
# Removed: from core.dev_watchlist_loader import DevWatchlistLoader (not needed - tracking ALL tokens)

# Import Geyser/LaserStream
import grpc
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))
import geyser_pb2
import geyser_pb2_grpc

logger = get_logger(__name__)


# Pump.fun program constants
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_CREATE_DISCRIMINATOR = struct.pack("<Q", 8576854823835016728)
PUMP_BUY_DISCRIMINATOR = bytes.fromhex("66063d1201daebea")


@dataclass
class TokenCreation:
    """Token creation event"""
    mint: str
    creator: str
    symbol: str
    name: str
    slot: int
    timestamp: datetime
    signature: str


@dataclass
class BuyEvent:
    """A buy event"""
    buyer: str
    is_dev: bool
    slot: int
    timestamp: datetime
    signature: str
    slots_after_creation: int
    ms_after_creation: float
    tokens_out: int
    max_sol_cost: int

    def to_dict(self):
        return {
            "buyer": self.buyer,
            "is_dev": self.is_dev,
            "slot": self.slot,
            "slots_after_creation": self.slots_after_creation,
            "ms_after_creation": round(self.ms_after_creation, 2),
            "tokens_out": self.tokens_out,
            "max_sol_cost_lamports": self.max_sol_cost,
            "max_sol_cost_sol": self.max_sol_cost / 1e9,
            "signature": self.signature
        }


@dataclass
class TokenTrace:
    """Complete trace of token's first 5 seconds"""
    mint: str
    creator: str
    symbol: str
    name: str
    creation_slot: int
    creation_time: datetime
    creation_signature: str
    buy_events: List[BuyEvent]  # All buys in first 5 seconds

    def to_dict(self):
        return {
            "mint": self.mint,
            "creator": self.creator,
            "symbol": self.symbol,
            "name": self.name,
            "creation_slot": self.creation_slot,
            "creation_time": self.creation_time.isoformat(),
            "creation_signature": self.creation_signature,
            "total_buys_in_5s": len(self.buy_events),
            "dev_buys": [b.to_dict() for b in self.buy_events if b.is_dev],
            "other_buys": [b.to_dict() for b in self.buy_events if not b.is_dev],
            "all_buys_timeline": [b.to_dict() for b in self.buy_events]
        }


class TokenActivityAnalyzer:
    """
    Captures detailed activity traces for newly created tokens

    Monitors ALL buys in first 5 seconds after token creation to reveal patterns.
    """

    def __init__(
        self,
        config_path: str,
        analysis_duration_minutes: int = 60,
        trace_limit: int = 20,
        trace_window_seconds: float = 5.0
    ):
        """
        Initialize analyzer

        Args:
            config_path: Path to config.yml
            analysis_duration_minutes: How long to run analysis
            trace_limit: Max number of token traces to collect
            trace_window_seconds: How long to track buys after creation (default: 5s)
        """
        self.analysis_duration = analysis_duration_minutes
        self.trace_limit = trace_limit
        self.trace_window_seconds = trace_window_seconds

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Tracking
        self.tracked_tokens: Dict[str, TokenCreation] = {}  # mint -> TokenCreation
        self.buy_events: Dict[str, List[BuyEvent]] = defaultdict(list)  # mint -> list of buys
        self.completed_traces: List[TokenTrace] = []  # Finalized traces

        # Get Helius config for Geyser
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        self.geyser_endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")
        self.geyser_api_key = helius_endpoint.laserstream_api_key

        logger.info(
            "analyzer_initialized",
            duration_minutes=analysis_duration_minutes,
            trace_limit=trace_limit,
            trace_window_seconds=trace_window_seconds,
            tracking="First 5s of activity per token"
        )

    def decode_create_instruction(self, ix_data: bytes, account_keys: List, accounts: List[int]) -> Optional[Dict]:
        """Decode token creation instruction"""
        try:
            offset = 8

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
                'creator': creator,
                'mint': get_account_key(0),
            }
        except Exception as e:
            logger.debug("create_decode_failed", error=str(e))
            return None

    def decode_buy_instruction(self, ix_data: bytes, account_keys: List, accounts: List[int]) -> Optional[Dict]:
        """Decode buy instruction"""
        try:
            # Skip discriminator (8 bytes)
            offset = 8

            # Read amount (tokens expected) - u64
            tokens_out = struct.unpack('<Q', ix_data[offset:offset + 8])[0]
            offset += 8

            # Read max_sol_cost - u64
            max_sol_cost = struct.unpack('<Q', ix_data[offset:offset + 8])[0]

            # Get buyer (account index 6 in buy instruction)
            def get_account_key(index: int) -> str:
                if index >= len(accounts):
                    return None
                account_index = accounts[index]
                if account_index >= len(account_keys):
                    return None
                key_bytes = account_keys[account_index]
                return base58.b58encode(bytes(key_bytes)).decode()

            buyer = get_account_key(6)  # User account
            mint = get_account_key(2)   # Mint account

            return {
                'buyer': buyer,
                'mint': mint,
                'tokens_out': tokens_out,
                'max_sol_cost': max_sol_cost,
            }
        except Exception as e:
            logger.debug("buy_decode_failed", error=str(e))
            return None

    async def run(self):
        """Run analysis for specified duration"""
        logger.info("=== TOKEN ACTIVITY ANALYZER STARTED ===")
        logger.info(f"Will capture {self.trace_limit} token traces over {self.analysis_duration} minutes")
        logger.info(f"Tracking: First {self.trace_window_seconds}s of buys after each creation")
        logger.info("Press Ctrl+C to stop early")

        # Start background task to finalize traces
        finalize_task = asyncio.create_task(self._finalize_traces_periodically())

        # Setup Geyser connection
        class HeliusAuthMetadata(grpc.AuthMetadataPlugin):
            def __init__(self, api_key: str):
                self.api_key = api_key

            def __call__(self, context, callback_fn):
                metadata = (('x-token', self.api_key),)
                callback_fn(metadata, None)

        auth = HeliusAuthMetadata(self.geyser_api_key)
        call_creds = grpc.metadata_call_credentials(auth)
        ssl_creds = grpc.ssl_channel_credentials()
        combined_creds = grpc.composite_channel_credentials(ssl_creds, call_creds)

        channel = grpc.aio.secure_channel(self.geyser_endpoint, credentials=combined_creds)
        stub = geyser_pb2_grpc.GeyserStub(channel)

        # Build subscription request
        subscription_request = geyser_pb2.SubscribeRequest()
        subscription_request.transactions["pump_txs"].account_include.append(str(PUMP_FUN_PROGRAM))
        subscription_request.transactions["pump_txs"].failed = False
        subscription_request.commitment = geyser_pb2.CommitmentLevel.PROCESSED

        logger.info("subscribing_to_pump_transactions")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=self.analysis_duration)

        try:
            async def request_generator():
                yield subscription_request
                while datetime.now() < end_time:
                    await asyncio.sleep(30)

            # Subscribe to stream
            async for message in stub.Subscribe(request_generator()):
                # Stop if we hit time limit or trace limit
                if datetime.now() >= end_time or len(self.completed_traces) >= self.trace_limit:
                    break

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

                    # Check each instruction
                    for ix in msg.instructions:
                        ix_data = bytes(ix.data)

                        # TOKEN CREATION
                        if ix_data.startswith(PUMP_CREATE_DISCRIMINATOR):
                            creation_data = self.decode_create_instruction(
                                ix_data,
                                msg.account_keys,
                                list(ix.accounts)
                            )

                            if not creation_data:
                                continue

                            creator = creation_data['creator']

                            # Track ALL tokens (no filtering)
                            token_creation = TokenCreation(
                                mint=creation_data['mint'],
                                creator=creator,
                                symbol=creation_data['symbol'],
                                name=creation_data['name'],
                                slot=slot,
                                timestamp=datetime.now(),
                                signature=signature
                            )

                            self.tracked_tokens[token_creation.mint] = token_creation

                            logger.info(
                                "📝 NEW TOKEN",
                                mint=token_creation.mint[:16] + "...",
                                creator=creator[:16] + "...",
                                symbol=token_creation.symbol,
                                slot=slot
                            )

                        # BUY TRANSACTION
                        elif ix_data.startswith(PUMP_BUY_DISCRIMINATOR):
                            buy_data = self.decode_buy_instruction(
                                ix_data,
                                msg.account_keys,
                                list(ix.accounts)
                            )

                            if not buy_data:
                                continue

                            mint = buy_data['mint']
                            buyer = buy_data['buyer']

                            # Check if this is for a tracked token
                            if mint not in self.tracked_tokens:
                                continue

                            creation = self.tracked_tokens[mint]

                            # Calculate timing relative to creation
                            slots_since_creation = slot - creation.slot
                            time_since_creation_ms = (datetime.now() - creation.timestamp).total_seconds() * 1000

                            # Only track buys within the trace window
                            if time_since_creation_ms > (self.trace_window_seconds * 1000):
                                continue

                            # Check if this is the dev buying
                            is_dev = buyer == creation.creator

                            # Record this buy event
                            buy_event = BuyEvent(
                                buyer=buyer,
                                is_dev=is_dev,
                                slot=slot,
                                timestamp=datetime.now(),
                                signature=signature,
                                slots_after_creation=slots_since_creation,
                                ms_after_creation=time_since_creation_ms,
                                tokens_out=buy_data['tokens_out'],
                                max_sol_cost=buy_data['max_sol_cost']
                            )

                            self.buy_events[mint].append(buy_event)

                            logger.info(
                                "💰 BUY EVENT",
                                mint=mint[:16] + "...",
                                buyer=buyer[:16] + "...",
                                is_dev=is_dev,
                                slot=slot,
                                slots_after=slots_since_creation,
                                ms_after=round(time_since_creation_ms, 2),
                                sol_amount=round(buy_data['max_sol_cost'] / 1e9, 4)
                            )

                except Exception as e:
                    logger.error("message_processing_error", error=str(e))

        finally:
            finalize_task.cancel()
            await channel.close()
            # Finalize any remaining traces
            await self._finalize_all_traces()
            await self._generate_report()

    async def _finalize_traces_periodically(self):
        """Periodically finalize traces that are past the window"""
        while True:
            await asyncio.sleep(1)  # Check every second
            await self._finalize_old_traces()

    async def _finalize_old_traces(self):
        """Finalize traces that are past the trace window"""
        now = datetime.now()
        mints_to_finalize = []

        for mint, creation in self.tracked_tokens.items():
            time_since_creation = (now - creation.timestamp).total_seconds()
            if time_since_creation > self.trace_window_seconds:
                mints_to_finalize.append(mint)

        for mint in mints_to_finalize:
            if len(self.completed_traces) >= self.trace_limit:
                break
            self._finalize_trace(mint)

    async def _finalize_all_traces(self):
        """Finalize all remaining traces"""
        for mint in list(self.tracked_tokens.keys()):
            if len(self.completed_traces) >= self.trace_limit:
                break
            self._finalize_trace(mint)

    def _finalize_trace(self, mint: str):
        """Finalize a single trace"""
        if mint not in self.tracked_tokens:
            return

        creation = self.tracked_tokens[mint]
        buys = self.buy_events.get(mint, [])

        trace = TokenTrace(
            mint=mint,
            creator=creation.creator,
            symbol=creation.symbol,
            name=creation.name,
            creation_slot=creation.slot,
            creation_time=creation.timestamp,
            creation_signature=creation.signature,
            buy_events=buys
        )

        self.completed_traces.append(trace)

        # Clean up tracking
        del self.tracked_tokens[mint]
        if mint in self.buy_events:
            del self.buy_events[mint]

        logger.info(
            "✅ TRACE COMPLETE",
            mint=mint[:16] + "...",
            symbol=creation.symbol,
            total_buys=len(buys),
            dev_buys=sum(1 for b in buys if b.is_dev)
        )

    async def _generate_report(self):
        """Generate activity trace report"""
        logger.info("\n=== TOKEN ACTIVITY ANALYSIS COMPLETE ===\n")

        total_traces = len(self.completed_traces)

        logger.info(
            "analysis_summary",
            total_traces=total_traces,
            trace_window_seconds=self.trace_window_seconds
        )

        if total_traces == 0:
            logger.warning("⚠️ No traces collected. Try running for longer.")
            return

        # Analyze patterns
        logger.info("\n=== ACTIVITY PATTERNS ===")

        for i, trace in enumerate(self.completed_traces[:5]):  # Show first 5 examples
            dev_buys = [b for b in trace.buy_events if b.is_dev]
            other_buys = [b for b in trace.buy_events if not b.is_dev]

            logger.info(f"\n--- Example {i+1}: {trace.symbol} ---")
            logger.info(f"Creator: {trace.creator[:16]}...")
            logger.info(f"Total buys in 5s: {len(trace.buy_events)}")
            logger.info(f"Dev buys: {len(dev_buys)}")
            logger.info(f"Other buys: {len(other_buys)}")

            if dev_buys:
                logger.info("\nDev buy timing:")
                for j, buy in enumerate(dev_buys):
                    logger.info(
                        f"  #{j+1}: Slot {buy.slot} ({buy.slots_after_creation} slots after creation, "
                        f"{buy.ms_after_creation:.0f}ms, {buy.max_sol_cost/1e9:.4f} SOL)"
                    )

            if other_buys:
                logger.info(f"\nOther buys: {len(other_buys)} total")
                for j, buy in enumerate(other_buys[:3]):  # Show first 3
                    logger.info(
                        f"  #{j+1}: {buy.buyer[:16]}... at slot {buy.slot} "
                        f"({buy.slots_after_creation} slots after, {buy.ms_after_creation:.0f}ms)"
                    )

        # Export detailed traces
        output_dir = Path("data/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"token_activity_traces_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "summary": {
                "total_traces": total_traces,
                "trace_window_seconds": self.trace_window_seconds
            },
            "traces": [t.to_dict() for t in self.completed_traces]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n✅ Detailed traces exported to: {output_file}")
        logger.info(f"📋 Captured {total_traces} complete token traces")
        logger.info("\n💡 Review the JSON to see exact patterns and timing!")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze dev first buy timing on ALL Pump.fun tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--config", default="config/config.yml", help="Path to config.yml")
    parser.add_argument("--duration", type=int, default=30, help="Analysis duration in minutes (default: 30)")
    parser.add_argument("--limit", type=int, default=20, help="Max number of token traces to collect (default: 20)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="json", output_file="logs/dev_buy_analysis.log")

    # Create and run analyzer
    analyzer = TokenActivityAnalyzer(
        config_path=args.config,
        analysis_duration_minutes=args.duration,
        trace_limit=args.limit
    )

    await analyzer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error("analysis_failed", error=str(e), error_type=type(e).__name__)
        raise
