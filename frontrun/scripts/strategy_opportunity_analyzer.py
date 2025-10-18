"""
Strategy Opportunity Analyzer

Mathematically proves whether profitable strategies exist by tracking:
1. Momentum Back-Running: Buy at slot +1, sell into retail wave
2. Large Buy Following: Back-run large buys (>0.5 SOL)

Measures:
- Buy velocity (buys per second)
- Price movements (bonding curve simulation)
- Momentum peaks (when does buy rate drop?)
- Large buy impact (how much does price bounce?)
- Optimal entry/exit timing
- Expected value for each strategy

Output: Clear metrics showing IF and WHEN edge exists

Usage:
    python scripts/strategy_opportunity_analyzer.py --duration 30 --tokens 20
"""

import asyncio
import sys
import json
import struct
import base58
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from solders.keypair import Keypair
from solders.pubkey import Pubkey

from core.config import ConfigurationManager
from core.logger import setup_logging, get_logger
from core.bonding_curve import (
    BondingCurveCalculator,
    BondingCurveState,
    INITIAL_VIRTUAL_TOKEN_RESERVES,
    INITIAL_VIRTUAL_SOL_RESERVES,
    INITIAL_REAL_TOKEN_RESERVES,
    INITIAL_REAL_SOL_RESERVES,
    INITIAL_TOKEN_TOTAL_SUPPLY
)

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
class BuyEvent:
    """A buy event with timing and amount"""
    buyer: str
    is_dev: bool
    slot: int
    timestamp: datetime
    signature: str
    seconds_after_creation: float
    sol_amount: float  # Actual SOL spent
    tokens_received: int

    def to_dict(self):
        return {
            "buyer": self.buyer[:16] + "...",
            "is_dev": self.is_dev,
            "slot": self.slot,
            "seconds_after_creation": round(self.seconds_after_creation, 2),
            "sol_amount": self.sol_amount,
            "tokens_received": self.tokens_received,
            "signature": self.signature
        }


@dataclass
class MomentumWindow:
    """Momentum metrics for a time window"""
    start_second: int
    end_second: int
    buy_count: int
    total_sol_volume: float
    unique_buyers: int
    avg_buy_size_sol: float
    buys_per_second: float


@dataclass
class LargeBuyOpportunity:
    """Analysis of following a large buy"""
    buy_event: BuyEvent
    price_before: float  # SOL per million tokens
    price_after: float   # SOL per million tokens
    price_impact_pct: float

    # If we bought immediately after
    simulated_entry_price: float
    simulated_profit_3s: float  # Profit if we held 3 seconds
    simulated_profit_5s: float  # Profit if we held 5 seconds
    simulated_profit_10s: float # Profit if we held 10 seconds


@dataclass
class TokenAnalysis:
    """Complete analysis of a token's first 60 seconds"""
    mint: str
    creator: str
    symbol: str
    name: str
    creation_slot: int
    creation_time: datetime

    # All buy events (sorted by time)
    buy_events: List[BuyEvent]

    # Momentum metrics
    momentum_windows: List[MomentumWindow]  # 5-second windows
    peak_momentum_second: int  # When was momentum highest?
    peak_buys_per_second: float

    # Large buy opportunities
    large_buy_opportunities: List[LargeBuyOpportunity]

    # Strategy 1: Momentum back-running analysis
    # If we bought at slot +1 (our current capability)
    our_entry_slot: int  # slot + 1
    our_entry_price: float  # SOL per million tokens

    # What if we sold at different times?
    profit_if_sell_5s: float   # % profit if sold at 5s
    profit_if_sell_10s: float  # % profit if sold at 10s
    profit_if_sell_15s: float  # % profit if sold at 15s
    profit_if_sell_20s: float  # % profit if sold at 20s
    profit_if_sell_30s: float  # % profit if sold at 30s
    profit_if_sell_at_peak: float  # % profit if sold at momentum peak

    # Was there actually a retail wave?
    retail_wave_detected: bool
    retail_wave_start_second: float
    retail_wave_buy_count: int

    def to_dict(self):
        return {
            "mint": self.mint,
            "creator": self.creator[:16] + "...",
            "symbol": self.symbol,
            "name": self.name,
            "creation_slot": self.creation_slot,
            "total_buys_60s": len(self.buy_events),
            "peak_momentum_second": self.peak_momentum_second,
            "peak_buys_per_second": round(self.peak_buys_per_second, 2),
            "large_buy_opportunities": len(self.large_buy_opportunities),
            "strategy_1_momentum_backrun": {
                "our_entry_slot": self.our_entry_slot,
                "our_entry_price_sol_per_m": round(self.our_entry_price, 6),
                "profit_if_sell_5s": round(self.profit_if_sell_5s, 2),
                "profit_if_sell_10s": round(self.profit_if_sell_10s, 2),
                "profit_if_sell_15s": round(self.profit_if_sell_15s, 2),
                "profit_if_sell_20s": round(self.profit_if_sell_20s, 2),
                "profit_if_sell_30s": round(self.profit_if_sell_30s, 2),
                "profit_if_sell_at_peak": round(self.profit_if_sell_at_peak, 2),
                "retail_wave_detected": self.retail_wave_detected,
                "retail_wave_start_second": round(self.retail_wave_start_second, 2) if self.retail_wave_detected else None,
                "retail_wave_buy_count": self.retail_wave_buy_count
            },
            "strategy_2_large_buy_following": {
                "total_opportunities": len(self.large_buy_opportunities),
                "opportunities": [
                    {
                        "at_second": round(opp.buy_event.seconds_after_creation, 2),
                        "sol_amount": opp.buy_event.sol_amount,
                        "price_impact_pct": round(opp.price_impact_pct, 2),
                        "profit_if_hold_3s": round(opp.simulated_profit_3s, 2),
                        "profit_if_hold_5s": round(opp.simulated_profit_5s, 2),
                        "profit_if_hold_10s": round(opp.simulated_profit_10s, 2)
                    }
                    for opp in self.large_buy_opportunities
                ]
            }
        }


class StrategyOpportunityAnalyzer:
    """
    Proves whether strategies are profitable by tracking real data
    """

    def __init__(
        self,
        config_path: str,
        analysis_duration_minutes: int = 60,
        token_limit: int = 20,
        tracking_window_seconds: float = 60.0
    ):
        """
        Initialize analyzer

        Args:
            config_path: Path to config.yml
            analysis_duration_minutes: How long to run analysis
            token_limit: Max number of tokens to analyze
            tracking_window_seconds: How long to track each token (default: 60s)
        """
        self.analysis_duration = analysis_duration_minutes
        self.token_limit = token_limit
        self.tracking_window = tracking_window_seconds

        # Load config
        config_mgr = ConfigurationManager(config_path)
        self.bot_config = config_mgr.load_config()

        # Bonding curve calculator
        self.calculator = BondingCurveCalculator()

        # Tracking
        self.tracked_tokens: Dict[str, Dict] = {}  # mint -> {creation, buys, curve_state}
        self.completed_analyses: List[TokenAnalysis] = []

        # Get Helius config for Geyser
        helius_endpoint = self.bot_config.rpc_config.endpoints[0]
        self.geyser_endpoint = helius_endpoint.laserstream_url.replace("https://", "").replace("http://", "")
        self.geyser_api_key = helius_endpoint.laserstream_api_key

        logger.info(
            "strategy_analyzer_initialized",
            duration_minutes=analysis_duration_minutes,
            token_limit=token_limit,
            tracking_window_seconds=tracking_window_seconds
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
            offset = 8

            # Read amount (tokens expected) - u64
            tokens_out = struct.unpack('<Q', ix_data[offset:offset + 8])[0]
            offset += 8

            # Read max_sol_cost - u64
            max_sol_cost = struct.unpack('<Q', ix_data[offset:offset + 8])[0]

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

    def calculate_price_per_million_tokens(self, curve_state: BondingCurveState) -> float:
        """Calculate current price in SOL per 1M tokens"""
        try:
            # Validate curve state
            if curve_state.virtual_token_reserves <= 0:
                logger.warning(
                    "invalid_virtual_token_reserves",
                    reserves=curve_state.virtual_token_reserves
                )
                return 0.0

            if curve_state.virtual_sol_reserves <= 0:
                logger.warning(
                    "invalid_virtual_sol_reserves",
                    reserves=curve_state.virtual_sol_reserves
                )
                return 0.0

            # Use the formula: price = virtual_sol / virtual_tokens
            # Price per single token (in lamports per base token unit)
            price_per_token = curve_state.virtual_sol_reserves / curve_state.virtual_token_reserves

            # Price per 1 million tokens (1M tokens = 1_000_000 * 1e6 base units)
            price_per_million = price_per_token * 1_000_000

            # Convert from lamports to SOL
            price_in_sol = price_per_million / 1e9

            return price_in_sol
        except Exception as e:
            logger.error(
                "price_calculation_failed",
                error=str(e),
                virtual_sol=curve_state.virtual_sol_reserves,
                virtual_tokens=curve_state.virtual_token_reserves
            )
            return 0.0

    def simulate_curve_after_buy(self, curve_state: BondingCurveState, sol_in_lamports: int) -> BondingCurveState:
        """Simulate bonding curve state after a buy"""
        try:
            # Validate input
            if sol_in_lamports <= 0:
                logger.warning(
                    "invalid_sol_amount_for_simulation",
                    sol_lamports=sol_in_lamports
                )
                return curve_state

            # Calculate quote
            quote = self.calculator.calculate_buy_price(curve_state, sol_in_lamports)

            # Validate quote
            if quote.tokens_out <= 0:
                logger.warning(
                    "invalid_quote_tokens_out",
                    tokens_out=quote.tokens_out,
                    sol_in=sol_in_lamports
                )
                return curve_state

            # Calculate new reserves
            new_virtual_sol = curve_state.virtual_sol_reserves + sol_in_lamports
            new_virtual_tokens = curve_state.virtual_token_reserves - quote.tokens_out
            new_real_sol = curve_state.real_sol_reserves + sol_in_lamports
            new_real_tokens = curve_state.real_token_reserves - quote.tokens_out

            # Validate new state
            if new_virtual_tokens < 0:
                logger.warning(
                    "negative_virtual_tokens_after_buy",
                    new_virtual_tokens=new_virtual_tokens,
                    sol_in=sol_in_lamports,
                    tokens_out=quote.tokens_out
                )
                # Return current state to avoid invalid state
                return curve_state

            return BondingCurveState(
                virtual_token_reserves=new_virtual_tokens,
                virtual_sol_reserves=new_virtual_sol,
                real_token_reserves=new_real_tokens,
                real_sol_reserves=new_real_sol,
                token_total_supply=curve_state.token_total_supply,
                complete=curve_state.complete
            )
        except Exception as e:
            logger.error(
                "curve_simulation_failed",
                error=str(e),
                sol_in=sol_in_lamports
            )
            return curve_state

    async def run(self):
        """Run analysis for specified duration"""
        logger.info("=== STRATEGY OPPORTUNITY ANALYZER STARTED ===")
        logger.info(f"Will analyze {self.token_limit} tokens over {self.analysis_duration} minutes")
        logger.info(f"Tracking each token for {self.tracking_window}s to find profitable patterns")
        logger.info("Press Ctrl+C to stop early\n")

        # Start background task to finalize analyses
        finalize_task = asyncio.create_task(self._finalize_analyses_periodically())

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
                # Stop if we hit limits
                if datetime.now() >= end_time or len(self.completed_analyses) >= self.token_limit:
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

                            mint = creation_data['mint']

                            # Initialize tracking for this token
                            self.tracked_tokens[mint] = {
                                'creation': {
                                    'mint': mint,
                                    'creator': creation_data['creator'],
                                    'symbol': creation_data['symbol'],
                                    'name': creation_data['name'],
                                    'slot': slot,
                                    'timestamp': datetime.now(),
                                    'signature': signature
                                },
                                'buys': [],
                                'curve_state': BondingCurveState(
                                    virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
                                    virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
                                    real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
                                    real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
                                    token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
                                    complete=False
                                )
                            }

                            logger.info(
                                "📝 TRACKING NEW TOKEN",
                                mint=mint[:16] + "...",
                                symbol=creation_data['symbol'],
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

                            # Check if this is for a tracked token
                            if mint not in self.tracked_tokens:
                                continue

                            token_data = self.tracked_tokens[mint]
                            creation = token_data['creation']

                            # Calculate timing
                            seconds_since_creation = (datetime.now() - creation['timestamp']).total_seconds()

                            # Only track buys within window
                            if seconds_since_creation > self.tracking_window:
                                continue

                            buyer = buy_data['buyer']
                            is_dev = buyer == creation['creator']
                            sol_amount = buy_data['max_sol_cost'] / 1e9

                            # Create buy event
                            buy_event = BuyEvent(
                                buyer=buyer,
                                is_dev=is_dev,
                                slot=slot,
                                timestamp=datetime.now(),
                                signature=signature,
                                seconds_after_creation=seconds_since_creation,
                                sol_amount=sol_amount,
                                tokens_received=buy_data['tokens_out']
                            )

                            # Update curve state
                            token_data['curve_state'] = self.simulate_curve_after_buy(
                                token_data['curve_state'],
                                buy_data['max_sol_cost']
                            )

                            # Record buy
                            token_data['buys'].append(buy_event)

                            logger.debug(
                                "💰 BUY",
                                mint=mint[:12] + "...",
                                is_dev=is_dev,
                                seconds_after=round(seconds_since_creation, 2),
                                sol=round(sol_amount, 4)
                            )

                except Exception as e:
                    logger.error("message_processing_error", error=str(e))

        finally:
            finalize_task.cancel()
            await channel.close()
            # Finalize any remaining analyses
            await self._finalize_all_analyses()
            await self._generate_report()

    async def _finalize_analyses_periodically(self):
        """Periodically finalize analyses that are past the window"""
        while True:
            await asyncio.sleep(2)  # Check every 2 seconds
            await self._finalize_old_analyses()

    async def _finalize_old_analyses(self):
        """Finalize analyses that are past the tracking window"""
        now = datetime.now()
        mints_to_finalize = []

        for mint, token_data in self.tracked_tokens.items():
            creation = token_data['creation']
            time_since_creation = (now - creation['timestamp']).total_seconds()
            if time_since_creation > self.tracking_window:
                mints_to_finalize.append(mint)

        for mint in mints_to_finalize:
            if len(self.completed_analyses) >= self.token_limit:
                break
            self._finalize_analysis(mint)

    async def _finalize_all_analyses(self):
        """Finalize all remaining analyses"""
        for mint in list(self.tracked_tokens.keys()):
            if len(self.completed_analyses) >= self.token_limit:
                break
            self._finalize_analysis(mint)

    def _finalize_analysis(self, mint: str):
        """Analyze a token's complete data and finalize"""
        if mint not in self.tracked_tokens:
            return

        token_data = self.tracked_tokens[mint]
        creation = token_data['creation']
        buy_events = token_data['buys']

        # Sort buys by time
        buy_events.sort(key=lambda b: b.seconds_after_creation)

        if len(buy_events) == 0:
            # No buys, skip this token
            del self.tracked_tokens[mint]
            return

        # Calculate momentum windows (5-second windows)
        momentum_windows = self._calculate_momentum_windows(buy_events)

        # Find peak momentum
        peak_window = max(momentum_windows, key=lambda w: w.buys_per_second) if momentum_windows else None
        peak_momentum_second = peak_window.start_second if peak_window else 0
        peak_buys_per_second = peak_window.buys_per_second if peak_window else 0

        # Identify large buy opportunities (>0.5 SOL)
        large_buy_opportunities = self._identify_large_buy_opportunities(buy_events, token_data)

        # Strategy 1: Momentum back-running analysis
        # Simulate buying at slot +1 (our capability)
        our_entry_slot = creation['slot'] + 1

        # Simulate our entry price (after slot 0 dev buy)
        initial_state = BondingCurveState(
            virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
            virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
            real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
            real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
            token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
            complete=False
        )

        # Find dev's buy at slot 0 (or earliest buy)
        dev_buys_slot_0 = [b for b in buy_events if b.is_dev and b.slot == creation['slot']]

        if dev_buys_slot_0:
            # Simulate after dev buy
            dev_buy = dev_buys_slot_0[0]
            state_after_dev = self.simulate_curve_after_buy(initial_state, int(dev_buy.sol_amount * 1e9))
            our_entry_price = self.calculate_price_per_million_tokens(state_after_dev)
        else:
            # No dev buy at slot 0, use initial price
            our_entry_price = self.calculate_price_per_million_tokens(initial_state)

        # Calculate profit at different hold times
        profit_5s = self._calculate_profit_at_time(buy_events, our_entry_price, 5.0)
        profit_10s = self._calculate_profit_at_time(buy_events, our_entry_price, 10.0)
        profit_15s = self._calculate_profit_at_time(buy_events, our_entry_price, 15.0)
        profit_20s = self._calculate_profit_at_time(buy_events, our_entry_price, 20.0)
        profit_30s = self._calculate_profit_at_time(buy_events, our_entry_price, 30.0)
        profit_at_peak = self._calculate_profit_at_time(buy_events, our_entry_price, peak_momentum_second)

        # Detect retail wave (3+ buys from non-dev wallets in a 5s window)
        retail_wave_detected, retail_wave_start, retail_wave_count = self._detect_retail_wave(buy_events)

        # Create analysis
        analysis = TokenAnalysis(
            mint=mint,
            creator=creation['creator'],
            symbol=creation['symbol'],
            name=creation['name'],
            creation_slot=creation['slot'],
            creation_time=creation['timestamp'],
            buy_events=buy_events,
            momentum_windows=momentum_windows,
            peak_momentum_second=peak_momentum_second,
            peak_buys_per_second=peak_buys_per_second,
            large_buy_opportunities=large_buy_opportunities,
            our_entry_slot=our_entry_slot,
            our_entry_price=our_entry_price,
            profit_if_sell_5s=profit_5s,
            profit_if_sell_10s=profit_10s,
            profit_if_sell_15s=profit_15s,
            profit_if_sell_20s=profit_20s,
            profit_if_sell_30s=profit_30s,
            profit_if_sell_at_peak=profit_at_peak,
            retail_wave_detected=retail_wave_detected,
            retail_wave_start_second=retail_wave_start,
            retail_wave_buy_count=retail_wave_count
        )

        self.completed_analyses.append(analysis)

        # Clean up tracking
        del self.tracked_tokens[mint]

        logger.info(
            "✅ ANALYSIS COMPLETE",
            mint=mint[:16] + "...",
            symbol=creation['symbol'],
            total_buys=len(buy_events),
            best_profit=f"{max(profit_5s, profit_10s, profit_15s, profit_20s, profit_30s):.1f}%"
        )

    def _calculate_momentum_windows(self, buy_events: List[BuyEvent]) -> List[MomentumWindow]:
        """Calculate 5-second momentum windows"""
        windows = []

        # Group buys into 5-second windows
        for start_sec in range(0, 60, 5):
            end_sec = start_sec + 5
            window_buys = [b for b in buy_events if start_sec <= b.seconds_after_creation < end_sec]

            if not window_buys:
                continue

            unique_buyers = len(set(b.buyer for b in window_buys))
            total_sol = sum(b.sol_amount for b in window_buys)
            avg_buy_size = total_sol / len(window_buys)
            buys_per_second = len(window_buys) / 5.0

            windows.append(MomentumWindow(
                start_second=start_sec,
                end_second=end_sec,
                buy_count=len(window_buys),
                total_sol_volume=total_sol,
                unique_buyers=unique_buyers,
                avg_buy_size_sol=avg_buy_size,
                buys_per_second=buys_per_second
            ))

        return windows

    def _identify_large_buy_opportunities(
        self,
        buy_events: List[BuyEvent],
        token_data: Dict
    ) -> List[LargeBuyOpportunity]:
        """Identify opportunities to back-run large buys"""
        opportunities = []

        # Reconstruct curve state at each buy
        current_state = BondingCurveState(
            virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
            virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
            real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
            real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
            token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
            complete=False
        )

        for i, buy_event in enumerate(buy_events):
            try:
                # Price before this buy
                price_before = self.calculate_price_per_million_tokens(current_state)

                # Skip if price is invalid
                if price_before <= 0:
                    logger.warning(
                        "invalid_price_before",
                        buy_index=i,
                        price=price_before
                    )
                    # Still update state for future buys
                    current_state = self.simulate_curve_after_buy(current_state, int(buy_event.sol_amount * 1e9))
                    continue

                # Simulate this buy
                current_state = self.simulate_curve_after_buy(current_state, int(buy_event.sol_amount * 1e9))

                # Price after this buy
                price_after = self.calculate_price_per_million_tokens(current_state)

                # Skip if price is invalid
                if price_after <= 0:
                    logger.warning(
                        "invalid_price_after",
                        buy_index=i,
                        price=price_after
                    )
                    continue

                # Check if this is a large buy (>0.5 SOL)
                if buy_event.sol_amount >= 0.5:
                    price_impact_pct = ((price_after - price_before) / price_before) * 100

                    # Simulate if we bought 0.01 SOL immediately after
                    our_entry_price = price_after

                    # Validate entry price
                    if our_entry_price <= 0:
                        logger.warning(
                            "invalid_large_buy_entry_price",
                            buy_index=i,
                            entry_price=our_entry_price
                        )
                        continue

                    # Find future buys and calculate profits
                    future_buys_3s = [b for b in buy_events[i+1:] if b.seconds_after_creation <= buy_event.seconds_after_creation + 3]
                    future_buys_5s = [b for b in buy_events[i+1:] if b.seconds_after_creation <= buy_event.seconds_after_creation + 5]
                    future_buys_10s = [b for b in buy_events[i+1:] if b.seconds_after_creation <= buy_event.seconds_after_creation + 10]

                    # Simulate price after future buys
                    state_3s = current_state
                    for fb in future_buys_3s:
                        state_3s = self.simulate_curve_after_buy(state_3s, int(fb.sol_amount * 1e9))
                    price_3s = self.calculate_price_per_million_tokens(state_3s)
                    profit_3s = ((price_3s - our_entry_price) / our_entry_price) * 100 if our_entry_price > 0 else -100.0

                    state_5s = current_state
                    for fb in future_buys_5s:
                        state_5s = self.simulate_curve_after_buy(state_5s, int(fb.sol_amount * 1e9))
                    price_5s = self.calculate_price_per_million_tokens(state_5s)
                    profit_5s = ((price_5s - our_entry_price) / our_entry_price) * 100 if our_entry_price > 0 else -100.0

                    state_10s = current_state
                    for fb in future_buys_10s:
                        state_10s = self.simulate_curve_after_buy(state_10s, int(fb.sol_amount * 1e9))
                    price_10s = self.calculate_price_per_million_tokens(state_10s)
                    profit_10s = ((price_10s - our_entry_price) / our_entry_price) * 100 if our_entry_price > 0 else -100.0

                    opportunities.append(LargeBuyOpportunity(
                        buy_event=buy_event,
                        price_before=price_before,
                        price_after=price_after,
                        price_impact_pct=price_impact_pct,
                        simulated_entry_price=our_entry_price,
                        simulated_profit_3s=profit_3s,
                        simulated_profit_5s=profit_5s,
                        simulated_profit_10s=profit_10s
                    ))

            except Exception as e:
                logger.warning(
                    "large_buy_analysis_failed",
                    buy_index=i,
                    error=str(e)
                )
                # Continue to next buy
                continue

        return opportunities

    def _calculate_profit_at_time(
        self,
        buy_events: List[BuyEvent],
        entry_price: float,
        sell_time_seconds: float
    ) -> float:
        """Calculate profit % if we bought at entry_price and sold at sell_time_seconds"""
        # Validate entry price
        if entry_price <= 0:
            logger.warning(
                "invalid_entry_price_for_profit_calc",
                entry_price=entry_price,
                sell_time=sell_time_seconds
            )
            return -100.0  # Invalid entry price = can't trade = 100% loss

        # Find all buys up to sell_time
        buys_before_sell = [b for b in buy_events if b.seconds_after_creation <= sell_time_seconds]

        if not buys_before_sell:
            return -100.0  # No buys = no price movement = loss from fees

        # Simulate curve state at sell time
        current_state = BondingCurveState(
            virtual_token_reserves=INITIAL_VIRTUAL_TOKEN_RESERVES,
            virtual_sol_reserves=INITIAL_VIRTUAL_SOL_RESERVES,
            real_token_reserves=INITIAL_REAL_TOKEN_RESERVES,
            real_sol_reserves=INITIAL_REAL_SOL_RESERVES,
            token_total_supply=INITIAL_TOKEN_TOTAL_SUPPLY,
            complete=False
        )

        for buy in buys_before_sell:
            current_state = self.simulate_curve_after_buy(current_state, int(buy.sol_amount * 1e9))

        sell_price = self.calculate_price_per_million_tokens(current_state)

        # Validate sell price
        if sell_price <= 0:
            logger.warning(
                "invalid_sell_price_for_profit_calc",
                sell_price=sell_price,
                sell_time=sell_time_seconds
            )
            return -100.0  # Invalid sell price = can't exit = 100% loss

        # Calculate profit % (accounting for ~2% fees)
        gross_profit_pct = ((sell_price - entry_price) / entry_price) * 100
        net_profit_pct = gross_profit_pct - 2.0  # Subtract fees

        return net_profit_pct

    def _detect_retail_wave(self, buy_events: List[BuyEvent]) -> Tuple[bool, float, int]:
        """Detect if there was a retail wave (3+ non-dev buys in 5s window)"""
        non_dev_buys = [b for b in buy_events if not b.is_dev]

        # Check each 5-second window
        for start_sec in range(0, 60, 1):  # Check every second
            end_sec = start_sec + 5
            window_buys = [b for b in non_dev_buys if start_sec <= b.seconds_after_creation < end_sec]

            if len(window_buys) >= 3:
                return True, start_sec, len(window_buys)

        return False, 0.0, 0

    async def _generate_report(self):
        """Generate strategy opportunity report"""
        logger.info("\n" + "="*80)
        logger.info("STRATEGY OPPORTUNITY ANALYSIS COMPLETE")
        logger.info("="*80 + "\n")

        total_analyses = len(self.completed_analyses)

        if total_analyses == 0:
            logger.warning("⚠️ No analyses completed. Try running for longer.")
            return

        # ===== STRATEGY 1: MOMENTUM BACK-RUNNING =====
        logger.info("\n" + "="*80)
        logger.info("📊 STRATEGY 1: MOMENTUM BACK-RUNNING ANALYSIS")
        logger.info("="*80)
        logger.info("Concept: Buy at slot +1, sell into retail wave\n")

        # Calculate average profits at each hold time
        avg_profit_5s = sum(a.profit_if_sell_5s for a in self.completed_analyses) / total_analyses
        avg_profit_10s = sum(a.profit_if_sell_10s for a in self.completed_analyses) / total_analyses
        avg_profit_15s = sum(a.profit_if_sell_15s for a in self.completed_analyses) / total_analyses
        avg_profit_20s = sum(a.profit_if_sell_20s for a in self.completed_analyses) / total_analyses
        avg_profit_30s = sum(a.profit_if_sell_30s for a in self.completed_analyses) / total_analyses
        avg_profit_at_peak = sum(a.profit_if_sell_at_peak for a in self.completed_analyses) / total_analyses

        # Win rates
        win_rate_5s = sum(1 for a in self.completed_analyses if a.profit_if_sell_5s > 0) / total_analyses * 100
        win_rate_10s = sum(1 for a in self.completed_analyses if a.profit_if_sell_10s > 0) / total_analyses * 100
        win_rate_15s = sum(1 for a in self.completed_analyses if a.profit_if_sell_15s > 0) / total_analyses * 100
        win_rate_20s = sum(1 for a in self.completed_analyses if a.profit_if_sell_20s > 0) / total_analyses * 100
        win_rate_30s = sum(1 for a in self.completed_analyses if a.profit_if_sell_30s > 0) / total_analyses * 100

        # Retail wave detection
        tokens_with_retail_wave = sum(1 for a in self.completed_analyses if a.retail_wave_detected)
        retail_wave_rate = tokens_with_retail_wave / total_analyses * 100

        logger.info(f"📈 AVERAGE PROFIT BY HOLD TIME:")
        logger.info(f"   5s hold:  {avg_profit_5s:+.2f}% (win rate: {win_rate_5s:.1f}%)")
        logger.info(f"  10s hold:  {avg_profit_10s:+.2f}% (win rate: {win_rate_10s:.1f}%)")
        logger.info(f"  15s hold:  {avg_profit_15s:+.2f}% (win rate: {win_rate_15s:.1f}%)")
        logger.info(f"  20s hold:  {avg_profit_20s:+.2f}% (win rate: {win_rate_20s:.1f}%)")
        logger.info(f"  30s hold:  {avg_profit_30s:+.2f}% (win rate: {win_rate_30s:.1f}%)")
        logger.info(f"  Peak hold: {avg_profit_at_peak:+.2f}%")

        logger.info(f"\n🌊 RETAIL WAVE DETECTION:")
        logger.info(f"   {tokens_with_retail_wave}/{total_analyses} tokens had retail wave ({retail_wave_rate:.1f}%)")

        # Find best hold time
        hold_times = [
            (5, avg_profit_5s, win_rate_5s),
            (10, avg_profit_10s, win_rate_10s),
            (15, avg_profit_15s, win_rate_15s),
            (20, avg_profit_20s, win_rate_20s),
            (30, avg_profit_30s, win_rate_30s)
        ]
        best_hold = max(hold_times, key=lambda x: x[1])

        logger.info(f"\n✅ OPTIMAL HOLD TIME: {best_hold[0]}s")
        logger.info(f"   Expected profit: {best_hold[1]:+.2f}%")
        logger.info(f"   Win rate: {best_hold[2]:.1f}%")

        # Expected value calculation
        ev = best_hold[1]  # Average profit
        logger.info(f"\n💰 EXPECTED VALUE: {ev:+.2f}% per trade")

        if ev > 0:
            logger.info(f"   ✅ STRATEGY 1 IS PROFITABLE!")
            logger.info(f"   With 20 trades/day: {ev * 20:+.2f}% daily expected return")
        else:
            logger.info(f"   ❌ STRATEGY 1 IS NOT PROFITABLE")
            logger.info(f"   Average loss: {ev:.2f}%")

        # ===== STRATEGY 2: LARGE BUY FOLLOWING =====
        logger.info("\n" + "="*80)
        logger.info("🚀 STRATEGY 2: LARGE BUY FOLLOWING ANALYSIS")
        logger.info("="*80)
        logger.info("Concept: Back-run large buys (>0.5 SOL), sell into bounce\n")

        # Collect all large buy opportunities
        all_large_buy_opps = []
        for analysis in self.completed_analyses:
            all_large_buy_opps.extend(analysis.large_buy_opportunities)

        if len(all_large_buy_opps) == 0:
            logger.info("❌ NO LARGE BUY OPPORTUNITIES DETECTED")
            logger.info("   (No buys >0.5 SOL found in dataset)")
        else:
            # Calculate average profits
            avg_profit_3s_lb = sum(opp.simulated_profit_3s for opp in all_large_buy_opps) / len(all_large_buy_opps)
            avg_profit_5s_lb = sum(opp.simulated_profit_5s for opp in all_large_buy_opps) / len(all_large_buy_opps)
            avg_profit_10s_lb = sum(opp.simulated_profit_10s for opp in all_large_buy_opps) / len(all_large_buy_opps)

            # Win rates
            win_rate_3s_lb = sum(1 for opp in all_large_buy_opps if opp.simulated_profit_3s > 0) / len(all_large_buy_opps) * 100
            win_rate_5s_lb = sum(1 for opp in all_large_buy_opps if opp.simulated_profit_5s > 0) / len(all_large_buy_opps) * 100
            win_rate_10s_lb = sum(1 for opp in all_large_buy_opps if opp.simulated_profit_10s > 0) / len(all_large_buy_opps) * 100

            # Opportunity frequency
            opps_per_token = len(all_large_buy_opps) / total_analyses

            logger.info(f"📊 OPPORTUNITIES DETECTED:")
            logger.info(f"   Total: {len(all_large_buy_opps)} large buys (>0.5 SOL)")
            logger.info(f"   Per token: {opps_per_token:.2f} opportunities")

            logger.info(f"\n📈 AVERAGE PROFIT BY HOLD TIME:")
            logger.info(f"   3s hold:  {avg_profit_3s_lb:+.2f}% (win rate: {win_rate_3s_lb:.1f}%)")
            logger.info(f"   5s hold:  {avg_profit_5s_lb:+.2f}% (win rate: {win_rate_5s_lb:.1f}%)")
            logger.info(f"  10s hold:  {avg_profit_10s_lb:+.2f}% (win rate: {win_rate_10s_lb:.1f}%)")

            # Find best hold time for large buys
            lb_hold_times = [
                (3, avg_profit_3s_lb, win_rate_3s_lb),
                (5, avg_profit_5s_lb, win_rate_5s_lb),
                (10, avg_profit_10s_lb, win_rate_10s_lb)
            ]
            best_lb_hold = max(lb_hold_times, key=lambda x: x[1])

            logger.info(f"\n✅ OPTIMAL HOLD TIME: {best_lb_hold[0]}s")
            logger.info(f"   Expected profit: {best_lb_hold[1]:+.2f}%")
            logger.info(f"   Win rate: {best_lb_hold[2]:.1f}%")

            # Expected value
            ev_lb = best_lb_hold[1]
            logger.info(f"\n💰 EXPECTED VALUE: {ev_lb:+.2f}% per opportunity")

            if ev_lb > 0:
                logger.info(f"   ✅ STRATEGY 2 IS PROFITABLE!")
                logger.info(f"   With {opps_per_token:.1f} opps/token × 20 tokens/day: {ev_lb * opps_per_token * 20:+.2f}% daily expected return")
            else:
                logger.info(f"   ❌ STRATEGY 2 IS NOT PROFITABLE")
                logger.info(f"   Average loss: {ev_lb:.2f}%")

        # ===== FINAL RECOMMENDATION =====
        logger.info("\n" + "="*80)
        logger.info("🎯 FINAL RECOMMENDATION")
        logger.info("="*80 + "\n")

        strategy_1_ev = best_hold[1]
        strategy_2_ev = avg_profit_5s_lb if len(all_large_buy_opps) > 0 else -100

        if strategy_1_ev > 0 and strategy_1_ev > strategy_2_ev:
            logger.info("✅ RECOMMEND: Strategy 1 (Momentum Back-Running)")
            logger.info(f"   Entry: Buy at slot +1 (detection + 400ms)")
            logger.info(f"   Hold: {best_hold[0]} seconds")
            logger.info(f"   Expected: {strategy_1_ev:+.2f}% per trade")
            logger.info(f"   Win rate: {best_hold[2]:.1f}%")
        elif strategy_2_ev > 0:
            logger.info("✅ RECOMMEND: Strategy 2 (Large Buy Following)")
            logger.info(f"   Entry: Detect large buy >0.5 SOL, buy immediately after")
            logger.info(f"   Hold: {best_lb_hold[0]} seconds")
            logger.info(f"   Expected: {strategy_2_ev:+.2f}% per opportunity")
            logger.info(f"   Win rate: {best_lb_hold[2]:.1f}%")
        else:
            logger.info("❌ NO PROFITABLE STRATEGY FOUND")
            logger.info("   Both strategies show negative expected value")
            logger.info("   Recommendation: DO NOT TRADE with current parameters")

        # Export detailed data
        output_dir = Path("data/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "summary": {
                "total_tokens_analyzed": total_analyses,
                "tracking_window_seconds": self.tracking_window,
                "strategy_1_momentum_backrun": {
                    "optimal_hold_time_seconds": best_hold[0],
                    "expected_profit_pct": round(best_hold[1], 2),
                    "win_rate_pct": round(best_hold[2], 1),
                    "expected_value": round(strategy_1_ev, 2),
                    "is_profitable": strategy_1_ev > 0
                },
                "strategy_2_large_buy_following": {
                    "total_opportunities": len(all_large_buy_opps),
                    "opportunities_per_token": round(opps_per_token, 2) if len(all_large_buy_opps) > 0 else 0,
                    "optimal_hold_time_seconds": best_lb_hold[0] if len(all_large_buy_opps) > 0 else 0,
                    "expected_profit_pct": round(strategy_2_ev, 2),
                    "win_rate_pct": round(best_lb_hold[2], 1) if len(all_large_buy_opps) > 0 else 0,
                    "expected_value": round(strategy_2_ev, 2),
                    "is_profitable": strategy_2_ev > 0
                }
            },
            "token_analyses": [a.to_dict() for a in self.completed_analyses]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n📄 Detailed analysis exported to: {output_file}")
        logger.info(f"📋 Analyzed {total_analyses} complete tokens")
        logger.info("\n" + "="*80 + "\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mathematically prove strategy profitability",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--config", default="config/config.yml", help="Path to config.yml")
    parser.add_argument("--duration", type=int, default=30, help="Analysis duration in minutes (default: 30)")
    parser.add_argument("--tokens", type=int, default=20, help="Number of tokens to analyze (default: 20)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="json", output_file="logs/strategy_analysis.log")

    # Create and run analyzer
    analyzer = StrategyOpportunityAnalyzer(
        config_path=args.config,
        analysis_duration_minutes=args.duration,
        token_limit=args.tokens,
        tracking_window_seconds=60.0
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
