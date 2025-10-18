"""
LaserStream Bonding Curve Monitor - Real-time bonding curve state tracking via Geyser

This module monitors bonding curve state changes for active positions using Helius LaserStream.
Gets instant updates (30-100ms) when ANY trade happens on monitored tokens.
Uses the same price source for exit decisions and actual execution.

Based on working LaserStreamWalletMonitor implementation.
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

# Import proto files from frontrun directory (working implementation)
proto_path = Path(__file__).parent.parent.parent / "frontrun" / "proto"
sys.path.insert(0, str(proto_path))

try:
    import geyser_pb2
    import geyser_pb2_grpc
    GEYSER_AVAILABLE = True
except ImportError:
    GEYSER_AVAILABLE = False
    logging.warning("Geyser proto files not available - cannot use LaserStream bonding curve monitor")

# Import bonding curve calculator
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "frontrun"))
from core.bonding_curve import BondingCurveState, BondingCurveCalculator

# Import TradeEvent parsing from LaserStreamWalletMonitor
from .laserstream_wallet_monitor import (
    parse_trade_event_from_logs,
    HeliusAuthMetadata,
    WalletBuyEvent,
    PUMP_FUN_PROGRAM_ID
)


@dataclass
class BondingCurveUpdate:
    """Event emitted when bonding curve state changes"""
    mint: str
    slot: int
    timestamp: datetime
    virtual_sol_reserves: int
    virtual_token_reserves: int
    real_sol_reserves: int
    real_token_reserves: int
    is_buy: bool
    user: str
    sol_amount_ui: float
    token_amount_ui: float


class LaserStreamBondingCurveMonitor:
    """
    Unified real-time monitor using Helius LaserStream

    Handles BOTH:
    1. Alpha wallet buy detection (copy trading signals)
    2. Position monitoring with bonding curve price updates

    Gets instant updates (30-100ms) on every Pump.fun trade via TradeEvent.
    Uses bonding curve reserves to calculate EXACT sell prices for exit decisions.

    Replaces separate LaserStreamWalletMonitor to use single gRPC connection.
    """

    def __init__(self, config: Dict, trading_engine, watched_wallets: Optional[List[str]] = None):
        """
        Initialize unified LaserStream monitor

        Args:
            config: Configuration dict with:
                - helius_grpc_endpoint: LaserStream gRPC endpoint
                - helius_grpc_token: Authentication token
                - pump_fun_program: Pump.fun program address
            trading_engine: Trading engine instance for exit decisions
            watched_wallets: List of alpha wallet addresses to monitor for buys
        """
        self.logger = logging.getLogger(__name__)

        if not GEYSER_AVAILABLE:
            raise RuntimeError("Geyser proto files not available - cannot use LaserStream monitor")

        # Configuration
        self.grpc_endpoint = config.get('helius_grpc_endpoint')
        self.grpc_token = config.get('helius_grpc_token')
        self.pump_fun_program = config.get('pump_fun_program', PUMP_FUN_PROGRAM_ID)
        self.trading_engine = trading_engine

        if not self.grpc_endpoint or not self.grpc_token:
            raise ValueError("helius_grpc_endpoint and helius_grpc_token required for LaserStream monitor")

        # Strip https:// prefix if present
        if self.grpc_endpoint.startswith('https://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('https://', '')
        elif self.grpc_endpoint.startswith('http://'):
            self.grpc_endpoint = self.grpc_endpoint.replace('http://', '')

        # Add :443 port if not specified
        if ':' not in self.grpc_endpoint:
            self.grpc_endpoint = f"{self.grpc_endpoint}:443"

        # Monitored positions (bonding curve monitoring)
        self.monitored_positions = {}  # mint -> position_info

        # Watched wallets (alpha buy detection)
        self.watched_wallets = set(watched_wallets) if watched_wallets else set()

        # Connection state
        self.channel = None
        self.stub = None
        self.running = False

        # Bonding curve calculator
        self.calculator = BondingCurveCalculator()

        # Event callbacks
        self.on_wallet_buy: Optional[Callable[[WalletBuyEvent], None]] = None

        # Statistics
        self.updates_received = 0
        self.exit_checks_performed = 0
        self.wallet_buys_detected = 0
        self.stale_positions_dumped = 0
        self.connection_time = None

        # Stale position detection (dump dead tokens)
        self.stale_timeout_seconds = config.get('stale_position_timeout_minutes', 3) * 60  # Default: 3 minutes

        self.logger.info("LaserStream Unified Monitor initialized")
        self.logger.info(f"  Endpoint: {self.grpc_endpoint}")
        self.logger.info(f"  Watching {len(self.watched_wallets)} wallets for alpha buys")
        self.logger.info(f"  Pump.fun program: {self.pump_fun_program}")
        self.logger.info(f"  Stale position timeout: {self.stale_timeout_seconds / 60:.0f} minutes")

    async def start(self):
        """Start monitoring bonding curve changes"""
        if self.running:
            self.logger.warning("LaserStream bonding curve monitor already running")
            return

        self.running = True
        self.connection_time = datetime.now()

        try:
            # Start both stream processing AND stale position checker
            await asyncio.gather(
                self._connect_and_stream(),
                self._periodic_stale_check(),
                return_exceptions=True
            )
        except Exception as e:
            self.logger.error(f"LaserStream bonding curve monitor error: {e}")
            self.running = False
            raise

    async def add_position_for_monitoring(self, mint_address: str, position_info: Dict):
        """
        Add a position for real-time bonding curve monitoring

        Args:
            mint_address: Token mint address
            position_info: Position metadata (symbol, entry_price, tokens, etc.)
        """
        symbol = position_info.get('symbol', mint_address[:8])

        # Check if position exists in trading engine
        if mint_address not in self.trading_engine.active_positions:
            self.logger.error(f"Position {mint_address} not found in trading_engine!")
            return

        position = self.trading_engine.active_positions[mint_address]

        self.logger.info(
            f"⚡ LASERSTREAM MONITORING {symbol} "
            f"Entry: ${position.entry_price:.8f}, "
            f"Tokens: {position.tokens_initial:,.0f}"
        )

        # Add to our tracking with volume tracking
        self.monitored_positions[mint_address] = {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'tokens': position.tokens_initial,
            'start_time': datetime.now(),
            'last_activity_time': datetime.now(),  # Track last time we saw activity
            'updates_count': 0,
            # Volume tracking (30-second windows)
            'volume_history': [],  # List of (timestamp, sol_volume, buy_count, sell_count)
            'recent_volume_sol': 0,  # Rolling 30s volume
            'volume_trend': 'neutral',  # 'increasing', 'decreasing', 'neutral'
        }

    async def remove_position_monitoring(self, mint_address: str):
        """Remove a position from monitoring"""
        if mint_address in self.monitored_positions:
            symbol = self.monitored_positions[mint_address]['symbol']
            updates = self.monitored_positions[mint_address]['updates_count']
            del self.monitored_positions[mint_address]
            self.logger.info(f"⚡ Stopped LaserStream monitoring for {symbol} ({updates} updates received)")

    def add_wallet(self, wallet: str):
        """Add a wallet to the alpha watch list (dynamically, no restart needed)"""
        self.watched_wallets.add(wallet)
        self.logger.debug(f"Added wallet to alpha watch list: {wallet[:8]}... (total: {len(self.watched_wallets)})")

    def remove_wallet(self, wallet: str):
        """Remove a wallet from the alpha watch list"""
        if wallet in self.watched_wallets:
            self.watched_wallets.remove(wallet)
            self.logger.debug(f"Removed wallet from alpha watch list: {wallet[:8]}... (total: {len(self.watched_wallets)})")

    def update_wallets(self, wallets: List[str]):
        """Update the entire alpha wallet watch list (dynamically, no restart needed)"""
        old_count = len(self.watched_wallets)
        self.watched_wallets = set(wallets)
        self.logger.info(f"Updated alpha wallet watch list: {old_count} -> {len(self.watched_wallets)} wallets")

    async def _connect_and_stream(self):
        """Connect to LaserStream and process bonding curve updates (from working implementation)"""
        self.logger.info(f"Connecting to LaserStream at {self.grpc_endpoint}...")

        # Setup authentication (using working method)
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

        # Build subscription request
        subscription_request = self._build_subscription_request()

        self.logger.info("✅ LaserStream bonding curve monitor connected")
        self.logger.info("Subscribing to Pump.fun transactions for bonding curve monitoring...")

        try:
            # Create request iterator (gRPC streaming requires this)
            async def request_generator():
                # Send initial subscription request
                yield subscription_request
                # Keep stream open
                while self.running:
                    await asyncio.sleep(30)  # Keep-alive

            # Subscribe to transaction stream (bidirectional streaming)
            self.logger.info("✅ LaserStream bonding curve monitor active")

            async for message in self.stub.Subscribe(request_generator()):
                if not self.running:
                    break

                try:
                    await self._process_message(message)
                except Exception as e:
                    self.logger.error(f"Bonding curve update processing error: {e}")

        finally:
            if self.channel:
                await self.channel.close()
                self.channel = None
                self.stub = None

    def _build_subscription_request(self) -> geyser_pb2.SubscribeRequest:
        """
        Build Yellowstone subscription request for Pump.fun transactions
        (Using working implementation from LaserStreamWalletMonitor)

        Returns:
            Subscription request
        """
        # Use PROCESSED commitment for fastest updates (~30-100ms)
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
                "pump_fun_bonding_curve": tx_filter  # Subscription name
            },
            commitment=commitment
        )

        return request

    async def _process_message(self, message: geyser_pb2.SubscribeUpdate):
        """
        Process incoming LaserStream message

        Handles BOTH:
        1. Alpha wallet buy detection
        2. Position bonding curve monitoring

        Args:
            message: Message from Yellowstone stream
        """
        # Check if this is a transaction update
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
            elif hasattr(transaction.meta, 'logMessages'):
                logs = list(transaction.meta.logMessages)

        # Parse TradeEvent (using working method)
        trade_event = parse_trade_event_from_logs(logs)
        if not trade_event:
            return

        mint = trade_event['mint']
        wallet = trade_event['user']
        is_buy = trade_event['is_buy']

        # ============================================
        # 1. ALPHA WALLET BUY DETECTION
        # ============================================
        if is_buy and wallet in self.watched_wallets:
            # Create wallet buy event
            buy_event = WalletBuyEvent(
                wallet=wallet,
                mint=mint,
                sol_amount=trade_event['sol_amount_ui'],
                timestamp=datetime.now(),
                slot=slot,
                signature=signature,
                token_amount=trade_event['token_amount_ui'],
                virtual_sol_reserves=trade_event['virtual_sol_reserves'],
                virtual_token_reserves=trade_event['virtual_token_reserves']
            )

            self.wallet_buys_detected += 1

            self.logger.info(
                f"🎯 Alpha wallet buy detected: {wallet[:8]}... bought {mint[:8]}... "
                f"({trade_event['sol_amount_ui']:.3f} SOL) @ slot {slot}"
            )

            # Emit event to callback
            if self.on_wallet_buy:
                await self._safe_callback(self.on_wallet_buy, buy_event)

        # ============================================
        # 2. POSITION BONDING CURVE MONITORING
        # ============================================
        if mint in self.monitored_positions:
            # Update statistics
            self.updates_received += 1
            self.monitored_positions[mint]['updates_count'] += 1

            # CRITICAL: Update last activity timestamp (token is still alive!)
            now = datetime.now()
            self.monitored_positions[mint]['last_activity_time'] = now

            # UPDATE VOLUME TRACKING
            sol_volume = trade_event['sol_amount_ui']
            position_info = self.monitored_positions[mint]

            # Add this trade to volume history (timestamp, sol_volume, is_buy)
            position_info['volume_history'].append({
                'timestamp': now,
                'sol_volume': sol_volume,
                'is_buy': is_buy
            })

            # Clean old entries (keep only last 60 seconds of data for analysis)
            cutoff_time = now.timestamp() - 60
            position_info['volume_history'] = [
                entry for entry in position_info['volume_history']
                if entry['timestamp'].timestamp() > cutoff_time
            ]

            # Calculate rolling 30-second volume
            recent_cutoff = now.timestamp() - 30
            recent_trades = [
                entry for entry in position_info['volume_history']
                if entry['timestamp'].timestamp() > recent_cutoff
            ]
            position_info['recent_volume_sol'] = sum(t['sol_volume'] for t in recent_trades)

            # Calculate volume trend (compare last 15s to previous 15s)
            mid_cutoff = now.timestamp() - 15
            last_15s_volume = sum(t['sol_volume'] for t in recent_trades if t['timestamp'].timestamp() > mid_cutoff)
            prev_15s_volume = sum(t['sol_volume'] for t in recent_trades if t['timestamp'].timestamp() <= mid_cutoff)

            # Determine trend (need at least some volume to calculate)
            if last_15s_volume > 0 or prev_15s_volume > 0:
                if prev_15s_volume == 0:
                    position_info['volume_trend'] = 'increasing'
                elif last_15s_volume == 0:
                    position_info['volume_trend'] = 'decreasing'
                else:
                    change_pct = ((last_15s_volume / prev_15s_volume) - 1) * 100
                    if change_pct > 30:
                        position_info['volume_trend'] = 'increasing'
                    elif change_pct < -30:
                        position_info['volume_trend'] = 'decreasing'
                    else:
                        position_info['volume_trend'] = 'neutral'

            # Calculate buy/sell ratio (last 30s)
            buy_volume = sum(t['sol_volume'] for t in recent_trades if t['is_buy'])
            sell_volume = sum(t['sol_volume'] for t in recent_trades if not t['is_buy'])
            position_info['buy_sell_ratio'] = buy_volume / sell_volume if sell_volume > 0 else (10.0 if buy_volume > 0 else 1.0)

            # Create bonding curve update event
            curve_update = BondingCurveUpdate(
                mint=mint,
                slot=slot,
                timestamp=now,
                virtual_sol_reserves=trade_event['virtual_sol_reserves'],
                virtual_token_reserves=trade_event['virtual_token_reserves'],
                real_sol_reserves=0,  # Not in TradeEvent, but we don't need it for price calc
                real_token_reserves=0,  # Not in TradeEvent, but we don't need it for price calc
                is_buy=is_buy,
                user=wallet,
                sol_amount_ui=trade_event['sol_amount_ui'],
                token_amount_ui=trade_event['token_amount_ui']
            )

            # Process bonding curve update and check exit conditions
            await self._check_exit_conditions(curve_update)

    def _analyze_volume_for_exit(self, mint_address: str) -> Dict:
        """
        Analyze volume trend and return recommended exit strategy

        Args:
            mint_address: Token mint address

        Returns:
            Dict with:
                - tp1_exit_pct: Percentage to sell at TP1 (25% gain)
                - tp2_exit_pct: Percentage to sell at TP2 (60% gain)
                - tp3_exit_pct: Percentage to sell at TP3 (100% gain)
                - volume_signal: 'hot', 'warm', 'cooling', 'dead'
                - reasoning: Explanation of exit strategy
        """
        position_info = self.monitored_positions.get(mint_address)
        if not position_info:
            return {
                'tp1_exit_pct': 1.0,
                'tp2_exit_pct': 0.0,
                'tp3_exit_pct': 0.0,
                'volume_signal': 'unknown',
                'reasoning': 'No position info'
            }

        # CRITICAL: Check if we have enough data to make volume-based decisions
        # We need at least 30 seconds of monitoring AFTER our buy to have a full 30s window
        start_time = position_info.get('start_time')
        time_since_entry = (datetime.now() - start_time).total_seconds() if start_time else 0

        # If we haven't been monitoring for at least 30 seconds, use default config exits
        if time_since_entry < 30:
            return {
                'tp1_exit_pct': 1.0,  # Use config default (100% at TP1)
                'tp2_exit_pct': 0.0,
                'tp3_exit_pct': 0.0,
                'volume_signal': 'insufficient_data',
                'reasoning': f'Insufficient volume data ({time_since_entry:.0f}s < 30s required) - using config defaults',
                'volume_sol_30s': 0,
                'trend': 'unknown',
                'buy_sell_ratio': 1.0
            }

        recent_volume = position_info.get('recent_volume_sol', 0)
        volume_trend = position_info.get('volume_trend', 'neutral')
        buy_sell_ratio = position_info.get('buy_sell_ratio', 1.0)
        trade_count = len(position_info.get('volume_history', []))

        # Analyze volume quality
        # HIGH VOLUME + INCREASING + BUY PRESSURE = Let it run (small exits)
        # LOW VOLUME + DECREASING = Get out fast (large exits)

        # Volume thresholds (SOL in last 30s)
        HOT_VOLUME = 5.0      # >5 SOL/30s = very active
        WARM_VOLUME = 2.0     # 2-5 SOL/30s = decent activity
        COOL_VOLUME = 0.5     # 0.5-2 SOL/30s = slow
        # <0.5 SOL/30s = dying

        # Determine volume signal
        if recent_volume > HOT_VOLUME and volume_trend == 'increasing' and buy_sell_ratio > 1.5:
            # HOT: High volume, increasing, buy pressure
            volume_signal = 'hot'
            tp1_exit_pct = 0.25  # Sell only 25% at TP1 - let winners run!
            tp2_exit_pct = 0.35  # Sell 35% at TP2
            tp3_exit_pct = 0.40  # Sell remaining 40% at TP3
            reasoning = f"Hot token ({recent_volume:.1f} SOL/30s, +trend, {buy_sell_ratio:.1f}x buy pressure) - small exits, let it run"

        elif recent_volume > WARM_VOLUME and volume_trend != 'decreasing':
            # WARM: Decent volume, stable or increasing
            volume_signal = 'warm'
            tp1_exit_pct = 0.50  # Sell 50% at TP1 - balanced approach
            tp2_exit_pct = 0.30  # Sell 30% at TP2
            tp3_exit_pct = 0.20  # Hold 20% for moon shot
            reasoning = f"Warm token ({recent_volume:.1f} SOL/30s, {volume_trend} trend) - balanced exits"

        elif recent_volume > COOL_VOLUME:
            # COOLING: Low but existing volume
            volume_signal = 'cooling'
            tp1_exit_pct = 0.75  # Sell 75% at TP1 - take profits
            tp2_exit_pct = 0.25  # Sell remaining 25% at TP2
            tp3_exit_pct = 0.0   # Don't wait for TP3
            reasoning = f"Cooling token ({recent_volume:.1f} SOL/30s, {volume_trend} trend) - take profits early"

        else:
            # DEAD: No volume
            volume_signal = 'dead'
            tp1_exit_pct = 1.0  # Sell 100% immediately - get out!
            tp2_exit_pct = 0.0
            tp3_exit_pct = 0.0
            reasoning = f"Dead token ({recent_volume:.1f} SOL/30s, {trade_count} trades) - exit immediately"

        return {
            'tp1_exit_pct': tp1_exit_pct,
            'tp2_exit_pct': tp2_exit_pct,
            'tp3_exit_pct': tp3_exit_pct,
            'volume_signal': volume_signal,
            'reasoning': reasoning,
            'volume_sol_30s': recent_volume,
            'trend': volume_trend,
            'buy_sell_ratio': buy_sell_ratio
        }

    async def _safe_callback(self, callback: Callable, event: WalletBuyEvent):
        """Safely execute callback without blocking stream processing"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
        except Exception as e:
            self.logger.error(f"Error in wallet buy callback: {e}")

    async def _periodic_stale_check(self):
        """
        Periodic checker for stale positions (no LaserStream activity)

        If a position hasn't seen ANY trading activity in X minutes, it's dead.
        Dump it at market to free up capital.
        """
        self.logger.info(f"🕐 Stale position checker started (timeout: {self.stale_timeout_seconds / 60:.0f}min)")

        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.now()
                stale_positions = []

                # Find all positions that haven't had activity in X minutes
                for mint_address, position_info in list(self.monitored_positions.items()):
                    last_activity = position_info.get('last_activity_time', position_info.get('start_time'))
                    time_since_activity = (now - last_activity).total_seconds()

                    if time_since_activity > self.stale_timeout_seconds:
                        symbol = position_info['symbol']
                        updates_count = position_info['updates_count']
                        minutes_stale = time_since_activity / 60

                        self.logger.warning(
                            f"💀 STALE POSITION DETECTED: {symbol} "
                            f"({minutes_stale:.1f}min since last activity, {updates_count} total updates)"
                        )

                        stale_positions.append((mint_address, symbol, minutes_stale))

                # Dump all stale positions at market
                for mint_address, symbol, minutes_stale in stale_positions:
                    try:
                        # Check if position still exists (might have been sold already)
                        if mint_address not in self.trading_engine.active_positions:
                            self.logger.info(f"Position {symbol} already closed - removing from monitoring")
                            if mint_address in self.monitored_positions:
                                del self.monitored_positions[mint_address]
                            continue

                        position = self.trading_engine.active_positions[mint_address]

                        # Calculate current loss (assume we'd get very little for a dead token)
                        # We don't have a current price, so just sell at market
                        self.logger.warning(
                            f"💀 DUMPING STALE POSITION: {symbol} "
                            f"(NO ACTIVITY FOR {minutes_stale:.1f}min - TOKEN IS DEAD)"
                        )

                        # Sell 100% at market (low priority - no rush for dead tokens)
                        # Uses minimal priority fee (~1,000 lamports) and skips Jito bundle
                        # No one is trading this token, so we don't need speed - saves $1-2 per dump!
                        sell_result = await self.trading_engine.sell_token(
                            mint_address,
                            1.0,  # 100% sell
                            self.trading_engine.config.paper_mode,
                            symbol=symbol,
                            exit_reason="stale_position",
                            current_price=0,  # Let sell_token query current price
                            low_priority=True  # Use minimal fees for dead tokens
                        )

                        if sell_result.get("success"):
                            self.logger.warning(
                                f"💀 STALE POSITION DUMPED: {symbol} "
                                f"(Proceeds: ${sell_result.get('usd_value', 0):.2f})"
                            )
                            self.stale_positions_dumped += 1

                            # Remove from monitoring
                            if mint_address in self.monitored_positions:
                                del self.monitored_positions[mint_address]
                        else:
                            self.logger.error(
                                f"Failed to dump stale position {symbol}: {sell_result.get('error')}"
                            )

                    except Exception as e:
                        self.logger.error(f"Error dumping stale position {symbol}: {e}")

            except Exception as e:
                self.logger.error(f"Stale position checker error: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _check_exit_conditions(self, curve_update: BondingCurveUpdate):
        """
        Check exit conditions using real-time bonding curve state

        This uses the EXACT same price calculation as our actual sell execution,
        eliminating the price discrepancy issue.

        NOW WITH VOLUME-BASED EXIT STRATEGY:
        - High volume tokens: Small exits (25% at TP1) - let winners run
        - Low volume tokens: Large exits (75-100% at TP1) - get out fast

        Args:
            curve_update: Bonding curve state update
        """
        mint_address = curve_update.mint

        if mint_address not in self.trading_engine.active_positions:
            return

        position = self.trading_engine.active_positions[mint_address]
        position_info = self.monitored_positions[mint_address]
        symbol = position_info['symbol']

        # Create BondingCurveState from update
        curve_state = BondingCurveState(
            virtual_token_reserves=curve_update.virtual_token_reserves,
            virtual_sol_reserves=curve_update.virtual_sol_reserves,
            real_token_reserves=curve_update.real_token_reserves,
            real_sol_reserves=curve_update.real_sol_reserves,
            token_total_supply=1_000_000_000_000_000,  # 1B tokens (Pump.fun standard)
            complete=False
        )

        # Calculate EXACT SOL we'd receive if we sold right now
        # This is the SAME calculation used in actual sell execution
        # Use CURRENT token amount (position.amount), not initial amount
        token_amount_raw = int(position.amount * 1e6)  # Convert UI to raw (6 decimals)

        try:
            quote = self.calculator.calculate_sell_price(curve_state, token_amount_raw)
            expected_sol = quote.sol_out / 1e9  # Convert lamports to SOL

            # Get SOL price for USD conversion
            sol_usd = getattr(self.trading_engine, 'sol_price', 209.0)
            expected_usd = expected_sol * sol_usd

            # Calculate P&L based on ACTUAL sell price (not spot price)
            cost_basis_sol = position.sol_invested
            pnl_pct = ((expected_sol / cost_basis_sol) - 1) * 100 if cost_basis_sol > 0 else 0

            # Update peak price tracking
            # Use CURRENT token amount for accurate price per token
            current_price_per_token = expected_usd / position.amount if position.amount > 0 else 0
            if current_price_per_token > position.peak_price:
                position.peak_price = current_price_per_token
                peak_gain = ((current_price_per_token / position.entry_price) - 1) * 100
                position.high_gain_peak = max(position.high_gain_peak, peak_gain)

            # Get volume-based exit strategy
            volume_analysis = self._analyze_volume_for_exit(mint_address)

            # Log price check with volume info (every update)
            self.logger.debug(
                f"⚡ {symbol} @ slot {curve_update.slot}: "
                f"Would receive {expected_sol:.6f} SOL (${expected_usd:.2f}) = {pnl_pct:+.1f}% P&L | "
                f"Peak: {position.high_gain_peak:.1f}% | "
                f"Vol: {volume_analysis['volume_signal']} ({volume_analysis['volume_sol_30s']:.1f} SOL/30s)"
            )

            self.exit_checks_performed += 1

            # Check exit conditions using existing trading engine logic
            # Pass the ACTUAL price we'd get from selling
            exit_result = await self.trading_engine.check_exit_conditions(
                mint_address,
                current_price_per_token
            )

            # Early return if no exit signal
            if not exit_result:
                return

            # Unpack exit result
            exit_reason, config_sell_percentage = exit_result

            # CRITICAL: Check if we have enough volume data for smart exits
            # For TP exits, trailing stops, AND ATR stops: WAIT for verification + volume data
            # For stop-loss/velocity/break-even exits: Execute IMMEDIATELY (risk mitigation)
            volume_signal = volume_analysis['volume_signal']
            is_tp_exit = any(tp in exit_reason.lower() for tp in ['tp1', 'tp2', 'tp3', 'take_profit', 'trailing', 'atr_stop'])
            is_risk_exit = any(risk in exit_reason.lower() for risk in ['stop_loss', 'velocity', 'stale', 'break_even', 'catastrophe'])

            # CRITICAL: Block TP exits until BOTH conditions met:
            # 1. Entry costs verified (prevent false profit calculations)
            # 2. 30s volume data collected (enable smart exit strategy)
            # Stop-loss exits are ALWAYS immediate (risk protection)

            # Check if entry costs have been verified
            is_verified = getattr(position, 'actual_costs_verified', False)

            if is_tp_exit:
                # Block TP exits until entry verification completes
                if not is_verified:
                    time_since_entry = (datetime.now() - position_info['start_time']).total_seconds()
                    self.logger.info(
                        f"⏳ {symbol} SKIP {exit_reason}: Entry costs UNVERIFIED "
                        f"(held {time_since_entry:.0f}s, waiting for blockchain confirmation) | "
                        f"Estimated P&L: {pnl_pct:+.1f}% (may be incorrect)"
                    )
                    return  # Skip this TP exit, wait for verification

                # Also block TP exits until we have volume data
                if volume_signal == 'insufficient_data':
                    time_since_entry = (datetime.now() - position_info['start_time']).total_seconds()
                    remaining_time = 30 - time_since_entry

                    self.logger.info(
                        f"⏳ {symbol} SKIP {exit_reason}: Need {remaining_time:.0f}s more volume data "
                        f"(held {time_since_entry:.0f}s / 30s min for TP exits) | "
                        f"Current: {pnl_pct:+.1f}% P&L"
                    )
                    return  # Skip this TP exit, wait for volume data

            # Check if token is HOT or WARM (climbing with volume)
            is_climbing = volume_signal in ['hot', 'warm']

            # CLIMBING TOKEN: Skip TP exits, use trailing stop instead
            if is_tp_exit and is_climbing:
                # Calculate trailing stop (15% drop from peak for hot, 20% for warm)
                trailing_stop_pct = 15 if volume_signal == 'hot' else 20
                peak_gain = position.high_gain_peak
                current_gain = ((current_price_per_token / position.entry_price) - 1) * 100
                drop_from_peak = peak_gain - current_gain

                if drop_from_peak < trailing_stop_pct:
                    # Still climbing or within trailing stop tolerance - SKIP this TP exit
                    self.logger.info(
                        f"⚡ {symbol} SKIP {exit_reason}: {volume_signal.upper()} token climbing "
                        f"(peak: +{peak_gain:.1f}%, current: +{current_gain:.1f}%, drop: {drop_from_peak:.1f}% < {trailing_stop_pct}% trailing stop) | "
                        f"Volume: {volume_analysis['reasoning']}"
                    )
                    return  # Skip this exit, let it run
                else:
                    # Trailing stop triggered - sell 100%
                    actual_sell_pct = 1.0
                    exit_reason = f"trailing_stop_{volume_signal}"
                    self.logger.info(
                        f"⚡ LASERSTREAM TRAILING STOP: {symbol} dropped {drop_from_peak:.1f}% from peak +{peak_gain:.1f}% "
                        f"(trigger: {trailing_stop_pct}%) - selling 100% @ slot {curve_update.slot} | "
                        f"Volume: {volume_analysis['volume_signal'].upper()}"
                    )
            else:
                # COOLING/DEAD token or non-TP exit: Use volume-based percentages
                if 'tp1' in exit_reason.lower():
                    actual_sell_pct = volume_analysis['tp1_exit_pct']
                elif 'tp2' in exit_reason.lower():
                    actual_sell_pct = volume_analysis['tp2_exit_pct']
                elif 'tp3' in exit_reason.lower():
                    actual_sell_pct = volume_analysis['tp3_exit_pct']
                else:
                    # For stop-loss or other exits, use config value
                    actual_sell_pct = config_sell_percentage

                self.logger.info(
                    f"⚡ LASERSTREAM EXIT: {symbol} {exit_reason} "
                    f"{actual_sell_pct*100:.0f}% @ slot {curve_update.slot} | "
                    f"Would receive {expected_sol:.6f} SOL ({pnl_pct:+.1f}% P&L) | "
                    f"Volume: {volume_analysis['volume_signal'].upper()} - {volume_analysis['reasoning']}"
                )

            # PRICE DISCREPANCY PROTECTION: Re-check price right before execution
            # If price dropped significantly in the last second, skip TP exits (natural volatility)
            # Stop-loss exits execute immediately (risk protection)
            try:
                latest_quote = self.calculator.calculate_sell_price(curve_state, token_amount_raw)
                latest_sol = latest_quote.sol_out / 1e9
                latest_usd = latest_sol * sol_usd
                latest_pnl_pct = ((latest_sol / cost_basis_sol) - 1) * 100 if cost_basis_sol > 0 else 0

                # Calculate price change from our exit decision
                price_change_pct = ((latest_sol / expected_sol) - 1) * 100 if expected_sol > 0 else 0

                # For TP exits: Skip if price dropped >3% (volatile, wait for next update)
                # For stop-loss: Execute immediately regardless
                if is_tp_exit and price_change_pct < -3.0:
                    self.logger.warning(
                        f"⚠️ {symbol} SKIP {exit_reason}: Price dropped {price_change_pct:.1f}% since exit signal "
                        f"({expected_sol:.6f} → {latest_sol:.6f} SOL) - waiting for stability"
                    )
                    return  # Skip this exit, price is too volatile

                # Update expected values with latest quote
                expected_sol = latest_sol
                expected_usd = latest_usd
                pnl_pct = latest_pnl_pct

            except Exception as e:
                self.logger.warning(f"Could not re-check price for {symbol}: {e} - proceeding with original quote")

            # Execute the exit
            sell_result = await self.trading_engine.sell_token(
                mint_address,
                actual_sell_pct,
                self.trading_engine.config.paper_mode,
                symbol=symbol,
                exit_reason=exit_reason,
                current_price=current_price_per_token
            )

            if sell_result.get("success"):
                self.logger.info(
                    f"⚡ LASERSTREAM SELL SUCCESS: {symbol} executed "
                    f"({actual_sell_pct*100:.0f}% sold - {volume_analysis['volume_signal']} token)"
                )
            else:
                self.logger.error(f"⚡ LASERSTREAM SELL FAILED: {symbol} - {sell_result.get('error')}")

        except Exception as e:
            self.logger.error(f"Error checking exit conditions for {symbol}: {e}")

    def is_monitoring(self, mint_address: str) -> bool:
        """Check if a token is being monitored"""
        return mint_address in self.monitored_positions

    def get_monitored_positions(self) -> Dict:
        """Get all monitored positions"""
        return self.monitored_positions.copy()

    async def stop(self):
        """Stop monitoring"""
        self.running = False

        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None

        uptime = (datetime.now() - self.connection_time).total_seconds() if self.connection_time else 0

        self.logger.info("LaserStream unified monitor stopped")
        self.logger.info(f"  Uptime: {uptime:.1f}s")
        self.logger.info(f"  Wallet buys detected: {self.wallet_buys_detected:,}")
        self.logger.info(f"  Position updates received: {self.updates_received:,}")
        self.logger.info(f"  Exit checks performed: {self.exit_checks_performed:,}")
        self.logger.info(f"  Stale positions dumped: {self.stale_positions_dumped:,}")

    def get_stats(self) -> Dict:
        """Get monitor statistics"""
        uptime = (datetime.now() - self.connection_time).total_seconds() if self.connection_time else 0

        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'watched_wallets': len(self.watched_wallets),
            'monitored_positions': len(self.monitored_positions),
            'wallet_buys_detected': self.wallet_buys_detected,
            'updates_received': self.updates_received,
            'exit_checks_performed': self.exit_checks_performed,
            'stale_positions_dumped': self.stale_positions_dumped,
            'updates_per_minute': (self.updates_received / (uptime / 60)) if uptime > 0 else 0,
            'buys_per_minute': (self.wallet_buys_detected / (uptime / 60)) if uptime > 0 else 0
        }
