"""
Opportunity Tracker - Track all alpha signals for performance analysis

Tracks every alpha wallet buy signal to analyze:
- How many opportunities at different thresholds (1, 2, 3, 4+ wallets)
- Token price performance after the signal
- Optimal entry threshold based on historical data
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class PriceSnapshot:
    """Price snapshot at a specific time"""
    timestamp: float
    price_usd: float
    seconds_after_signal: float

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'price_usd': self.price_usd,
            'seconds_after_signal': self.seconds_after_signal
        }


@dataclass
class ThresholdEntry:
    """Entry point at a specific wallet threshold"""
    wallet_count: int
    timestamp: float
    price_usd: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'wallet_count': self.wallet_count,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'price_usd': self.price_usd,
            'seconds_after_first_signal': self.timestamp  # Will be calculated relative
        }


@dataclass
class AlphaSignal:
    """Single alpha wallet buy signal with price tracking"""
    token_mint: str
    timestamp: float
    wallet_count: int
    wallets: List[str]
    sol_amounts: List[float]
    virtual_sol_reserves: Optional[int] = None
    virtual_token_reserves: Optional[int] = None
    entry_price_usd: Optional[float] = None  # Deprecated - use threshold_entries instead
    price_snapshots: List[PriceSnapshot] = None
    threshold_entries: List[ThresholdEntry] = None  # Entry points at each threshold

    def __post_init__(self):
        if self.price_snapshots is None:
            self.price_snapshots = []
        if self.threshold_entries is None:
            self.threshold_entries = []

    def to_dict(self) -> Dict:
        data = {
            'token_mint': self.token_mint,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'wallet_count': self.wallet_count,
            'wallets': self.wallets,
            'sol_amounts': self.sol_amounts,
            'total_sol': sum(self.sol_amounts),
            'virtual_sol_reserves': self.virtual_sol_reserves,
            'virtual_token_reserves': self.virtual_token_reserves,
            'entry_price_usd': self.entry_price_usd,  # Deprecated - for backward compatibility
            'price_snapshots': [s.to_dict() for s in self.price_snapshots],
            'threshold_entries': []
        }

        # Add threshold entry data with relative timestamps
        for entry in self.threshold_entries:
            entry_dict = entry.to_dict()
            entry_dict['seconds_after_first_signal'] = entry.timestamp - self.timestamp
            data['threshold_entries'].append(entry_dict)

        # Calculate performance metrics for each threshold entry
        if self.price_snapshots:
            peak_price = max(s.price_usd for s in self.price_snapshots)
            latest_price = self.price_snapshots[-1].price_usd

            # Overall metrics (using first threshold entry)
            data['peak_price_usd'] = peak_price
            data['latest_price_usd'] = latest_price

            # Performance by threshold
            threshold_performance = {}
            for entry in self.threshold_entries:
                if entry.price_usd and entry.price_usd > 0:
                    threshold = entry.wallet_count

                    # Calculate gains from this entry point
                    peak_gain = ((peak_price / entry.price_usd) - 1) * 100
                    current_gain = ((latest_price / entry.price_usd) - 1) * 100

                    threshold_performance[f'threshold_{threshold}'] = {
                        'entry_price_usd': entry.price_usd,
                        'peak_gain_pct': peak_gain,
                        'current_gain_pct': current_gain
                    }

            data['threshold_performance'] = threshold_performance

            # Backward compatibility - use 1-wallet threshold as default
            if 1 in [e.wallet_count for e in self.threshold_entries]:
                entry_1 = next(e for e in self.threshold_entries if e.wallet_count == 1)
                if entry_1.price_usd and entry_1.price_usd > 0:
                    data['peak_gain_pct'] = ((peak_price / entry_1.price_usd) - 1) * 100
                    data['current_gain_pct'] = ((latest_price / entry_1.price_usd) - 1) * 100

        return data


class OpportunityTracker:
    """
    Track all alpha signals for performance analysis

    Records every token that gets alpha wallet buys, regardless of whether
    we execute a trade. Later analysis will show which thresholds work best.
    """

    def __init__(self, data_dir: str = "data/opportunities", moralis_client=None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.moralis_client = moralis_client

        # Active signals - keyed by token mint
        self.active_signals: Dict[str, AlphaSignal] = {}

        # Track active price polling tasks
        self.price_tracking_tasks: Dict[str, asyncio.Task] = {}

        # Session file for current run
        self.session_start = datetime.now()
        self.session_file = self.data_dir / f"signals_{self.session_start.strftime('%Y%m%d_%H%M%S')}.json"

        self.logger.info(f"OpportunityTracker initialized - logging to {self.session_file}")

    def record_alpha_buy(self, token_mint: str, wallet: str, sol_amount: float,
                        virtual_sol_reserves: Optional[int] = None,
                        virtual_token_reserves: Optional[int] = None) -> int:
        """
        Record an alpha wallet buy signal

        Args:
            token_mint: Token mint address
            wallet: Wallet that bought
            sol_amount: Amount of SOL used
            virtual_sol_reserves: Current SOL reserves
            virtual_token_reserves: Current token reserves

        Returns:
            Current wallet count for this token
        """
        timestamp = datetime.now().timestamp()
        is_new_token = token_mint not in self.active_signals

        if is_new_token:
            # First signal for this token
            self.active_signals[token_mint] = AlphaSignal(
                token_mint=token_mint,
                timestamp=timestamp,
                wallet_count=1,
                wallets=[wallet],
                sol_amounts=[sol_amount],
                virtual_sol_reserves=virtual_sol_reserves,
                virtual_token_reserves=virtual_token_reserves
            )
            wallet_count = 1

            # Start price tracking for new token
            if self.moralis_client:
                self._start_price_tracking(token_mint)
            else:
                self.logger.debug(f"No Moralis client - skipping price tracking for {token_mint[:8]}...")
        else:
            # Additional signal for existing token
            signal = self.active_signals[token_mint]

            # Only add if wallet hasn't bought yet (avoid double counting)
            if wallet not in signal.wallets:
                signal.wallet_count += 1
                signal.wallets.append(wallet)
                signal.sol_amounts.append(sol_amount)

                # Update reserves with latest data
                if virtual_sol_reserves:
                    signal.virtual_sol_reserves = virtual_sol_reserves
                if virtual_token_reserves:
                    signal.virtual_token_reserves = virtual_token_reserves

                # Record threshold entry point for this wallet count
                # This will be populated with price asynchronously
                threshold_entry = ThresholdEntry(
                    wallet_count=signal.wallet_count,
                    timestamp=timestamp,
                    price_usd=None  # Will be fetched asynchronously
                )
                signal.threshold_entries.append(threshold_entry)

                # Fetch price for this threshold asynchronously
                if self.moralis_client:
                    asyncio.create_task(self._fetch_threshold_price(token_mint, signal.wallet_count))

            wallet_count = signal.wallet_count

        # Save after each update
        self._save_signals()

        return wallet_count

    def _save_signals(self):
        """Save all signals to JSON file"""
        try:
            signals_data = {
                'session_start': self.session_start.isoformat(),
                'last_update': datetime.now().isoformat(),
                'total_tokens': len(self.active_signals),
                'signals': [signal.to_dict() for signal in self.active_signals.values()]
            }

            with open(self.session_file, 'w') as f:
                json.dump(signals_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save signals: {e}")

    def _start_price_tracking(self, token_mint: str):
        """Start background price tracking for a token"""
        if token_mint in self.price_tracking_tasks:
            self.logger.debug(f"Price tracking already active for {token_mint[:8]}...")
            return

        # Create and store the tracking task
        task = asyncio.create_task(self._price_polling_task(token_mint))
        self.price_tracking_tasks[token_mint] = task
        self.logger.info(f"Started price tracking for {token_mint[:8]}...")

    async def _fetch_threshold_price(self, token_mint: str, wallet_count: int):
        """Fetch and record price for a specific threshold entry"""
        try:
            signal = self.active_signals.get(token_mint)
            if not signal:
                return

            # Wait 2 seconds (conservative entry delay)
            await asyncio.sleep(2)

            # Get current price
            price = await self.moralis_client.get_current_price(token_mint, fresh=True)

            if price > 0:
                # Find the threshold entry and update its price
                for entry in signal.threshold_entries:
                    if entry.wallet_count == wallet_count and entry.price_usd is None:
                        entry.price_usd = price
                        seconds_after = entry.timestamp - signal.timestamp
                        self.logger.info(
                            f"Threshold {wallet_count} entry for {token_mint[:8]}: "
                            f"${price:.8f} (+{seconds_after:.1f}s)"
                        )
                        self._save_signals()
                        break
            else:
                self.logger.warning(f"Failed to get price for threshold {wallet_count} on {token_mint[:8]}")

        except Exception as e:
            self.logger.error(f"Error fetching threshold price: {e}")

    async def _price_polling_task(self, token_mint: str):
        """
        Background task to poll token price at intervals

        Polls at: entry (+2s conservative), +30s, +1m, +2m, +5m, +10m, +15m, +30m
        """
        try:
            signal = self.active_signals.get(token_mint)
            if not signal:
                return

            signal_time = signal.timestamp

            # Wait 2 seconds after first signal (conservative entry)
            await asyncio.sleep(2)

            # Get entry price (for backward compatibility and first threshold)
            entry_price = await self.moralis_client.get_current_price(token_mint, fresh=True)
            if entry_price > 0:
                signal.entry_price_usd = entry_price

                # Record first threshold entry (wallet_count = 1)
                threshold_entry = ThresholdEntry(
                    wallet_count=1,
                    timestamp=signal.timestamp,
                    price_usd=entry_price
                )
                signal.threshold_entries.append(threshold_entry)

                # Add to price snapshots for ongoing tracking
                snapshot = PriceSnapshot(
                    timestamp=datetime.now().timestamp(),
                    price_usd=entry_price,
                    seconds_after_signal=2.0
                )
                signal.price_snapshots.append(snapshot)
                self.logger.info(f"Entry price for {token_mint[:8]}: ${entry_price:.8f}")
                self._save_signals()
            else:
                self.logger.warning(f"Failed to get entry price for {token_mint[:8]}")

            # Polling intervals in seconds: 30s, 1m, 2m, 5m, 10m, 15m, 30m
            intervals = [30, 60, 120, 300, 600, 900, 1800]
            last_poll_time = 2  # We already polled at 2 seconds

            for interval in intervals:
                # Calculate how long to wait
                wait_time = interval - last_poll_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                # Poll current price
                current_time = datetime.now().timestamp()
                seconds_after = current_time - signal_time

                price = await self.moralis_client.get_current_price(token_mint, fresh=True)
                if price > 0:
                    snapshot = PriceSnapshot(
                        timestamp=current_time,
                        price_usd=price,
                        seconds_after_signal=seconds_after
                    )
                    signal.price_snapshots.append(snapshot)

                    # Calculate gain if we have entry price
                    if signal.entry_price_usd:
                        gain_pct = ((price / signal.entry_price_usd) - 1) * 100
                        self.logger.info(
                            f"Price check {token_mint[:8]} @ +{int(seconds_after)}s: "
                            f"${price:.8f} ({gain_pct:+.2f}%)"
                        )

                    self._save_signals()

                last_poll_time = interval

            self.logger.info(f"Price tracking completed for {token_mint[:8]}")

        except asyncio.CancelledError:
            self.logger.debug(f"Price tracking cancelled for {token_mint[:8]}")
        except Exception as e:
            self.logger.error(f"Error in price tracking for {token_mint[:8]}: {e}")
        finally:
            # Clean up task reference
            if token_mint in self.price_tracking_tasks:
                del self.price_tracking_tasks[token_mint]

    def get_signal_count(self, token_mint: str) -> int:
        """Get current wallet count for a token"""
        if token_mint in self.active_signals:
            return self.active_signals[token_mint].wallet_count
        return 0

    def get_signals_by_threshold(self, min_wallets: int) -> List[AlphaSignal]:
        """Get all signals that meet a wallet threshold"""
        return [
            signal for signal in self.active_signals.values()
            if signal.wallet_count >= min_wallets
        ]

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.active_signals:
            return {
                'total_tokens': 0,
                'by_threshold': {}
            }

        # Count tokens by threshold
        by_threshold = {}
        for threshold in [1, 2, 3, 4, 5]:
            count = len([s for s in self.active_signals.values() if s.wallet_count >= threshold])
            by_threshold[f"{threshold}_plus"] = count

        return {
            'session_start': self.session_start.isoformat(),
            'runtime_minutes': (datetime.now() - self.session_start).total_seconds() / 60,
            'total_tokens': len(self.active_signals),
            'by_threshold': by_threshold,
            'session_file': str(self.session_file)
        }

    def log_summary(self):
        """Log a summary of tracked opportunities"""
        summary = self.get_summary()

        if summary['total_tokens'] == 0:
            self.logger.info("No opportunities tracked yet")
            return

        self.logger.info("=" * 60)
        self.logger.info("📊 OPPORTUNITY TRACKER SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Session runtime: {summary['runtime_minutes']:.1f} minutes")
        self.logger.info(f"Total unique tokens: {summary['total_tokens']}")
        self.logger.info("")
        self.logger.info("Opportunities by threshold:")
        for threshold, count in summary['by_threshold'].items():
            self.logger.info(f"  {threshold.replace('_', ' ')}: {count} tokens")
        self.logger.info("")
        self.logger.info(f"Data saved to: {summary['session_file']}")
        self.logger.info("=" * 60)
