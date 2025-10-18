"""
Wallet classification engine with rule-based labeling

Production-ready version with:
- Adaptive thresholds (rolling percentiles)
- Funding graph (RPC-based sybil detection)
- Feature-based confidence scoring (not len(reasons) * constant)
"""

from typing import Dict, Set, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import logging

from anti_bot.core.types import (
    WalletProfile, WalletLabel, MintState, TxFeatures,
    BurstEvent, ClusterProfile, ClassificationMetrics
)
from anti_bot.core.feature_extractor import calculate_gini_coefficient
from anti_bot.core.adaptive_thresholds import (
    AdaptiveThresholdManager,
    calculate_inter_arrival_cv,
    calculate_smallest_k_mass,
    calculate_preferred_delay_std
)
from anti_bot.core.funding_graph import FundingGraph, detect_coordinated_cluster
from anti_bot.core.confidence_scoring import (
    score_sybil,
    score_metric_bot,
    score_priority_racer,
    score_early_exit
)


# Classification thresholds (tunable)
class Thresholds:
    # Burst detection
    BURST_MIN_BUYS = 25
    BURST_MIN_SOL = 4.0
    BURST_WINDOW_SECONDS = 3.0

    # Sybil detection
    SYBIL_MIN_WALLETS = 6
    SYBIL_FUNDING_WINDOW_SECONDS = 90.0
    SYBIL_BUY_WINDOW_SECONDS = 5.0
    SYBIL_MAX_BUY_SIZE = 0.02  # SOL

    # Metric bot detection
    METRIC_BOT_MIN_OCCURRENCES = 5  # Lowered from 7 to 5
    METRIC_BOT_DELAY_MIN = 0.5  # Expanded from 1.0 to 0.5
    METRIC_BOT_DELAY_MAX = 5.0  # Expanded from 4.0 to 5.0
    METRIC_BOT_MEDIAN_HOLD = 25.0  # seconds

    # Copy bot detection
    COPY_BOT_MIN_OCCURRENCES = 6
    COPY_BOT_DELAY_MIN = 0.5
    COPY_BOT_DELAY_MAX = 2.0

    # Priority racer detection
    PRIORITY_RACER_CU_MULTIPLIER = 1.3  # p90 + 30%
    PRIORITY_RACER_MIN_ENTRY_INDEX = 2  # Median entry index <= 2

    # Organic cohort
    ORGANIC_MIN_UNIQUE_BUYERS = 30
    ORGANIC_TIME_WINDOW = 30.0  # seconds
    ORGANIC_GINI_MIN = 0.35
    ORGANIC_GINI_MAX = 0.65
    ORGANIC_SYBIL_RATIO_MAX = 0.25
    ORGANIC_MEDIAN_HOLD = 120.0  # seconds

    # Early exit
    EARLY_EXIT_NO_SECOND_BURST = 30.0  # seconds
    EARLY_EXIT_MIN_UNIQUE_BUYERS = 12
    EARLY_EXIT_MAX_SYBIL_RATIO = 0.5

    # Cluster detection
    CLUSTER_MIN_MINTS_PER_DAY = 3


class WalletClassifier:
    """Maintains wallet profiles and classifies based on behavior patterns

    Production-ready with adaptive thresholds and funding graph analysis
    """

    def __init__(self, rpc_manager=None):
        """
        Args:
            rpc_manager: Optional RPC manager for funding graph lookups
        """
        # State tracking
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.mint_states: Dict[str, MintState] = {}
        self.bursts: Dict[str, List[BurstEvent]] = defaultdict(list)  # mint -> bursts
        self.clusters: Dict[str, ClusterProfile] = {}

        # Rolling windows for time-based features
        self.wallet_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.mint_trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))

        # Production-ready components
        self.threshold_manager = AdaptiveThresholdManager()
        self.funding_graph = FundingGraph(
            rpc_manager=rpc_manager,
            cache_ttl_hours=24.0,
            max_cache_size=50000  # Increased from 10K to 50K for longer runs
        )

        # Metrics
        self.metrics = ClassificationMetrics()

        # Track wallets needing re-classification (after funding graph updates)
        self.needs_reclassification: Set[str] = set()
        self.last_reclassification_check = datetime.now()

    async def start_funding_graph_worker(self):
        """Start background worker for funding graph RPC lookups"""
        await self.funding_graph.start_lookup_worker()

    def periodic_reclassification(self):
        """
        Periodically re-classify wallets with newly discovered funding relationships

        Call this every 30-60 seconds to catch sybils after RPC lookups complete
        """
        now = datetime.now()
        if (now - self.last_reclassification_check).total_seconds() < 30:
            return  # Too soon

        self.last_reclassification_check = now

        # Re-classify suspicious wallets that now have cached funding data
        reclassified = 0
        for wallet in list(self.funding_graph.suspicious_wallets):
            # Skip if not tracked
            if wallet not in self.wallet_profiles:
                continue

            # Skip if already classified as sybil
            profile = self.wallet_profiles[wallet]
            if profile.label == WalletLabel.SYBIL:
                continue

            # Check if funding data now available
            parent = self.funding_graph.get_funding_parent_sync(wallet)
            if parent:
                # Funding data available, re-classify
                self._classify_wallet(wallet)
                reclassified += 1

        if reclassified > 0:
            logger = logging.getLogger(__name__)
            logger.info(f"🔄 Re-classified {reclassified} wallets with new funding data")

    def process_transaction(self, tx: TxFeatures):
        """Process a transaction and update wallet/mint state"""
        wallet = tx.signer
        mint = tx.mint

        # Update wallet profile
        if wallet not in self.wallet_profiles:
            self.wallet_profiles[wallet] = WalletProfile(address=wallet)
            self.metrics.total_wallets_tracked += 1

        profile = self.wallet_profiles[wallet]
        profile.observations += 1
        profile.last_seen = tx.timestamp
        profile.trades_count += 1

        if tx.is_buy:
            profile.buys_count += 1
        else:
            profile.sells_count += 1

        # Update trade history
        self.wallet_trades[wallet].append({
            'timestamp': tx.timestamp,
            'mint': mint,
            'is_buy': tx.is_buy,
            'sol_amount': tx.sol_amount,
            'cu_price': tx.cu_price,
            'jito_tip': tx.jito_tip_lamports,
            'entry_index': tx.entry_index,
        })

        # Update mint state
        if mint and mint not in self.mint_states:
            self.mint_states[mint] = MintState(
                mint=mint,
                created_at=tx.timestamp,
                created_at_slot=tx.slot
            )
            self.metrics.total_mints_tracked += 1

        if mint:
            mint_state = self.mint_states[mint]
            mint_state.last_update = tx.timestamp

            if tx.is_buy:
                mint_state.unique_buyers.add(wallet)
                mint_state.buy_sizes.append(tx.sol_amount)
            else:
                mint_state.unique_sellers.add(wallet)

            # Update curve state
            if tx.virtual_sol_reserves > 0:
                mint_state.current_sol_reserves = tx.virtual_sol_reserves
                mint_state.peak_sol_reserves = max(mint_state.peak_sol_reserves, tx.virtual_sol_reserves)

            # Update rolling windows
            self._update_mint_rolling_windows(mint, tx)

            # Update CU price distribution
            mint_state.cu_prices.append(tx.cu_price)
            if len(mint_state.cu_prices) >= 10:
                mint_state.median_cu_price = int(statistics.median(mint_state.cu_prices[-50:]))
                sorted_prices = sorted(mint_state.cu_prices[-50:])
                p90_idx = int(len(sorted_prices) * 0.9)
                mint_state.p90_cu_price = sorted_prices[p90_idx]

            # Detect bursts
            self._detect_burst(mint, tx)

        # Track Jito bundle usage
        if tx.jito_tip_lamports > 0:
            recent_trades = list(self.wallet_trades[wallet])[-50:]
            jito_count = sum(1 for t in recent_trades if t['jito_tip'] > 0)
            profile.bundle_rate = jito_count / len(recent_trades) if recent_trades else 0.0

        # Track entry index
        if tx.entry_index > 0:
            recent_trades = list(self.wallet_trades[wallet])[-50:]
            entry_indices = [t['entry_index'] for t in recent_trades if t['entry_index'] > 0]
            if entry_indices:
                profile.mean_entry_index = statistics.mean(entry_indices)

        # Track mean CU price
        recent_trades = list(self.wallet_trades[wallet])[-50:]
        cu_prices = [t['cu_price'] for t in recent_trades if t['cu_price'] > 0]
        if cu_prices:
            profile.mean_cu_price = statistics.mean(cu_prices)

        # Track mean buy size
        recent_buys = [t for t in recent_trades if t['is_buy']]
        if recent_buys:
            profile.mean_buy_size_sol = statistics.mean([t['sol_amount'] for t in recent_buys])

        # Track mints
        if mint:
            profile.mints_traded.add(mint)
            if "Create" in tx.ix_types:
                profile.mints_created.add(mint)

        # SMART TRIGGER: Flag suspicious wallets for funding graph lookup
        self._check_and_flag_suspicious(wallet, tx)

        # Run classification
        self._classify_wallet(wallet)

    def _check_and_flag_suspicious(self, wallet: str, tx: TxFeatures):
        """
        Smart trigger: Flag suspicious wallets for funding graph RPC lookup

        Only flags wallets showing coordinated patterns to minimize RPC calls
        """
        mint = tx.mint
        if not mint:
            return

        # Trigger 1: Micro-buy (≤0.02 SOL) - strong sybil signal
        if tx.is_buy and tx.sol_amount <= 0.02:
            self.funding_graph.flag_suspicious_wallet(wallet, "micro_buy")
            return

        # Trigger 2: Part of coordinated burst (5+ wallets buying within 3s)
        # Tightened from 3+ in 5s to reduce false positives
        if tx.is_buy and mint in self.mint_trades:
            recent_buys = [
                t for t in self.mint_trades[mint]
                if t['is_buy'] and (tx.timestamp - t['timestamp']).total_seconds() <= 3.0
            ]
            unique_buyers = set(t['wallet'] for t in recent_buys)
            if len(unique_buyers) >= 5:
                # Flag all wallets in this coordinated group
                for buyer in unique_buyers:
                    self.funding_graph.flag_suspicious_wallet(buyer, "coordinated_burst")
                return

        # Trigger 3: Synchronized sell (sold within 1s of another wallet)
        if not tx.is_buy and mint in self.mint_trades:
            recent_sells = [
                t for t in self.mint_trades[mint]
                if not t['is_buy'] and (tx.timestamp - t['timestamp']).total_seconds() <= 1.0
            ]
            if len(recent_sells) >= 2:
                # Flag wallets doing synchronized exits
                for sell in recent_sells:
                    self.funding_graph.flag_suspicious_wallet(sell['wallet'], "synchronized_sell")
                return

        # Trigger 4: Very similar CU price to recent trades (template bot)
        trades = list(self.wallet_trades[wallet])
        if len(trades) >= 3:
            cu_prices = [t['cu_price'] for t in trades[-5:] if t['cu_price'] > 0]
            if len(cu_prices) >= 3:
                cu_variance = statistics.variance(cu_prices)
                if cu_variance < 5000:  # Tightened from 10000 to 5000 (very low variance = template)
                    # Check if other wallets on this mint have similar CU
                    if mint in self.mint_trades:
                        # Convert deque to list for slicing
                        mint_trades_list = list(self.mint_trades[mint])
                        mint_recent = [t for t in mint_trades_list[-20:] if t['wallet'] != wallet]
                        # If multiple wallets with same template CU, flag them (require 3+ matches)
                        similar_wallets = []
                        for mt in mint_recent:
                            other_wallet = mt['wallet']
                            other_trades = list(self.wallet_trades[other_wallet])
                            if len(other_trades) >= 2:
                                other_cu = [t['cu_price'] for t in other_trades[-3:] if t['cu_price'] > 0]
                                if len(other_cu) >= 2:
                                    other_variance = statistics.variance(other_cu)
                                    if other_variance < 5000:
                                        similar_wallets.append(other_wallet)

                        if len(similar_wallets) >= 3:  # Increased from 2 to 3
                            self.funding_graph.flag_suspicious_wallet(wallet, "template_cu")
                            for sw in similar_wallets:
                                self.funding_graph.flag_suspicious_wallet(sw, "template_cu")

    def _update_mint_rolling_windows(self, mint: str, tx: TxFeatures):
        """Update rolling time windows for mint activity"""
        mint_state = self.mint_states[mint]
        now = tx.timestamp

        # Store trade history
        self.mint_trades[mint].append({
            'timestamp': now,
            'wallet': tx.signer,
            'is_buy': tx.is_buy,
            'sol_amount': tx.sol_amount,
        })

        # Calculate rolling windows (1s, 3s, 10s)
        recent_trades = list(self.mint_trades[mint])

        # 1s window
        trades_1s = [t for t in recent_trades if (now - t['timestamp']).total_seconds() <= 1.0]
        mint_state.buy_count_1s = sum(1 for t in trades_1s if t['is_buy'])
        mint_state.sol_volume_1s = sum(t['sol_amount'] for t in trades_1s if t['is_buy'])

        # 3s window
        trades_3s = [t for t in recent_trades if (now - t['timestamp']).total_seconds() <= 3.0]
        mint_state.buy_count_3s = sum(1 for t in trades_3s if t['is_buy'])
        mint_state.sol_volume_3s = sum(t['sol_amount'] for t in trades_3s if t['is_buy'])

        # 10s window
        trades_10s = [t for t in recent_trades if (now - t['timestamp']).total_seconds() <= 10.0]
        mint_state.buy_count_10s = sum(1 for t in trades_10s if t['is_buy'])
        mint_state.sol_volume_10s = sum(t['sol_amount'] for t in trades_10s if t['is_buy'])

        # Update Gini
        if len(mint_state.buy_sizes) >= 5:
            mint_state.gini_buy_size = calculate_gini_coefficient(mint_state.buy_sizes[-100:])

        # Feed data to adaptive threshold manager
        token_age_seconds = (now - mint_state.created_at).total_seconds()
        if len(mint_state.buy_sizes) >= 10:  # Minimum sample for meaningful stats
            hold_times = []
            # Calculate median hold time (simplified - just from recent buys)
            buys_in_window = [t for t in recent_trades if t['is_buy'] and
                            (now - t['timestamp']).total_seconds() <= 60.0]
            if buys_in_window:
                # Rough estimate: assume half held for avg time
                median_hold = 30.0  # Placeholder

            self.threshold_manager.update_mint_stats(
                mint=mint,
                token_age_seconds=token_age_seconds,
                buy_count_3s=mint_state.buy_count_3s,
                sol_volume_3s=mint_state.sol_volume_3s,
                gini=mint_state.gini_buy_size,
                median_hold=30.0  # Simplified
            )

        # Track CU prices for adaptive priority threshold
        if tx.cu_price > 0:
            self.threshold_manager.add_cu_price(mint, token_age_seconds, tx.cu_price)

    def _detect_burst(self, mint: str, tx: TxFeatures):
        """Detect burst events on a mint using adaptive thresholds"""
        mint_state = self.mint_states[mint]

        # Calculate token age
        token_age_seconds = (tx.timestamp - mint_state.created_at).total_seconds()

        # Use adaptive burst detection
        is_burst = self.threshold_manager.is_burst(
            mint=mint,
            token_age_seconds=token_age_seconds,
            buy_count_3s=mint_state.buy_count_3s,
            sol_volume_3s=mint_state.sol_volume_3s
        )

        if is_burst:
            # Create burst event
            burst = BurstEvent(
                mint=mint,
                start_time=tx.timestamp - timedelta(seconds=3),
                end_time=tx.timestamp,
                buy_count=mint_state.buy_count_3s,
                sol_volume=mint_state.sol_volume_3s,
                unique_wallets=len(mint_state.unique_buyers)
            )

            self.bursts[mint].append(burst)
            mint_state.burst_count += 1

            # Log burst detection (first 3 for each mint to avoid spam)
            if mint_state.burst_count <= 3:
                logger = logging.getLogger(__name__)
                logger.info(f"🔥 Burst #{mint_state.burst_count} detected on {mint[:8]}: "
                          f"{burst.buy_count} buys, {burst.sol_volume:.2f} SOL in 3s")

            # Check for second burst (momentum)
            age = (tx.timestamp - mint_state.created_at).total_seconds()
            if age > 10 and mint_state.burst_count >= 2:
                mint_state.has_second_burst = True

    def _classify_wallet(self, wallet: str):
        """Apply classification rules to a wallet"""
        profile = self.wallet_profiles[wallet]

        # Save previous label to detect changes
        previous_label = profile.label

        # Reset reasons
        profile.label_reasons = []

        # Rule 1: Deployer
        if len(profile.mints_created) > 0:
            profile.label = WalletLabel.DEPLOYER
            profile.confidence = 1.0
            profile.label_reasons.append(f"Created {len(profile.mints_created)} token(s)")
            return

        # Rule 2: Sybil
        sybil_result = self._check_sybil_pattern(wallet)
        if sybil_result[0]:
            profile.label = WalletLabel.SYBIL
            profile.confidence = sybil_result[1]
            profile.label_reasons.extend(sybil_result[2])

            # Only increment if this is a NEW sybil detection (label changed)
            if previous_label != WalletLabel.SYBIL:
                self.metrics.sybil_clusters_found += 1

                # Log first 10 sybil detections for verification
                if self.metrics.sybil_clusters_found <= 10:
                    logger = logging.getLogger(__name__)
                    logger.info(f"🎯 Sybil #{self.metrics.sybil_clusters_found} detected: {wallet[:8]}... "
                              f"(confidence={sybil_result[1]:.2f}, cluster_size={self.funding_graph.get_cluster_size(wallet)})")
            return

        # Rule 3: Metric bot
        metric_result = self._check_metric_bot_pattern(wallet)
        if metric_result[0]:
            profile.label = WalletLabel.METRIC_BOT
            profile.confidence = metric_result[1]
            profile.label_reasons.extend(metric_result[2])

            # Only increment if this is a NEW metric bot detection (label changed)
            if previous_label != WalletLabel.METRIC_BOT:
                self.metrics.metric_bots_found += 1

                # Log first 10 metric bot detections for verification
                if self.metrics.metric_bots_found <= 10:
                    logger = logging.getLogger(__name__)
                    logger.info(f"🤖 Metric bot #{self.metrics.metric_bots_found} detected: {wallet[:8]}... "
                              f"(confidence={metric_result[1]:.2f}, reasons={len(metric_result[2])})")
            return

        # Rule 4: Copy bot
        copy_result = self._check_copy_bot_pattern(wallet)
        if copy_result[0]:
            profile.label = WalletLabel.COPY_BOT
            profile.confidence = copy_result[1]
            profile.label_reasons.extend(copy_result[2])
            return

        # Rule 5: Priority racer
        priority_result = self._check_priority_racer(wallet)
        if priority_result[0]:
            profile.label = WalletLabel.PRIORITY_RACER
            profile.confidence = priority_result[1]
            profile.label_reasons.extend(priority_result[2])
            return

        # Rule 6: Early exit specialist
        exit_result = self._check_early_exit_specialist(wallet)
        if exit_result[0]:
            profile.label = WalletLabel.EARLY_EXIT
            profile.confidence = exit_result[1]
            profile.label_reasons.extend(exit_result[2])
            return

        # Default: Unknown (may become organic over time)
        if profile.observations >= 20:
            profile.label = WalletLabel.ORGANIC
            profile.confidence = 0.3
            profile.label_reasons.append("Default organic (needs more data)")

    def _check_sybil_pattern(self, wallet: str) -> Tuple[bool, float, List[str]]:
        """Check if wallet exhibits sybil behavior using feature-based scoring"""
        profile = self.wallet_profiles[wallet]
        reasons = []

        # Check for coordinated small buys
        trades = list(self.wallet_trades[wallet])
        if len(trades) < 3:
            return (False, 0.0, [])

        # Get funding cluster size
        cluster_size = self.funding_graph.get_cluster_size(wallet, max_age_seconds=90.0)

        # Calculate timing gaps for buys
        buy_trades = [t for t in trades if t['is_buy']]
        timing_gaps_ms = []
        if len(buy_trades) >= 2:
            for i in range(len(buy_trades) - 1):
                gap_seconds = (buy_trades[i+1]['timestamp'] - buy_trades[i]['timestamp']).total_seconds()
                timing_gaps_ms.append(gap_seconds * 1000)

        # Calculate median buy size
        buy_sizes = [t['sol_amount'] for t in buy_trades]
        median_buy_sol = statistics.median(buy_sizes) if buy_sizes else 0.02

        # Calculate smallest-k mass (sybil indicator)
        smallest_k_mass = calculate_smallest_k_mass(buy_sizes, k=8) if len(buy_sizes) >= 8 else 0.0

        # Feature-based scoring
        confidence = score_sybil(
            cluster_size=cluster_size,
            timing_gaps_ms=timing_gaps_ms,
            median_buy_sol=median_buy_sol,
            smallest_k_mass=smallest_k_mass
        )

        # Build reasons for explainability
        if cluster_size >= 6:
            reasons.append(f"Funding cluster: {cluster_size} wallets")
        if timing_gaps_ms and statistics.median(timing_gaps_ms) < 250:
            reasons.append(f"Tight timing: {statistics.median(timing_gaps_ms):.0f}ms median gap")
        if median_buy_sol <= 0.02:
            reasons.append(f"Micro-buys: {median_buy_sol:.4f} SOL median")
        if smallest_k_mass >= 0.5:
            reasons.append(f"Concentrated volume: {smallest_k_mass:.1%} in smallest 8 buys")

        is_sybil = confidence >= 0.5  # Lowered from 0.6 to 0.5 (same as metric_bot)

        return (is_sybil, confidence, reasons)

    def _check_metric_bot_pattern(self, wallet: str) -> Tuple[bool, float, List[str]]:
        """Check if wallet is a metric-triggered bot using feature-based scoring"""
        profile = self.wallet_profiles[wallet]
        reasons = []

        trades = list(self.wallet_trades[wallet])
        if len(trades) < Thresholds.METRIC_BOT_MIN_OCCURRENCES:
            return (False, 0.0, [])

        # Count burst-triggered buys and delays
        burst_follows = 0
        delays_after_bursts = []
        for trade in trades:
            if not trade['is_buy']:
                continue

            mint = trade['mint']
            if mint not in self.bursts:
                continue

            # Check if buy occurred 0.5-5s after a burst (expanded window)
            burst_list = self.bursts[mint]
            for burst in burst_list:
                time_after_burst = (trade['timestamp'] - burst.end_time).total_seconds()
                if Thresholds.METRIC_BOT_DELAY_MIN <= time_after_burst <= Thresholds.METRIC_BOT_DELAY_MAX:
                    burst_follows += 1
                    delays_after_bursts.append(time_after_burst)
                    break

        total_buys = sum(1 for t in trades if t['is_buy'])
        distinct_mints = len(set(t['mint'] for t in trades if t['mint']))

        # Calculate inter-arrival CV (timing consistency)
        buy_timestamps = [t['timestamp'].timestamp() for t in trades if t['is_buy']]
        inter_arrival_cv = calculate_inter_arrival_cv(buy_timestamps) if len(buy_timestamps) >= 3 else 1.0

        # Calculate preferred delay std (consistency after bursts)
        preferred_delay_std = calculate_preferred_delay_std(delays_after_bursts) if delays_after_bursts else 0.6

        # Calculate CU variance
        cu_prices = [t['cu_price'] for t in trades if t['cu_price'] > 0]
        cu_variance = statistics.variance(cu_prices) if len(cu_prices) >= 2 else 100000.0

        # Calculate median hold time
        hold_times = self._calculate_hold_times(wallet)
        median_hold_sec = statistics.median(hold_times) if hold_times else 30.0

        # Feature-based scoring
        confidence = score_metric_bot(
            burst_follows=burst_follows,
            total_buys=total_buys,
            distinct_mints=distinct_mints,
            inter_arrival_cv=inter_arrival_cv,
            preferred_delay_std=preferred_delay_std,
            cu_variance=cu_variance,
            median_hold_sec=median_hold_sec
        )

        # Build reasons for explainability
        if burst_follows >= 5:  # Lowered from 7
            reasons.append(f"Burst-triggered: {burst_follows}/{total_buys} buys after bursts")
        if inter_arrival_cv < 0.4:
            reasons.append(f"Template timing: CV={inter_arrival_cv:.2f}")
        if preferred_delay_std < 0.3:
            reasons.append(f"Consistent delay: std={preferred_delay_std:.2f}s")
        if cu_variance < 100000:
            reasons.append(f"Template CU: variance={cu_variance:.0f}")
        if median_hold_sec < 25.0:
            reasons.append(f"Fast flips: {median_hold_sec:.1f}s median hold")

        is_metric_bot = confidence >= 0.50  # Lowered from 0.65 to 0.50

        return (is_metric_bot, confidence, reasons)

    def _check_copy_bot_pattern(self, wallet: str) -> Tuple[bool, float, List[str]]:
        """Check if wallet is a copy trading bot"""
        # This requires tracking "alpha wallets" which we'll implement in the main analyzer
        return (False, 0.0, [])

    def _check_priority_racer(self, wallet: str) -> Tuple[bool, float, List[str]]:
        """Check if wallet is a priority fee racer using feature-based scoring"""
        profile = self.wallet_profiles[wallet]
        reasons = []

        trades = list(self.wallet_trades[wallet])
        if len(trades) < 20:
            return (False, 0.0, [])

        # Calculate high CU rate (using adaptive thresholds)
        high_cu_trades = 0
        total_evaluated = 0
        for trade in trades:
            mint = trade['mint']
            if not mint or mint not in self.mint_states:
                continue

            mint_state = self.mint_states[mint]
            token_age = (trade['timestamp'] - mint_state.created_at).total_seconds()

            # Get adaptive priority threshold
            threshold = self.threshold_manager.get_priority_threshold(mint, token_age)
            if trade['cu_price'] >= threshold:
                high_cu_trades += 1
            total_evaluated += 1

        high_cu_rate = high_cu_trades / total_evaluated if total_evaluated > 0 else 0.0

        # Calculate early lander rate (entry_index <= 10% of typical slot size ~64)
        early_trades = sum(1 for t in trades if t.get('entry_index', 100) <= 6)
        early_lander_rate = early_trades / len(trades)

        # Jito rate
        jito_rate = profile.bundle_rate

        # Leader stickiness (placeholder - would need validator tracking)
        leader_concentration = 0.1  # Simplified for now

        # Feature-based scoring
        confidence = score_priority_racer(
            early_lander_rate=early_lander_rate,
            high_cu_rate=high_cu_rate,
            jito_rate=jito_rate,
            leader_concentration=leader_concentration
        )

        # Build reasons for explainability
        if high_cu_rate > 0.6:
            reasons.append(f"High CU: {high_cu_rate:.1%} above adaptive threshold")
        if early_lander_rate > 0.3:
            reasons.append(f"Early entries: {early_lander_rate:.1%} in first 10% of slot")
        if jito_rate > 0.5:
            reasons.append(f"Jito bundles: {jito_rate:.1%}")
        if profile.mean_entry_index <= 2:
            reasons.append(f"Very early: mean entry_index={profile.mean_entry_index:.1f}")

        is_racer = confidence >= 0.7  # Threshold

        return (is_racer, confidence, reasons)

    def _check_early_exit_specialist(self, wallet: str) -> Tuple[bool, float, List[str]]:
        """Check if wallet specializes in early exits from failed launches"""
        profile = self.wallet_profiles[wallet]
        reasons = []

        trades = list(self.wallet_trades[wallet])
        if len(trades) < 15:
            return (False, 0.0, [])

        # Count first seller occurrences and prediction accuracy
        first_seller_count = 0
        correct_predictions = 0
        total_predictions = 0

        for trade in trades:
            if trade['is_buy']:
                continue

            mint = trade['mint']
            if mint not in self.mint_states:
                continue

            mint_state = self.mint_states[mint]

            # Check if mint failed to gain momentum
            age_at_sell = (trade['timestamp'] - mint_state.created_at).total_seconds()
            if age_at_sell < 60:
                total_predictions += 1

                # Check if wallet was among first 3 sellers
                mint_trade_history = list(self.mint_trades[mint])
                sells = [t for t in mint_trade_history if not t['is_buy']]
                sell_index = sum(1 for s in sells if s['timestamp'] < trade['timestamp'])
                if sell_index < 3:
                    first_seller_count += 1

                # Check if prediction was correct (failed launch = no second burst)
                if not mint_state.has_second_burst:
                    correct_predictions += 1

        total_sells = sum(1 for t in trades if not t['is_buy'])

        # Calculate median P&L (simplified - would need buy/sell matching)
        hold_times = self._calculate_hold_times(wallet)
        median_pnl_pct = 2.0  # Placeholder (would need actual P&L calculation)

        # Feature-based scoring
        confidence = score_early_exit(
            first_seller_count=first_seller_count,
            total_sells=total_sells,
            correct_predictions=correct_predictions,
            total_predictions=max(1, total_predictions),
            median_pnl_pct=median_pnl_pct
        )

        # Build reasons for explainability
        if first_seller_count >= 5:
            reasons.append(f"First seller: {first_seller_count}/{total_sells} times")
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            if accuracy > 0.6:
                reasons.append(f"Prediction accuracy: {accuracy:.1%}")
        if len(hold_times) > 0 and statistics.median(hold_times) < 30:
            reasons.append(f"Quick exits: {statistics.median(hold_times):.1f}s median")

        is_early_exit = confidence >= 0.65  # Threshold

        return (is_early_exit, confidence, reasons)

    def _calculate_hold_times(self, wallet: str) -> List[float]:
        """Calculate hold times for a wallet's positions"""
        trades = list(self.wallet_trades[wallet])

        # Group by mint
        positions: Dict[str, List] = defaultdict(list)
        for trade in trades:
            positions[trade['mint']].append(trade)

        hold_times = []
        for mint, mint_trades in positions.items():
            buys = [t for t in mint_trades if t['is_buy']]
            sells = [t for t in mint_trades if not t['is_buy']]

            # Match buys to sells
            for buy in buys:
                # Find next sell after this buy
                later_sells = [s for s in sells if s['timestamp'] > buy['timestamp']]
                if later_sells:
                    first_sell = min(later_sells, key=lambda s: s['timestamp'])
                    hold_time = (first_sell['timestamp'] - buy['timestamp']).total_seconds()
                    hold_times.append(hold_time)

        return hold_times

    def get_wallet_label(self, wallet: str) -> Tuple[WalletLabel, float, List[str]]:
        """Get classification for a wallet"""
        if wallet not in self.wallet_profiles:
            return (WalletLabel.UNKNOWN, 0.0, ["Wallet not tracked"])

        profile = self.wallet_profiles[wallet]
        return (profile.label, profile.confidence, profile.label_reasons)

    def get_statistics(self) -> ClassificationMetrics:
        """Get overall classification statistics"""
        # Update label counts
        self.metrics.label_counts = defaultdict(int)
        for profile in self.wallet_profiles.values():
            self.metrics.label_counts[profile.label] += 1

        return self.metrics
