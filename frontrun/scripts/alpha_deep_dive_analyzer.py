#!/usr/bin/env python3
"""
Alpha Deep Dive Analyzer

Analyzes alpha pattern data to answer critical questions:
1. What curve state (SOL reserves) do alphas enter at?
2. Are we counting wash traders in our analysis?
3. What triggered the alpha's buy? Was it manipulation?
4. What peak price do alphas buy at (curve position)?
5. Are our entry/exit metrics even good?

This goes DEEP into the data to validate our strategy assumptions.

Usage:
    python frontrun/scripts/alpha_deep_dive_analyzer.py \
        --data data/alpha_pattern_analysis/alpha_pattern_analysis_20251015_145916.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logging, get_logger

logger = get_logger(__name__)


# Pump.fun bonding curve constants
INITIAL_SOL_RESERVES = 30_000_000_000  # 30 SOL in lamports
INITIAL_TOKEN_RESERVES = 1_073_000_000_000_000  # 1.073B tokens in smallest units (6 decimals)
GRADUATION_SOL = 85_000_000_000  # 85 SOL in lamports
K_CONSTANT = INITIAL_SOL_RESERVES * INITIAL_TOKEN_RESERVES  # Constant product


def estimate_sol_reserves_from_price(price_per_million: float) -> float:
    """
    Estimate SOL reserves from price using constant product formula.

    Price = (virtual_sol / virtual_token) * 1e6  (SOL per million tokens)
    k = virtual_sol * virtual_token = constant

    Solving:
    virtual_sol = sqrt(k * price / 1e6)
    """
    if price_per_million <= 0:
        return 30.0  # Default to initial

    # price_per_million = (sol / token) * 1e6
    # price_ratio = sol / token
    price_ratio = price_per_million / 1e6

    # From k = sol * token and price_ratio = sol / token:
    # sol = sqrt(k * price_ratio)
    virtual_sol_lamports = (K_CONSTANT * price_ratio) ** 0.5
    virtual_sol = virtual_sol_lamports / 1e9

    # Clamp to reasonable range (30 SOL min, 85 SOL max)
    return max(30.0, min(85.0, virtual_sol))


@dataclass
class AlphaAnalysis:
    """Deep analysis of an alpha buy event"""
    token: str
    signature: str

    # Entry conditions
    buy_size_sol: float
    buy_position: int
    token_age_seconds: float
    pattern_type: str

    # PRE-ALPHA CONTEXT (critical!)
    holders_before: int
    volume_before_sol: float
    avg_buy_before: float  # Average size of buys before this alpha

    # CURVE STATE AT ENTRY (critical!)
    entry_price: float
    estimated_entry_sol_reserves: float
    entry_curve_progress: float  # % from 30 SOL to 85 SOL
    entry_price_markup: float  # % above initial 30 SOL price

    # WASH TRADING INDICATORS
    is_likely_wash_trade: bool
    wash_confidence: float
    wash_reasons: List[str]

    # POST-ALPHA OUTCOME
    followers: int
    volume_added_sol: float
    peak_gain_pct: float
    peak_time_seconds: float
    triggered_swarm: bool

    # PEAK CURVE ANALYSIS
    peak_price: float
    estimated_peak_sol_reserves: float
    peak_curve_progress: float

    # FEE DATA
    total_fee_sol: float
    priority_fee_sol: float
    used_jito: bool

    # EXIT ANALYSIS
    best_exit_gain: float
    best_exit_time: float
    profitable_exits: int  # How many exit windows were profitable


class AlphaDeepDiveAnalyzer:
    """Comprehensive analyzer for alpha pattern data"""

    def __init__(self, data_file: str):
        self.data_file = data_file
        self.raw_data = None
        self.alphas: List[AlphaAnalysis] = []

        # Thresholds for wash trading detection
        self.WASH_MIN_AVG_BUY = 5.0  # If avg buy before is > 5 SOL, suspicious
        self.WASH_MIN_HOLDERS = 3  # Need at least 3 holders before
        self.WASH_VOLUME_RATIO = 0.3  # Alpha size should be < 30% of prior volume

    def load_data(self):
        """Load raw alpha pattern data"""
        logger.info(f"Loading data from: {self.data_file}")

        with open(self.data_file, 'r') as f:
            self.raw_data = json.load(f)

        observations = self.raw_data.get('observations', [])
        logger.info(f"Loaded {len(observations)} alpha observations")

        # Convert to deep analysis
        for obs in observations:
            alpha = self._analyze_alpha(obs)
            if alpha:
                self.alphas.append(alpha)

        logger.info(f"Created deep analysis for {len(self.alphas)} alphas")

    def _analyze_alpha(self, obs: Dict) -> Optional[AlphaAnalysis]:
        """Create deep analysis from raw observation"""
        try:
            # Calculate pre-alpha metrics
            holders_before = obs.get('holder_count_before', 0)
            volume_before = obs.get('volume_before_sol', 0)
            buy_size = obs.get('buy_size_sol', 0)

            # Calculate average buy size before this alpha
            avg_buy_before = 0.0
            if holders_before > 0 and volume_before > 0:
                avg_buy_before = volume_before / holders_before

            # Get entry price (use first price history entry as proxy for buy price)
            entry_price = obs.get('buy_price', 0)
            if not entry_price and obs.get('price_history'):
                entry_price = obs['price_history'][0].get('price', 0)

            # Estimate curve state at entry
            entry_sol_reserves = estimate_sol_reserves_from_price(entry_price)
            entry_progress = ((entry_sol_reserves - 30) / (85 - 30)) * 100  # % to graduation

            # Calculate entry price markup vs initial 30 SOL
            initial_price = (INITIAL_SOL_RESERVES / INITIAL_TOKEN_RESERVES) * 1e6
            entry_markup = ((entry_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0

            # Detect wash trading
            wash_indicators = self._detect_wash_trading(
                holders_before, volume_before, avg_buy_before, buy_size
            )

            # Peak analysis
            peak_price = obs.get('peak_gain_pct', 0) + entry_price if entry_price > 0 else 0
            peak_price_abs = 0

            # Find actual peak price from price history
            price_history = obs.get('price_history', [])
            if price_history:
                prices = [p['price'] for p in price_history]
                peak_price_abs = max(prices) if prices else entry_price
            else:
                peak_price_abs = entry_price

            peak_sol_reserves = estimate_sol_reserves_from_price(peak_price_abs)
            peak_progress = ((peak_sol_reserves - 30) / (85 - 30)) * 100

            # Exit analysis
            exit_windows = obs.get('profitable_exit_windows', [])
            profitable_exits = sum(1 for w in exit_windows if w.get('profitable', False))
            best_exit_gain = max([w.get('gain_pct', 0) for w in exit_windows], default=0)
            best_exit_time = 0
            if exit_windows:
                best_window = max(exit_windows, key=lambda w: w.get('gain_pct', 0))
                best_exit_time = best_window.get('exit_time', 0)

            return AlphaAnalysis(
                token=obs.get('token', ''),
                signature=obs.get('signature', ''),
                buy_size_sol=buy_size,
                buy_position=obs.get('buy_position', 0),
                token_age_seconds=obs.get('token_age_seconds', 0),
                pattern_type=obs.get('pattern_type', ''),
                holders_before=holders_before,
                volume_before_sol=volume_before,
                avg_buy_before=avg_buy_before,
                entry_price=entry_price,
                estimated_entry_sol_reserves=entry_sol_reserves,
                entry_curve_progress=entry_progress,
                entry_price_markup=entry_markup,
                is_likely_wash_trade=wash_indicators['is_wash'],
                wash_confidence=wash_indicators['confidence'],
                wash_reasons=wash_indicators['reasons'],
                followers=obs.get('unique_followers', 0),
                volume_added_sol=obs.get('volume_added_sol', 0),
                peak_gain_pct=obs.get('peak_gain_pct', 0),
                peak_time_seconds=obs.get('peak_time_seconds', 0),
                triggered_swarm=obs.get('did_trigger_swarm', False),
                peak_price=peak_price_abs,
                estimated_peak_sol_reserves=peak_sol_reserves,
                peak_curve_progress=peak_progress,
                total_fee_sol=obs.get('total_fee_sol', 0),
                priority_fee_sol=obs.get('priority_fee_sol', 0),
                used_jito=obs.get('used_jito_bundle', False),
                best_exit_gain=best_exit_gain,
                best_exit_time=best_exit_time,
                profitable_exits=profitable_exits
            )

        except Exception as e:
            logger.error(f"Error analyzing alpha: {e}")
            return None

    def _detect_wash_trading(self, holders: int, volume: float, avg_buy: float, alpha_size: float) -> Dict:
        """Detect if this looks like wash trading"""
        reasons = []
        confidence = 0.0

        # Indicator 1: Very few holders before alpha
        if holders < self.WASH_MIN_HOLDERS:
            reasons.append(f"Only {holders} holders before alpha")
            confidence += 0.3

        # Indicator 2: Average buy size is suspiciously large
        if avg_buy > self.WASH_MIN_AVG_BUY:
            reasons.append(f"Avg buy {avg_buy:.2f} SOL (> {self.WASH_MIN_AVG_BUY} SOL)")
            confidence += 0.4

        # Indicator 3: Alpha is huge portion of prior volume
        if volume > 0:
            alpha_ratio = alpha_size / volume
            if alpha_ratio > self.WASH_VOLUME_RATIO:
                reasons.append(f"Alpha is {alpha_ratio:.1%} of prior volume")
                confidence += 0.3

        is_wash = confidence >= 0.5

        return {
            'is_wash': is_wash,
            'confidence': min(confidence, 1.0),
            'reasons': reasons
        }

    def analyze_entry_curve_states(self):
        """Analyze what curve states alphas enter at"""
        logger.info("\n" + "="*80)
        logger.info("CURVE STATE AT ENTRY ANALYSIS")
        logger.info("="*80)

        # Group by entry curve position
        very_early = [a for a in self.alphas if a.estimated_entry_sol_reserves < 33]  # 30-33 SOL
        early = [a for a in self.alphas if 33 <= a.estimated_entry_sol_reserves < 38]  # 33-38 SOL
        mid = [a for a in self.alphas if 38 <= a.estimated_entry_sol_reserves < 50]  # 38-50 SOL
        late = [a for a in self.alphas if a.estimated_entry_sol_reserves >= 50]  # 50+ SOL

        logger.info(f"\n📊 Entry Distribution:")
        logger.info(f"   Very early (30-33 SOL): {len(very_early)} ({len(very_early)/len(self.alphas)*100:.1f}%)")
        logger.info(f"   Early (33-38 SOL): {len(early)} ({len(early)/len(self.alphas)*100:.1f}%)")
        logger.info(f"   Mid (38-50 SOL): {len(mid)} ({len(mid)/len(self.alphas)*100:.1f}%)")
        logger.info(f"   Late (50+ SOL): {len(late)} ({len(late)/len(self.alphas)*100:.1f}%)")

        # Success rate by entry position
        logger.info(f"\n✅ Success Rate by Entry Position:")
        for name, group in [("Very early (30-33 SOL)", very_early),
                           ("Early (33-38 SOL)", early),
                           ("Mid (38-50 SOL)", mid),
                           ("Late (50+ SOL)", late)]:
            if group:
                successful = [a for a in group if a.triggered_swarm]
                success_rate = len(successful) / len(group) * 100
                avg_peak = statistics.mean([a.peak_gain_pct for a in group])
                avg_entry_markup = statistics.mean([a.entry_price_markup for a in group])

                logger.info(f"   {name}:")
                logger.info(f"      Success rate: {success_rate:.1f}% ({len(successful)}/{len(group)})")
                logger.info(f"      Avg entry markup: {avg_entry_markup:.1f}%")
                logger.info(f"      Avg peak gain: {avg_peak:.1f}%")

        # CRITICAL QUESTION: Are we entering too late?
        dangerous_entries = [a for a in self.alphas if a.estimated_entry_sol_reserves > 38]
        logger.info(f"\n⚠️ CRITICAL INSIGHT:")
        logger.info(f"   {len(dangerous_entries)} alphas ({len(dangerous_entries)/len(self.alphas)*100:.1f}%) entered above 38 SOL")
        logger.info(f"   (Like your TIVO trade at 44.85 SOL)")

        if dangerous_entries:
            successful_late = [a for a in dangerous_entries if a.triggered_swarm]
            late_success_rate = len(successful_late) / len(dangerous_entries) * 100
            logger.info(f"   Late entry success rate: {late_success_rate:.1f}%")
            logger.info(f"   → Entering above 38 SOL is {'DANGEROUS' if late_success_rate < 40 else 'RISKY'}")

    def analyze_wash_trading(self):
        """Analyze wash trading patterns"""
        logger.info("\n" + "="*80)
        logger.info("WASH TRADING ANALYSIS")
        logger.info("="*80)

        wash_trades = [a for a in self.alphas if a.is_likely_wash_trade]
        clean_trades = [a for a in self.alphas if not a.is_likely_wash_trade]

        logger.info(f"\n🔍 Detection Results:")
        logger.info(f"   Likely wash trades: {len(wash_trades)} ({len(wash_trades)/len(self.alphas)*100:.1f}%)")
        logger.info(f"   Clean trades: {len(clean_trades)} ({len(clean_trades)/len(self.alphas)*100:.1f}%)")

        if wash_trades:
            logger.info(f"\n⚠️ Wash Trade Characteristics:")
            avg_holders_wash = statistics.mean([a.holders_before for a in wash_trades])
            avg_holders_clean = statistics.mean([a.holders_before for a in clean_trades]) if clean_trades else 0

            logger.info(f"   Avg holders before (wash): {avg_holders_wash:.1f}")
            logger.info(f"   Avg holders before (clean): {avg_holders_clean:.1f}")

            # Success rate comparison
            wash_success = [a for a in wash_trades if a.triggered_swarm]
            clean_success = [a for a in clean_trades if a.triggered_swarm]

            wash_success_rate = len(wash_success) / len(wash_trades) * 100 if wash_trades else 0
            clean_success_rate = len(clean_success) / len(clean_trades) * 100 if clean_trades else 0

            logger.info(f"\n📊 Success Rate Comparison:")
            logger.info(f"   Wash trades: {wash_success_rate:.1f}% ({len(wash_success)}/{len(wash_trades)})")
            logger.info(f"   Clean trades: {clean_success_rate:.1f}% ({len(clean_success)}/{len(clean_trades)})")

            if wash_success_rate < clean_success_rate:
                logger.info(f"\n✅ INSIGHT: Clean trades are {clean_success_rate - wash_success_rate:.1f}% more successful!")
                logger.info(f"   → Filter out tokens with < {self.WASH_MIN_HOLDERS} holders")
                logger.info(f"   → Filter out tokens with avg buy > {self.WASH_MIN_AVG_BUY} SOL")

    def analyze_pre_alpha_patterns(self):
        """Analyze what happens BEFORE alphas enter"""
        logger.info("\n" + "="*80)
        logger.info("PRE-ALPHA PATTERN ANALYSIS")
        logger.info("="*80)

        # Group by pre-alpha volume
        no_volume = [a for a in self.alphas if a.volume_before_sol < 0.5]
        low_volume = [a for a in self.alphas if 0.5 <= a.volume_before_sol < 2.0]
        med_volume = [a for a in self.alphas if 2.0 <= a.volume_before_sol < 5.0]
        high_volume = [a for a in self.alphas if a.volume_before_sol >= 5.0]

        logger.info(f"\n📊 Pre-Alpha Volume Distribution:")
        logger.info(f"   No volume (< 0.5 SOL): {len(no_volume)}")
        logger.info(f"   Low volume (0.5-2 SOL): {len(low_volume)}")
        logger.info(f"   Medium volume (2-5 SOL): {len(med_volume)}")
        logger.info(f"   High volume (5+ SOL): {len(high_volume)}")

        logger.info(f"\n✅ Success Rate by Pre-Alpha Volume:")
        for name, group in [("No volume (< 0.5 SOL)", no_volume),
                           ("Low volume (0.5-2 SOL)", low_volume),
                           ("Medium volume (2-5 SOL)", med_volume),
                           ("High volume (5+ SOL)", high_volume)]:
            if group:
                successful = [a for a in group if a.triggered_swarm]
                success_rate = len(successful) / len(group) * 100
                avg_followers = statistics.mean([a.followers for a in group])

                logger.info(f"   {name}:")
                logger.info(f"      Success rate: {success_rate:.1f}% ({len(successful)}/{len(group)})")
                logger.info(f"      Avg followers: {avg_followers:.1f}")

    def validate_entry_metrics(self):
        """Validate if our entry metrics (5+ buyers, 2.5+ SOL) are good"""
        logger.info("\n" + "="*80)
        logger.info("ENTRY METRICS VALIDATION")
        logger.info("="*80)

        # Simulate our current strategy: 5+ buyers, 2.5+ SOL volume
        MIN_BUYERS = 5
        MIN_VOLUME = 2.5

        # Filter alphas that would pass our criteria
        would_enter = [a for a in self.alphas
                      if a.holders_before >= MIN_BUYERS and
                      a.volume_before_sol >= MIN_VOLUME and
                      not a.is_likely_wash_trade and
                      a.estimated_entry_sol_reserves <= 38]  # Include new curve filter

        would_reject = [a for a in self.alphas if a not in would_enter]

        logger.info(f"\n🎯 Current Strategy Simulation:")
        logger.info(f"   Would enter: {len(would_enter)} alphas")
        logger.info(f"   Would reject: {len(would_reject)} alphas")

        if would_enter:
            entered_success = [a for a in would_enter if a.triggered_swarm]
            success_rate = len(entered_success) / len(would_enter) * 100

            avg_peak = statistics.mean([a.peak_gain_pct for a in would_enter])
            avg_followers = statistics.mean([a.followers for a in would_enter])

            logger.info(f"\n📊 Strategy Performance:")
            logger.info(f"   Success rate: {success_rate:.1f}% ({len(entered_success)}/{len(would_enter)})")
            logger.info(f"   Avg peak gain: {avg_peak:.1f}%")
            logger.info(f"   Avg followers: {avg_followers:.1f}")

            # Best exit analysis
            profitable_trades = [a for a in would_enter if a.best_exit_gain > 5]
            logger.info(f"   Profitable trades (>5% gain): {len(profitable_trades)} ({len(profitable_trades)/len(would_enter)*100:.1f}%)")

            if profitable_trades:
                avg_best_exit_gain = statistics.mean([a.best_exit_gain for a in profitable_trades])
                avg_best_exit_time = statistics.mean([a.best_exit_time for a in profitable_trades])
                logger.info(f"   Avg best exit gain: {avg_best_exit_gain:.1f}%")
                logger.info(f"   Avg best exit time: {avg_best_exit_time:.1f}s")

        # Test different thresholds
        logger.info(f"\n🔬 Testing Different Thresholds:")

        for min_buyers in [3, 5, 7, 10]:
            for min_volume in [1.0, 2.5, 5.0]:
                candidates = [a for a in self.alphas
                            if a.holders_before >= min_buyers and
                            a.volume_before_sol >= min_volume and
                            not a.is_likely_wash_trade and
                            a.estimated_entry_sol_reserves <= 38]

                if candidates:
                    successful = [a for a in candidates if a.triggered_swarm]
                    success_rate = len(successful) / len(candidates) * 100
                    logger.info(f"   {min_buyers}+ buyers, {min_volume}+ SOL: {success_rate:.1f}% success ({len(candidates)} opportunities)")

    def generate_recommendations(self):
        """Generate actionable recommendations"""
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATIONS")
        logger.info("="*80)

        # Calculate optimal metrics
        clean_alphas = [a for a in self.alphas if not a.is_likely_wash_trade]
        successful_alphas = [a for a in clean_alphas if a.triggered_swarm]

        if not successful_alphas:
            logger.info("\n⚠️ Not enough successful data to generate recommendations")
            return

        # Entry curve state
        entry_reserves = [a.estimated_entry_sol_reserves for a in successful_alphas]
        avg_entry = statistics.mean(entry_reserves)
        median_entry = statistics.median(entry_reserves)

        logger.info(f"\n✅ Optimal Entry Conditions (from successful alphas):")
        logger.info(f"   Avg entry reserves: {avg_entry:.1f} SOL")
        logger.info(f"   Median entry reserves: {median_entry:.1f} SOL")
        logger.info(f"   → RECOMMENDATION: Only enter below {median_entry + 5:.0f} SOL reserves")

        # Pre-alpha volume
        pre_volumes = [a.volume_before_sol for a in successful_alphas]
        avg_pre_volume = statistics.mean(pre_volumes)

        logger.info(f"\n📊 Pre-Alpha Activity:")
        logger.info(f"   Avg volume before: {avg_pre_volume:.2f} SOL")
        logger.info(f"   → RECOMMENDATION: Require {avg_pre_volume * 0.7:.1f}+ SOL volume before entry")

        # Wash trading
        wash_filtered = [a for a in successful_alphas if not a.is_likely_wash_trade]
        logger.info(f"\n🔍 Wash Trading Filter:")
        logger.info(f"   Clean trades: {len(wash_filtered)}/{len(successful_alphas)} ({len(wash_filtered)/len(successful_alphas)*100:.1f}%)")
        logger.info(f"   → RECOMMENDATION: Filter out avg buy > {self.WASH_MIN_AVG_BUY} SOL")

        # Exit timing
        exit_times = [a.best_exit_time for a in successful_alphas if a.best_exit_time > 0]
        if exit_times:
            avg_exit = statistics.mean(exit_times)
            logger.info(f"\n⏰ Exit Timing:")
            logger.info(f"   Avg best exit: {avg_exit:.1f}s")
            logger.info(f"   → RECOMMENDATION: Hold for {avg_exit:.0f}s")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deep dive alpha pattern analyzer")
    parser.add_argument("--data", required=True, help="Path to alpha pattern analysis JSON")

    args = parser.parse_args()

    setup_logging(level="INFO", format="text", output_file="logs/alpha_deep_dive.log")

    logger.info("="*80)
    logger.info("ALPHA DEEP DIVE ANALYZER")
    logger.info("="*80)
    logger.info("Analyzing fundamental market conditions, not just 'big buy triggers bots'")
    logger.info("="*80)

    # Load and analyze
    analyzer = AlphaDeepDiveAnalyzer(args.data)
    analyzer.load_data()

    # Run all analyses
    analyzer.analyze_entry_curve_states()
    analyzer.analyze_wash_trading()
    analyzer.analyze_pre_alpha_patterns()
    analyzer.validate_entry_metrics()
    analyzer.generate_recommendations()

    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
