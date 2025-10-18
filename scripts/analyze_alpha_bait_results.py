#!/usr/bin/env python3
"""
Analysis helper for Alpha Bait Strategy results

This script loads the JSON output from alpha_bait_analyzer.py and provides
detailed analysis to answer the three key questions:
1. WHEN to buy (timing analysis)
2. HOW MUCH to buy (position size analysis)
3. WHAT coins to buy (token characteristics)

Usage:
    python scripts/analyze_alpha_bait_results.py data/alpha_bait_analysis/alpha_bait_analysis_20251015_143022.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
from statistics import mean, median, stdev
from collections import defaultdict


def load_results(file_path: str) -> Dict:
    """Load analysis results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}")


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{'-'*80}")
    print(f"{text}")
    print(f"{'-'*80}")


def analyze_metadata(data: Dict):
    """Print metadata summary"""
    print_header("ANALYSIS METADATA")
    meta = data['metadata']
    print(f"Start time:        {meta['start_time']}")
    print(f"End time:          {meta['end_time']}")
    print(f"Runtime:           {meta['runtime_hours']:.2f} hours")
    print(f"Total buys seen:   {meta['total_buys_seen']:,}")
    print(f"Tokens tracked:    {meta['total_tokens_tracked']}")
    print(f"Completed:         {meta['completed_tokens']}")
    print(f"Min buy threshold: {meta['min_significant_buy']} SOL")


def analyze_bot_swarms(tokens: List[Dict]):
    """Analyze bot swarm patterns"""
    print_header("BOT SWARM ANALYSIS")

    swarm_tokens = [t for t in tokens if t['bot_swarm_detected']]
    no_swarm_tokens = [t for t in tokens if not t['bot_swarm_detected']]

    print(f"\nBot Swarm Frequency:")
    print(f"  Swarms detected: {len(swarm_tokens)}/{len(tokens)} ({len(swarm_tokens)/len(tokens)*100:.1f}%)")
    print(f"  No swarms:       {len(no_swarm_tokens)}/{len(tokens)} ({len(no_swarm_tokens)/len(tokens)*100:.1f}%)")

    if swarm_tokens:
        swarm_sizes = [t['bot_swarm_size'] for t in swarm_tokens]
        print(f"\nSwarm Size Statistics:")
        print(f"  Average:  {mean(swarm_sizes):.1f} bots")
        print(f"  Median:   {median(swarm_sizes):.0f} bots")
        print(f"  Min:      {min(swarm_sizes)} bots")
        print(f"  Max:      {max(swarm_sizes)} bots")

        # Compare performance
        swarm_peaks = [t['peak_gain_pct'] for t in swarm_tokens if t['peak_gain_pct'] > 0]
        no_swarm_peaks = [t['peak_gain_pct'] for t in no_swarm_tokens if t['peak_gain_pct'] > 0]

        if swarm_peaks:
            print(f"\nPerformance Comparison:")
            print(f"  With swarm:    {mean(swarm_peaks):.1f}% average peak gain")
        if no_swarm_peaks:
            print(f"  Without swarm: {mean(no_swarm_peaks):.1f}% average peak gain")


def analyze_timing(tokens: List[Dict]):
    """Analyze WHEN to buy"""
    print_header("TIMING ANALYSIS - WHEN TO BUY")

    swarm_tokens = [t for t in tokens if t['bot_swarm_detected']]

    if not swarm_tokens:
        print("\n⚠️ No bot swarms detected - cannot analyze timing patterns")
        return

    print_section("Buy Activity Distribution")

    # Aggregate timing
    total_0_10s = sum(t['timing_analysis']['buys_0_10s'] for t in swarm_tokens)
    total_10_30s = sum(t['timing_analysis']['buys_10_30s'] for t in swarm_tokens)
    total_30_60s = sum(t['timing_analysis']['buys_30_60s'] for t in swarm_tokens)
    total_60_120s = sum(t['timing_analysis']['buys_60_120s'] for t in swarm_tokens)
    total = total_0_10s + total_10_30s + total_30_60s + total_60_120s

    if total > 0:
        print(f"\n  0-10 seconds:   {total_0_10s:4d} buys ({total_0_10s/total*100:5.1f}%)")
        print(f"  10-30 seconds:  {total_10_30s:4d} buys ({total_10_30s/total*100:5.1f}%)")
        print(f"  30-60 seconds:  {total_30_60s:4d} buys ({total_30_60s/total*100:5.1f}%)")
        print(f"  60-120 seconds: {total_60_120s:4d} buys ({total_60_120s/total*100:5.1f}%)")

    print_section("Peak Timing Analysis")

    peak_times = [t['peak_time_seconds'] for t in swarm_tokens if t['peak_time_seconds'] > 0]
    if peak_times:
        print(f"\nPrice Peak Statistics:")
        print(f"  Average peak time: {mean(peak_times):.1f} seconds")
        print(f"  Median peak time:  {median(peak_times):.1f} seconds")
        print(f"  Earliest peak:     {min(peak_times):.1f} seconds")
        print(f"  Latest peak:       {max(peak_times):.1f} seconds")

        # Distribution
        peaks_0_30s = sum(1 for t in peak_times if t <= 30)
        peaks_30_60s = sum(1 for t in peak_times if 30 < t <= 60)
        peaks_60_120s = sum(1 for t in peak_times if t > 60)

        print(f"\nPeak Distribution:")
        print(f"  0-30s:   {peaks_0_30s}/{len(peak_times)} ({peaks_0_30s/len(peak_times)*100:.1f}%)")
        print(f"  30-60s:  {peaks_30_60s}/{len(peak_times)} ({peaks_30_60s/len(peak_times)*100:.1f}%)")
        print(f"  60-120s: {peaks_60_120s}/{len(peak_times)} ({peaks_60_120s/len(peak_times)*100:.1f}%)")

    print_section("RECOMMENDATION: WHEN TO BUY")

    if total > 0:
        early_pct = (total_0_10s + total_10_30s) / total * 100
        if early_pct > 60:
            print("\n✅ Most bot activity happens in first 30 seconds")
            print("   → Buy as early as possible (within 10s of launch)")
        else:
            print("\n⚠️ Bot activity is spread throughout 2 minutes")
            print("   → More time to position, but less predictable")

    if peak_times:
        avg_peak = mean(peak_times)
        if avg_peak < 45:
            print(f"\n✅ Average peak at {avg_peak:.0f}s - Fast exits work")
            print(f"   → Set exit timeouts at {avg_peak+10:.0f}-{avg_peak+20:.0f} seconds")
        else:
            print(f"\n⚠️ Average peak at {avg_peak:.0f}s - Need patience")
            print(f"   → Set exit timeouts at {avg_peak+10:.0f}-{avg_peak+20:.0f} seconds")


def analyze_position_sizes(tokens: List[Dict]):
    """Analyze HOW MUCH to buy"""
    print_header("POSITION SIZE ANALYSIS - HOW MUCH TO BUY")

    swarm_tokens = [t for t in tokens if t['bot_swarm_detected'] and t.get('first_significant_buy')]
    no_swarm_tokens = [t for t in tokens if not t['bot_swarm_detected'] and t.get('first_significant_buy')]

    if not swarm_tokens:
        print("\n⚠️ No bot swarms detected - cannot analyze position size impact")
        return

    print_section("First Buy Size vs Bot Swarm")

    # Group by first buy size
    small_first = [t for t in swarm_tokens if t['first_significant_buy']['sol_amount'] < 0.5]
    medium_first = [t for t in swarm_tokens if 0.5 <= t['first_significant_buy']['sol_amount'] < 1.0]
    large_first = [t for t in swarm_tokens if t['first_significant_buy']['sol_amount'] >= 1.0]

    print(f"\nTokens by First Buy Size:")
    print(f"  Small (<0.5 SOL):  {len(small_first):3d} tokens")
    print(f"  Medium (0.5-1.0):  {len(medium_first):3d} tokens")
    print(f"  Large (>1.0 SOL):  {len(large_first):3d} tokens")

    # Compare performance
    if small_first:
        avg_swarm = mean(t['bot_swarm_size'] for t in small_first)
        avg_peak = mean(t['peak_gain_pct'] for t in small_first)
        avg_volume = mean(t['total_volume_sol'] for t in small_first)
        print(f"\n  Small first buys → {avg_swarm:.1f} avg bots, {avg_peak:.1f}% avg peak, {avg_volume:.1f} SOL volume")

    if medium_first:
        avg_swarm = mean(t['bot_swarm_size'] for t in medium_first)
        avg_peak = mean(t['peak_gain_pct'] for t in medium_first)
        avg_volume = mean(t['total_volume_sol'] for t in medium_first)
        print(f"  Medium first buys → {avg_swarm:.1f} avg bots, {avg_peak:.1f}% avg peak, {avg_volume:.1f} SOL volume")

    if large_first:
        avg_swarm = mean(t['bot_swarm_size'] for t in large_first)
        avg_peak = mean(t['peak_gain_pct'] for t in large_first)
        avg_volume = mean(t['total_volume_sol'] for t in large_first)
        print(f"  Large first buys → {avg_swarm:.1f} avg bots, {avg_peak:.1f}% avg peak, {avg_volume:.1f} SOL volume")

    print_section("Overall Buy Size Distribution")

    # Aggregate all buys
    total_small = sum(t['position_size_analysis']['small_buys'] for t in swarm_tokens)
    total_medium = sum(t['position_size_analysis']['medium_buys'] for t in swarm_tokens)
    total_large = sum(t['position_size_analysis']['large_buys'] for t in swarm_tokens)
    total_whale = sum(t['position_size_analysis']['whale_buys'] for t in swarm_tokens)
    total = total_small + total_medium + total_large + total_whale

    if total > 0:
        print(f"\nAll buys in bot swarm tokens:")
        print(f"  Small (<0.1 SOL):    {total_small:4d} ({total_small/total*100:5.1f}%)")
        print(f"  Medium (0.1-0.5):    {total_medium:4d} ({total_medium/total*100:5.1f}%)")
        print(f"  Large (0.5-2.0):     {total_large:4d} ({total_large/total*100:5.1f}%)")
        print(f"  Whale (>2.0 SOL):    {total_whale:4d} ({total_whale/total*100:5.1f}%)")

    print_section("RECOMMENDATION: HOW MUCH TO BUY")

    # Find optimal size
    if small_first and medium_first:
        small_peak = mean(t['peak_gain_pct'] for t in small_first)
        medium_peak = mean(t['peak_gain_pct'] for t in medium_first)

        if medium_peak > small_peak * 1.2:
            print("\n✅ Medium buys (0.5-1.0 SOL) attract significantly better returns")
            print("   → Recommended position size: 0.5-0.8 SOL")
        elif small_peak > 30:
            print("\n✅ Small buys (<0.5 SOL) provide good returns with less risk")
            print("   → Recommended position size: 0.3-0.5 SOL")
        else:
            print("\n⚠️ Position size doesn't significantly impact returns")
            print("   → Start with small positions (0.2-0.4 SOL) to test")


def analyze_detection_latency(tokens: List[Dict]):
    """Analyze our detection speed - can we beat snipers?"""
    print_header("DETECTION LATENCY ANALYSIS - CAN WE BEAT SNIPERS?")

    # Extract latency data
    latencies = [t.get('detection_latency_analysis', {}).get('detection_delay_seconds', 0)
                 for t in tokens if t.get('detection_latency_analysis', {}).get('detection_delay_seconds', 0) > 0]

    if not latencies:
        print("\n⚠️ No latency data available")
        return

    print_section("Detection Speed Statistics")

    avg_latency = mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    median_latency = median(latencies)

    print(f"\nOur Detection Latency (time between first buy on-chain and our detection):")
    print(f"  Average:  {avg_latency:.3f} seconds")
    print(f"  Median:   {median_latency:.3f} seconds")
    print(f"  Best:     {min_latency:.3f} seconds")
    print(f"  Worst:    {max_latency:.3f} seconds")

    # Count how many we could potentially beat
    very_fast = sum(1 for l in latencies if l < 0.5)
    fast = sum(1 for l in latencies if 0.5 <= l < 1.0)
    moderate = sum(1 for l in latencies if 1.0 <= l < 2.0)
    slow = sum(1 for l in latencies if l >= 2.0)

    print(f"\nLatency Distribution:")
    print(f"  < 0.5s:   {very_fast}/{len(latencies)} ({very_fast/len(latencies)*100:.1f}%) - Might beat snipers")
    print(f"  0.5-1.0s: {fast}/{len(latencies)} ({fast/len(latencies)*100:.1f}%) - Early but not first")
    print(f"  1.0-2.0s: {moderate}/{len(latencies)} ({moderate/len(latencies)*100:.1f}%) - In early swarm")
    print(f"  > 2.0s:   {slow}/{len(latencies)} ({slow/len(latencies)*100:.1f}%) - Miss initial snipers")

    print_section("CAN WE BEAT SNIPERS?")

    if avg_latency < 0.5:
        print("\n✅ YES - Our detection is very fast!")
        print("   → We're seeing tokens within 0.5s of first buy")
        print("   → Can potentially beat slower snipe bots")
        print("   → Should be in the first 3-5 buys")
    elif avg_latency < 1.0:
        print("\n⚠️ MAYBE - Our detection is fast but not blazing")
        print("   → We're seeing tokens within 1s of first buy")
        print("   → Won't beat the fastest snipers")
        print("   → But can be in the early swarm (first 5-10 buys)")
    elif avg_latency < 2.0:
        print("\n⚠️ PROBABLY NOT - Our detection is moderate")
        print("   → We're seeing tokens 1-2s after first buy")
        print("   → Will miss the fastest snipers")
        print("   → But can still catch early swarm activity")
    else:
        print("\n❌ NO - Our detection is too slow")
        print("   → We're seeing tokens 2+ seconds after first buy")
        print("   → By then, snipers have already positioned")
        print("   → Strategy: Be part of secondary swarm, not first")

    print_section("IMPLICATIONS FOR STRATEGY")

    if avg_latency < 1.0:
        print("\n✅ Alpha Bait Strategy is VIABLE")
        print("   → We're fast enough to be an 'alpha' that others follow")
        print("   → Our buys will be visible to slower bots quickly")
        print("   → Can position before the main bot swarm arrives")
    else:
        print("\n⚠️ Alpha Bait Strategy may be DIFFICULT")
        print("   → We're not fast enough to be the 'alpha'")
        print("   → We'll be part of the swarm, not leading it")
        print("   → Consider pivoting to following faster alphas instead")


def analyze_token_characteristics(tokens: List[Dict]):
    """Analyze WHAT coins to buy"""
    print_header("TOKEN CHARACTERISTICS - WHAT TO BUY")

    swarm_tokens = [t for t in tokens if t['bot_swarm_detected']]

    if not swarm_tokens:
        print("\n⚠️ No bot swarms detected - cannot analyze token characteristics")
        return

    print_section("Basic Statistics")

    avg_buys = mean(t['total_buys'] for t in swarm_tokens)
    avg_volume = mean(t['total_volume_sol'] for t in swarm_tokens)
    avg_peak = mean(t['peak_gain_pct'] for t in swarm_tokens if t['peak_gain_pct'] > 0)
    avg_wallets = mean(len(t['unique_wallets']) if isinstance(t.get('unique_wallets'), list) else t.get('unique_wallets', 0) for t in swarm_tokens)

    print(f"\nBot Swarm Token Averages:")
    print(f"  Total buys:      {avg_buys:.1f}")
    print(f"  Unique wallets:  {avg_wallets:.1f}")
    print(f"  Total volume:    {avg_volume:.2f} SOL")
    print(f"  Peak gain:       {avg_peak:.1f}%")

    print_section("Atomic Dev Buy Analysis")

    # Analyze atomic vs non-atomic dev buys
    atomic_swarms = [t for t in swarm_tokens if t.get('atomic_dev_buy_analysis', {}).get('is_atomic_dev_buy', False)]
    non_atomic_swarms = [t for t in swarm_tokens if not t.get('atomic_dev_buy_analysis', {}).get('is_atomic_dev_buy', False)]

    print(f"\nAtomic vs Non-Atomic Dev Buys:")
    print(f"  Atomic dev buys:     {len(atomic_swarms)}/{len(swarm_tokens)} ({len(atomic_swarms)/len(swarm_tokens)*100:.1f}%)")
    print(f"  Non-atomic dev buys: {len(non_atomic_swarms)}/{len(swarm_tokens)} ({len(non_atomic_swarms)/len(swarm_tokens)*100:.1f}%)")

    if atomic_swarms and non_atomic_swarms:
        atomic_peak = mean(t['peak_gain_pct'] for t in atomic_swarms if t['peak_gain_pct'] > 0)
        non_atomic_peak = mean(t['peak_gain_pct'] for t in non_atomic_swarms if t['peak_gain_pct'] > 0)
        atomic_swarm_size = mean(t['bot_swarm_size'] for t in atomic_swarms)
        non_atomic_swarm_size = mean(t['bot_swarm_size'] for t in non_atomic_swarms)

        print(f"\n  Performance Comparison:")
        print(f"    Atomic:     {atomic_peak:.1f}% avg peak, {atomic_swarm_size:.1f} avg bots")
        print(f"    Non-atomic: {non_atomic_peak:.1f}% avg peak, {non_atomic_swarm_size:.1f} avg bots")

        if atomic_peak > non_atomic_peak * 1.2:
            print(f"\n  ✅ Atomic dev buys perform significantly better!")
        elif atomic_peak < non_atomic_peak * 0.8:
            print(f"\n  ⚠️ Non-atomic dev buys perform better (unexpected!)")
        else:
            print(f"\n  ➡️ No significant performance difference")

    print_section("Developer Analysis")
    print("\n⚠️ Developer history tracking not yet implemented")
    print("   TODO: Query on-chain data for dev wallet previous token launches")

    print_section("RECOMMENDATION: WHAT TO BUY")

    if atomic_swarms:
        atomic_pct = len(atomic_swarms) / len(swarm_tokens) * 100
        if atomic_pct > 60:
            print("\n✅ Atomic dev buys are common and correlate with bot swarms")
            print("   → Prioritize tokens where first significant buy is atomic (slot 0)")
            print("   → These may be more 'legitimate' launches")
        else:
            print("\n⚠️ Non-atomic dev buys are more common")
            print("   → May indicate devs buying after observing initial interest")

    print("\n  General approach:")
    print("  → Buy all new launches that meet timing criteria")
    print("  → Consider atomic dev buys as positive signal")
    print("  → Filter out obvious rugs (no liquidity, suspicious metadata)")
    print("  → Use aggressive stops to limit downside")


def generate_summary(data: Dict):
    """Generate actionable summary"""
    print_header("ACTIONABLE SUMMARY")

    tokens = data['tokens']
    swarm_tokens = [t for t in tokens if t['bot_swarm_detected']]

    print("\nStrategy Viability Assessment:\n")

    # Criterion 1: Bot swarm frequency
    swarm_freq = len(swarm_tokens) / len(tokens) * 100 if tokens else 0
    print(f"1. Bot Swarm Frequency: {swarm_freq:.1f}%")
    if swarm_freq > 20:
        print(f"   ✅ PASS - Enough opportunities ({swarm_freq:.1f}% > 20%)")
    else:
        print(f"   ❌ FAIL - Too few opportunities ({swarm_freq:.1f}% < 20%)")

    # Criterion 2: Average peak gain
    if swarm_tokens:
        avg_peak = mean(t['peak_gain_pct'] for t in swarm_tokens if t['peak_gain_pct'] > 0)
        print(f"\n2. Average Peak Gain: {avg_peak:.1f}%")
        if avg_peak > 30:
            print(f"   ✅ PASS - Good reward:risk ratio ({avg_peak:.1f}% > 30%)")
        else:
            print(f"   ⚠️ BORDERLINE - Marginal reward:risk ({avg_peak:.1f}% ~ 30%)")

        # Criterion 3: Peak timing
        peak_times = [t['peak_time_seconds'] for t in swarm_tokens if t['peak_time_seconds'] > 0]
        if peak_times:
            avg_peak_time = mean(peak_times)
            print(f"\n3. Average Peak Timing: {avg_peak_time:.1f} seconds")
            if avg_peak_time < 60:
                print(f"   ✅ PASS - Fast enough for aggressive exits ({avg_peak_time:.1f}s < 60s)")
            else:
                print(f"   ⚠️ CONCERN - May need longer hold times ({avg_peak_time:.1f}s > 60s)")

        # Criterion 4: Volume
        avg_volume = mean(t['total_volume_sol'] for t in swarm_tokens)
        print(f"\n4. Average Swarm Volume: {avg_volume:.2f} SOL")
        if avg_volume > 5:
            print(f"   ✅ PASS - Sufficient liquidity ({avg_volume:.2f} SOL > 5 SOL)")
        else:
            print(f"   ⚠️ CONCERN - Limited liquidity ({avg_volume:.2f} SOL < 5 SOL)")

    print("\n" + "="*80)
    print("FINAL RECOMMENDATION:")
    print("="*80)

    if swarm_freq > 20 and swarm_tokens:
        avg_peak = mean(t['peak_gain_pct'] for t in swarm_tokens if t['peak_gain_pct'] > 0)
        if avg_peak > 25:
            print("\n✅ Strategy appears VIABLE")
            print("   → Proceed with implementation")
            print("   → Start with small positions (0.2-0.3 SOL)")
            print("   → Use aggressive stops (15-20%)")
            print("   → Set profit targets based on peak timing analysis")
        else:
            print("\n⚠️ Strategy is MARGINAL")
            print("   → Returns may not justify risk")
            print("   → Consider collecting more data")
            print("   → Or adjust strategy parameters")
    else:
        print("\n❌ Strategy does NOT appear viable")
        print("   → Not enough bot swarm opportunities")
        print("   → Collect more data or explore different approach")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_alpha_bait_results.py <json_file>")
        print("\nExample:")
        print("  python scripts/analyze_alpha_bait_results.py data/alpha_bait_analysis/alpha_bait_analysis_20251015_143022.json")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Load results
    print("Loading results...")
    data = load_results(file_path)
    tokens = data['tokens']

    # Run analysis
    analyze_metadata(data)
    analyze_bot_swarms(tokens)
    analyze_detection_latency(tokens)  # NEW: Can we beat snipers?
    analyze_timing(tokens)
    analyze_position_sizes(tokens)
    analyze_token_characteristics(tokens)
    generate_summary(data)

    print("\n")


if __name__ == "__main__":
    main()
