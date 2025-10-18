"""
Analyze Opportunity Tracker Data

Analyzes recorded alpha signals to determine:
- How many opportunities at each threshold
- Which threshold would have been most profitable
- Token performance after alpha signals
- Optimal stop loss and take profit strategies

Run after bot session to analyze: python scripts/analyze_opportunities.py <session_file>
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeSimulation:
    """Result of a simulated trade"""
    entry_threshold: int
    entry_price: float
    exit_price: float
    exit_reason: str  # 'stop_loss', 'take_profit', 'trailing_stop', 'hold'
    exit_time_seconds: float
    profit_pct: float
    strategy_name: str


def load_signals(filepath: str) -> Dict:
    """Load signals from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def simulate_trade_with_strategy(
    entry_price: float,
    price_snapshots: List[Dict],
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    partial_exits: Optional[List[Tuple[float, float]]] = None
) -> Tuple[float, str, float, float]:
    """
    Simulate a trade with given strategy parameters

    Args:
        entry_price: Entry price
        price_snapshots: List of price snapshots over time
        stop_loss_pct: Stop loss percentage (e.g., -15 for -15%)
        take_profit_pct: Take profit percentage (e.g., 50 for +50%)
        trailing_stop_pct: Trailing stop from peak (e.g., 20 for -20% from peak)
        partial_exits: List of (target_pct, exit_portion) tuples

    Returns:
        Tuple of (exit_price, exit_reason, exit_time, total_profit_pct)
    """
    if not price_snapshots:
        return entry_price, 'no_data', 0, 0

    peak_price = entry_price
    position_size = 1.0  # Track remaining position (for partial exits)
    total_profit = 0.0
    remaining_exits = list(partial_exits) if partial_exits else []
    hit_targets = set()

    for snapshot in price_snapshots:
        current_price = snapshot['price_usd']
        current_time = snapshot['seconds_after_signal']
        current_pct = ((current_price / entry_price) - 1) * 100

        # Update peak for trailing stop
        if current_price > peak_price:
            peak_price = current_price

        # Check stop loss (full exit)
        if stop_loss_pct and current_pct <= stop_loss_pct:
            total_profit += position_size * current_pct
            return current_price, 'stop_loss', current_time, total_profit

        # Check partial exits first (before full TP)
        if remaining_exits and position_size > 0:
            for target_pct, exit_portion in remaining_exits[:]:
                if current_pct >= target_pct and target_pct not in hit_targets:
                    # Exit this portion
                    exit_size = position_size * exit_portion
                    total_profit += exit_size * current_pct
                    position_size -= exit_size
                    hit_targets.add(target_pct)

                    # If position is fully closed
                    if position_size <= 0.01:
                        return current_price, 'partial_exit_full', current_time, total_profit

        # Check take profit (full exit)
        if take_profit_pct and current_pct >= take_profit_pct:
            total_profit += position_size * current_pct
            return current_price, 'take_profit', current_time, total_profit

        # Check trailing stop
        if trailing_stop_pct and peak_price > entry_price:
            drawdown_from_peak = ((current_price / peak_price) - 1) * 100
            if drawdown_from_peak <= -trailing_stop_pct:
                total_profit += position_size * current_pct
                return current_price, 'trailing_stop', current_time, total_profit

    # Held until end
    final_snapshot = price_snapshots[-1]
    final_price = final_snapshot['price_usd']
    final_time = final_snapshot['seconds_after_signal']
    final_pct = ((final_price / entry_price) - 1) * 100

    total_profit += position_size * final_pct

    return final_price, 'hold', final_time, total_profit


def simulate_tiered_exit_with_trailing(
    entry_price: float,
    price_snapshots: List[Dict],
    exits: List[Tuple[float, float]],  # (target_pct, exit_portion)
    trailing_stop_pct: float,
    hard_stop_pct: float
) -> Tuple[float, str, float, float]:
    """
    Simulate a sophisticated tiered exit strategy with trailing stop on remainder

    This mimics real trading: take profits at levels, keep moon bag with trailing stop
    """
    if not price_snapshots:
        return entry_price, 'no_data', 0, 0

    position_size = 1.0
    total_profit = 0.0
    peak_price = entry_price
    hit_targets = set()

    for snapshot in price_snapshots:
        current_price = snapshot['price_usd']
        current_time = snapshot['seconds_after_signal']
        current_pct = ((current_price / entry_price) - 1) * 100

        # Update peak
        if current_price > peak_price:
            peak_price = current_price

        # Hard stop loss (full exit)
        if current_pct <= hard_stop_pct:
            total_profit += position_size * current_pct
            return current_price, 'stop_loss', current_time, total_profit

        # Check tiered profit targets
        for target_pct, exit_portion in exits:
            if current_pct >= target_pct and target_pct not in hit_targets:
                exit_size = position_size * exit_portion
                total_profit += exit_size * current_pct
                position_size -= exit_size
                hit_targets.add(target_pct)

        # Trailing stop on remaining position (only if in profit)
        if position_size > 0 and peak_price > entry_price:
            drawdown_from_peak = ((current_price / peak_price) - 1) * 100
            if drawdown_from_peak <= -trailing_stop_pct:
                total_profit += position_size * current_pct
                return current_price, 'trailing_stop', current_time, total_profit

    # Position held to end
    final_snapshot = price_snapshots[-1]
    final_price = final_snapshot['price_usd']
    final_time = final_snapshot['seconds_after_signal']
    final_pct = ((final_price / entry_price) - 1) * 100
    total_profit += position_size * final_pct

    return final_price, 'hold', final_time, total_profit


def get_trading_strategies() -> List[Dict]:
    """
    Define realistic trading strategies that professionals would use

    Each strategy has a clear philosophy and use case
    """
    return [
        # 1. CONSERVATIVE SCALPER
        {
            'name': 'Conservative Scalper',
            'description': 'Take quick +25% profit with tight 10% trailing',
            'exits': [(25, 1.0)],  # 100% at +25%
            'trailing': 10,
            'hard_stop': -20,
            'philosophy': 'Quick in, quick out - accumulate small wins'
        },

        # 2. CAUTIOUS TRADER
        {
            'name': 'Cautious Trader',
            'description': '50% at +30%, 50% at +60% with -20% SL',
            'exits': [(30, 0.5), (60, 0.5)],
            'trailing': 15,
            'hard_stop': -20,
            'philosophy': 'Two clear profit targets, tight risk control'
        },

        # 3. BALANCED 3-TIER
        {
            'name': 'Balanced 3-Tier',
            'description': '40% at +40%, 30% at +80%, 30% trails -20%',
            'exits': [(40, 0.4), (80, 0.3)],
            'trailing': 20,
            'hard_stop': -25,
            'philosophy': 'Scale out methodically, keep runner'
        },

        # 4. AGGRESSIVE 3-TIER
        {
            'name': 'Aggressive 3-Tier',
            'description': '30% at +75%, 30% at +150%, 40% trails -25%',
            'exits': [(75, 0.3), (150, 0.3)],
            'trailing': 25,
            'hard_stop': -30,
            'philosophy': 'Hold for bigger gains, larger moon bag'
        },

        # 5. SMART SCALING
        {
            'name': 'Smart Scaling',
            'description': '40% at +50%, 20% at +100%, 15% at +200%, 25% trails -25%',
            'exits': [(50, 0.4), (100, 0.2), (200, 0.15)],
            'trailing': 25,
            'hard_stop': -25,
            'philosophy': 'Your suggestion - progressive scaling with moon bag'
        },

        # 6. MOON CHASER
        {
            'name': 'Moon Chaser',
            'description': '20% at +50%, 20% at +100%, 60% trails -30%',
            'exits': [(50, 0.2), (100, 0.2)],
            'trailing': 30,
            'hard_stop': -30,
            'philosophy': 'Keep most for massive gains, wide trailing'
        },

        # 7. QUICK DOUBLE
        {
            'name': 'Quick Double',
            'description': '100% exit at +100% with -25% SL',
            'exits': [(100, 1.0)],
            'trailing': 15,
            'hard_stop': -25,
            'philosophy': 'Target 2x, nothing more'
        },

        # 8. CONSERVATIVE 4-TIER
        {
            'name': 'Conservative 4-Tier',
            'description': '35% at +30%, 25% at +50%, 20% at +75%, 20% trails -15%',
            'exits': [(30, 0.35), (50, 0.25), (75, 0.2)],
            'trailing': 15,
            'hard_stop': -20,
            'philosophy': 'Take profits early and often, small moon bag'
        },

        # 9. PATIENT HOLDER
        {
            'name': 'Patient Holder',
            'description': '25% at +150%, 25% at +300%, 50% trails -35%',
            'exits': [(150, 0.25), (300, 0.25)],
            'trailing': 35,
            'hard_stop': -30,
            'philosophy': 'Diamond hands - wait for huge moves'
        },

        # 10. RISK-MANAGED SCALING
        {
            'name': 'Risk-Managed Scaling',
            'description': '50% at +40%, 25% at +80%, 25% trails -20%',
            'exits': [(40, 0.5), (80, 0.25)],
            'trailing': 20,
            'hard_stop': -20,
            'philosophy': 'Secure profits quickly, controlled risk'
        },

        # 11. TIGHT SCALPER
        {
            'name': 'Tight Scalper',
            'description': '100% exit at +15% with trailing -8%',
            'exits': [(15, 1.0)],
            'trailing': 8,
            'hard_stop': -15,
            'philosophy': 'Maximum safety, accept small gains'
        },

        # 12. RUNNER FOCUSED
        {
            'name': 'Runner Focused',
            'description': '30% at +60%, 20% at +120%, 50% trails -30%',
            'exits': [(60, 0.3), (120, 0.2)],
            'trailing': 30,
            'hard_stop': -30,
            'philosophy': 'Half position for moon shots'
        },

        # 13. SWING TRADER
        {
            'name': 'Swing Trader',
            'description': '40% at +100%, 30% at +200%, 30% trails -30%',
            'exits': [(100, 0.4), (200, 0.3)],
            'trailing': 30,
            'hard_stop': -30,
            'philosophy': 'Target large swings, wide stops'
        },

        # 14. CONSERVATIVE MOON BAG
        {
            'name': 'Conservative Moon Bag',
            'description': '60% at +40%, 20% at +80%, 20% trails -20%',
            'exits': [(40, 0.6), (80, 0.2)],
            'trailing': 20,
            'hard_stop': -20,
            'philosophy': 'Lock profits early, small lottery ticket'
        },

        # 15. DEGENERATE HOLD
        {
            'name': 'Degenerate Hold',
            'description': '10% at +200%, 90% trails -40%',
            'exits': [(200, 0.1)],
            'trailing': 40,
            'hard_stop': -35,
            'philosophy': 'Maximum risk - hold for 10x or bust'
        }
    ]


def test_strategies_for_signal(signal: Dict) -> List[TradeSimulation]:
    """
    Test all realistic trading strategies for a single signal

    Returns list of TradeSimulation results
    """
    results = []

    threshold_performance = signal.get('threshold_performance', {})
    price_snapshots = signal.get('price_snapshots', [])

    if not threshold_performance or not price_snapshots:
        return results

    # Get all strategies to test
    strategies = get_trading_strategies()

    # Test each entry threshold
    for threshold in [1, 2, 3]:
        threshold_key = f'threshold_{threshold}'
        threshold_data = threshold_performance.get(threshold_key)

        if not threshold_data:
            continue

        entry_price = threshold_data['entry_price_usd']
        if not entry_price or entry_price <= 0:
            continue

        # Test each strategy
        for strategy in strategies:
            exit_price, reason, exit_time, profit = simulate_tiered_exit_with_trailing(
                entry_price,
                price_snapshots,
                exits=strategy['exits'],
                trailing_stop_pct=strategy['trailing'],
                hard_stop_pct=strategy['hard_stop']
            )

            results.append(TradeSimulation(
                entry_threshold=threshold,
                entry_price=entry_price,
                exit_price=exit_price,
                exit_reason=reason,
                exit_time_seconds=exit_time,
                profit_pct=profit,
                strategy_name=strategy['name']
            ))

    return results


def analyze_signals(signals_data: Dict, filepath: str):
    """Analyze signal data and print summary"""
    signals = signals_data['signals']

    if not signals:
        print("No signals found in this session")
        return

    print("=" * 80)
    print("OPPORTUNITY ANALYSIS")
    print("=" * 80)
    print(f"Session: {signals_data['session_start']}")
    print(f"Last update: {signals_data['last_update']}")
    print(f"Total unique tokens: {len(signals)}")
    print()

    # Count by threshold
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    threshold_counts = {}

    for threshold in thresholds:
        count = len([s for s in signals if s['wallet_count'] >= threshold])
        threshold_counts[threshold] = count

    print("Opportunities by Alpha Wallet Threshold:")
    print("-" * 80)
    for threshold in thresholds:
        count = threshold_counts[threshold]
        if count > 0:
            pct = (count / len(signals)) * 100
            print(f"  {threshold}+ wallets: {count:4d} tokens ({pct:5.1f}%)")
    print()

    # Analyze signal quality
    print("Signal Quality Analysis:")
    print("-" * 80)

    # Average SOL amounts by threshold
    for threshold in [1, 2, 3, 4, 5]:
        threshold_signals = [s for s in signals if s['wallet_count'] >= threshold]
        if threshold_signals:
            avg_total_sol = sum(s['total_sol'] for s in threshold_signals) / len(threshold_signals)
            avg_wallets = sum(s['wallet_count'] for s in threshold_signals) / len(threshold_signals)
            print(f"  {threshold}+ wallets: avg {avg_total_sol:.3f} SOL, avg {avg_wallets:.1f} wallets")
    print()

    # Price Performance Analysis
    signals_with_threshold_data = [s for s in signals if s.get('threshold_performance')]
    signals_with_price = [s for s in signals if s.get('entry_price_usd') and s.get('peak_gain_pct') is not None]

    if signals_with_threshold_data:
        print("Price Performance Analysis (By Entry Threshold):")
        print("-" * 80)
        print(f"Tracked {len(signals_with_threshold_data)} tokens with threshold price data ({len(signals_with_threshold_data)/len(signals)*100:.1f}%)")
        print()

        # Collect all threshold performance data
        threshold_stats = {}
        for signal in signals_with_threshold_data:
            threshold_perf = signal.get('threshold_performance', {})
            for threshold_key, perf in threshold_perf.items():
                threshold = int(threshold_key.split('_')[1])  # Extract number from 'threshold_X'

                if threshold not in threshold_stats:
                    threshold_stats[threshold] = []

                threshold_stats[threshold].append({
                    'peak_gain': perf['peak_gain_pct'],
                    'current_gain': perf['current_gain_pct'],
                    'entry_price': perf['entry_price_usd']
                })

        # Display stats for each threshold
        print("Performance if entering at each threshold:")
        print("-" * 80)
        for threshold in sorted(threshold_stats.keys()):
            stats = threshold_stats[threshold]
            avg_peak = sum(s['peak_gain'] for s in stats) / len(stats)
            avg_current = sum(s['current_gain'] for s in stats) / len(stats)
            winners = len([s for s in stats if s['peak_gain'] > 0])
            win_rate = (winners / len(stats)) * 100

            # Show how many opportunities exist at this threshold
            tokens_at_threshold = len([s for s in signals if s['wallet_count'] >= threshold])

            print(f"  {threshold} wallet(s): {tokens_at_threshold} opportunities, "
                  f"avg peak {avg_peak:+.2f}%, avg current {avg_current:+.2f}%, "
                  f"win rate {win_rate:.1f}% ({winners}/{len(stats)} tracked)")
        print()

    elif signals_with_price:
        # Fallback to old analysis if no threshold data
        print("Price Performance Analysis:")
        print("-" * 80)
        print(f"Tracked {len(signals_with_price)} tokens with price data ({len(signals_with_price)/len(signals)*100:.1f}%)")
        print()

        # Average performance by threshold (old method - measures from first signal)
        for threshold in [1, 2, 3, 4, 5]:
            threshold_price_signals = [s for s in signals_with_price if s['wallet_count'] >= threshold]
            if threshold_price_signals:
                avg_peak = sum(s['peak_gain_pct'] for s in threshold_price_signals) / len(threshold_price_signals)
                avg_current = sum(s['current_gain_pct'] for s in threshold_price_signals) / len(threshold_price_signals)
                winners = len([s for s in threshold_price_signals if s['peak_gain_pct'] > 0])
                win_rate = (winners / len(threshold_price_signals)) * 100
                print(f"  {threshold}+ wallets: avg peak {avg_peak:+.2f}%, avg current {avg_current:+.2f}%, win rate {win_rate:.1f}% ({winners}/{len(threshold_price_signals)})")
        print()

        # Top performers with threshold data
        if signals_with_threshold_data:
            print("Top 20 Price Performers (by 1-wallet entry peak gain):")
            print("-" * 80)
            print(f"{'Token':<13} {'Wallets':>7} {'1W Entry $':>12} {'Peak $':>12} {'1W Peak':>11} {'3W Entry $':>12} {'3W Peak':>11}")
            print("-" * 80)

            # Sort by 1-wallet threshold performance
            sortable = []
            for signal in signals_with_threshold_data:
                threshold_1 = signal.get('threshold_performance', {}).get('threshold_1', {})
                if threshold_1:
                    sortable.append((signal, threshold_1.get('peak_gain_pct', 0)))

            sorted_by_price = sorted(sortable, key=lambda x: x[1], reverse=True)[:20]

            for signal, _ in sorted_by_price:
                token = signal['token_mint'][:8] + "..."
                wallets = signal['wallet_count']
                peak = signal.get('peak_price_usd', 0)

                # Get 1-wallet and 3-wallet threshold data
                threshold_1 = signal.get('threshold_performance', {}).get('threshold_1', {})
                threshold_3 = signal.get('threshold_performance', {}).get('threshold_3', {})

                entry_1 = threshold_1.get('entry_price_usd', 0)
                gain_1 = threshold_1.get('peak_gain_pct', 0)
                entry_3 = threshold_3.get('entry_price_usd', 0) if threshold_3 else 0
                gain_3 = threshold_3.get('peak_gain_pct', 0) if threshold_3 else 0

                if entry_3 > 0:
                    print(f"{token:<13} {wallets:>7} ${entry_1:>11.8f} ${peak:>11.8f} {gain_1:>+10.2f}% ${entry_3:>11.8f} {gain_3:>+10.2f}%")
                else:
                    print(f"{token:<13} {wallets:>7} ${entry_1:>11.8f} ${peak:>11.8f} {gain_1:>+10.2f}% {'N/A':>12} {'N/A':>11}")
            print()

        # Top performers (fallback to old format)
        elif signals_with_price:
            print("Top 20 Price Performers (by peak gain):")
            print("-" * 80)
            print(f"{'Token':<13} {'Wallets':>7} {'Entry $':>12} {'Peak $':>12} {'Peak Gain':>11} {'Current $':>12} {'Current Gain':>13}")
            print("-" * 80)

            sorted_by_price = sorted(signals_with_price, key=lambda x: x['peak_gain_pct'], reverse=True)[:20]
            for signal in sorted_by_price:
                token = signal['token_mint'][:8] + "..."
                wallets = signal['wallet_count']
                entry = signal['entry_price_usd']
                peak = signal['peak_price_usd']
                peak_gain = signal['peak_gain_pct']
                latest = signal['latest_price_usd']
                current_gain = signal['current_gain_pct']
                print(f"{token:<13} {wallets:>7} ${entry:>11.8f} ${peak:>11.8f} {peak_gain:>+10.2f}% ${latest:>11.8f} {current_gain:>+12.2f}%")
            print()

    # Top signals
    print("Top 20 Signals (by wallet count):")
    print("-" * 80)
    print(f"{'Token':<45} {'Wallets':>8} {'Total SOL':>10} {'Time':<20}")
    print("-" * 80)

    sorted_signals = sorted(signals, key=lambda x: x['wallet_count'], reverse=True)[:20]
    for signal in sorted_signals:
        token = signal['token_mint'][:8] + "..."
        wallets = signal['wallet_count']
        total_sol = signal['total_sol']
        time = signal['datetime'].split('T')[1][:8]  # Just HH:MM:SS
        print(f"{token:<45} {wallets:>8} {total_sol:>10.3f} {time:<20}")
    print()

    # Distribution analysis
    print("Wallet Count Distribution:")
    print("-" * 80)
    wallet_count_dist = {}
    for signal in signals:
        count = signal['wallet_count']
        wallet_count_dist[count] = wallet_count_dist.get(count, 0) + 1

    for count in sorted(wallet_count_dist.keys()):
        tokens = wallet_count_dist[count]
        pct = (tokens / len(signals)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {count:2d} wallets: {tokens:4d} tokens ({pct:5.1f}%) {bar}")
    print()

    # Strategy Simulation Analysis
    print("\n" + "=" * 80)
    print("STRATEGY SIMULATION ANALYSIS")
    print("=" * 80)
    print()
    print("Testing 15 REALISTIC trading strategies across 3 entry thresholds (1, 2, 3 alpha wallets)")
    print()
    print("Strategies include:")
    print("  • Conservative Scalper: Quick +25% with tight trailing")
    print("  • Smart Scaling: Your suggestion - 40%@+50%, 20%@+100%, 15%@+200%, 25% moon bag")
    print("  • Moon Chaser: Hold 60% for massive gains with wide trailing")
    print("  • Quick Double: Target 2x and exit")
    print("  • Balanced/Aggressive 3-Tier exits with moon bags")
    print("  • Patient Holder: Diamond hands for 3x-10x")
    print("  • Risk-Managed, Runner Focused, Swing Trader, and more...")
    print()
    print("Each strategy has:")
    print("  • Tiered profit targets (TP1, TP2, TP3)")
    print("  • Position sizing at each level (e.g., 40%, 20%, 15%)")
    print("  • Trailing stop on remaining 'moon bag'")
    print("  • Hard stop loss for risk management")
    print()
    print("This simulates REAL trading - not just simple SL/TP combos.")
    print("Based on actual price movements captured during alpha signals.")
    print()

    # Run simulations for all signals with price data
    all_simulations = []
    signals_with_full_data = [s for s in signals if s.get('threshold_performance') and s.get('price_snapshots')]

    if signals_with_full_data:
        print(f"Running strategy simulations for {len(signals_with_full_data)} tokens with complete data...")
        print()

        for signal in signals_with_full_data:
            signal_sims = test_strategies_for_signal(signal)
            all_simulations.extend(signal_sims)

        if all_simulations:
            # Group by strategy and threshold
            strategy_stats = {}
            for sim in all_simulations:
                key = (sim.entry_threshold, sim.strategy_name)
                if key not in strategy_stats:
                    strategy_stats[key] = []
                strategy_stats[key].append(sim.profit_pct)

            # Display results organized by threshold
            for threshold in [1, 2, 3]:
                print(f"\n{'='*80}")
                print(f"THRESHOLD {threshold} WALLET{'S' if threshold > 1 else ''} - STRATEGY COMPARISON")
                print(f"{'='*80}")

                threshold_strategies = [(k, v) for k, v in strategy_stats.items() if k[0] == threshold]

                if not threshold_strategies:
                    print(f"No data available for {threshold}-wallet threshold")
                    continue

                print(f"{'Strategy':<45} {'Trades':>7} {'Avg P/L':>10} {'Win Rate':>9} {'Median':>10} {'Best':>10} {'Worst':>10}")
                print("-" * 80)

                # Sort by average P/L
                sorted_strategies = sorted(threshold_strategies, key=lambda x: sum(x[1])/len(x[1]), reverse=True)

                for (thresh, strat_name), profits in sorted_strategies:
                    trades = len(profits)
                    avg_profit = sum(profits) / trades
                    winners = len([p for p in profits if p > 0])
                    win_rate = (winners / trades) * 100
                    median_profit = sorted(profits)[len(profits)//2]
                    best = max(profits)
                    worst = min(profits)

                    print(f"{strat_name:<45} {trades:>7} {avg_profit:>9.2f}% {win_rate:>8.1f}% {median_profit:>9.2f}% {best:>9.2f}% {worst:>9.2f}%")

            # Overall best strategies
            print(f"\n{'='*80}")
            print("TOP 10 BEST STRATEGIES (ALL THRESHOLDS)")
            print(f"{'='*80}")
            print(f"{'Rank':>4} {'Threshold':>10} {'Strategy':<35} {'Avg P/L':>10} {'Win Rate':>9} {'Trades':>7}")
            print("-" * 80)

            all_strategy_stats = []
            for (thresh, strat_name), profits in strategy_stats.items():
                avg_profit = sum(profits) / len(profits)
                win_rate = (len([p for p in profits if p > 0]) / len(profits)) * 100
                all_strategy_stats.append((thresh, strat_name, avg_profit, win_rate, len(profits)))

            sorted_all = sorted(all_strategy_stats, key=lambda x: x[2], reverse=True)[:10]

            for rank, (thresh, strat, avg_pnl, win_rate, trades) in enumerate(sorted_all, 1):
                print(f"{rank:>4} {thresh:>10} {strat:<35} {avg_pnl:>9.2f}% {win_rate:>8.1f}% {trades:>7}")

            # Show strategy descriptions for top 5
            print(f"\n{'='*80}")
            print("TOP 5 STRATEGY DETAILS")
            print(f"{'='*80}")

            strategies_dict = {s['name']: s for s in get_trading_strategies()}
            for rank, (thresh, strat, avg_pnl, win_rate, trades) in enumerate(sorted_all[:5], 1):
                if strat in strategies_dict:
                    strategy = strategies_dict[strat]
                    print(f"\n#{rank}. {strat} @ {thresh}-wallet threshold")
                    print(f"    Description: {strategy['description']}")
                    print(f"    Philosophy: {strategy['philosophy']}")
                    print(f"    Performance: Avg {avg_pnl:+.2f}%, Win Rate {win_rate:.1f}% ({trades} trades)")

            print()

        else:
            print("No simulations could be run - signals lack price snapshot data")
    else:
        print("No signals with complete threshold and price data for simulation")

    # Output file info
    print("\n" + "=" * 80)
    print(f"Data file: {filepath}")
    if signals_with_price:
        print(f"Price data available for {len(signals_with_price)}/{len(signals)} tokens")
        print(f"Complete simulation data for {len(signals_with_full_data)} tokens")
        print("Use this analysis to determine optimal entry threshold and exit strategy")
    else:
        print("No price data available - run bot with Moralis client to track price performance")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        # Find most recent session file
        data_dir = Path("data/opportunities")
        if not data_dir.exists():
            print("No opportunity data found. Run the bot first to collect data.")
            return

        session_files = sorted(data_dir.glob("signals_*.json"))
        if not session_files:
            print("No session files found in data/opportunities/")
            return

        filepath = session_files[-1]
        print(f"Using most recent session: {filepath.name}")
        print()
    else:
        filepath = Path(sys.argv[1])
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return

    signals_data = load_signals(filepath)
    analyze_signals(signals_data, str(filepath))


if __name__ == "__main__":
    main()
