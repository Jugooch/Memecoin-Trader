#!/usr/bin/env python3
"""
Run the new profitable trading strategies simulation
Target: $100-125 daily profit with realistic fees and slippage
"""

import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sims.strategy_simulator_v2 import StrategySimulator

async def main():
    """Run the profitable strategies simulation"""
    
    print("=" * 60)
    print("PROFITABLE STRATEGIES SIMULATION")
    print("=" * 60)
    print()
    print("Testing 3 strategies with realistic fees:")
    print("1. Quick Scalp   - $150 pos, 6% TP,  Target: $40/day")
    print("2. Momentum Ride - $100 pos, 12% TP, Target: $45/day")  
    print("3. Moon Chase    - $200 pos, 15%+ TP, Target: $40/day")
    print()
    print("Combined target: $100-125/day profit")
    print("=" * 60)
    print()
    
    # Get duration from command line or default to 24 hours
    duration_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    
    print(f"Simulation duration: {duration_hours} hours")
    print()
    
    # Initialize simulator
    print("Initializing simulator...")
    simulator = StrategySimulator('config.yml')
    
    # Verify all strategies are loaded
    loaded_strategies = list(simulator.strategies.keys())
    print(f"Loaded strategies: {loaded_strategies}")
    print()
    
    required_strategies = ['quick_scalp', 'momentum_ride', 'moon_chase']
    missing = [s for s in required_strategies if s not in loaded_strategies]
    
    if missing:
        print(f"ERROR: Missing strategies: {missing}")
        print("Please ensure all strategy config files are in sims/configs/")
        return 1
    
    # Clear the ALL_TRADES.txt file
    trades_file = Path("sims/ALL_TRADES.txt")
    trades_file.write_text("")
    print("Cleared ALL_TRADES.txt for fresh results")
    print()
    
    # Display strategy details
    print("Strategy Configuration Summary:")
    print("-" * 40)
    
    for strategy_name in required_strategies:
        config = simulator.strategies[strategy_name]
        entry = config.entry_config
        exits = config.exit_config
        sim = config.simulation_config
        
        print(f"\n{strategy_name.upper()}:")
        print(f"  Position Size: ${entry.get('position_size_usd', 30)}")
        print(f"  Min Alphas: {entry.get('min_alpha_wallets', 1)}")
        
        # Get TP1 info
        tp_config = exits.get('take_profit', {})
        tp1 = tp_config.get('tp1_percentage', 0)
        tp1_sell = tp_config.get('tp1_sell_amount', 0) * 100
        print(f"  TP1: +{tp1}% (sell {tp1_sell:.0f}%)")
        
        # Show fees
        print(f"  Entry Fee: ${sim.get('entry_priority_fee_usd', 1.0):.2f}")
        print(f"  Entry Slippage: {sim.get('entry_slippage_pct', 2.0)}%")
    
    print()
    print("-" * 40)
    print()
    
    # Inform about Discord notifications
    print("Discord Notifications:")
    print("- Trade exits will be notified immediately")
    print("- Summary updates every 10 minutes")
    print("- Final report at simulation end")
    print()
    
    # Run the simulation
    print("Starting simulation...")
    print("=" * 60)
    print()
    
    try:
        await simulator.run_simulation(duration_hours=duration_hours)
        print()
        print("=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print()
        
        # Print final statistics
        print("Final Statistics:")
        print(f"- Total Alpha Signals: {simulator.stats['total_alpha_signals']}")
        print(f"- Positions Created: {simulator.stats['positions_created']}")
        print(f"- Positions Closed: {simulator.stats['positions_closed']}")
        print(f"- Tokens Analyzed: {simulator.stats['tokens_analyzed']}")
        print()
        
        # Calculate totals per strategy
        print("Results by Strategy:")
        print("-" * 40)
        
        for strategy_name in required_strategies:
            completed = simulator.completed_positions.get(strategy_name, [])
            if completed:
                total_pnl = sum(p.cumulative_profit_usd for p in completed)
                wins = len([p for p in completed if p.cumulative_profit_pct > 0])
                losses = len([p for p in completed if p.cumulative_profit_pct < 0])
                win_rate = (wins / len(completed) * 100) if completed else 0
                
                print(f"\n{strategy_name.upper()}:")
                print(f"  Trades: {len(completed)}")
                print(f"  Win Rate: {win_rate:.1f}% ({wins}W/{losses}L)")
                print(f"  Total P&L: ${total_pnl:+.2f}")
                
                if len(completed) > 0:
                    avg_pnl = total_pnl / len(completed)
                    print(f"  Avg per trade: ${avg_pnl:+.2f}")
                    
                    # Project daily based on trade frequency
                    if duration_hours > 0:
                        trades_per_hour = len(completed) / duration_hours
                        projected_daily = trades_per_hour * 24 * avg_pnl
                        print(f"  Projected daily: ${projected_daily:+.2f}")
        
        print()
        print("-" * 40)
        print()
        
        # Calculate combined totals
        all_completed = []
        for positions in simulator.completed_positions.values():
            all_completed.extend(positions)
        
        if all_completed:
            total_pnl = sum(p.cumulative_profit_usd for p in all_completed)
            total_trades = len(all_completed)
            avg_per_trade = total_pnl / total_trades
            
            print("COMBINED RESULTS:")
            print(f"Total Trades: {total_trades}")
            print(f"Total P&L: ${total_pnl:+.2f}")
            print(f"Average per trade: ${avg_per_trade:+.2f}")
            
            if duration_hours > 0:
                hourly_rate = total_pnl / duration_hours
                daily_projection = hourly_rate * 24
                print(f"Hourly rate: ${hourly_rate:+.2f}")
                print(f"Daily projection: ${daily_projection:+.2f}")
                
                if daily_projection >= 100:
                    print("✅ TARGET ACHIEVED! $100+/day profit rate")
                elif daily_projection >= 75:
                    print("⚠️  Close to target: Needs optimization")
                else:
                    print("❌ Below target: Strategy adjustment needed")
        
        print()
        print("=" * 60)
        print()
        print("Results saved to:")
        print("- Trade log: sims/ALL_TRADES.txt")
        print("- JSON data: sims/data/*.json")
        print("- Logs: sims/logs/")
        print()
        print("Run 'python3 sims/analyze_trades.py' for detailed analysis")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)