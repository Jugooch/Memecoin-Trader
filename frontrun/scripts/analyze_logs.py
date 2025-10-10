"""
Analyze Phase 1 logs to extract opportunity data
Useful when script was stopped before final export
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

log_file = Path("logs/phase1_monitor.log")

if not log_file.exists():
    print(f"Log file not found: {log_file}")
    exit(1)

opportunities = []
pump_fun_txs = 0

# Parse log file
with open(log_file, 'r') as f:
    for line in f:
        try:
            # Parse JSON log line
            data = json.loads(line.strip())

            # Count Pump.fun transactions
            if data.get('event') == 'laserstream_transactions_seen':
                pump_fun_txs += 1

            # Extract frontrun opportunities
            if data.get('event') == 'frontrun_opportunity_detected':
                opportunities.append({
                    'timestamp': data.get('timestamp'),
                    'signature': data.get('signature'),
                    'buy_amount_sol': data.get('buy_amount_sol', 0),
                    'fee': data.get('fee', 0),
                    'would_frontrun': True,
                    'reason': f"Large buy detected ({data.get('buy_amount_sol', 0):.2f} SOL)"
                })
        except:
            pass

# Analyze
print(f"\n=== PHASE 1 LOG ANALYSIS ===\n")
print(f"Total Pump.fun transactions: {pump_fun_txs}")
print(f"Frontrun opportunities detected: {len(opportunities)}")

if opportunities:
    buy_amounts = [o['buy_amount_sol'] for o in opportunities]
    fees = [o['fee'] for o in opportunities]

    print(f"\nBuy Amount Stats:")
    print(f"  Average: {sum(buy_amounts)/len(buy_amounts):.3f} SOL")
    print(f"  Max: {max(buy_amounts):.3f} SOL")
    print(f"  Min: {min(buy_amounts):.3f} SOL")

    print(f"\nCompetitor Fee Stats:")
    print(f"  Average: {int(sum(fees)/len(fees)):,} lamports")
    print(f"  Max: {max(fees):,} lamports")
    print(f"  Min: {min(fees):,} lamports")

    # Hourly distribution
    hourly = defaultdict(int)
    for opp in opportunities:
        if opp.get('timestamp'):
            try:
                dt = datetime.fromisoformat(opp['timestamp'].replace('Z', '+00:00'))
                hourly[dt.hour] += 1
            except:
                pass

    if hourly:
        print(f"\nOpportunities by Hour:")
        for hour in sorted(hourly.keys()):
            print(f"  {hour:02d}:00 - {hourly[hour]} opportunities")

    # Export to JSON
    output_dir = Path("data/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "opportunities_from_logs.json"
    with open(output_file, 'w') as f:
        json.dump(opportunities, f, indent=2)

    print(f"\n✅ Exported {len(opportunities)} opportunities to: {output_file}")

else:
    print("\n⚠️ No opportunities found in logs")
    print("Make sure the log file contains 'frontrun_opportunity_detected' events")
