"""
Alpha Wallet Accumulator Runner
Main production script for continuous alpha wallet discovery

RECOMMENDED USAGE (for production):
  python scripts/run_alpha_accumulator.py --loop

This will run the accumulator continuously every 2 minutes, building up
a database of alpha wallets over time. Leave this running 24/7.

SINGLE RUN (for testing):
  python scripts/run_alpha_accumulator.py

This runs once and exits. Useful for testing or debugging.

CUSTOM INTERVAL:
  python scripts/run_alpha_accumulator.py --loop --interval 180

Run every 3 minutes instead of 2 minutes. Useful if you want to be more
conservative with API usage.

The accumulator automatically updates your config.yml with discovered alpha wallets,
so your trading bot will start following them immediately.
"""

import asyncio
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.discovery.alpha_accumulator import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run alpha wallet accumulator')
    parser.add_argument('--loop', action='store_true', help='Run continuously every 2 minutes')
    parser.add_argument('--interval', type=int, default=120, help='Seconds between runs (default: 120)')
    args = parser.parse_args()
    
    if args.loop:
        print(f"Running alpha accumulator every {args.interval} seconds...")
        print("Press Ctrl+C to stop")
        while True:
            try:
                asyncio.run(main())
                print(f"\nWaiting {args.interval} seconds before next run...\n")
                asyncio.run(asyncio.sleep(args.interval))
            except KeyboardInterrupt:
                print("\nStopped by user")
                break
    else:
        asyncio.run(main())