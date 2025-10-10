"""
Build Developer Watchlist for Token Creation Frontrunning

Strategy: Identify developers who consistently buy into tokens they create,
then frontrun their initial buys by detecting token creation events.

This queries Bitquery to find devs who:
- Create tokens regularly (X+ tokens in last 7 days)
- Buy into their own tokens >X% of the time
- Are ACTIVE recently (not stale)

Output: JSON watchlist of high-quality devs to monitor

Usage:
    python scripts/build_dev_watchlist.py --min-tokens 3 --min-buy-rate 80 --lookback-days 7
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.clients.bitquery_client import BitqueryClient
from src.utils.config_loader import load_config
from frontrun.core.logger import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class DevProfile:
    """Developer profile with token creation stats"""
    wallet: str
    tokens_created: int
    tokens_bought_into: int
    buy_rate: float  # Percentage of tokens they bought into
    tokens_created_7d: int
    tokens_created_30d: int
    first_seen: str
    last_seen: str
    example_tokens: List[str]  # Sample tokens they created
    updated_at: str = None  # When this profile was last updated
    all_tokens: List[str] = None  # All tokens created by this dev (across all scans)

    def to_dict(self):
        return asdict(self)


class DevWatchlistBuilder:
    """
    Build developer watchlist by analyzing historical token creation patterns
    """

    def __init__(self, bitquery_client: BitqueryClient, lookback_days: int = 7, max_pages: int = 200, incremental: bool = True):
        """
        Initialize watchlist builder

        Args:
            bitquery_client: Bitquery API client
            lookback_days: How many days to look back for activity
            max_pages: Maximum pages to fetch from Bitquery (more pages = more history)
            incremental: If True, merge with existing watchlist instead of overwriting
        """
        self.client = bitquery_client
        self.lookback_days = lookback_days
        self.max_pages = max_pages
        self.incremental = incremental
        logger.info(f"dev_watchlist_builder_initialized", lookback_days=lookback_days, max_pages=max_pages, incremental=incremental)

    def load_existing_watchlist(self, filepath: Path) -> Dict[str, DevProfile]:
        """Load existing watchlist and convert to dict for easy merging"""
        if not filepath.exists():
            logger.info("no_existing_watchlist_found")
            return {}

        try:
            with open(filepath, 'r') as f:
                existing_data = json.load(f)

            # Convert list to dict keyed by wallet
            existing_profiles = {}
            for profile_data in existing_data:
                wallet = profile_data.get('wallet')
                if wallet:
                    # Convert dict back to DevProfile
                    profile = DevProfile(
                        wallet=profile_data['wallet'],
                        tokens_created=profile_data['tokens_created'],
                        tokens_bought_into=profile_data['tokens_bought_into'],
                        buy_rate=profile_data['buy_rate'],
                        tokens_created_7d=profile_data.get('tokens_created_7d', 0),
                        tokens_created_30d=profile_data.get('tokens_created_30d', 0),
                        first_seen=profile_data['first_seen'],
                        last_seen=profile_data['last_seen'],
                        example_tokens=profile_data['example_tokens'],
                        updated_at=profile_data.get('updated_at'),
                        all_tokens=profile_data.get('all_tokens', profile_data['example_tokens'])
                    )
                    existing_profiles[wallet] = profile

            logger.info(f"loaded_{len(existing_profiles)}_existing_dev_profiles")
            return existing_profiles

        except Exception as e:
            logger.error(f"error_loading_existing_watchlist", error=str(e))
            return {}

    async def get_all_token_creators(self) -> Dict[str, Dict]:
        """
        Get all token creators from Pump.fun in the lookback window

        Returns:
            Dict mapping creator_wallet -> {tokens: [mint_addresses], timestamps: [creation_times]}
        """
        logger.info("fetching_all_pump_fun_token_creations")

        # Calculate time range
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(days=self.lookback_days)
        start_iso = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_iso = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        logger.info(f"querying_bitquery", start=start_iso, end=end_iso)

        # Use paginated fetch to get ALL trades in the window
        # CRITICAL: We need the Transaction.Signer field to identify token creators
        # IMPORTANT: 10 pages covers ~30 mins, 100 pages covers ~5 hours, 500 pages covers ~1 day
        trades = await self.client.get_trades_windowed_paginated(
            start_iso=start_iso,
            end_iso=end_iso,
            page_limit=3000,
            max_pages=self.max_pages
        )

        if not trades:
            logger.warning("no_trades_found")
            return {}

        logger.info(f"analyzing_{len(trades)}_trades")

        # Group by creator (Transaction.Signer) and token
        creator_tokens = defaultdict(lambda: {
            'tokens': set(),  # Unique tokens
            'timestamps': [],
            'first_trades': {}  # token -> first trade data
        })

        for trade in trades:
            try:
                signer = trade.get('Transaction', {}).get('Signer')
                if not signer:
                    continue

                # Get token mint (not SOL)
                trade_data = trade.get('Trade', {})
                buy_currency = trade_data.get('Buy', {}).get('Currency', {})
                sell_currency = trade_data.get('Sell', {}).get('Currency', {})

                mint = None
                if buy_currency.get('MintAddress') not in [None, 'So11111111111111111111111111111112', '11111111111111111111111111111111']:
                    mint = buy_currency.get('MintAddress')
                elif sell_currency.get('MintAddress') not in [None, 'So11111111111111111111111111111112', '11111111111111111111111111111111']:
                    mint = sell_currency.get('MintAddress')

                if not mint:
                    continue

                timestamp = trade.get('Block', {}).get('Time')
                if not timestamp:
                    continue

                # Track this token for this creator
                creator_data = creator_tokens[signer]

                # If this is the first time we see this token for this creator, record it
                if mint not in creator_data['tokens']:
                    creator_data['tokens'].add(mint)
                    creator_data['timestamps'].append(timestamp)
                    creator_data['first_trades'][mint] = {
                        'timestamp': timestamp,
                        'buy_side': buy_currency.get('MintAddress') == mint,
                        'sell_side': sell_currency.get('MintAddress') == mint,
                        'signature': trade.get('Transaction', {}).get('Signature')
                    }

            except Exception as e:
                logger.warning(f"error_parsing_trade", error=str(e))
                continue

        # Convert sets to lists for JSON serialization
        for creator, data in creator_tokens.items():
            data['tokens'] = list(data['tokens'])

        logger.info(f"found_{len(creator_tokens)}_unique_creators")

        return dict(creator_tokens)

    async def analyze_dev_buy_behavior(self, dev_wallet: str, tokens_created: List[str], first_trades: Dict) -> Dict:
        """
        Analyze if a dev bought into their own tokens

        Args:
            dev_wallet: Developer wallet address
            tokens_created: List of token mints they created
            first_trades: Dict of mint -> first trade data

        Returns:
            Dict with buy behavior stats
        """
        bought_into = 0

        # For each token, check if the first trade was a BUY by this dev
        for mint in tokens_created:
            first_trade = first_trades.get(mint, {})

            # If the first trade has the token on the BUY side, the dev bought it
            # (This means dev traded SOL -> Token, which is a buy)
            if first_trade.get('buy_side'):
                bought_into += 1

        buy_rate = (bought_into / len(tokens_created) * 100) if tokens_created else 0

        return {
            'tokens_created': len(tokens_created),
            'tokens_bought_into': bought_into,
            'buy_rate': buy_rate
        }

    async def build_watchlist(
        self,
        min_tokens: int = 3,
        min_buy_rate: float = 80.0,
        min_tokens_7d: int = 2,
        existing_profiles: Dict[str, DevProfile] = None
    ) -> List[DevProfile]:
        """
        Build watchlist of high-quality devs, merging with existing data if available

        Args:
            min_tokens: Minimum tokens created in lookback window
            min_buy_rate: Minimum % of tokens they bought into (e.g., 80%)
            min_tokens_7d: Minimum tokens created in last 7 days (activity filter)
            existing_profiles: Existing dev profiles to merge with (dict keyed by wallet)

        Returns:
            List of DevProfile objects, sorted by buy_rate DESC
        """
        logger.info(
            "building_watchlist",
            min_tokens=min_tokens,
            min_buy_rate=min_buy_rate,
            min_tokens_7d=min_tokens_7d,
            lookback_days=self.lookback_days,
            existing_count=len(existing_profiles) if existing_profiles else 0
        )

        if existing_profiles is None:
            existing_profiles = {}

        # Get all creators from current scan
        creators = await self.get_all_token_creators()

        if not creators:
            logger.error("no_creators_found")
            # Return existing profiles if we couldn't fetch new data
            return list(existing_profiles.values()) if existing_profiles else []

        # Analyze each creator
        profiles_dict = {}  # Use dict for easy merging
        seven_days_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)
        now = datetime.now(tz=timezone.utc).isoformat()

        for wallet, data in creators.items():
            tokens = data['tokens']
            timestamps = data['timestamps']
            first_trades = data['first_trades']

            # Get existing profile if it exists
            existing = existing_profiles.get(wallet)

            # Merge tokens with existing data
            if existing and existing.all_tokens:
                all_tokens_set = set(existing.all_tokens) | set(tokens)
                all_tokens = list(all_tokens_set)
            else:
                all_tokens = tokens

            # Skip if below minimum threshold (using current scan data)
            if len(tokens) < min_tokens:
                # But keep existing profile if it already qualified
                if existing:
                    profiles_dict[wallet] = existing
                continue

            # Analyze buy behavior
            buy_stats = await self.analyze_dev_buy_behavior(wallet, tokens, first_trades)

            # Skip if buy rate too low
            if buy_stats['buy_rate'] < min_buy_rate:
                # But keep existing profile if it already qualified
                if existing:
                    profiles_dict[wallet] = existing
                continue

            # Count tokens created in last 7 days
            tokens_7d = 0
            tokens_30d = len(tokens)  # All tokens in lookback window

            for ts_str in timestamps:
                try:
                    from dateutil import parser
                    ts = parser.parse(ts_str)
                    if ts >= seven_days_ago:
                        tokens_7d += 1
                except:
                    continue

            # Skip if not active enough
            if tokens_7d < min_tokens_7d:
                # But keep existing profile if it already qualified
                if existing:
                    profiles_dict[wallet] = existing
                continue

            # Parse timestamps for first/last seen
            parsed_timestamps = []
            for ts_str in timestamps:
                try:
                    from dateutil import parser
                    parsed_timestamps.append(parser.parse(ts_str))
                except:
                    continue

            if not parsed_timestamps:
                continue

            current_first = min(parsed_timestamps).isoformat()
            current_last = max(parsed_timestamps).isoformat()

            # Merge with existing profile
            if existing:
                # Use earliest first_seen from both scans
                from dateutil import parser as date_parser
                existing_first = date_parser.parse(existing.first_seen)
                current_first_dt = date_parser.parse(current_first)
                first_seen = min(existing_first, current_first_dt).isoformat()

                # Use latest last_seen
                existing_last = date_parser.parse(existing.last_seen)
                current_last_dt = date_parser.parse(current_last)
                last_seen = max(existing_last, current_last_dt).isoformat()
            else:
                first_seen = current_first
                last_seen = current_last

            # Create/update profile
            profile = DevProfile(
                wallet=wallet,
                tokens_created=len(all_tokens),  # Total across all scans
                tokens_bought_into=buy_stats['tokens_bought_into'],
                buy_rate=round(buy_stats['buy_rate'], 2),
                tokens_created_7d=tokens_7d,
                tokens_created_30d=tokens_30d,
                first_seen=first_seen,
                last_seen=last_seen,
                example_tokens=tokens[:5],  # Most recent tokens as examples
                updated_at=now,
                all_tokens=all_tokens
            )

            profiles_dict[wallet] = profile

        # Add any existing profiles that weren't in the current scan
        for wallet, existing_profile in existing_profiles.items():
            if wallet not in profiles_dict:
                profiles_dict[wallet] = existing_profile

        # Convert to list and sort
        profiles = list(profiles_dict.values())
        profiles.sort(key=lambda p: (p.buy_rate, p.tokens_created_7d), reverse=True)

        logger.info(
            "watchlist_built",
            total_creators=len(creators),
            new_devs=len([p for p in profiles if p.wallet in creators]),
            total_devs=len(profiles),
            top_buy_rate=profiles[0].buy_rate if profiles else 0
        )

        return profiles


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build developer watchlist for token creation frontrunning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy:
  Instead of frontrunning retail buyers, frontrun the DEV's initial buy!

  1. Find devs who consistently buy into their own tokens (>80% of the time)
  2. Monitor for their token CREATION events (not buys)
  3. Buy IMMEDIATELY after creation (before dev's buy)
  4. Sell into dev buy + retail wave

Examples:
  # Conservative: 5+ tokens, 90%+ buy rate, 3+ tokens in last 7 days
  python scripts/build_dev_watchlist.py --min-tokens 5 --min-buy-rate 90 --min-tokens-7d 3

  # Aggressive: 3+ tokens, 80%+ buy rate, 2+ tokens in last 7 days
  python scripts/build_dev_watchlist.py --min-tokens 3 --min-buy-rate 80 --min-tokens-7d 2

  # Wide net: Any dev with 2+ tokens and 70%+ buy rate
  python scripts/build_dev_watchlist.py --min-tokens 2 --min-buy-rate 70 --min-tokens-7d 1

Output:
  - data/dev_watchlist.json: List of qualified devs with stats
  - Can be used by modified LaserStream monitor to detect token creations
        """
    )

    parser.add_argument(
        "--config",
        default="../config/config.yml",
        help="Path to config.yml (for Bitquery API keys)"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum tokens created in lookback window (default: 3)"
    )
    parser.add_argument(
        "--min-buy-rate",
        type=float,
        default=80.0,
        help="Minimum %% of tokens they bought into (default: 80%%)"
    )
    parser.add_argument(
        "--min-tokens-7d",
        type=int,
        default=2,
        help="Minimum tokens created in last 7 days (activity filter, default: 2)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="How many days to look back for token creations (default: 7)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=200,
        help="Maximum pages to fetch from Bitquery. ~10 pages = 30 mins, ~200 pages = ~10 hours, ~500 pages = 1 day (default: 200)"
    )
    parser.add_argument(
        "--output",
        default="data/dev_watchlist.json",
        help="Output file path (default: data/dev_watchlist.json)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO", format="console")

    logger.info("=== DEV WATCHLIST BUILDER ===")
    logger.info(f"Analyzing Pump.fun token creations from last {args.lookback_days} days...")
    logger.info(f"Filters: {args.min_tokens}+ tokens, {args.min_buy_rate}%+ buy rate, {args.min_tokens_7d}+ tokens in last 7 days")

    # Load config (using same pattern as alpha_discovery_scheduler)
    config = load_config(args.config)

    # Get Bitquery API keys (try both 'bitquery_tokens' and 'bitquery_token')
    bitquery_keys = config.get('bitquery_tokens', config.get('bitquery_token', []))
    if not bitquery_keys:
        logger.error("No Bitquery API keys found in config!")
        logger.error("Add bitquery.api_keys to config/config.yml")
        sys.exit(1)

    logger.info(f"Using {len(bitquery_keys)} Bitquery API token(s)")

    # Initialize Bitquery client
    bitquery = BitqueryClient(bitquery_keys)
    await bitquery.initialize()

    # Build watchlist
    builder = DevWatchlistBuilder(bitquery, lookback_days=args.lookback_days, max_pages=args.max_pages, incremental=True)

    # Load existing watchlist if it exists
    output_path = Path(args.output)
    existing_profiles = builder.load_existing_watchlist(output_path)

    try:
        profiles = await builder.build_watchlist(
            min_tokens=args.min_tokens,
            min_buy_rate=args.min_buy_rate,
            min_tokens_7d=args.min_tokens_7d,
            existing_profiles=existing_profiles
        )

        if not profiles:
            logger.warning("No devs matched your criteria!")
            logger.warning("Try lowering --min-buy-rate or --min-tokens")
            sys.exit(0)

        # Print summary
        logger.info("\n=== TOP 10 DEVS ===")
        for i, profile in enumerate(profiles[:10], 1):
            logger.info(
                f"{i}. {profile.wallet[:16]}... | "
                f"{profile.tokens_created} tokens created | "
                f"{profile.buy_rate}% buy rate | "
                f"{profile.tokens_created_7d} in last 7d"
            )

        # Export to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump([p.to_dict() for p in profiles], f, indent=2)

        logger.info(f"\n✅ Watchlist exported to: {output_path}")
        logger.info(f"Total devs: {len(profiles)}")
        logger.info(f"New devs this scan: {len(profiles) - len(existing_profiles)}")
        logger.info(f"Avg buy rate: {sum(p.buy_rate for p in profiles) / len(profiles):.1f}%")
        logger.info(f"Avg tokens/dev: {sum(p.tokens_created for p in profiles) / len(profiles):.1f}")

        # Strategy notes
        logger.info("\n=== NEXT STEPS ===")
        logger.info("1. Review the watchlist in data/dev_watchlist.json")
        logger.info("2. Modify LaserStream monitor to detect TOKEN CREATION events")
        logger.info("3. Check if creator is on watchlist")
        logger.info("4. Buy IMMEDIATELY after creation (before dev's first buy)")
        logger.info("5. Sell into dev buy + retail wave")
        logger.info("\nThis strategy frontrun's the DEV, not retail! Much better edge.")

    except Exception as e:
        logger.error(f"watchlist_build_failed", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("fatal_error", error=str(e), exc_info=True)
        sys.exit(1)
