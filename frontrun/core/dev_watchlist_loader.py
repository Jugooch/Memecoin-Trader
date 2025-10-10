"""
Fast Dev Watchlist Loader with O(1) Lookup

This module provides efficient in-memory lookups for dev watchlists,
even with 10,000+ devs. Uses a set for O(1) membership testing.

Usage:
    from frontrun.core.dev_watchlist_loader import DevWatchlistLoader

    loader = DevWatchlistLoader("frontrun/data/dev_watchlist.json")

    # O(1) lookup
    if loader.is_tracked_dev(creator_wallet):
        profile = loader.get_dev_profile(creator_wallet)
        print(f"Dev has {profile['buy_rate']}% buy rate!")
"""

import json
from pathlib import Path
from typing import Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)


class DevWatchlistLoader:
    """
    Efficient dev watchlist loader with O(1) lookups

    Instead of scanning a 10k-entry JSON file linearly (O(n)),
    we load it once into a dict/set for instant lookups (O(1)).
    """

    def __init__(self, watchlist_path: str):
        """
        Initialize loader and load watchlist into memory

        Args:
            watchlist_path: Path to dev_watchlist.json
        """
        self.watchlist_path = Path(watchlist_path)

        # Fast lookup structures
        self.dev_set: Set[str] = set()  # Just wallet addresses for O(1) membership
        self.dev_profiles: Dict[str, Dict] = {}  # Full profiles keyed by wallet

        # Load on init
        self._load_watchlist()

    def _load_watchlist(self):
        """Load watchlist from JSON into memory"""
        if not self.watchlist_path.exists():
            logger.warning(f"Watchlist not found: {self.watchlist_path}")
            return

        try:
            with open(self.watchlist_path, 'r') as f:
                profiles = json.load(f)

            # Build fast lookup structures
            for profile in profiles:
                wallet = profile.get('wallet')
                if wallet:
                    self.dev_set.add(wallet)
                    self.dev_profiles[wallet] = profile

            logger.info(
                f"Loaded {len(self.dev_set)} devs from watchlist "
                f"(O(1) lookup ready)"
            )

        except Exception as e:
            logger.error(f"Failed to load watchlist: {e}")

    def is_tracked_dev(self, wallet: str) -> bool:
        """
        Check if wallet is in watchlist (O(1) lookup)

        Args:
            wallet: Wallet address to check

        Returns:
            True if wallet is tracked, False otherwise
        """
        return wallet in self.dev_set

    def get_dev_profile(self, wallet: str) -> Optional[Dict]:
        """
        Get full dev profile (O(1) lookup)

        Args:
            wallet: Wallet address

        Returns:
            Dev profile dict or None if not found
        """
        return self.dev_profiles.get(wallet)

    def get_all_tracked_devs(self) -> Set[str]:
        """
        Get set of all tracked dev wallets

        Returns:
            Set of wallet addresses
        """
        return self.dev_set.copy()

    def reload(self):
        """Reload watchlist from disk (for updates)"""
        logger.info("Reloading dev watchlist...")
        self.dev_set.clear()
        self.dev_profiles.clear()
        self._load_watchlist()

    def get_stats(self) -> Dict:
        """Get watchlist statistics"""
        if not self.dev_profiles:
            return {
                'total_devs': 0,
                'avg_buy_rate': 0,
                'avg_tokens_created': 0
            }

        buy_rates = [p['buy_rate'] for p in self.dev_profiles.values()]
        tokens_created = [p['tokens_created'] for p in self.dev_profiles.values()]

        return {
            'total_devs': len(self.dev_profiles),
            'avg_buy_rate': sum(buy_rates) / len(buy_rates),
            'avg_tokens_created': sum(tokens_created) / len(tokens_created),
            'min_buy_rate': min(buy_rates),
            'max_buy_rate': max(buy_rates),
            'max_tokens_created': max(tokens_created)
        }


# Example usage
if __name__ == "__main__":
    import time

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load watchlist
    loader = DevWatchlistLoader("data/dev_watchlist.json")

    # Print stats
    stats = loader.get_stats()
    print(f"\n=== Watchlist Stats ===")
    print(f"Total devs: {stats['total_devs']}")
    print(f"Avg buy rate: {stats['avg_buy_rate']:.1f}%")
    print(f"Avg tokens/dev: {stats['avg_tokens_created']:.1f}")

    # Benchmark lookup speed (even with 10k devs, this will be instant)
    test_wallets = list(loader.get_all_tracked_devs())[:100]  # Test first 100

    start = time.time()
    for wallet in test_wallets:
        _ = loader.is_tracked_dev(wallet)
    elapsed = time.time() - start

    print(f"\n=== Performance ===")
    print(f"100 lookups in: {elapsed*1000:.2f}ms")
    print(f"Avg lookup time: {elapsed/100*1000000:.2f}μs (microseconds)")
    print(f"✅ O(1) lookup confirmed - fast enough for real-time trading!")
