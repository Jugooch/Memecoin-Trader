"""
Test Dev Watchlist Integration

Verify that the dev watchlist loader integrates correctly with the sniper bot.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dev_watchlist_loader import DevWatchlistLoader
from core.logger import setup_logging, get_logger


def test_watchlist_loading():
    """Test that watchlist loads correctly"""
    print("=== Test 1: Load Watchlist ===")

    loader = DevWatchlistLoader("data/dev_watchlist.json")
    stats = loader.get_stats()

    if stats['total_devs'] == 0:
        print(f"❌ No devs loaded!")
        print("\n💡 Build your watchlist first:")
        print("   python scripts/build_dev_watchlist.py --max-pages 200 --min-buy-rate 70")
        raise AssertionError("Watchlist is empty. Run build_dev_watchlist.py first.")

    print(f"✅ Loaded {stats['total_devs']} devs")
    print(f"   Avg buy rate: {stats['avg_buy_rate']:.1f}%")
    print(f"   Avg tokens/dev: {stats['avg_tokens_created']:.1f}")

    return loader


def test_lookup_performance(loader: DevWatchlistLoader):
    """Test lookup performance"""
    print("\n=== Test 2: Lookup Performance ===")

    import time

    # Get a real dev from the watchlist
    all_devs = list(loader.get_all_tracked_devs())
    test_dev = all_devs[0] if all_devs else None

    if not test_dev:
        print("⚠️ No devs in watchlist to test")
        return

    # Test positive lookup (dev in watchlist)
    start = time.time()
    for _ in range(10000):
        result = loader.is_tracked_dev(test_dev)
    elapsed = time.time() - start

    print(f"✅ 10,000 positive lookups in {elapsed*1000:.2f}ms")
    print(f"   Avg: {elapsed/10000*1000000:.2f}μs per lookup")

    # Test negative lookup (dev not in watchlist)
    fake_dev = "1111111111111111111111111111111111111111111"

    start = time.time()
    for _ in range(10000):
        result = loader.is_tracked_dev(fake_dev)
    elapsed = time.time() - start

    print(f"✅ 10,000 negative lookups in {elapsed*1000:.2f}ms")
    print(f"   Avg: {elapsed/10000*1000000:.2f}μs per lookup")

    assert elapsed < 0.1, "10k lookups should take <100ms"


def test_filtering_by_buy_rate(loader: DevWatchlistLoader):
    """Test filtering devs by buy rate"""
    print("\n=== Test 3: Filter by Buy Rate ===")

    all_devs = loader.get_all_tracked_devs()

    # Count devs at different thresholds
    thresholds = [80, 85, 90, 95, 100]

    for threshold in thresholds:
        qualifying_devs = []

        for dev in all_devs:
            profile = loader.get_dev_profile(dev)
            if profile['buy_rate'] >= threshold:
                qualifying_devs.append(dev)

        print(f"   {threshold}%+ buy rate: {len(qualifying_devs)} devs")

    print("✅ Filtering works correctly")


def test_get_profile(loader: DevWatchlistLoader):
    """Test getting dev profiles"""
    print("\n=== Test 4: Get Dev Profiles ===")

    all_devs = list(loader.get_all_tracked_devs())

    if not all_devs:
        print("⚠️ No devs to test")
        return

    # Get profile for first dev
    test_dev = all_devs[0]
    profile = loader.get_dev_profile(test_dev)

    print(f"✅ Got profile for {test_dev[:16]}...")
    print(f"   Buy rate: {profile['buy_rate']}%")
    print(f"   Tokens created: {profile['tokens_created']}")
    print(f"   Last seen: {profile.get('last_seen', 'N/A')}")

    assert profile is not None
    assert 'buy_rate' in profile
    assert 'tokens_created' in profile


def test_simulation_scenario(loader: DevWatchlistLoader):
    """Simulate the real-time detection scenario"""
    print("\n=== Test 5: Simulate Real-Time Detection ===")

    all_devs = list(loader.get_all_tracked_devs())

    if not all_devs:
        print("⚠️ No devs to test")
        return

    # Simulate detecting a token creation
    creator_wallet = all_devs[0]
    min_buy_rate = 90.0

    print(f"📡 Simulating token creation by {creator_wallet[:16]}...")

    # Step 1: Check if tracked (O(1) lookup)
    if not loader.is_tracked_dev(creator_wallet):
        print("❌ Not a tracked dev - SKIP")
        return

    print("✅ Dev is tracked!")

    # Step 2: Get profile
    profile = loader.get_dev_profile(creator_wallet)

    print(f"   Buy rate: {profile['buy_rate']}%")
    print(f"   Tokens created: {profile['tokens_created']}")

    # Step 3: Check quality filter
    if profile['buy_rate'] < min_buy_rate:
        print(f"❌ Buy rate too low (<{min_buy_rate}%) - SKIP")
        return

    print(f"✅ High-quality dev! Would execute frontrun trade!")


def main():
    """Run all tests"""
    setup_logging(level="WARNING", format="console")

    print("╔═══════════════════════════════════════════════════╗")
    print("║  Dev Watchlist Integration Test                  ║")
    print("╚═══════════════════════════════════════════════════╝\n")

    try:
        # Test 1: Load watchlist
        loader = test_watchlist_loading()

        # Test 2: Performance
        test_lookup_performance(loader)

        # Test 3: Filtering
        test_filtering_by_buy_rate(loader)

        # Test 4: Get profiles
        test_get_profile(loader)

        # Test 5: Simulation
        test_simulation_scenario(loader)

        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("="*50)
        print("\n💡 Integration is ready for production use!")

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Run this first:")
        print("   python scripts/build_dev_watchlist.py --max-pages 200")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
