# Dev Watchlist Usage Guide

## Overview

This guide shows how to use the dev watchlist for real-time token creation frontrunning.

## Building the Watchlist

The watchlist now supports **incremental updates** - new scans merge with existing data instead of overwriting.

### Initial Build (First Time)

```bash
cd frontrun
python scripts/build_dev_watchlist.py \
  --max-pages 1000 \
  --lookback-days 7 \
  --min-tokens 2 \
  --min-buy-rate 70
```

This will:
- Scan ~1-2 days of historical data (1000 pages)
- Find devs who created 2+ tokens with 70%+ buy rate
- Save to `data/dev_watchlist.json`

### Incremental Updates (Run Daily/Weekly)

```bash
# Run the same command again - it will merge with existing data!
cd frontrun
python scripts/build_dev_watchlist.py \
  --max-pages 500 \
  --lookback-days 3 \
  --min-tokens 2 \
  --min-buy-rate 70
```

This will:
- Load existing watchlist (e.g., 500 devs)
- Scan recent data (500 pages = ~12 hours)
- Find new devs + update existing devs
- Merge and save (e.g., now 650 devs total)

**The file grows over time - you're building a historical database!**

## Using in Real-Time Detection

### Fast O(1) Lookup (Recommended for 1000+ devs)

```python
from frontrun.core.dev_watchlist_loader import DevWatchlistLoader

# Load once at startup
dev_watchlist = DevWatchlistLoader("frontrun/data/dev_watchlist.json")
print(f"Loaded {dev_watchlist.get_stats()['total_devs']} devs")

# In your event handler (LaserStream callback)
async def on_token_creation(event):
    creator_wallet = event.get('creator')
    token_mint = event.get('mint')

    # O(1) lookup - instant even with 10,000 devs!
    if dev_watchlist.is_tracked_dev(creator_wallet):
        profile = dev_watchlist.get_dev_profile(creator_wallet)

        print(f"🎯 TRACKED DEV DETECTED!")
        print(f"  Dev: {creator_wallet[:16]}...")
        print(f"  Buy Rate: {profile['buy_rate']}%")
        print(f"  Tokens Created: {profile['tokens_created']}")
        print(f"  Token: {token_mint}")

        # Execute frontrun trade
        await execute_buy(token_mint, amount_sol=0.1)
```

### Simple List Approach (OK for <100 devs)

```python
import json

# Load watchlist
with open("frontrun/data/dev_watchlist.json", 'r') as f:
    watchlist = json.load(f)

# Convert to set for faster lookup
dev_wallets = {profile['wallet'] for profile in watchlist}

# Check in event handler
if creator_wallet in dev_wallets:
    await execute_buy(token_mint)
```

## Performance Comparison

| Watchlist Size | Linear Scan (O(n)) | Set Lookup (O(1)) |
|----------------|-------------------|-------------------|
| 100 devs       | ~10μs             | ~0.1μs           |
| 1,000 devs     | ~100μs            | ~0.1μs           |
| 10,000 devs    | ~1ms              | ~0.1μs           |

**For frontrunning, every microsecond counts!** Use `DevWatchlistLoader` for O(1) lookups.

## Watchlist Maintenance

### Reload After Updates

If you rebuild the watchlist while your bot is running:

```python
# In your bot's update handler
dev_watchlist.reload()
print(f"Reloaded {dev_watchlist.get_stats()['total_devs']} devs")
```

### View Stats

```bash
cd frontrun
python -c "from frontrun.core.dev_watchlist_loader import DevWatchlistLoader; \
  loader = DevWatchlistLoader('data/dev_watchlist.json'); \
  print(loader.get_stats())"
```

## Integration with LaserStream

Example integration with your Phase 1 monitor:

```python
from frontrun.core.dev_watchlist_loader import DevWatchlistLoader
from frontrun.core.logger import get_logger

logger = get_logger(__name__)

class DevCreationSniperMonitor:
    def __init__(self, watchlist_path: str):
        self.dev_watchlist = DevWatchlistLoader(watchlist_path)
        logger.info(f"Monitoring {self.dev_watchlist.get_stats()['total_devs']} devs")

    async def process_transaction(self, tx_data: dict):
        """Process transaction from LaserStream"""
        creator = tx_data.get('signer')

        if self.dev_watchlist.is_tracked_dev(creator):
            profile = self.dev_watchlist.get_dev_profile(creator)

            # Only frontrun high-quality devs
            if profile['buy_rate'] >= 90:
                logger.info(f"HIGH QUALITY DEV: {creator[:16]}... ({profile['buy_rate']}%)")
                await self.execute_frontrun_buy(tx_data)
            else:
                logger.info(f"Tracked dev (medium quality): {creator[:16]}...")
```

## Tips

1. **Run updates weekly** to keep catching new devs
2. **Lower filters over time** - as your watchlist grows, you can be more selective in real-time
3. **Monitor performance** - even 10k devs should be <1ms lookup with `DevWatchlistLoader`
4. **Check buy_rate in real-time** - filter for 90%+ devs during actual trading for best quality

## File Size Estimates

| # Devs | File Size | Load Time |
|--------|-----------|-----------|
| 100    | ~50KB     | <1ms      |
| 1,000  | ~500KB    | ~5ms      |
| 10,000 | ~5MB      | ~50ms     |

Even with 10,000 devs, the watchlist loads in 50ms at startup - totally acceptable!
