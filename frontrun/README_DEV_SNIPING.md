# Dev Token Creation Sniping - Complete Guide

## Overview

This system lets you **frontrun developer token buys** by detecting token creation events. Instead of competing with other bots after a dev buys 5 SOL, you buy **immediately after creation**, before the dev even buys!

## How It Works

```
Traditional (Phase 2):
  1. Dev creates token
  2. Dev buys 5 SOL → Price pumps
  3. Your bot detects the buy ← You're LATE!
  4. You buy → Competing with 100 other bots
  5. You sell → Maybe profit?

New Strategy (Dev Creation Sniping):
  1. Dev creates token ← YOU DETECT THIS!
  2. You buy IMMEDIATELY (before dev!) ← You're FIRST!
  3. Dev buys 5 SOL → Price pumps ← You profit!
  4. Retail wave → Price pumps more ← You profit more!
  5. You sell → Guaranteed better position!
```

## System Components

### 1. Dev Watchlist Builder (`scripts/build_dev_watchlist.py`)

Scans historical data to find devs who consistently buy their own tokens.

**Features:**
- ✅ **Incremental updates** - Merges with existing data
- ✅ **Pagination** - Covers days/weeks of history
- ✅ **Quality filters** - Only high buy-rate devs
- ✅ **Grows over time** - Build a database of 1000+ devs

**Usage:**
```bash
# Initial build (1000 pages = ~1-2 days of data)
cd frontrun
python scripts/build_dev_watchlist.py --max-pages 1000 --min-buy-rate 70

# Weekly updates (merges with existing)
python scripts/build_dev_watchlist.py --max-pages 500 --lookback-days 3
```

### 2. Fast Watchlist Loader (`core/dev_watchlist_loader.py`)

O(1) lookup for real-time detection. Works with 10,000+ devs.

**Features:**
- ✅ **O(1) lookup** - <1μs per check
- ✅ **Profile access** - Get buy rate, token count, etc.
- ✅ **Reloadable** - Update without restarting bot
- ✅ **Statistics** - Get watchlist stats

**Example:**
```python
from frontrun.core.dev_watchlist_loader import DevWatchlistLoader

loader = DevWatchlistLoader("frontrun/data/dev_watchlist.json")

# O(1) check
if loader.is_tracked_dev(creator_wallet):
    profile = loader.get_dev_profile(creator_wallet)

    if profile['buy_rate'] >= 95:
        # High-quality dev - execute trade!
        await frontrun_buy(token_mint)
```

### 3. Dev Creation Sniper (`scripts/dev_creation_sniper.py`)

Real-time bot that monitors token creations and frontrun high-quality devs.

**Features:**
- ✅ **LaserStream integration** - Real-time token creation events
- ✅ **Quality filtering** - Only frontrun devs with X%+ buy rate
- ✅ **Auto buy/sell** - Full trade execution
- ✅ **Performance tracking** - Logs all trades

## Quick Start

### Step 1: Build Your Watchlist

```bash
cd frontrun

# First time (scan ~2 days of data)
python scripts/build_dev_watchlist.py \
  --max-pages 1000 \
  --lookback-days 7 \
  --min-tokens 2 \
  --min-buy-rate 70
```

**Output:**
```
✅ Watchlist exported to: data/dev_watchlist.json
Total devs: 1945
Avg buy rate: 93.4%
Avg tokens/dev: 4.7
```

### Step 2: Test the Integration

```bash
python tests/test_dev_watchlist_integration.py
```

**Expected output:**
```
✅ Loaded 1945 devs
   Avg buy rate: 93.4%

✅ 10,000 positive lookups in 0.00ms
   Avg: 0.00μs per lookup

✅ ALL TESTS PASSED!
```

### Step 3: Run the Sniper Bot

```bash
# Simulation mode (no real trades)
python scripts/dev_creation_sniper.py \
  --wallet data/phase2-wallet.json \
  --max-trades 5 \
  --min-buy-rate 95

# Real trades (WARNING: Uses real capital!)
python scripts/dev_creation_sniper.py \
  --wallet data/phase2-wallet.json \
  --real-trades \
  --max-trades 10 \
  --buy-amount 0.05 \
  --min-buy-rate 90
```

## Configuration Options

### Watchlist Builder

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-pages` | 200 | Pages to fetch (~10 pages = 30 mins) |
| `--lookback-days` | 7 | How far back to scan |
| `--min-tokens` | 3 | Minimum tokens created |
| `--min-buy-rate` | 80 | Minimum % of tokens they bought |
| `--min-tokens-7d` | 2 | Minimum recent activity |

### Sniper Bot

| Argument | Default | Description |
|----------|---------|-------------|
| `--min-buy-rate` | 90.0 | Only frontrun devs with X%+ buy rate |
| `--buy-amount` | 0.01 | SOL to spend per trade |
| `--hold-time` | 3.0 | Seconds to hold before selling |
| `--max-trades` | 5 | Max trades to execute |
| `--real-trades` | False | Enable real trading |

## Performance Characteristics

### Watchlist Size vs Performance

| Devs | File Size | Load Time | Lookup Time |
|------|-----------|-----------|-------------|
| 100 | ~50KB | <1ms | 0.1μs |
| 1,000 | ~500KB | ~5ms | 0.1μs |
| 10,000 | ~5MB | ~50ms | 0.1μs |

**Conclusion:** Even with 10,000 devs, the system is real-time ready!

### Coverage vs API Calls

| Pages | Coverage | API Calls | Devs Found (Est.) |
|-------|----------|-----------|-------------------|
| 10 | ~30 mins | 10 | 10-50 |
| 100 | ~5 hours | 100 | 100-300 |
| 500 | ~1 day | 500 | 500-1500 |
| 1000 | ~2 days | 1000 | 1000-2500 |

## Maintenance

### Weekly Updates (Recommended)

```bash
# Run every week to catch new devs
cd frontrun
python scripts/build_dev_watchlist.py \
  --max-pages 500 \
  --lookback-days 7 \
  --min-buy-rate 70
```

This will:
- Scan the last 7 days
- Find new devs
- Update existing devs
- Merge with your existing watchlist

### Reload Watchlist Without Restart

```python
# In your bot's signal handler or admin endpoint
dev_watchlist.reload()
logger.info(f"Reloaded {dev_watchlist.get_stats()['total_devs']} devs")
```

## Strategy Tips

### 1. **Quality Over Quantity**

```bash
# Conservative: Only 95%+ buy rate devs (fewer trades, higher quality)
--min-buy-rate 95

# Balanced: 90%+ buy rate (good balance)
--min-buy-rate 90

# Aggressive: 80%+ buy rate (more trades, lower quality)
--min-buy-rate 80
```

### 2. **Position Sizing**

```bash
# Small positions (testing)
--buy-amount 0.01

# Medium positions (production)
--buy-amount 0.05

# Large positions (only if proven profitable!)
--buy-amount 0.1
```

### 3. **Hold Time**

```bash
# Quick flip (catch just dev buy)
--hold-time 2.0

# Standard (catch dev + some retail)
--hold-time 3.0

# Patient (catch full retail wave)
--hold-time 5.0
```

## Monitoring

### View Watchlist Stats

```bash
cd frontrun
python -c "
from core.dev_watchlist_loader import DevWatchlistLoader
loader = DevWatchlistLoader('data/dev_watchlist.json')
stats = loader.get_stats()
print(f'Devs: {stats[\"total_devs\"]}')
print(f'Avg Buy Rate: {stats[\"avg_buy_rate\"]:.1f}%')
print(f'Max Buy Rate: {stats[\"max_buy_rate\"]}%')
"
```

### Check Sniper Performance

```bash
# Trades are saved to data/sniper/
ls -lh frontrun/data/sniper/

# View latest trades
cat frontrun/data/sniper/snipes_*.json | jq '.'
```

## Troubleshooting

### "No devs found"

- Lower `--min-buy-rate` to 60-70
- Lower `--min-tokens` to 1-2
- Increase `--max-pages` to get more data

### "Lookups are slow"

- You're NOT using `DevWatchlistLoader`
- Check that you imported from `core.dev_watchlist_loader`
- Should be <1μs per lookup

### "Not finding any trades"

- Check your `--min-buy-rate` filter (try lowering it)
- Verify LaserStream is connected
- Check logs for "creator_not_on_watchlist" messages

## File Structure

```
frontrun/
├── data/
│   └── dev_watchlist.json          # Your growing dev database
├── scripts/
│   ├── build_dev_watchlist.py      # Build/update watchlist
│   └── dev_creation_sniper.py      # Real-time sniping bot
├── core/
│   └── dev_watchlist_loader.py     # Fast O(1) lookup
├── tests/
│   └── test_dev_watchlist_integration.py
└── docs/
    ├── DEV_WATCHLIST_USAGE.md
    └── README_DEV_SNIPING.md (this file)
```

## Next Steps

1. ✅ Build your initial watchlist
2. ✅ Test the integration
3. ✅ Run in simulation mode first
4. ✅ Analyze results
5. ✅ Enable real trades if profitable
6. ✅ Run weekly updates to grow your database

## Questions?

- Check logs in `frontrun/logs/`
- Run integration tests
- Review `DEV_WATCHLIST_USAGE.md` for more examples
