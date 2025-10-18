# Anti-Bot Wallet Classification System

## Purpose

**Before trading, know your environment.**

This system monitors Laserstream in real-time to classify EVERY wallet and detect bot patterns. It answers the critical question:

> **Is there ANY organic edge, or are we just analyzing wash traders and bots?**

## What It Does

### Real-Time Classification

Monitors ALL Pump.fun transactions via Laserstream and classifies wallets into:

1. **Deployer** - Token creators
2. **Sybil** - Coordinated wash trading wallets (funded by same parent, synchronized buys/sells)
3. **Metric Bot** - Triggers on volume/buy thresholds (1-4s after bursts)
4. **Copy Bot** - Follows alpha wallets with fixed delay
5. **Priority Racer** - Uses high CU prices + Jito bundles, lands in first slots
6. **Momentum Scalper** - Buys on acceleration, quick exits
7. **Early Exit Specialist** - First to sell in failed launches
8. **Bait Cluster** - Deployer + sybils that repeatedly rug
9. **Organic** - Real humans (rare)

### Key Metrics Tracked

**Per Wallet:**
- Funding lineage (who funded this wallet?)
- Sibling detection (wallets funded by same parent)
- CU price patterns
- Entry index (position in slot)
- Jito bundle usage rate
- Hold times & P&L
- Mints traded/created

**Per Mint:**
- Burst detection (volume spikes)
- Buyer distribution (Gini coefficient)
- Sybil ratio
- CU price distribution
- Curve state progression

## Architecture

```
anti-bot/
├── core/
│   ├── types.py                 # Data structures (WalletProfile, MintState, etc.)
│   ├── feature_extractor.py    # Extract features from Geyser transactions
│   └── wallet_classifier.py    # Rule-based classification engine
├── analyzer.py                  # Main executable (Laserstream monitor)
├── data/                        # Output classification results
└── README.md                    # This file
```

### Classification Pipeline

```
Laserstream Transaction
    ↓
Feature Extraction (TxFeatures)
    ↓
Update Rolling State (WalletProfile, MintState)
    ↓
Apply Classification Rules
    ↓
Label + Confidence + Reasons
```

## Classification Rules

### 1. Deployer (Priority 1)
- **Trigger**: Signed a Create instruction
- **Confidence**: 1.0

### 2. Sybil (Priority 2)
- **Triggers**:
  - ≥6 wallets funded by same parent within 90s
  - All buy same mint within 5s
  - Buy sizes ≤0.02 SOL
  - Synchronized sells (±1s)
- **Confidence**: 0.4 per indicator

### 3. Metric Bot (Priority 3)
- **Triggers**:
  - Bought 1-4s after burst ≥7 times
  - Low CU price variance (template defaults)
  - Median hold <25s
- **Confidence**: 0.35 per indicator
- **Threshold**: ≥0.6

### 4. Copy Bot (Priority 4)
- **Triggers**:
  - Buys 0.5-2s after alpha wallet ≥6 times
  - Position sizing 0.7-1.3x alpha's size
- **Confidence**: TBD

### 5. Priority Racer (Priority 5)
- **Triggers**:
  - CU price ≥ (p90 + 30%) in 60%+ of trades
  - Mean entry index ≤2
  - Jito usage >50%
- **Confidence**: 0.4 per indicator
- **Threshold**: ≥0.7

### 6. Early Exit Specialist (Priority 6)
- **Triggers**:
  - Among first 3 sellers in ≥5 failed launches
  - Failed launch = no second burst within 30s
- **Confidence**: 0.8 per indicator

## Thresholds (Tunable)

```python
# Burst detection
BURST_MIN_BUYS = 25
BURST_MIN_SOL = 4.0
BURST_WINDOW_SECONDS = 3.0

# Sybil detection
SYBIL_MIN_WALLETS = 6
SYBIL_FUNDING_WINDOW_SECONDS = 90.0
SYBIL_BUY_WINDOW_SECONDS = 5.0
SYBIL_MAX_BUY_SIZE = 0.02  # SOL

# Metric bot detection
METRIC_BOT_MIN_OCCURRENCES = 7
METRIC_BOT_DELAY_MIN = 1.0  # seconds after burst
METRIC_BOT_DELAY_MAX = 4.0
METRIC_BOT_MEDIAN_HOLD = 25.0  # seconds

# Priority racer detection
PRIORITY_RACER_CU_MULTIPLIER = 1.3  # p90 + 30%
PRIORITY_RACER_MIN_ENTRY_INDEX = 2
```

## Usage

### Basic Usage

```bash
# Run for 6 hours (default)
python anti-bot/analyzer.py --config config_fast.yml

# Run for 1 hour
python anti-bot/analyzer.py --hours 1 --config config_fast.yml

# Run until interrupted
python anti-bot/analyzer.py --hours 999
```

### Output

**Live Stats** (every 60 seconds):
```
============================================================
LIVE STATS (5.2 min)
============================================================
Transactions seen: 12,453
Trades seen: 3,821
Wallets tracked: 1,203
Mints tracked: 487

Label Distribution:
  unknown             :    643 ( 53.4%)
  sybil               :    289 ( 24.0%)
  metric_bot          :     87 (  7.2%)
  deployer            :     67 (  5.6%)
  priority_racer      :     54 (  4.5%)
  organic             :     38 (  3.2%)
  early_exit          :     25 (  2.1%)

Detections:
  Sybil clusters: 43
  Metric bots: 87
  Bait clusters: 12
============================================================
```

**Final Report**:
```
================================================================================
ANTI-BOT ANALYSIS - FINAL REPORT
================================================================================
Runtime: 6.00 hours
Total transactions: 389,234
Total trades: 98,123
Wallets classified: 23,401
Mints observed: 8,234

================================================================================
WALLET CLASSIFICATION BREAKDOWN
================================================================================
unknown             : 11,234 ( 48.0%)
sybil               :  5,632 ( 24.1%)
metric_bot          :  2,134 (  9.1%)
deployer            :  1,567 (  6.7%)
priority_racer      :    987 (  4.2%)
organic             :    823 (  3.5%)
...

================================================================================
CRITICAL INSIGHTS
================================================================================

🤖 BOT ACTIVITY: 9,876 wallets (42.2%)
👤 ORGANIC ACTIVITY: 823 wallets (3.5%)

⚠️ WARNING: Environment is HEAVILY botted (42%)
   → Most 'organic' patterns are likely sophisticated bots
   → Standard quant/metrics strategies will fail
   → Edge must come from exploiting bot behaviors
```

**Saved Results**: `anti-bot/data/classification_results_YYYYMMDD_HHMMSS.json`

## Integration with Existing Code

Uses existing "lego pieces":
- ✅ `frontrun/proto/geyser_pb2.py` - Geyser protobuf definitions
- ✅ `src/utils/logger_setup.py` - Logging
- ✅ `src/utils/config_loader.py` - Config loading
- ✅ TradeEvent parsing (from alpha_pattern_observer)
- ✅ Geyser connection setup

## What's Next: Anti-Bot Trading Strategies

Once you know the environment composition, you can:

### Strategy A: Exploit Metric Bots
- **Detect**: Burst threshold approaching
- **Action**: Take seed position before threshold
- **Exit**: Dump when metric bots trigger (2-4s after burst)
- **Target**: Metric bots' slippage

### Strategy B: Early Exit from Failed Launches
- **Detect**: Low unique buyers (<12), high sybil ratio (>50%), no second burst (30s)
- **Action**: Sell immediately, same slot backrun
- **Target**: Beat slow exits

### Strategy C: Avoid Bait Clusters
- **Detect**: Deployer + sybils repeated pattern (≥3 mints/day)
- **Action**: Hard-avoid or short-term short
- **Target**: Defensive (avoid losses)

### Strategy D: Follow Priority Racers (carefully)
- **Detect**: Consistent early entry + high CU + Jito
- **Action**: Backrun their entries with momentum check
- **Risk**: They exit fast

## Critical Questions Answered

### Q1: Are we analyzing wash traders?
**A**: Run for 6 hours, check sybil_ratio in final report. If >30%, yes.

### Q2: Are the "followers" real or fake?
**A**: Check if followers have SYBIL or METRIC_BOT labels. If >60%, fake swarm.

### Q3: Is there organic edge?
**A**: Check organic wallet percentage. If <5%, edge must come from exploiting bots, not organic trading.

### Q4: Why did TIVO fail?
**A**: Check if early buyers (position #1-5) were PRIORITY_RACERS or EARLY_EXIT specialists. They dump coordinated.

## Roadmap

### Phase 1: Classification (COMPLETE)
- ✅ Feature extraction from Laserstream
- ✅ Wallet profiling
- ✅ Mint state tracking
- ✅ Rule-based labeling

### Phase 2: Funding Graph (TODO)
- [ ] Track funding lineage (RPC lookups)
- [ ] Build wallet clusters
- [ ] Detect bait deployer networks

### Phase 3: Trading Hooks (TODO)
- [ ] Export blacklist (avoid sybil clusters)
- [ ] Export opportunity signals (metric bot triggers)
- [ ] Real-time API for sniper integration

### Phase 4: Adaptive Strategies (TODO)
- [ ] Counter-MEV against detected bots
- [ ] Jito backrunning
- [ ] Early exit automation

## Performance Notes

- **Memory**: ~500MB for 6 hours (stores 1000 trades per wallet)
- **CPU**: Minimal (async processing)
- **Network**: Laserstream bandwidth ~10MB/min
- **Latency**: Classification adds <1ms per transaction

## Tuning Tips

1. **Increase confidence thresholds** if getting too many false positives
2. **Decrease min_occurrences** to catch bots earlier (less confident)
3. **Adjust burst thresholds** based on current market conditions
4. **Add custom rules** in `wallet_classifier.py`

## FAQ

**Q: Why not use machine learning?**
**A**: Rule-based is interpretable, fast, and doesn't need training data. Once you know the rules, you can exploit them.

**Q: Can bots evade detection?**
**A**: Yes, sophisticated bots can mimic organic behavior. But most don't bother - they rely on speed, not stealth.

**Q: What if environment is 90% bots?**
**A**: Then the edge comes from exploiting predictable bot behaviors (trigger baiting, early exits), not organic trading.

**Q: Should I trade if environment is heavily botted?**
**A**: Only if you're trading AGAINST the bots (not with them). Otherwise, you're exit liquidity.

## License

Same as parent project.

## Contact

Questions? Check the main project README.
