# Trading Bot Feature Reference

**Quick reference for all active trading features and their configurations**

## 🚀 Entry Quality Controls

### Temporal Clustering Check
- **What**: Ensures alpha wallets buy within 90 seconds of each other
- **Why**: Filters out uncoordinated signals spread over minutes
- **Config**: `alpha_enhanced.max_time_spread_seconds: 90`
- **Code**: `wallet_tracker.py:376-412`
- **Log**: `"TEMPORAL CLUSTERING FAILED: Buys spread over Xs"`

### Price Extension Block
- **What**: Hard blocks trades when price is above 90th percentile of last 60s
- **Why**: Prevents buying at local price peaks
- **Config**: `price_extension_percentile: 90`
- **Code**: `main.py:530-539`
- **Log**: `"PRICE EXTENSION BLOCK: Price at Xth percentile"`

### Entry Timing Check
- **What**: Rejects trades >180 seconds after first alpha wallet buy
- **Why**: Prevents late entries after momentum dissipates
- **Config**: `max_entry_delay_seconds: 180`
- **Code**: `main.py:487-507`
- **Log**: `"ENTRY TOO LATE: Xs after first alpha"`

### Stricter Consensus Requirements
- **What**: Requires 5 wallets (was 4), 3.5 weight score (was 3.0)
- **Why**: Ensures stronger consensus before entry
- **Config**: 
  - `threshold_alpha_buys: 5`
  - `alpha_weight_min: 3.5`
  - `require_one_wallet_pge_55: true`

## 🎯 Exit Strategy Features

### Scratch Exits
- **What**: Exit at -2% to -3% if 4% drawdown in first 45 seconds
- **Why**: Prevents small losses from becoming -8% stop losses
- **Config**: `scratch_rule.enabled: true`
- **Code**: `trading_engine.py:397-405`
- **Log**: `"SCRATCH EXIT triggered: Drawdown: X%, P&L: Y%"`

### Dynamic TP1 (Take Profit 1)
- **What**: Variable profit taking based on speed to target
- **Percentages**:
  - <30s: 8% sell (ultra-fast)
  - 30-60s: 12% sell (fast)  
  - 60-180s: 20% sell (normal)
  - >180s: 25% sell (slow)
- **Code**: `trading_engine.py:407-430`
- **Log**: `"Dynamic TP1: Selling X% after Ys to TP"`

### Moonshot Detection
- **What**: Skip TP1 entirely if already at 60%+ gain at TP target
- **Why**: Lets exceptional runs continue with just trailing stops
- **Code**: `trading_engine.py:413-418`
- **Log**: `"MOONSHOT DETECTED: X% gain, skipping TP1"`

### Enhanced Trailing Stops
- **What**: Looser trailing stops for exceptional gains
- **Percentages**:
  - 100%+ gains: Trail at 75% of peak
  - 60%+ gains: Trail at 80% of peak
  - 30%+ fast gains: Trail at 85% of peak
  - Standard: Trail at 85% of peak
- **Code**: `trading_engine.py:434-450`
- **Log**: `"MOONSHOT EXIT: X% peak, exiting at Y%"`

### Volatility-Based Stop Losses
- **What**: Dynamic stop losses based on hold time and token volatility
- **Levels**:
  - 0-30s: 4% stop (tight)
  - 30-60s: 5-15% (calculated from volatility)
  - 60-180s: 8% stop (standard)
  - 180s+: 10% stop (wider)
- **Code**: `trading_engine.py:574-646`
- **Log**: `"DYNAMIC STOP HIT: Volatility-adjusted stop at X%"`

## 📊 Risk Management Levels

### Adaptive Parameters Based on Daily P&L
| P&L Level | Min Wallets | Min Weight | Max Daily | Min Time Between |
|-----------|-------------|------------|-----------|------------------|
| Normal (0% to -2%) | 3 | 2.5 | 20 | 120s |
| Cautious (-2% to -4%) | 4 | 3.0 | 15 | 180s |
| Conservative (-4% to -6%) | 5 | 3.5 | 10 | 300s |
| Defensive (-6%+) | 6 | 4.0 | 5 | 600s |

## 🔍 Enhanced Logging Patterns

### Entry Decision Logs
```bash
# Approved Trade
✅ TRADE APPROVED: mint=abc123... weight=3.8 confidence=65.0 
   risk_level=normal wallets=[wallet1(0.8), wallet2(0.9)] 
   safety=45 mult=1.2x distinct=5 liquidity=$35,000 
   entry_delay=45s spread=30s price_percentile=75th

# Rejected Trade  
❌ TRADE REJECTED: mint=xyz789... 
   reasons=[temporal_clustering(120s), low_weight(2.5<3.5)]
   confidence=45.0 wallets=4 distinct=4 weight=2.5
```

### Exit Decision Logs
```bash
# Scratch Exit
🔸 SCRATCH EXIT triggered: Drawdown: 4.2%, Accel: -0.8, P&L: -2.1%

# Moonshot Detection
MOONSHOT DETECTED: 75% gain, skipping TP1 to capture tail

# Dynamic Stop Loss
DYNAMIC STOP HIT: Volatility-adjusted stop at -6.5%

# Trailing Stop
MOONSHOT EXIT: 120% peak, exiting at 90%

# Dynamic TP1
Dynamic TP1: Selling 12% after 45s to TP
```

## ⚙️ Configuration Checklist

### Required Config Updates
```yaml
# Entry Requirements (STRICTER)
threshold_alpha_buys: 5              # Up from 4
alpha_weight_min: 3.5               # Up from 3.0
require_one_wallet_pge_55: true     # Was false

# New Timing Controls
alpha_enhanced:
  max_time_spread_seconds: 90       # NEW: Temporal clustering
max_entry_delay_seconds: 180        # NEW: Entry timing
price_extension_percentile: 90      # NEW: Price extension

# Exit Features (Auto-Active)
scratch_rule:
  enabled: true
  max_time_seconds: 45
  drawdown_threshold: 0.04
  exit_range_low: -0.03
  exit_range_high: -0.02

# Quality Requirements
min_liquidity_usd: 25000            # Higher quality tokens
```

## 📈 Key Performance Indicators

### Entry Quality Metrics
- **Temporal Clustering Pass Rate**: >70%
- **Price Extension Block Rate**: <20% of rejections  
- **Entry Timing Block Rate**: <15% of rejections
- **Overall Approval Rate**: 10-20% of detected tokens

### Exit Effectiveness Metrics  
- **Scratch Exit Rate**: 10-15% of losing trades
- **Moonshot Capture Rate**: 2-5% of all trades
- **Dynamic TP1 Average**: 15-20% (not fixed 30%)
- **Volatility Stop Reduction**: 30%+ fewer -8% stops

### Overall Performance Targets
- **Win Rate**: 35-45% (up from 20%)
- **Average Win**: +25% (up from +20%)
- **Average Loss**: -2.5% (down from -6%)  
- **Daily P&L Volatility**: ±3-5% on $500 capital
- **Trade Frequency**: 5-15/day (down from 20+)

## 🚨 Common Issues & Solutions

### Too Few Trades
**Problem**: Very strict filters blocking most opportunities
**Solutions**: 
- Reduce `threshold_alpha_buys` to 4
- Lower `alpha_weight_min` to 3.0  
- Increase `max_time_spread_seconds` to 120

### Low Win Rate Still
**Problem**: Still getting bad signals through filters
**Solutions**:
- Lower `price_extension_percentile` to 85
- Reduce `max_entry_delay_seconds` to 120  
- Increase `min_liquidity_usd` to 35000

### Missing Moonshots
**Problem**: Taking too much profit on big runners
**Solutions**:
- Enable moonshot detection (should be auto-active)
- Check scratch exits aren't too aggressive
- Verify trailing stops are working (75-85% levels)

---

*All features listed are active in the current codebase and ready for production use.*