# Memecoin Trading Bot - Current State & Recent Updates

**Last Updated**: December 2024  
**Version**: 2.0 (Post-Optimization Update)

## 📊 Current Performance Targets

- **Target Win Rate**: 35-45% (improved from 20%)
- **Target Gains**: 15-25% per winning trade
- **Maximum Loss**: 2-8% per losing trade (volatility-adjusted)
- **Hold Time**: 30 seconds to 5 minutes average

## 🚀 Recent Major Updates (December 2024)

### Entry Quality Improvements

#### 1. **Temporal Clustering Requirement** ✅
- **File**: `src/core/wallet_tracker.py`
- **Change**: Added check to ensure alpha wallet buys are clustered within 90 seconds
- **Impact**: Filters out uncoordinated signals where wallets buy spread over minutes
- **Config**: `alpha_enhanced.max_time_spread_seconds: 90`
- **Log Pattern**: `"TEMPORAL CLUSTERING FAILED: Buys spread over Xs"`

#### 2. **Price Extension Hard Block** ✅
- **File**: `main.py`
- **Change**: Hard blocks trades when price is above 90th percentile of last 60 seconds
- **Impact**: Prevents buying at local peaks
- **Config**: `price_extension_percentile: 90`
- **Log Pattern**: `"PRICE EXTENSION BLOCK: Price at Xth percentile"`

#### 3. **Entry Timing Enforcement** ✅
- **File**: `main.py`
- **Change**: Rejects trades more than 180 seconds after first alpha buy
- **Impact**: Prevents late entries after momentum has dissipated
- **Config**: `max_entry_delay_seconds: 180`
- **Log Pattern**: `"ENTRY TOO LATE: Xs after first alpha"`

### Exit Strategy Optimizations

#### 4. **Volatility-Based Stop Losses** ✅
- **File**: `src/core/trading_engine.py`
- **Change**: Dynamic stop losses based on hold time and token volatility
- **Time-based adjustments**:
  - 0-30s: 4% stop (tight for quick scratch)
  - 30-60s: 5-15% (volatility-calculated)
  - 60-180s: 8% stop (standard)
  - 180s+: 10% stop (wider for established positions)
- **Log Pattern**: `"DYNAMIC STOP HIT: Volatility-adjusted stop at X%"`

#### 5. **Enhanced TP1 Logic** ✅
- **File**: `src/core/trading_engine.py`
- **Change**: Dynamic profit-taking based on momentum speed
- **Moonshot Detection**: Skip TP1 entirely if already at 60%+ gain
- **TP1 Percentages**:
  - <30s to target: 8% sell (ultra-fast momentum)
  - 30-60s: 12% sell
  - 60-180s: 20% sell
  - 180s+: 25% sell
- **Log Pattern**: `"MOONSHOT DETECTED: X% gain, skipping TP1"`

#### 6. **Optimized Trailing Stops** ✅
- **File**: `src/core/trading_engine.py`
- **Change**: Looser trailing for exceptional gains
- **Trailing Percentages**:
  - 100%+ gains: Trail at 75% of peak (capture moonshots)
  - 60%+ gains: Trail at 80% of peak
  - 30%+ gains (fast): Trail at 85% of peak
  - Standard: Trail at 85% of peak
- **Log Pattern**: `"MOONSHOT EXIT: X% peak, exiting at Y%"`

### Configuration Updates

#### 7. **Stricter Entry Requirements** ✅
- **File**: `config/config.yml.example`
- **Changes**:
  - `threshold_alpha_buys`: 4 → **5** (more wallets required)
  - `alpha_weight_min`: 3.0 → **3.5** (higher weighted threshold)
  - `require_one_wallet_pge_55`: false → **true** (confidence requirement enabled)
  - `min_liquidity_usd`: **25,000** (higher quality tokens)

## 📈 Current Trading Logic Flow

### Entry Decision Tree
```
1. Token Detected
   ↓
2. Check Alpha Wallets (minimum 5 required)
   ↓
3. Temporal Clustering Check (must be within 90s)
   ↓
4. Entry Timing Check (must be within 180s of first alpha)
   ↓
5. Price Extension Check (must be below 90th percentile)
   ↓
6. Safety Checks (sellability, liquidity, rug score)
   ↓
7. Risk Manager Approval (adaptive based on P&L)
   ↓
8. Execute Trade
```

### Exit Decision Priority
```
1. Scratch Exit (first 45s, -2% to -3% with 4% drawdown)
   ↓
2. Dynamic TP1 (at 1.25x, selling 8-25% based on speed)
   ↓
3. Moonshot Detection (skip TP1 if already 60%+ gain)
   ↓
4. Trailing Stops (75-85% of peak based on gain level)
   ↓
5. Break-even Protection (armed at +8%, 60s protection)
   ↓
6. Volatility Stop Loss (4-10% based on time and volatility)
   ↓
7. Time Exit (15 minutes if <5% gain)
```

## 🔧 Current Configuration Defaults

### Entry Requirements
- **Minimum Alpha Wallets**: 5
- **Minimum Weight Score**: 3.5
- **Maximum Time Spread**: 90 seconds
- **Maximum Entry Delay**: 180 seconds
- **Price Extension Percentile**: 90
- **Minimum Liquidity**: $25,000
- **One Wallet ≥55% Confidence**: Required

### Risk Management Levels
| P&L Status | Min Wallets | Min Weight | Max Daily Trades | Min Time Between |
|------------|-------------|------------|------------------|------------------|
| Normal (0% to -2%) | 3 | 2.5 | 20 | 120s |
| Cautious (-2% to -4%) | 4 | 3.0 | 15 | 180s |
| Conservative (-4% to -6%) | 5 | 3.5 | 10 | 300s |
| Defensive (-6%+) | 6 | 4.0 | 5 | 600s |

### Exit Parameters
- **TP1 Target**: 1.25x (25% gain)
- **Scratch Window**: First 45 seconds
- **Scratch Range**: -2% to -3%
- **Peak Drawdown Threshold**: 4%
- **Break-even Armed**: At +8% gain
- **Break-even Duration**: 60 seconds
- **Maximum Hold Time**: 15 minutes (if <5% gain)

## 📊 Enhanced Logging

### Entry Logs
```
✅ TRADE APPROVED: mint=abc123... weight=3.8 confidence=65.0 
   risk_level=normal wallets=[wallet1(0.8), wallet2(0.9)] 
   safety=45 mult=1.2x distinct=5 liquidity=$35,000 
   entry_delay=45s spread=30s price_percentile=75th

❌ TRADE REJECTED: mint=xyz789... 
   reasons=[temporal_clustering(120s), low_weight(2.5<3.5)]
   confidence=45.0 wallets=4 distinct=4 weight=2.5
```

### Exit Logs
```
🔸 SCRATCH EXIT triggered: Drawdown: 4.2%, Accel: -0.8, P&L: -2.1%
MOONSHOT DETECTED: 75% gain, skipping TP1 to capture tail
DYNAMIC STOP HIT: Volatility-adjusted stop at -6.5%
MOONSHOT EXIT: 120% peak, exiting at 90%
Dynamic TP1: Selling 12% after 45s to TP
```

## 🎯 Key Metrics to Monitor

### Entry Quality
- **Temporal Clustering Failures**: Should be <30% of rejections
- **Price Extension Blocks**: Should be <20% of rejections
- **Late Entry Blocks**: Should be <15% of rejections
- **Average Entry Delay**: Target 30-90 seconds after first alpha

### Exit Effectiveness
- **Scratch Exit Rate**: Target 10-15% of losing trades
- **Moonshot Captures**: Target 2-5% of trades reaching 60%+
- **Average TP1 Size**: Should average 15-20% (not always 30%)
- **Volatility Stop Effectiveness**: Should reduce -8% stops by 30%+

## 🚨 Known Issues & Limitations

1. **Wallet Rotation Impact**: Fresh wallets still get 65% initial confidence
2. **Volatility Calculation**: Requires 5+ recent trades (may use default on new tokens)
3. **Temporal Data**: Relies on realtime cache which may miss some timestamps
4. **Price Extension**: Only checks last 60 seconds (may miss longer-term trends)

## 🔄 Next Planned Improvements

1. **Graduated Trust System**: Time-based confidence decay for fresh wallets
2. **Cohort Validation**: Require 3+ fresh wallets to agree for validation
3. **Advanced Volatility**: Use ATR or Bollinger Bands for better volatility measurement
4. **Multi-timeframe Analysis**: Check price extension on multiple timeframes
5. **Buyer Acceleration**: Implement actual calculation instead of placeholder

## 📝 Configuration Checklist

When setting up the bot with these updates, ensure:

- [ ] Set `threshold_alpha_buys` to at least 5
- [ ] Set `alpha_weight_min` to at least 3.5
- [ ] Enable `require_one_wallet_pge_55: true`
- [ ] Configure `max_time_spread_seconds: 90`
- [ ] Configure `max_entry_delay_seconds: 180`
- [ ] Configure `price_extension_percentile: 90`
- [ ] Set `min_liquidity_usd` to at least 25000
- [ ] Enable `scratch_rule.enabled: true`
- [ ] Review and adjust risk management thresholds

## 📈 Performance Expectations

With all optimizations active, you should expect:

- **Win Rate**: 35-45% (up from 20%)
- **Average Win**: +20-25% (with occasional moonshots)
- **Average Loss**: -2.5% (down from -6%)
- **Daily P&L Volatility**: ±3-5% on $500 capital
- **Trade Frequency**: 5-15 trades per day (down from 20+)
- **Quality Score**: 70%+ of trades should pass all quality checks

---

*This document represents the current state of the bot as of December 2024. All features listed are fully implemented and active in the codebase.*