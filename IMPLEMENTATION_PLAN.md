# Memecoin Trading Bot - Performance Improvement Implementation Plan

## Overview
This implementation plan addresses the bot's current ~32% win rate and aims to achieve 40%+ win rate with positive EV per trade. All changes are designed to work identically in paper and live trading modes.

## Target Metrics
- **Current State**: Win Rate ~32%, Avg Winner +20%, EV -2%/trade
- **Week 1 Target**: Win Rate ~36%, Avg Winner +25%, EV +0.5%/trade
- **Week 2 Target**: Win Rate ~40%, Avg Winner +30%, EV +2%/trade

## Implementation Phases

---

## Phase 1: Critical Bug Fixes (Day 1 - 4 hours)
**Goal**: Fix broken attribution and remove artificial constraints

### 1.1 Fix Wallet Attribution Bug ✅
**File**: `src/utils/wallet_performance.py`

- [x] Change line 116: Store full mint address instead of `mint[:16]`
- [x] Add transaction signature field to trade records
- [x] Update `record_trade_outcome()` to match on full mint address
- [ ] Add unit tests for attribution:
  - [ ] Test two mints with same first 16 chars
  - [ ] Test multiple buys by same wallet on same mint
  - [ ] Test outcome matching with lookback window
- [x] **Discord**: Notify when attribution fix is deployed

### 1.2 Remove Artificial Constraints ✅
**File**: `config/config.yml`

- [x] Change `max_hold_seconds: 300` → `900` (15 minutes)
- [x] Change `dedup_cache_ttl: 900` → `75` (75 seconds)
- [x] Verify changes don't break existing positions

### 1.3 Add Comprehensive Logging ✅
**Files**: `main.py`, `src/core/trading_engine.py`

- [x] Log entry decisions with scores: `"TradeDecision: mint={} weight={} wallets={}"`
- [x] Log exit reasons: `"Exit: mint={} reason={} hold_sec={} pnl={}"`
- [x] Log wallet performance: `"Wallet {}: win_rate={}% trades={}"`
- [x] Ensure logs work identically in paper/live mode

### 1.4 Discord Notification System ✅
**File**: `src/utils/discord_notifier.py` (enhance existing)

Important notifications to add:
- [x] **Daily Summary** (once at 00:00 UTC):
  ```
  📊 Daily Summary
  P&L: +$X (Y%)
  Trades: X (W wins, L losses)
  Win Rate: X%
  Best Trade: +X%
  Worst Trade: -X%
  Top Wallet: [address] X% WR
  ```
- [x] **Risk Management Triggers**:
  ```
  ⚠️ Risk Management Activated
  Daily P&L: -X%
  Action: Increased selectivity
  New Requirements: X wallets, Y weight
  ```
- [x] **Exceptional Trades**:
  ```
  🚀 Big Winner: +X% on [token]
  💀 Stop Loss Hit: -X% on [token]
  ```
- [x] **Wallet Performance Alerts**:
  ```
  ⭐ New S-Tier Wallet: [address]
  📉 Wallet Demoted: [address] (X% → Y% WR)
  ```
- [x] **System Health** (every 6 hours):
  ```
  ❤️ System Health Check
  Uptime: X hours
  API Usage: X/Y (Moralis), X/Y (Bitquery)
  Active Positions: X
  Memory: X MB
  ```

---

## Phase 2: Bayesian Wallet Scoring (Day 2-3 - 8 hours)
**Goal**: Replace binary threshold with intelligent weighted voting

### 2.1 Implement Bayesian Scorer ✅
**New File**: `src/core/wallet_scorer.py`

```python
class WalletScorer:
    def __init__(self, prior_alpha=3, prior_beta=5):
        """Beta(3,5) prior = 37.5% baseline win rate"""
        self.wallet_stats = {}  # {wallet: deque of {ts, outcome}}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.window_seconds = 3600  # 60-minute rolling window
```

- [x] Implement `update_wallet_outcome(wallet, win_bool, timestamp)`
- [x] Implement `get_wallet_score(wallet)` returning posterior mean
- [x] Implement time-based pruning of old outcomes
- [x] Add recency boost calculation (1.0-1.3x based on last activity)
- [x] Add size factor calculation (1.0-1.5x based on bet size)
- [ ] Unit test with sample win/loss sequences
- [x] **Discord**: Notify when Bayesian scoring goes live

### 2.2 Replace Threshold Check with Weighted Voting ✅
**File**: `src/core/wallet_tracker.py`

- [x] Replace `check_alpha_activity()` threshold logic
- [x] Implement weighted voting:
  ```python
  def calculate_weighted_score(alpha_buyers):
      total_weight = 0
      for wallet in alpha_buyers:
          score = scorer.get_wallet_score(wallet)
          recency = get_recency_boost(wallet)
          size = get_size_factor(wallet)
          total_weight += score * recency * size
      return total_weight
  ```
- [x] Add config parameters:
  - [x] `alpha_weight_min: 2.5` (replaces threshold_alpha_buys)
  - [x] `require_one_wallet_pge_55: true`
  - [x] `alpha_vote_window_sec: 90`

### 2.3 Score Calibration from Stream ✅
**File**: `src/core/wallet_scorer.py`

- [x] Track wallet buys from Bitquery stream
- [x] Define "win" as VWAP +20% before -8% within 15 minutes
- [x] Calculate VWAP from last 20 trades
- [x] Update wallet scores based on stream-derived outcomes
- [x] No additional API calls required

---

## Phase 3: Enhanced Exit Strategy (Day 3-4 - 6 hours)
**Goal**: Capture right-tail gains through dynamic exits

### 3.1 Dynamic TP1 Sizing ✅
**File**: `src/core/trading_engine.py`

- [x] Implement time-based TP1 sizing:
  ```python
  def calculate_tp1_percentage(time_to_tp1):
      if time_to_tp1 < 60:
          return 0.12  # 12% if reached in <1 minute
      elif time_to_tp1 < 180:
          return 0.20  # 20% if reached in 1-3 minutes
      else:
          return 0.30  # 30% if reached after 3 minutes
  ```
- [x] Update position management to track time_to_tp1
- [x] Ensure paper and live modes use identical logic
- [x] **Discord**: Notify on dynamic TP1 triggers with timing

### 3.2 Intelligent Trailing Stops ✅
**File**: `src/core/trading_engine.py`

- [x] Implement peak-based trailing:
  - [x] If peak >= +30% in first 2 min → trail at 85% of peak
  - [x] If peak >= +60% anytime → trail at 82% of peak
- [x] Add break-even stop:
  - [x] Arm at +8% gain
  - [x] Set stop at entry price for 60 seconds
  - [x] Then switch to peak trailing
- [ ] Test trailing logic with simulated price movements

### 3.3 Remove Hard Time Exit ✅
**File**: `src/core/trading_engine.py`

- [x] Remove or extend 5-minute forced exit
- [x] Set `max_hold_seconds: 900` as safety maximum
- [x] Log when max hold time is reached
- [x] **Discord**: Alert if position held for max time

---

## Phase 4: Safety Checks & Risk Management (Day 4-5 - 8 hours)
**Goal**: Prevent losses through intelligent filtering and adaptation

### 4.1 Stream-Based Safety Checks ✅
**File**: `src/core/safety_checks.py`

```python
class SafetyChecker:
    def check_sellability(self, mint, recent_trades):
        """Require at least 1 sell by non-dev wallet in last 60s"""
        
    def estimate_price_impact(self, mint, order_size, recent_trades):
        """Estimate slippage, skip if >1%"""
```

- [x] Implement sellability check (honeypot detection)
- [x] Implement micro-impact estimation
- [x] Add to entry criteria without additional API calls
- [x] Log when trades are skipped for safety
- [ ] **Discord**: Daily summary of skipped trades by reason

### 4.2 Adaptive Risk Management ✅
**File**: `src/core/risk_manager.py`

```python
class AdaptiveRiskManager:
    def get_trading_params(self):
        """Adjust selectivity based on daily P&L"""
        if daily_pnl_pct < -2:
            return {'min_wallet_score': 0.60, 'min_weight': 3.0, ...}
```

- [x] Implement graduated response system
- [x] Track daily P&L from PnLStore
- [x] Adjust thresholds, not position size
- [x] Add cooldown periods after consecutive losses
- [x] Ensure paper/live modes use same risk logic
- [ ] **Discord**: Notify when risk parameters change

### 4.3 Quality-Based Circuit Breakers ✅
**File**: `src/core/risk_manager.py`

- [x] Implement smart cooldowns:
  - [x] 30-min cooldown after 3 consecutive losses
  - [x] 60-min cooldown after 5 consecutive losses
- [x] Add trade frequency limits when down:
  - [x] Normal: 20 trades/day max
  - [x] -2% day: 15 trades/day max
  - [x] -4% day: 10 trades/day max
  - [x] -6% day: 5 trades/day max
- [x] Space out trades when losing:
  - [x] Normal: 2 min between trades
  - [x] -3% day: 5 min between trades (180s)
  - [x] -5% day: 10 min between trades (600s)
- [ ] **Discord**: Alert on cooldown activation/deactivation

---

## Phase 5: Wallet Quality Validation (Day 5-6 - 6 hours)
**Goal**: Ensure we're following actual alpha wallets

### 5.1 VWAP Calibration System ✅
**File**: `src/core/vwap_calibrator.py`

```python
class VWAPCalibrator:
    def __init__(self, sample_rate=0.05, daily_cap=30):
        """Sample 5% of trades for ground truth verification"""
```

- [ ] Sample 1 in 20 wallet outcomes for verification
- [ ] Fetch real price via Moralis (capped at 30/day)
- [ ] Compare VWAP proxy vs real price
- [ ] Adjust win/loss thresholds if bias detected
- [ ] Log calibration metrics
- [ ] **Discord**: Weekly calibration report

### 5.2 Daily Wallet Audit ✅
**File**: `src/discovery/wallet_auditor.py`

```python
class WalletAuditor:
    async def audit_wallets_stratified(self):
        """Daily validation of wallet quality"""
        # Sample 10 top, 10 mid, 10 low performers
        # Fetch last 10 real trades per wallet
        # Calculate actual win rates
        # Demote if <30%, promote if >55%
```

- [ ] Implement stratified sampling (30 wallets/day)
- [ ] Fetch limited history (10 trades per wallet)
- [ ] Calculate real win rates from blockchain
- [ ] Update wallet tiers based on audit
- [ ] API budget: max 60 calls/day
- [ ] Schedule to run once daily
- [ ] **Discord**: Daily audit summary with promotions/demotions

### 5.3 Sybil Attack Detection ✅
**File**: `src/discovery/wallet_auditor.py`

- [ ] Track wallet correlation
- [ ] If two wallets co-buy >70% of same tokens
- [ ] Keep only the better performer
- [ ] Log suspected sybil relationships
- [ ] **Discord**: Alert on sybil detection

---

## Phase 6: Testing Framework (Day 6-7 - 6 hours)
**Goal**: Comprehensive testing to ensure reliability

### 6.1 Test Directory Structure ✅
Create the following structure:
```
tests/
├── README.md                    # Testing guide
├── requirements-test.txt        # Test dependencies
├── run_all_tests.py            # Master test runner
├── unit/                       # Unit tests
│   ├── test_wallet_scorer.py
│   ├── test_attribution.py
│   ├── test_risk_manager.py
│   └── test_vwap_calc.py
├── integration/                # Integration tests
│   ├── test_trading_flow.py
│   ├── test_paper_trading.py
│   └── test_api_limits.py
├── performance/                # Performance tests
│   ├── test_memory_usage.py
│   ├── test_latency.py
│   └── test_throughput.py
├── fixtures/                   # Test data
│   ├── sample_trades.json
│   ├── sample_wallets.json
│   └── mock_stream_data.json
└── reports/                    # Test results
    └── .gitkeep
```

### 6.2 Unit Tests ✅
**File**: `tests/unit/test_wallet_scorer.py`

```python
"""
Test Bayesian Wallet Scorer

Run: python -m pytest tests/unit/test_wallet_scorer.py -v

Expected output:
test_prior_score ✓ (Prior should give 37.5% win rate)
test_perfect_wallet ✓ (10/10 wins should give ~68% score)
test_poor_wallet ✓ (0/10 wins should give ~21% score)
test_time_window_pruning ✓ (Old trades should be removed)
"""

import pytest
from src.core.wallet_scorer import WalletScorer

def test_prior_score():
    """Test that prior gives expected 37.5% win rate"""
    scorer = WalletScorer(prior_alpha=3, prior_beta=5)
    score = scorer.get_wallet_score("new_wallet")
    assert abs(score - 0.375) < 0.001
    
def test_perfect_wallet():
    """Test high-performing wallet gets high score"""
    scorer = WalletScorer()
    for i in range(10):
        scorer.update_wallet_outcome("wallet1", True, time.time())
    score = scorer.get_wallet_score("wallet1")
    assert score > 0.65  # Should be around 0.68
```

### 6.3 Integration Tests ✅
**File**: `tests/integration/test_trading_flow.py`

```python
"""
Test Complete Trading Flow

Run: python -m pytest tests/integration/test_trading_flow.py -v

Expected output:
test_paper_trade_execution ✓ (Paper trade should execute)
test_risk_management_trigger ✓ (Risk should activate at -3%)
test_wallet_voting ✓ (Weighted voting should work)
"""

async def test_paper_trade_execution():
    """Test that paper trading executes correctly"""
    engine = TradingEngine(config, paper_mode=True)
    result = await engine.execute_trade("fake_mint", 25.0, "BUY")
    assert result['success']
    assert result['fees'] > 0
    assert result['paper_mode'] == True
```

### 6.4 Performance Tests ✅
**File**: `tests/performance/test_memory_usage.py`

```python
"""
Test Memory Usage

Run: python tests/performance/test_memory_usage.py

Expected output:
Initial memory: 120 MB
After 1000 trades: 125 MB
After cleanup: 121 MB
✓ Memory usage stable
"""

import tracemalloc
import gc

def test_memory_stability():
    tracemalloc.start()
    
    # Initial snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Simulate 1000 trades
    for i in range(1000):
        process_trade(...)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # Check memory growth
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_growth = sum(stat.size_diff for stat in stats)
    
    assert total_growth < 10_000_000  # Less than 10MB growth
```

### 6.5 Test Runner ✅
**File**: `tests/run_all_tests.py`

```python
#!/usr/bin/env python3
"""
Master Test Runner

Usage: python tests/run_all_tests.py [unit|integration|performance|all]

Example output:
Running unit tests...
✓ 15/15 unit tests passed
Running integration tests...
✓ 8/8 integration tests passed
Running performance tests...
✓ Memory usage: 125MB (stable)
✓ Latency: 45ms average
✓ Throughput: 150 trades/min

Overall: 23/23 tests passed ✅
"""

import sys
import subprocess

def run_tests(category="all"):
    if category in ["unit", "all"]:
        subprocess.run(["pytest", "tests/unit/", "-v"])
    if category in ["integration", "all"]:
        subprocess.run(["pytest", "tests/integration/", "-v"])
    if category in ["performance", "all"]:
        subprocess.run(["python", "tests/performance/test_memory_usage.py"])
```

### 6.6 Statistical Validation ✅
**File**: `src/utils/ab_testing.py`

```python
class SPRTester:
    """Sequential Probability Ratio Test for early stopping"""
    def __init__(self, p0=0.32, p1=0.40, alpha=0.05, beta=0.20):
        # H0: win rate = 32% (current)
        # H1: win rate = 40% (target)
```

- [ ] Implement SPRT for sequential testing
- [ ] Add early stop conditions
- [ ] Track control vs treatment performance
- [ ] Generate daily reports
- [ ] Stop test when statistical significance reached
- [ ] **Discord**: Daily A/B test progress update

---

## Phase 7: Configuration & Deployment (Day 7 - 2 hours)
**Goal**: Deploy production-ready system

### 7.1 Update Configuration ✅
**File**: `config/config.yml`

```yaml
# Core changes
alpha_vote_mode: weighted
alpha_weight_min: 2.5
require_one_wallet_pge_55: true
max_hold_seconds: 900
dedup_cache_ttl: 75

# Scoring
beta_prior_alpha: 3
beta_prior_beta: 5
wallet_score_window_minutes: 60

# Safety
require_sellability: true
max_estimated_impact_pct: 1.0

# Risk management (position size stays constant)
position_size_pct: 0.05  # Never change due to fees

# Discord notifications
discord:
  daily_summary_hour: 0  # UTC
  health_check_hours: 6  # Every 6 hours
  big_win_threshold: 0.5  # 50%+ gains
  risk_alert_threshold: -0.03  # -3% daily
  wallet_audit_summary: true

# Adaptive thresholds
adaptive_safety:
  normal:
    min_wallets: 3
    min_weight: 2.5
    max_daily_trades: 20
    min_time_between: 120
  
  down_2pct:
    min_wallets: 4
    min_weight: 3.0
    max_daily_trades: 15
    min_time_between: 180
  
  down_4pct:
    min_wallets: 5
    min_weight: 3.5
    max_daily_trades: 10
    min_time_between: 300
```

### 7.2 Pre-Deployment Testing ✅
- [ ] Run full test suite: `python tests/run_all_tests.py all`
- [ ] Verify paper trading calculations
- [ ] Test wallet scoring with sample data
- [ ] Test risk management triggers
- [ ] Verify Discord notifications working
- [ ] Check memory usage over time
- [ ] Load test with simulated high-volume stream

### 7.3 Deployment ✅
- [ ] Deploy to VM
- [ ] Start in paper trading mode
- [ ] Monitor for 24 hours
- [ ] Check all logging working
- [ ] Verify Discord notifications arriving
- [ ] Confirm API usage within limits
- [ ] **Discord**: Deployment confirmation message

---

## Discord Notification Schedule

### Continuous Notifications
- **Trade Execution**: On each buy/sell (with confidence score)
- **Big Winners**: Trades closing >50% profit
- **Stop Losses**: Trades hitting stop loss
- **Risk Changes**: When risk parameters adjust

### Periodic Notifications
- **Every 6 Hours**: System health check
- **Daily (00:00 UTC)**: Full performance summary
- **Daily (12:00 UTC)**: Wallet audit results
- **Weekly**: VWAP calibration report

### Alert Notifications
- **Cooldown Activation**: When consecutive losses trigger cooldown
- **API Limits**: When approaching API limits (80% usage)
- **Memory Warning**: If memory usage exceeds threshold
- **Sybil Detection**: When correlated wallets detected
- **Wallet Tier Changes**: S-tier promotions, demotions

---

## Testing Commands Reference

```bash
# Run all tests
python tests/run_all_tests.py all

# Run specific test categories
python tests/run_all_tests.py unit
python tests/run_all_tests.py integration
python tests/run_all_tests.py performance

# Run specific test file
python -m pytest tests/unit/test_wallet_scorer.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run memory profiling
python -m memory_profiler tests/performance/test_memory_usage.py

# Run continuous testing during development
python -m pytest tests/ --watch
```

---

## Critical Implementation Notes

### Paper vs Live Trading Consistency
1. **Execution Layer Only**: Paper and live modes must use identical decision logic
2. **Mock at Execution**: Only mock the actual buy/sell execution, not the decision process
3. **Same Fees**: Use realistic fee estimates in paper mode
4. **Same Timing**: Paper trades must respect same timing constraints as live
5. **Same Risk Logic**: Risk management must work identically in both modes

### Code Structure for Paper/Live Compatibility
```python
class TradingEngine:
    async def execute_trade(self, mint, amount, action):
        """Single execution point for paper/live divergence"""
        if self.paper_mode:
            return await self._execute_paper_trade(mint, amount, action)
        else:
            return await self._execute_live_trade(mint, amount, action)
    
    # All decision logic is shared
    async def should_enter_position(self, signals):
        """Same logic for paper and live"""
        score = self.calculate_score(signals)
        if self.risk_manager.can_trade() and score > threshold:
            return True
        return False
```

---

## Success Metrics (Week 1)
- [ ] Win rate improves from 32% to 36%+
- [ ] Average winner increases from 20% to 25%+
- [ ] Daily P&L shows positive trend
- [ ] API usage stays within limits
- [ ] No memory leaks or crashes
- [ ] Statistical significance via SPRT
- [ ] Discord notifications working reliably

## Success Metrics (Week 2)
- [ ] Win rate reaches 40%+
- [ ] Average winner reaches 30%+
- [ ] Positive EV per trade
- [ ] Consistent daily profits
- [ ] All tests passing
- [ ] Ready for live trading

---

## Rollback Plan
If performance degrades:
1. Keep attribution fix (always beneficial)
2. Revert to binary threshold temporarily
3. Analyze failure patterns via Discord alerts
4. Adjust parameters based on data
5. Re-test with smaller changes

---

## Daily Checklist During Testing

### Morning (via Discord summary)
- [ ] Check overnight performance
- [ ] Review error alerts
- [ ] Check API usage
- [ ] Review wallet audit results
- [ ] Adjust thresholds if needed

### Evening (via Discord summary)
- [ ] Review day's trades
- [ ] Check win rate trend
- [ ] Review exit reasons
- [ ] Check SPRT test status
- [ ] Plan next day adjustments

---

## Emergency Procedures
1. **If bot crashes**: Discord alert → Check logs → Identify cause → Restart with fix
2. **If API rate limited**: Discord alert → Reduce audit frequency → Rotate keys
3. **If losing streak**: Risk management auto-engages → Discord notification
4. **If memory leak**: Discord health check alert → Profile → Fix → Restart
5. **If results diverge from paper**: Discord alert → Stop immediately → Verify logic

---

## Notes
- All timestamps in UTC
- All monetary values in USD
- All percentages in decimal (0.05 = 5%)
- Maintain backward compatibility with existing positions
- Document all parameter changes
- Test every component before deployment
- Monitor Discord for real-time alerts