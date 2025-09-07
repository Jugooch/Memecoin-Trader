# Farming Detection Hardening Plan
## Preventing Alpha Wallet Exploitation and Reducing Trading Losses

### Executive Summary

Our trading bot is losing money (-$20.23 in recent session) primarily due to "farming" - malicious wallets that bait our algorithm into buying tokens, then immediately dump for profit. This document provides a comprehensive, actionable plan to detect and prevent farming while maintaining our speed advantage.

**Key Finding**: 71% of our losses come from velocity stops triggered within 60 seconds of entry, indicating we're being systematically farmed by wallets that understand our entry triggers.

---

## Table of Contents
1. [Problem Analysis](#problem-analysis)
2. [Solution Architecture](#solution-architecture)
3. [Implementation Plan](#implementation-plan)
4. [Code Implementation](#code-implementation)
5. [Testing & Validation](#testing-validation)
6. [Monitoring & Maintenance](#monitoring-maintenance)

---

## Problem Analysis

### Current Failure Modes

From our trade analysis, we identified these specific patterns causing losses:

1. **Immediate Dumps (Most Critical)**
   - Pattern: Alpha wallet buys → We follow within seconds → Alpha wallet dumps
   - Example: Token `23ijqeuL` - dropped 18% within 40 seconds
   - Frequency: 5 out of 7 trades ended in velocity stops
   - Impact: -$15.72 from velocity stops alone

2. **Fake Momentum Creation**
   - Pattern: Multiple small buys to create upward price movement
   - Example: Token `BpnoXWiL` - artificial pump followed by -19.5% loss
   - Detection Gap: We don't check for ladder buying patterns

3. **Coordination Networks**
   - Pattern: Multiple wallets working together to create false signals
   - Example: Wallets `8aXnwvvw` and `4oANmb5H` both triggered same token
   - Current Issue: No cross-wallet correlation detection

### Why Current Detection Fails

Our current `farming_detector.py` has several critical weaknesses:

1. **No Causality Measurement**: We measure general patterns (hold times, win rates) but not "does following this wallet harm us?"
2. **Wrong Math**: Exit impact uses volume share, not actual price impact
3. **Time-Broken Metrics**: Token survival checks current price, not price 24h after trade
4. **Static Thresholds**: Hand-picked values don't adapt to market regimes
5. **Missing Real-Time Gates**: No pre-entry validation of wallet behavior

---

## Solution Architecture

### Design Principles

1. **Speed First**: Maintain <500ms decision time for alpha signals
2. **Learn & Adapt**: Build reputation system from actual outcomes
3. **Tiered Response**: Different verification levels based on risk
4. **Non-Blocking**: Verification happens in parallel, not sequentially

### Three-Layer Defense System

```
Layer 1: Pre-Entry Gates (0-500ms)
├── Blacklist Check (0ms)
├── Trusted List Check (0ms)  
├── Recent Behavior Check (100-500ms)
└── Risk Score Lookup (0ms)

Layer 3: Post-Entry Monitoring (async)
├── 30-Second Dump Watch
├── Emergency Exit Trigger
└── Reputation Update
```

---

## Implementation Plan

### Phase 1: Critical Fixes (Day 1)
**Goal**: Stop immediate losses from obvious farming

#### Step 1.1: Fix Exit Impact Calculation
**Current Problem**: Using volume share instead of price impact
```python
# WRONG (current)
impact = exit_amount / total_volume_60s

# CORRECT (new)
impact = (price_after_sell - price_before_sell) / price_before_sell
```

**Implementation**:
1. Locate `_analyze_exit_impact` in `farming_detector.py`
2. Replace volume calculation with price delta
3. Use 5-second before and 10-second after windows
4. Handle missing price data gracefully

#### Step 1.2: Add Wallet Blacklist System
**Purpose**: Immediately reject known farmers

**Implementation**:
1. Create `data/blacklisted_wallets.json`
2. Add wallet after 3 trap events
3. Check blacklist before any buy signal
4. Persist blacklist between sessions

#### Step 1.3: Quick Recent-Trade Check
**Purpose**: Detect if wallet just dumped the token

**Implementation**:
1. Before following buy, check wallet's last 3 trades
2. Reject if wallet sold this token in last 90 seconds
3. Use cached data when possible for speed

### Phase 2: Tiered Entry System (Day 2-3)
**Goal**: Balance speed with safety through intelligent position sizing

#### Step 2.1: Create Entry Gate Manager
**File**: `src/core/entry_gate_manager.py`

```python
class EntryGateManager:
    def __init__(self, bitquery_client, config):
        self.bitquery = bitquery_client
        self.config = config
        
        # Reputation tracking
        self.trusted_wallets = set()
        self.blacklisted_wallets = set()
        self.wallet_scores = {}
        
        # Real-time caches
        self.recent_entries = defaultdict(list)
        self.recent_exits = defaultdict(list)
        
        # Load persisted data
        self._load_reputation_data()
    
    async def should_follow_buy(self, wallet: str, token: str, timestamp: float):
        """Main entry point - must return in <500ms"""
        # Implementation details in code section below
```

**Integration Points**:
1. Import in `trading_engine.py`
2. Replace current entry logic with gate manager
3. Pass confidence score to position calculator


#### Step 2.3: Add Post-Entry Monitoring
**Purpose**: Detect farming after entry and emergency exit

**Implementation**:
1. Create async monitoring task after each buy
2. Check every 5 seconds for 30 seconds
3. If alpha wallet sells → emergency exit
4. Update wallet reputation based on outcome

### Phase 3: Statistical Validation (Day 4-5)
**Goal**: Reduce false positives through proper statistics

#### Step 3.1: Implement Follower Harm Metric
**Purpose**: Measure actual damage to those who follow wallets

**Key Metrics**:
- Trap Rate: % of times followers lose >10% within 60s
- Median Follower Return: Typical outcome for followers
- Follower Entry Price: Account for slippage followers pay

**Implementation**:
```python
async def measure_follower_harm(self, wallet, trades_data, bitquery_client):
    """
    Returns:
    - trap_rate: 0.0 to 1.0
    - median_follower_return: -1.0 to +1.0
    - is_trap_wallet: Boolean
    """
```

#### Step 3.2: Add Statistical Significance Testing
**Purpose**: Distinguish real followers from random co-occurrence

**Method**: Binomial test for co-occurrence significance
```python
from scipy.stats import binom_test

# Test if co-occurrences are statistically significant
p_value = binom_test(
    observed_cooccurrences,
    total_opportunities,
    expected_probability,
    alternative='greater'
)

is_significant = p_value < 0.01
```

#### Step 3.3: Fix Token Survival Metric
**Current Bug**: Checks current price instead of t+24h price

**Fix**:
```python
# For each trade, check price exactly 24h later
survival_price = await get_price(token, trade_time + 86400)
survived = (survival_price > entry_price * 0.5)  # >50% of entry
```

### Phase 4: Advanced Detection (Week 2)
**Goal**: Catch sophisticated farming patterns

#### Step 4.1: Ladder Buying Detection
**Pattern**: 5+ small buys within 2 minutes to pump price

**Detection**:
```python
def detect_ladder_buying(trades):
    trades_per_2min = group_by_time_window(trades, 120)
    for window in trades_per_2min:
        if len(window) >= 5 and all(t['type'] == 'buy' for t in window):
            return True
    return False
```

#### Step 4.2: Coordination Network Detection
**Purpose**: Identify wallet clusters working together

**Method**:
1. Track wallet pairs that trade together frequently
2. Calculate Lead-Lag Asymmetry Score (LLAS)
3. Flag coordinated groups

---

## Code Implementation

### Complete Entry Gate Manager Implementation

```python
# src/core/entry_gate_manager.py

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import binom_test

class EntryGateManager:
    """
    High-speed entry validation system for preventing farming exploitation
    while maintaining <500ms decision time for alpha signals
    """
    
    def __init__(self, bitquery_client, config_path: str = "config/farming_config.yml"):
        self.bitquery = bitquery_client
        self.config = self._load_config(config_path)
        
        # Reputation tracking
        self.trusted_wallets = set()
        self.blacklisted_wallets = set()
        self.wallet_scores = {}
        
        # Real-time caches (for speed)
        self.recent_entries = defaultdict(list)  # wallet -> [(token, timestamp)]
        self.recent_exits = defaultdict(list)    # wallet -> [(token, timestamp)]
        self.token_buy_rates = {}               # token -> buys per second
        
        # Performance tracking
        self.decision_times = []
        self.outcomes = []
        
        # Load persisted reputation data
        self._load_reputation_data()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._persist_reputation_data())
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration settings"""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Default values if not in config
        defaults = {
            'max_decision_time_ms': 500,
            'blacklist_after_traps': 3,
            'trust_after_successful': 10,
            'trust_min_success_rate': 0.7,
            'recent_trade_window': 90,
            'monitoring_window': 30,
            'emergency_exit_threshold': -0.15,
            'position_confidence_thresholds': {
                'full': 0.9,
                'large': 0.7,
                'medium': 0.5,
                'small': 0.25
            }
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _load_reputation_data(self):
        """Load persisted reputation data from disk"""
        reputation_file = Path("data/wallet_reputation.json")
        
        if reputation_file.exists():
            with open(reputation_file, 'r') as f:
                data = json.load(f)
                self.trusted_wallets = set(data.get('trusted', []))
                self.blacklisted_wallets = set(data.get('blacklisted', []))
                self.wallet_scores = data.get('scores', {})
        
        # Log loaded data
        print(f"Loaded {len(self.trusted_wallets)} trusted wallets")
        print(f"Loaded {len(self.blacklisted_wallets)} blacklisted wallets")
        print(f"Loaded scores for {len(self.wallet_scores)} wallets")
    
    async def _persist_reputation_data(self):
        """Periodically save reputation data to disk"""
        while True:
            await asyncio.sleep(60)  # Save every minute
            
            data = {
                'trusted': list(self.trusted_wallets),
                'blacklisted': list(self.blacklisted_wallets),
                'scores': self.wallet_scores,
                'last_updated': time.time()
            }
            
            reputation_file = Path("data/wallet_reputation.json")
            reputation_file.parent.mkdir(exist_ok=True)
            
            with open(reputation_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    async def _cleanup_old_data(self):
        """Remove old entries from caches to prevent memory bloat"""
        while True:
            await asyncio.sleep(300)  # Clean every 5 minutes
            
            current_time = time.time()
            cutoff_time = current_time - 3600  # Keep last hour
            
            # Clean recent entries/exits
            for wallet in list(self.recent_entries.keys()):
                self.recent_entries[wallet] = [
                    (token, ts) for token, ts in self.recent_entries[wallet]
                    if ts > cutoff_time
                ]
                if not self.recent_entries[wallet]:
                    del self.recent_entries[wallet]
            
            for wallet in list(self.recent_exits.keys()):
                self.recent_exits[wallet] = [
                    (token, ts) for token, ts in self.recent_exits[wallet]
                    if ts > cutoff_time
                ]
                if not self.recent_exits[wallet]:
                    del self.recent_exits[wallet]
    
    async def should_follow_buy(
        self, 
        wallet: str, 
        token: str, 
        timestamp: float,
        alpha_score: float = 0.5
    ) -> Tuple[bool, float, str, dict]:
        """
        Main entry point for buy decision
        
        Args:
            wallet: Alpha wallet address
            token: Token address
            timestamp: Current timestamp
            alpha_score: Original alpha score from discovery
        
        Returns:
            Tuple of (should_enter, confidence, reason, metadata)
            
        Must complete in <500ms
        """
        start_time = time.time()
        metadata = {'checks_performed': []}
        
        try:
            # INSTANT CHECKS (0ms)
            
            # Check 1: Blacklist
            if wallet in self.blacklisted_wallets:
                return False, 0.0, "blacklisted", metadata
            
            # Check 2: Trusted list
            if wallet in self.trusted_wallets:
                metadata['checks_performed'].append('trusted_list')
                return True, 0.95, "trusted_wallet", metadata
            
            # QUICK CHECKS (<100ms)
            
            # Check 3: Recent exit check
            recent_exit = self._check_recent_exit(wallet, token, timestamp)
            if recent_exit['found']:
                metadata['checks_performed'].append('recent_exit')
                metadata['recent_exit_details'] = recent_exit
                return False, 0.0, f"recent_exit_{recent_exit['seconds_ago']}s_ago", metadata
            
            # Check 4: Rapid buying pattern
            rapid_buying = self._check_rapid_buying(wallet, timestamp)
            if rapid_buying['is_rapid']:
                metadata['checks_performed'].append('rapid_buying')
                metadata['rapid_buying_details'] = rapid_buying
                return False, 0.0, "rapid_buying_pattern", metadata
            
            # Check 5: Historical performance
            risk_score = self._calculate_risk_score(wallet)
            metadata['risk_score'] = risk_score
            
            if risk_score < 0.3:
                confidence = 0.8
                reason = "low_risk"
            elif risk_score > 0.7:
                return False, 0.0, "high_risk", metadata
            else:
                # MEDIUM RISK - Need more checks
                
                # API CALL CHECK (100-500ms with timeout)
                try:
                    recent_trades = await asyncio.wait_for(
                        self.bitquery.get_wallet_token_trades(
                            wallet, token, limit=5
                        ),
                        timeout=0.4  # 400ms timeout
                    )
                    
                    metadata['checks_performed'].append('recent_trades_api')
                    
                    # Check if last action was sell
                    if recent_trades and recent_trades[0]['type'] == 'sell':
                        metadata['last_action'] = 'sell'
                        return False, 0.0, "last_action_was_sell", metadata
                    
                    # Check for pump pattern
                    buy_count = sum(1 for t in recent_trades if t['type'] == 'buy')
                    if buy_count >= 4:  # 4+ buys out of 5 trades
                        metadata['pump_pattern'] = True
                        confidence = 0.4
                        reason = "possible_pump_pattern"
                    else:
                        confidence = 0.6
                        reason = "medium_risk_passed"
                        
                except asyncio.TimeoutError:
                    # API timeout - proceed with caution
                    confidence = 0.5
                    reason = "api_timeout_medium_risk"
                    metadata['api_timeout'] = True
            
            # Record decision time
            decision_time = (time.time() - start_time) * 1000
            self.decision_times.append(decision_time)
            metadata['decision_time_ms'] = decision_time
            
            # Update recent entries cache
            self.recent_entries[wallet].append((token, timestamp))
            
            return True, confidence, reason, metadata
            
        except Exception as e:
            # On error, be conservative
            metadata['error'] = str(e)
            return False, 0.0, "error_in_validation", metadata
    
    def _check_recent_exit(self, wallet: str, token: str, timestamp: float) -> dict:
        """Check if wallet recently exited this token"""
        if wallet not in self.recent_exits:
            return {'found': False}
        
        for exit_token, exit_time in self.recent_exits[wallet]:
            if exit_token == token:
                seconds_ago = timestamp - exit_time
                if seconds_ago < self.config['recent_trade_window']:
                    return {
                        'found': True,
                        'seconds_ago': int(seconds_ago),
                        'exit_time': exit_time
                    }
        
        return {'found': False}
    
    def _check_rapid_buying(self, wallet: str, timestamp: float) -> dict:
        """Check for rapid buying pattern (pump indicator)"""
        if wallet not in self.recent_entries:
            return {'is_rapid': False, 'buy_count': 0}
        
        # Count buys in last 60 seconds
        recent_buys = [
            ts for _, ts in self.recent_entries[wallet]
            if timestamp - ts < 60
        ]
        
        buy_count = len(recent_buys)
        is_rapid = buy_count > 5  # More than 5 buys per minute
        
        return {
            'is_rapid': is_rapid,
            'buy_count': buy_count,
            'time_window': 60
        }
    
    def _calculate_risk_score(self, wallet: str) -> float:
        """Calculate risk score based on historical performance"""
        if wallet not in self.wallet_scores:
            return 0.5  # Unknown wallet = medium risk
        
        scores = self.wallet_scores[wallet]
        
        # Weighted risk calculation
        risk = 0.5  # Base risk
        
        # Adjust based on trap events
        trap_rate = scores.get('trap_events', 0) / max(scores.get('trades_followed', 1), 1)
        risk += trap_rate * 0.3
        
        # Adjust based on success rate
        if scores.get('trades_followed', 0) >= 3:
            success_rate = scores.get('profitable_follows', 0) / scores.get('trades_followed', 1)
            risk -= success_rate * 0.2
        
        # Adjust based on total PnL
        total_pnl = scores.get('total_pnl', 0)
        if total_pnl < -50:  # Lost us $50+
            risk += 0.2
        elif total_pnl > 100:  # Made us $100+
            risk -= 0.1
        
        return max(0.0, min(1.0, risk))
    
    async def verify_entry_post_buy(
        self,
        wallet: str,
        token: str,
        entry_price: float,
        entry_tx: str,
        position_size: float
    ) -> dict:
        """
        Monitor position after entry for farming behavior
        Runs asynchronously, doesn't block trading
        """
        monitoring_start = time.time()
        monitoring_window = self.config['monitoring_window']
        check_interval = 5  # Check every 5 seconds
        
        monitoring_result = {
            'wallet': wallet,
            'token': token,
            'entry_price': entry_price,
            'detected_dump': False,
            'emergency_exit': False,
            'monitoring_duration': 0
        }
        
        for i in range(monitoring_window // check_interval):
            await asyncio.sleep(check_interval)
            
            try:
                # Get wallet's recent trades on this token
                recent_trades = await self.bitquery.get_wallet_trades(
                    wallet, token,
                    start_time=monitoring_start
                )
                
                # Check for sells
                sells = [t for t in recent_trades if t['type'] == 'sell']
                
                if sells:
                    monitoring_result['detected_dump'] = True
                    monitoring_result['dump_time'] = sells[0]['timestamp']
                    monitoring_result['dump_price'] = sells[0]['price']
                    
                    # Calculate impact
                    price_impact = (sells[0]['price'] - entry_price) / entry_price
                    monitoring_result['dump_impact'] = price_impact
                    
                    # Trigger emergency exit if severe
                    if price_impact < self.config['emergency_exit_threshold']:
                        monitoring_result['emergency_exit'] = True
                        # This would trigger actual exit logic
                        print(f"EMERGENCY EXIT: {token} - Alpha wallet dumped at {price_impact:.1%}")
                    
                    # Update wallet reputation
                    await self._update_wallet_reputation(
                        wallet, 
                        outcome='trap',
                        loss_amount=position_size * abs(price_impact)
                    )
                    
                    # Add to recent exits cache
                    self.recent_exits[wallet].append((token, time.time()))
                    
                    break
                    
            except Exception as e:
                print(f"Error monitoring {wallet} on {token}: {e}")
        
        monitoring_result['monitoring_duration'] = time.time() - monitoring_start
        
        # If no dump detected, that's good signal
        if not monitoring_result['detected_dump']:
            await self._update_wallet_reputation(wallet, outcome='held')
        
        return monitoring_result
    
    async def _update_wallet_reputation(
        self,
        wallet: str,
        outcome: str,
        pnl: float = 0,
        loss_amount: float = 0
    ):
        """Update wallet reputation based on trade outcome"""
        if wallet not in self.wallet_scores:
            self.wallet_scores[wallet] = {
                'trades_followed': 0,
                'profitable_follows': 0,
                'trap_events': 0,
                'total_pnl': 0,
                'last_updated': time.time()
            }
        
        scores = self.wallet_scores[wallet]
        scores['trades_followed'] += 1
        scores['last_updated'] = time.time()
        
        if outcome == 'trap':
            scores['trap_events'] += 1
            scores['total_pnl'] -= loss_amount
            
            # Auto-blacklist if too many traps
            if scores['trap_events'] >= self.config['blacklist_after_traps']:
                self.blacklisted_wallets.add(wallet)
                print(f"BLACKLISTED: {wallet} after {scores['trap_events']} trap events")
                
        elif outcome == 'profitable':
            scores['profitable_follows'] += 1
            scores['total_pnl'] += pnl
            
            # Auto-trust if consistently good
            if (scores['trades_followed'] >= self.config['trust_after_successful'] and
                scores['profitable_follows'] / scores['trades_followed'] > self.config['trust_min_success_rate']):
                self.trusted_wallets.add(wallet)
                print(f"TRUSTED: {wallet} after {scores['trades_followed']} successful trades")
                
        elif outcome == 'held':
            # Held through monitoring window without dumping - positive signal
            scores['hold_count'] = scores.get('hold_count', 0) + 1
    
    def calculate_position_size(self, base_amount: float, confidence: float) -> float:
        """
        Calculate position size based on confidence level
        
        Args:
            base_amount: Normal position size
            confidence: Confidence score from should_follow_buy
            
        Returns:
            Adjusted position size
        """
        thresholds = self.config['position_confidence_thresholds']
        
        if confidence >= thresholds['full']:
            return base_amount * 1.0
        elif confidence >= thresholds['large']:
            return base_amount * 0.8
        elif confidence >= thresholds['medium']:
            return base_amount * 0.5
        else:
            return base_amount * 0.25
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        if not self.decision_times:
            return {}
        
        return {
            'avg_decision_time_ms': np.mean(self.decision_times),
            'p95_decision_time_ms': np.percentile(self.decision_times, 95),
            'total_decisions': len(self.decision_times),
            'trusted_wallets': len(self.trusted_wallets),
            'blacklisted_wallets': len(self.blacklisted_wallets),
            'tracked_wallets': len(self.wallet_scores)
        }


# Additional helper class for follower harm analysis
class FollowerHarmAnalyzer:
    """
    Analyzes whether wallets harm their followers
    This is the most important metric for detecting farmers
    """
    
    def __init__(self, bitquery_client):
        self.bitquery = bitquery_client
        self.follower_cache = {}  # Cache follower relationships
    
    async def analyze_follower_outcomes(
        self,
        wallet: str,
        trades_data: List[Dict],
        time_window: int = 60
    ) -> Dict:
        """
        Measure outcomes for wallets that follow this wallet's buys
        
        Returns:
            - trap_rate: % of times followers lose >10%
            - median_follower_return: Typical follower outcome
            - follower_count: Number of unique followers
            - is_trap_wallet: Boolean determination
        """
        harm_events = []
        total_events = 0
        unique_followers = set()
        
        for trade in trades_data:
            if trade['type'] != 'buy':
                continue
            
            # Get buyers who entered after this wallet
            followers = await self.bitquery.get_buyers_in_window(
                token=trade['token'],
                start_time=trade['timestamp'] + 5,  # 5s buffer
                end_time=trade['timestamp'] + time_window
            )
            
            if len(followers) < 2:  # Need at least 2 followers
                continue
            
            total_events += 1
            unique_followers.update([f['wallet'] for f in followers])
            
            # Calculate average follower entry price (includes slippage)
            follower_prices = [f['price'] for f in followers]
            avg_follower_price = np.mean(follower_prices)
            
            # Get price 60 seconds after wallet's buy
            price_60s = await self.bitquery.get_price(
                token=trade['token'],
                timestamp=trade['timestamp'] + 60
            )
            
            if not price_60s:
                continue
            
            # Calculate follower return
            follower_return = (price_60s - avg_follower_price) / avg_follower_price
            
            harm_events.append({
                'token': trade['token'],
                'timestamp': trade['timestamp'],
                'wallet_entry': trade['price'],
                'avg_follower_entry': avg_follower_price,
                'price_60s': price_60s,
                'follower_return': follower_return,
                'follower_count': len(followers),
                'is_trap': follower_return < -0.10  # Lost >10%
            })
        
        # Calculate statistics
        if not harm_events:
            return {
                'trap_rate': 0,
                'median_follower_return': 0,
                'total_events': 0,
                'unique_followers': 0,
                'is_trap_wallet': False
            }
        
        trap_count = sum(1 for e in harm_events if e['is_trap'])
        returns = [e['follower_return'] for e in harm_events]
        
        return {
            'trap_rate': trap_count / len(harm_events),
            'median_follower_return': np.median(returns),
            'avg_follower_return': np.mean(returns),
            'total_events': total_events,
            'unique_followers': len(unique_followers),
            'harm_events': harm_events,
            'is_trap_wallet': (
                trap_count / len(harm_events) > 0.35 and  # >35% trap rate
                len(harm_events) >= 3  # Minimum sample size
            )
        }
    
    async def detect_coordination_patterns(
        self,
        wallet: str,
        related_wallets: List[str],
        time_window: int = 300
    ) -> Dict:
        """
        Detect if wallet coordinates with others to farm
        
        Uses Lead-Lag Asymmetry Score (LLAS) to find coordination
        """
        coordination_scores = {}
        
        for other_wallet in related_wallets:
            if other_wallet == wallet:
                continue
            
            # Get trades for both wallets
            wallet_trades = await self.bitquery.get_wallet_trades(
                wallet, start_time=time.time() - time_window
            )
            other_trades = await self.bitquery.get_wallet_trades(
                other_wallet, start_time=time.time() - time_window
            )
            
            # Calculate LLAS
            llas = self._calculate_llas(wallet_trades, other_trades)
            
            if abs(llas) > 0.3:  # Significant asymmetry
                coordination_scores[other_wallet] = {
                    'llas': llas,
                    'is_leader': llas > 0,
                    'is_follower': llas < 0
                }
        
        return {
            'coordinated_wallets': coordination_scores,
            'coordination_count': len(coordination_scores),
            'is_coordinated': len(coordination_scores) >= 2
        }
    
    def _calculate_llas(
        self,
        wallet_trades: List[Dict],
        other_trades: List[Dict],
        window: int = 60
    ) -> float:
        """
        Calculate Lead-Lag Asymmetry Score
        Positive = wallet leads, Negative = wallet follows
        """
        wallet_leads = 0
        wallet_follows = 0
        
        for w_trade in wallet_trades:
            if w_trade['type'] != 'buy':
                continue
                
            # Check if other wallet bought same token soon after
            for o_trade in other_trades:
                if (o_trade['token'] == w_trade['token'] and
                    o_trade['type'] == 'buy' and
                    0 < o_trade['timestamp'] - w_trade['timestamp'] <= window):
                    wallet_leads += 1
                    break
        
        for o_trade in other_trades:
            if o_trade['type'] != 'buy':
                continue
                
            # Check if wallet bought same token soon after
            for w_trade in wallet_trades:
                if (w_trade['token'] == o_trade['token'] and
                    w_trade['type'] == 'buy' and
                    0 < w_trade['timestamp'] - o_trade['timestamp'] <= window):
                    wallet_follows += 1
                    break
        
        total_interactions = wallet_leads + wallet_follows
        if total_interactions == 0:
            return 0.0
        
        return (wallet_leads - wallet_follows) / total_interactions
```

### Integration with Trading Engine

```python
# src/core/trading_engine.py - Modified section

from src.core.entry_gate_manager import EntryGateManager, FollowerHarmAnalyzer

class TradingEngine:
    def __init__(self, config):
        # ... existing initialization ...
        
        # Initialize farming protection
        self.entry_gate = EntryGateManager(self.bitquery_client)
        self.harm_analyzer = FollowerHarmAnalyzer(self.bitquery_client)
        
    async def handle_alpha_signal(
        self,
        wallet: str,
        token: str,
        signal_strength: float
    ):
        """
        Process alpha signal with farming protection
        """
        timestamp = time.time()
        
        # Step 1: Validate entry through gate manager
        should_enter, confidence, reason, metadata = await self.entry_gate.should_follow_buy(
            wallet=wallet,
            token=token,
            timestamp=timestamp,
            alpha_score=signal_strength
        )
        
        # Log decision
        self.logger.info(
            f"Entry decision for {token[:8]}... from {wallet[:8]}...: "
            f"{'APPROVED' if should_enter else 'REJECTED'} - {reason} "
            f"(confidence: {confidence:.2f}, time: {metadata.get('decision_time_ms', 0):.1f}ms)"
        )
        
        if not should_enter:
            # Track rejected signals for analysis
            self.rejected_signals.append({
                'wallet': wallet,
                'token': token,
                'reason': reason,
                'timestamp': timestamp,
                'metadata': metadata
            })
            return
        
        # Step 2: Calculate position size based on confidence
        base_amount = self.config['position_size_usd']
        adjusted_amount = self.entry_gate.calculate_position_size(
            base_amount, confidence
        )
        
        self.logger.info(
            f"Position sizing: ${base_amount} -> ${adjusted_amount:.2f} "
            f"(confidence: {confidence:.2f})"
        )
        
        # Step 3: Execute buy
        try:
            buy_result = await self.execute_buy(
                token=token,
                amount_usd=adjusted_amount,
                slippage=self.config['slippage']
            )
            
            if buy_result['success']:
                # Step 4: Start post-entry monitoring (non-blocking)
                asyncio.create_task(
                    self._monitor_position(
                        wallet=wallet,
                        token=token,
                        entry_price=buy_result['price'],
                        entry_tx=buy_result['tx_hash'],
                        position_size=adjusted_amount
                    )
                )
                
                # Track active position
                self.active_positions[token] = {
                    'wallet': wallet,
                    'entry_price': buy_result['price'],
                    'entry_time': timestamp,
                    'position_size': adjusted_amount,
                    'confidence': confidence,
                    'tx_hash': buy_result['tx_hash']
                }
                
        except Exception as e:
            self.logger.error(f"Buy execution failed: {e}")
    
    async def _monitor_position(
        self,
        wallet: str,
        token: str,
        entry_price: float,
        entry_tx: str,
        position_size: float
    ):
        """
        Monitor position for farming behavior
        """
        monitoring_result = await self.entry_gate.verify_entry_post_buy(
            wallet=wallet,
            token=token,
            entry_price=entry_price,
            entry_tx=entry_tx,
            position_size=position_size
        )
        
        if monitoring_result['emergency_exit']:
            # Execute emergency exit
            await self.emergency_exit(
                token=token,
                reason=f"Alpha wallet dump detected at {monitoring_result['dump_impact']:.1%}"
            )
            
            # Record the farming event
            self.farming_events.append({
                'wallet': wallet,
                'token': token,
                'entry_price': entry_price,
                'dump_price': monitoring_result.get('dump_price'),
                'loss': position_size * abs(monitoring_result.get('dump_impact', 0)),
                'timestamp': time.time()
            })
    
    async def emergency_exit(self, token: str, reason: str):
        """
        Emergency exit from position
        """
        if token not in self.active_positions:
            return
        
        position = self.active_positions[token]
        
        self.logger.warning(f"EMERGENCY EXIT: {token[:8]}... - {reason}")
        
        try:
            # Execute market sell
            sell_result = await self.execute_sell(
                token=token,
                amount=position['tokens_held'],
                min_price=0  # Accept any price in emergency
            )
            
            if sell_result['success']:
                # Calculate loss
                exit_price = sell_result['price']
                pnl = (exit_price - position['entry_price']) / position['entry_price']
                
                self.logger.info(
                    f"Emergency exit completed: {pnl:.1%} "
                    f"(entry: ${position['entry_price']:.6f}, exit: ${exit_price:.6f})"
                )
                
                # Update wallet reputation
                await self.entry_gate._update_wallet_reputation(
                    wallet=position['wallet'],
                    outcome='trap',
                    loss_amount=abs(pnl * position['position_size'])
                )
                
                # Remove from active positions
                del self.active_positions[token]
                
        except Exception as e:
            self.logger.error(f"Emergency exit failed: {e}")
```

---

## Testing & Validation

### Backtesting Framework

```python
# tests/test_farming_detection.py

import asyncio
from datetime import datetime, timedelta
import pandas as pd
from src.core.entry_gate_manager import EntryGateManager

async def backtest_farming_detection(historical_trades_file: str):
    """
    Backtest farming detection on historical data
    """
    # Load historical trades
    trades = pd.read_csv(historical_trades_file)
    
    # Initialize gate manager
    gate_manager = EntryGateManager(bitquery_client=MockBitqueryClient())
    
    results = []
    
    for _, trade in trades.iterrows():
        # Simulate entry decision
        should_enter, confidence, reason, _ = await gate_manager.should_follow_buy(
            wallet=trade['alpha_wallet'],
            token=trade['token'],
            timestamp=trade['timestamp']
        )
        
        # Compare with actual outcome
        actual_farming = trade['was_farmed']  # From your labeled data
        
        results.append({
            'predicted': not should_enter,  # True if we think it's farming
            'actual': actual_farming,
            'confidence': confidence,
            'reason': reason,
            'pnl': trade['pnl'] if should_enter else 0
        })
    
    # Calculate metrics
    df = pd.DataFrame(results)
    
    metrics = {
        'true_positives': len(df[(df['predicted'] == True) & (df['actual'] == True)]),
        'false_positives': len(df[(df['predicted'] == True) & (df['actual'] == False)]),
        'true_negatives': len(df[(df['predicted'] == False) & (df['actual'] == False)]),
        'false_negatives': len(df[(df['predicted'] == False) & (df['actual'] == True)]),
        'total_pnl': df['pnl'].sum(),
        'avoided_losses': df[df['actual'] == True]['pnl'].sum()  # Would have lost this
    }
    
    metrics['precision'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
    metrics['recall'] = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    
    return metrics

# Run backtest
if __name__ == "__main__":
    metrics = asyncio.run(backtest_farming_detection("data/historical_trades.csv"))
    print(f"Backtest Results:")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1 Score: {metrics['f1_score']:.2%}")
    print(f"Avoided Losses: ${metrics['avoided_losses']:.2f}")
```

### Unit Tests

```python
# tests/unit/test_entry_gate.py

import pytest
from src.core.entry_gate_manager import EntryGateManager

@pytest.mark.asyncio
async def test_blacklist_rejection():
    """Test that blacklisted wallets are rejected"""
    gate = EntryGateManager(None)
    gate.blacklisted_wallets.add("bad_wallet")
    
    should_enter, confidence, reason, _ = await gate.should_follow_buy(
        "bad_wallet", "token", time.time()
    )
    
    assert should_enter == False
    assert reason == "blacklisted"
    assert confidence == 0.0

@pytest.mark.asyncio
async def test_trusted_approval():
    """Test that trusted wallets are approved"""
    gate = EntryGateManager(None)
    gate.trusted_wallets.add("good_wallet")
    
    should_enter, confidence, reason, _ = await gate.should_follow_buy(
        "good_wallet", "token", time.time()
    )
    
    assert should_enter == True
    assert reason == "trusted_wallet"
    assert confidence >= 0.9

@pytest.mark.asyncio
async def test_position_sizing():
    """Test position sizing based on confidence"""
    gate = EntryGateManager(None)
    
    assert gate.calculate_position_size(100, 0.95) == 100  # Full size
    assert gate.calculate_position_size(100, 0.75) == 80   # 80%
    assert gate.calculate_position_size(100, 0.55) == 50   # 50%
    assert gate.calculate_position_size(100, 0.20) == 25   # 25%
```

---

## Monitoring & Maintenance

### Key Metrics to Track

1. **Decision Speed**
   - Average decision time (target: <200ms)
   - P95 decision time (target: <500ms)
   - API timeout rate (target: <5%)

2. **Detection Accuracy**
   - True positive rate (farming correctly identified)
   - False positive rate (good wallets rejected)
   - F1 score (balance of precision and recall)

3. **Financial Impact**
   - PnL from trusted wallets
   - Losses avoided from blacklisted wallets
   - Emergency exit frequency and impact

4. **System Health**
   - Cache hit rates
   - Memory usage
   - Reputation database size

### Weekly Maintenance Tasks

1. **Threshold Calibration**
   ```python
   # scripts/calibrate_thresholds.py
   
   def calibrate_thresholds(recent_trades):
       """Recalibrate detection thresholds based on recent performance"""
       # Group trades by outcome
       farmed = recent_trades[recent_trades['was_farmed'] == True]
       clean = recent_trades[recent_trades['was_farmed'] == False]
       
       # Find optimal thresholds
       best_threshold = optimize_threshold(
           farmed['risk_scores'],
           clean['risk_scores'],
           metric='f1_score'
       )
       
       return {
           'farmer_threshold': best_threshold + 0.1,
           'suspicious_threshold': best_threshold - 0.1
       }
   ```

2. **Reputation Cleanup**
   - Remove wallets inactive >30 days
   - Reset scores for reformed wallets (after 30 days good behavior)
   - Archive old farming events

3. **Performance Analysis**
   - Review rejected signals that would have been profitable
   - Analyze emergency exits for patterns
   - Update blacklist with newly identified farmers

### Configuration Updates

```yaml
# config/farming_config.yml

# Decision Speed
max_decision_time_ms: 500
api_timeout_ms: 400

# Reputation Thresholds  
blacklist_after_traps: 3
trust_after_successful: 10
trust_min_success_rate: 0.7

# Risk Thresholds (update weekly)
farmer_threshold: 0.60
suspicious_threshold: 0.40

# Monitoring
monitoring_window_seconds: 30
emergency_exit_threshold: -0.15

# Position Sizing
position_confidence_thresholds:
  full: 0.90
  large: 0.70
  medium: 0.50
  small: 0.25

# Time Windows
recent_trade_window: 90
rapid_buying_threshold: 5
rapid_buying_window: 60
```

---

## Success Criteria

### Week 1 Goals
- [ ] Reduce velocity stop rate by 50%
- [ ] Maintain decision speed <500ms
- [ ] Identify and blacklist top 10 farming wallets
- [ ] Achieve 0% losses from blacklisted wallets

### Month 1 Goals
- [ ] Build trusted list of 20+ reliable wallets
- [ ] Achieve F1 score >0.8 for farming detection
- [ ] Reduce overall losses from farming by 80%
- [ ] Maintain false positive rate <10%

### Long-term Goals
- [ ] Fully automated reputation system
- [ ] Real-time threshold adaptation
- [ ] Cross-token pattern recognition
- [ ] Network-level coordination detection

---

## Conclusion

This plan addresses the core issue causing your losses: sophisticated farming attacks that exploit your alpha-following strategy. By implementing a tiered defense system that maintains speed while adding intelligent validation, you can:

1. **Immediately stop** obvious farming through blacklists and recent-trade checks
2. **Reduce risk** through confidence-based position sizing
3. **Learn and adapt** from actual outcomes to improve over time
4. **Maintain your edge** by keeping decision times under 500ms

The key insight is that you don't need perfect detection - you need fast detection that improves over time. Start with Phase 1 (critical fixes) today, and you should see immediate improvement in your PnL.

Remember: The farmers succeed because they know your patterns. This system breaks those patterns while maintaining your speed advantage.