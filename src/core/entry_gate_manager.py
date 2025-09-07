"""
Entry Gate Manager - High-speed farming detection and entry validation
Prevents farming exploitation while maintaining <500ms decision time
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics


class EntryGateManager:
    """
    High-speed entry validation system for preventing farming exploitation
    while maintaining <500ms decision time for alpha signals
    """
    
    def __init__(self, bitquery_client=None, config: Dict = None):
        self.bitquery = bitquery_client
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get configuration from config dict
        gate_config = self.config.get('entry_gate', {})
        
        # Reputation tracking
        self.trusted_wallets = set()
        self.blacklisted_wallets = set()
        self.wallet_scores = {}
        
        # Real-time caches (for speed)
        self.recent_entries = defaultdict(list)  # wallet -> [(token, timestamp)]
        self.recent_exits = defaultdict(list)    # wallet -> [(token, timestamp)]
        self.recent_dumps = defaultdict(list)    # wallet -> [(token, timestamp, price_impact)]
        
        # Performance tracking
        self.decision_times = []
        self.outcomes = []
        
        # Configuration
        self.max_decision_time_ms = gate_config.get('max_decision_time_ms', 500)
        self.blacklist_after_traps = gate_config.get('blacklist_after_traps', 3)
        self.trust_after_successful = gate_config.get('trust_after_successful', 10)
        self.trust_min_success_rate = gate_config.get('trust_min_success_rate', 0.7)
        self.recent_trade_window = gate_config.get('recent_trade_window', 90)
        self.monitoring_window = gate_config.get('monitoring_window', 30)
        self.emergency_exit_threshold = gate_config.get('emergency_exit_threshold', -0.15)
        
        # Load persisted reputation data
        self._load_reputation_data()
        
        # Start background tasks
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._persist_reputation_data())
        
        self.logger.info(f"EntryGateManager initialized: {len(self.blacklisted_wallets)} blacklisted, "
                        f"{len(self.trusted_wallets)} trusted wallets")
    
    def _load_reputation_data(self):
        """Load persisted reputation data from disk"""
        reputation_file = Path("data/wallet_reputation.json")
        
        if reputation_file.exists():
            try:
                with open(reputation_file, 'r') as f:
                    data = json.load(f)
                    self.trusted_wallets = set(data.get('trusted', []))
                    self.blacklisted_wallets = set(data.get('blacklisted', []))
                    self.wallet_scores = data.get('scores', {})
                    
                    self.logger.info(f"Loaded reputation data: {len(self.trusted_wallets)} trusted, "
                                   f"{len(self.blacklisted_wallets)} blacklisted wallets")
            except Exception as e:
                self.logger.error(f"Error loading reputation data: {e}")
    
    async def _persist_reputation_data(self):
        """Periodically save reputation data to disk"""
        while True:
            await asyncio.sleep(60)  # Save every minute
            
            try:
                data = {
                    'trusted': list(self.trusted_wallets),
                    'blacklisted': list(self.blacklisted_wallets),
                    'scores': self.wallet_scores,
                    'last_updated': time.time()
                }
                
                reputation_file = Path("data/wallet_reputation.json")
                reputation_file.parent.mkdir(exist_ok=True)
                
                # Write to temp file first then rename (atomic operation)
                temp_file = reputation_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                temp_file.rename(reputation_file)
                
            except Exception as e:
                self.logger.error(f"Error persisting reputation data: {e}")
    
    async def _cleanup_old_data(self):
        """Remove old entries from caches to prevent memory bloat"""
        while True:
            await asyncio.sleep(300)  # Clean every 5 minutes
            
            current_time = time.time()
            cutoff_time = current_time - 3600  # Keep last hour
            
            # Clean recent entries/exits
            for cache in [self.recent_entries, self.recent_exits, self.recent_dumps]:
                for wallet in list(cache.keys()):
                    cache[wallet] = [
                        entry for entry in cache[wallet]
                        if (entry[1] if len(entry) > 1 else 0) > cutoff_time
                    ]
                    if not cache[wallet]:
                        del cache[wallet]
    
    async def should_follow_buy(
        self, 
        wallet: str, 
        token: str, 
        timestamp: float,
        alpha_score: float = 0.5
    ) -> Tuple[bool, float, str, dict]:
        """
        Main entry point for buy decision - must complete in <500ms
        
        Args:
            wallet: Alpha wallet address
            token: Token address
            timestamp: Current timestamp
            alpha_score: Original alpha score from discovery
        
        Returns:
            Tuple of (should_enter, confidence, reason, metadata)
        """
        start_time = time.time()
        metadata = {'checks_performed': []}
        
        try:
            # INSTANT CHECKS (0ms)
            
            # Check 1: Blacklist
            if wallet in self.blacklisted_wallets:
                metadata['checks_performed'].append('blacklist')
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                self.logger.warning(f"BLOCKED: Blacklisted wallet {wallet[:8]}... attempted to buy {token[:8]}...")
                return False, 0.0, "blacklisted", metadata
            
            # Check 2: Trusted list
            if wallet in self.trusted_wallets:
                metadata['checks_performed'].append('trusted_list')
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                # Update recent entries cache
                self.recent_entries[wallet].append((token, timestamp))
                return True, 0.95, "trusted_wallet", metadata
            
            # QUICK CHECKS (<100ms)
            
            # Check 3: Recent exit/dump check
            recent_exit = self._check_recent_exit(wallet, token, timestamp)
            if recent_exit['found']:
                metadata['checks_performed'].append('recent_exit')
                metadata['recent_exit_details'] = recent_exit
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                self.logger.warning(f"BLOCKED: Wallet {wallet[:8]}... sold {token[:8]}... "
                                  f"{recent_exit['seconds_ago']}s ago")
                return False, 0.0, f"recent_exit_{recent_exit['seconds_ago']}s_ago", metadata
            
            # Check 4: Recent dump pattern
            recent_dump = self._check_recent_dumps(wallet, timestamp)
            if recent_dump['is_dumper']:
                metadata['checks_performed'].append('recent_dumps')
                metadata['dump_details'] = recent_dump
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                self.logger.warning(f"BLOCKED: Wallet {wallet[:8]}... has {recent_dump['dump_count']} "
                                  f"recent dumps (avg impact: {recent_dump['avg_impact']:.1%})")
                return False, 0.0, "recent_dumper", metadata
            
            # Check 5: Rapid buying pattern
            rapid_buying = self._check_rapid_buying(wallet, timestamp)
            if rapid_buying['is_rapid']:
                metadata['checks_performed'].append('rapid_buying')
                metadata['rapid_buying_details'] = rapid_buying
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                self.logger.warning(f"BLOCKED: Wallet {wallet[:8]}... rapid buying "
                                  f"({rapid_buying['buy_count']} buys in {rapid_buying['time_window']}s)")
                return False, 0.0, "rapid_buying_pattern", metadata
            
            # Check 6: Historical performance
            risk_score = self._calculate_risk_score(wallet)
            metadata['risk_score'] = risk_score
            
            confidence = 0.5  # Default medium confidence
            reason = "unknown_wallet"
            
            if risk_score < 0.3:
                confidence = 0.8
                reason = "low_risk"
            elif risk_score > 0.7:
                decision_time = (time.time() - start_time) * 1000
                metadata['decision_time_ms'] = decision_time
                self.logger.warning(f"BLOCKED: High risk wallet {wallet[:8]}... (score: {risk_score:.2f})")
                return False, 0.0, "high_risk", metadata
            else:
                # Medium risk - proceed with caution
                confidence = 0.6
                reason = "medium_risk"
            
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
            self.logger.error(f"Error in entry validation: {e}")
            return False, 0.0, "error_in_validation", metadata
    
    def _check_recent_exit(self, wallet: str, token: str, timestamp: float) -> dict:
        """Check if wallet recently exited this token"""
        if wallet not in self.recent_exits:
            return {'found': False}
        
        for exit_token, exit_time in self.recent_exits[wallet]:
            if exit_token == token:
                seconds_ago = int(timestamp - exit_time)
                if seconds_ago < self.recent_trade_window:
                    return {
                        'found': True,
                        'seconds_ago': seconds_ago,
                        'exit_time': exit_time
                    }
        
        return {'found': False}
    
    def _check_recent_dumps(self, wallet: str, timestamp: float) -> dict:
        """Check if wallet has been dumping tokens recently"""
        if wallet not in self.recent_dumps:
            return {'is_dumper': False, 'dump_count': 0}
        
        # Get dumps in last 5 minutes
        recent_time = timestamp - 300
        recent_dumps = [
            dump for dump in self.recent_dumps[wallet]
            if dump[1] > recent_time
        ]
        
        if len(recent_dumps) >= 2:  # 2+ dumps in 5 minutes
            impacts = [dump[2] for dump in recent_dumps if len(dump) > 2]
            avg_impact = statistics.mean(impacts) if impacts else 0
            
            return {
                'is_dumper': True,
                'dump_count': len(recent_dumps),
                'avg_impact': avg_impact,
                'time_window': 300
            }
        
        return {'is_dumper': False, 'dump_count': len(recent_dumps)}
    
    def _check_rapid_buying(self, wallet: str, timestamp: float) -> dict:
        """Check for rapid buying pattern (pump indicator)"""
        if wallet not in self.recent_entries:
            return {'is_rapid': False, 'buy_count': 0}
        
        # Count buys in last 60 seconds
        recent_buys = [
            entry for entry in self.recent_entries[wallet]
            if timestamp - entry[1] < 60
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
        monitoring_window = self.monitoring_window
        check_interval = 5  # Check every 5 seconds
        
        monitoring_result = {
            'wallet': wallet,
            'token': token,
            'entry_price': entry_price,
            'detected_dump': False,
            'emergency_exit': False,
            'monitoring_duration': 0,
            'dump_time': None,
            'dump_price': None,
            'dump_impact': None
        }
        
        if not self.bitquery:
            return monitoring_result
        
        for i in range(monitoring_window // check_interval):
            await asyncio.sleep(check_interval)
            
            try:
                # Get wallet's recent trades on this token
                recent_trades = await self.bitquery.get_wallet_token_trades(
                    wallet, token, limit=5
                )
                
                # Check for sells
                sells = [t for t in recent_trades if t.get('side') == 'sell' 
                        and t.get('timestamp', 0) > monitoring_start]
                
                if sells:
                    sell_trade = sells[0]  # Most recent sell
                    monitoring_result['detected_dump'] = True
                    monitoring_result['dump_time'] = sell_trade.get('timestamp')
                    monitoring_result['dump_price'] = sell_trade.get('price', 0)
                    
                    # Calculate impact
                    if entry_price > 0 and monitoring_result['dump_price'] > 0:
                        price_impact = (monitoring_result['dump_price'] - entry_price) / entry_price
                        monitoring_result['dump_impact'] = price_impact
                        
                        # Update recent dumps cache
                        self.recent_dumps[wallet].append((token, time.time(), abs(price_impact)))
                        
                        # Trigger emergency exit if severe
                        if price_impact < self.emergency_exit_threshold:
                            monitoring_result['emergency_exit'] = True
                            self.logger.error(f"EMERGENCY EXIT TRIGGERED: {token[:8]}... "
                                            f"Alpha wallet {wallet[:8]}... dumped at {price_impact:.1%}")
                        
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
                self.logger.error(f"Error monitoring {wallet[:8]}... on {token[:8]}...: {e}")
        
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
            if scores['trap_events'] >= self.blacklist_after_traps:
                self.blacklisted_wallets.add(wallet)
                self.logger.warning(f"AUTO-BLACKLISTED: {wallet[:8]}... after "
                                  f"{scores['trap_events']} trap events")
                
        elif outcome == 'profitable':
            scores['profitable_follows'] += 1
            scores['total_pnl'] += pnl
            
            # Auto-trust if consistently good
            if (scores['trades_followed'] >= self.trust_after_successful and
                scores['profitable_follows'] / scores['trades_followed'] > self.trust_min_success_rate):
                self.trusted_wallets.add(wallet)
                self.logger.info(f"AUTO-TRUSTED: {wallet[:8]}... after "
                               f"{scores['trades_followed']} successful trades")
                
        elif outcome == 'held':
            # Held through monitoring window without dumping - positive signal
            scores['hold_count'] = scores.get('hold_count', 0) + 1
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        if not self.decision_times:
            return {}
        
        return {
            'avg_decision_time_ms': statistics.mean(self.decision_times) if self.decision_times else 0,
            'p95_decision_time_ms': sorted(self.decision_times)[int(len(self.decision_times) * 0.95)] if self.decision_times else 0,
            'total_decisions': len(self.decision_times),
            'trusted_wallets': len(self.trusted_wallets),
            'blacklisted_wallets': len(self.blacklisted_wallets),
            'tracked_wallets': len(self.wallet_scores)
        }