"""
Risk scoring and management for dev evaluation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional


class RiskManager:
    """Score devs and manage risk gates"""
    
    def __init__(self, config: Dict, store):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.store = store
        
        # Risk scoring weights
        weights = config['risk_scoring']['weights']
        self.weight_peak_mc = weights['peak_mc'] / 100
        self.weight_rug_count = weights['rug_count'] / 100
        self.weight_holders = weights['holder_count'] / 100
        self.weight_launch_freq = weights['launch_freq'] / 100
        self.weight_lp = weights['lp_quality'] / 100
        self.weight_social = weights['social'] / 100
        
        # Risk gates
        self.min_score = config['risk_scoring']['min_score_to_trade']
        self.dev_cooldown_hours = config['risk_scoring']['dev_cooldown_hours']
        
        # Track recent attempts
        self.recent_attempts = {}  # dev_wallet -> last_attempt_time
        
        # Global limits
        self.max_open_positions = config['trade']['max_open_positions']
        self.max_trades_per_hour = config['trade']['max_trades_per_hour']
        self.max_risk_per_hour = config['trade']['max_risk_per_hour_sol']
        
        # Track hourly metrics
        self.hourly_trades = []  # List of timestamps
        self.hourly_risk = []  # List of (timestamp, sol_amount)
    
    def score_dev(self, dev_wallet: str, event: Dict) -> float:
        """
        Calculate risk score for dev (0-100)
        Higher score = lower risk / better quality
        """
        profile = event.get('profile', {})
        scores = {}
        
        # 1. Peak Market Cap Score (30%)
        best_peak_mc = profile.get('best_peak_mc_usd', 0)
        if best_peak_mc >= 400_000:  # $400k+
            scores['peak_mc'] = 100
        elif best_peak_mc >= 300_000:  # $300k+
            scores['peak_mc'] = 85
        elif best_peak_mc >= 200_000:  # $200k+
            scores['peak_mc'] = 70
        elif best_peak_mc >= 100_000:  # $100k+
            scores['peak_mc'] = 55
        elif best_peak_mc >= 50_000:  # $50K+
            scores['peak_mc'] = 40
        else:
            scores['peak_mc'] = max(0, (best_peak_mc / 50_000) * 40)
        
        # 2. Rug Count Score (20%) - Inverse scoring
        num_rugs = profile.get('num_rugs_90d', 0)
        if num_rugs == 0:
            scores['rug_count'] = 100
        elif num_rugs == 1:
            scores['rug_count'] = 30
        else:
            scores['rug_count'] = 0
        
        # 3. Holder Count Score (15%)
        avg_holders = profile.get('avg_holder_count_30m', 0)
        if avg_holders >= 1000:
            scores['holders'] = 100
        elif avg_holders >= 500:
            scores['holders'] = 80
        elif avg_holders >= 250:
            scores['holders'] = 60
        elif avg_holders >= 100:
            scores['holders'] = 40
        else:
            scores['holders'] = max(0, (avg_holders / 100) * 40)
        
        # 4. Launch Frequency Score (15%) - Inverse scoring
        tokens_7d = profile.get('tokens_launched_7d', 0)
        if tokens_7d <= 1:
            scores['launch_freq'] = 100
        elif tokens_7d == 2:
            scores['launch_freq'] = 70
        elif tokens_7d == 3:
            scores['launch_freq'] = 40
        else:
            scores['launch_freq'] = max(0, 20 - (tokens_7d - 3) * 10)
        
        # 5. LP Quality Score (10%)
        lp_events = profile.get('lp_events', [])
        lp_locks = len([e for e in lp_events if e.get('type') == 'lock'])
        if lp_locks > 0:
            scores['lp_quality'] = 100
        else:
            lp_sol = event.get('lp_init', 0)
            if lp_sol >= 20:
                scores['lp_quality'] = 90
            elif lp_sol >= 10:
                scores['lp_quality'] = 70
            elif lp_sol >= 5:
                scores['lp_quality'] = 50
            else:
                scores['lp_quality'] = max(0, (lp_sol / 5) * 50)
        
        # 6. Social Score (10%) - Optional
        # Could integrate Twitter followers, Telegram members, etc.
        scores['social'] = 50  # Default neutral score
        
        # Calculate weighted total
        total_score = (
            scores['peak_mc'] * self.weight_peak_mc +
            scores['rug_count'] * self.weight_rug_count +
            scores['holders'] * self.weight_holders +
            scores['launch_freq'] * self.weight_launch_freq +
            scores['lp_quality'] * self.weight_lp +
            scores['social'] * self.weight_social
        )
        
        # Store component scores for analysis
        event['risk_scores'] = scores
        event['total_score'] = total_score
        
        self.logger.info(f"Risk score for {dev_wallet[:8]}...: {total_score:.1f} "
                        f"(MC:{scores['peak_mc']:.0f}, Rug:{scores['rug_count']:.0f}, "
                        f"Hold:{scores['holders']:.0f}, Freq:{scores['launch_freq']:.0f})")
        
        return total_score
    
    def allowed(self, score: float, dev_wallet: str = None, sol_amount: float = 0) -> tuple[bool, str]:
        """Check if trade is allowed based on score and risk limits
        Returns: (allowed, reason_if_not_allowed)
        """
        # Check minimum score
        if score < self.min_score:
            reason = f"Score too low: {score:.1f} < {self.min_score}"
            self.logger.warning(reason)
            return False, reason
        
        # Check dev cooldown
        if dev_wallet and not self._check_dev_cooldown(dev_wallet):
            reason = f"Dev cooldown active (24h between launches)"
            self.logger.warning(f"Dev {dev_wallet[:8]}... still in cooldown")
            return False, reason
        
        # Check global position limit
        open_positions = len(self.store.get_open_positions())
        if open_positions >= self.max_open_positions:
            reason = f"Max positions reached: {open_positions}/{self.max_open_positions}"
            self.logger.warning(reason)
            return False, reason
        
        # Check hourly trade limit
        if not self._check_hourly_trades():
            reason = f"Max trades/hour reached: {self.max_trades_per_hour}"
            self.logger.warning(reason)
            return False, reason
        
        # Check hourly risk limit
        if sol_amount > 0 and not self._check_hourly_risk(sol_amount):
            reason = f"Max risk/hour reached: {self.max_risk_per_hour} SOL"
            self.logger.warning(reason)
            return False, reason
        
        return True, ""
    
    def _check_dev_cooldown(self, dev_wallet: str) -> bool:
        """Check if dev is outside cooldown period"""
        last_attempt = self.store.get_dev_last_attempt(dev_wallet)
        
        if not last_attempt:
            return True
        
        cooldown_end = last_attempt + timedelta(hours=self.dev_cooldown_hours)
        if datetime.now() < cooldown_end:
            remaining = (cooldown_end - datetime.now()).total_seconds() / 3600
            self.logger.info(f"Dev {dev_wallet[:8]}... in cooldown for {remaining:.1f} more hours")
            return False
        
        return True
    
    def _check_hourly_trades(self) -> bool:
        """Check if under hourly trade limit"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Clean old entries
        self.hourly_trades = [t for t in self.hourly_trades if t > cutoff]
        
        return len(self.hourly_trades) < self.max_trades_per_hour
    
    def _check_hourly_risk(self, sol_amount: float) -> bool:
        """Check if under hourly risk limit"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Clean old entries
        self.hourly_risk = [(t, amt) for t, amt in self.hourly_risk if t > cutoff]
        
        # Calculate total risk
        total_risk = sum(amt for _, amt in self.hourly_risk)
        
        return (total_risk + sol_amount) <= self.max_risk_per_hour
    
    def record_attempt(self, dev_wallet: str, sol_amount: float):
        """Record a trade attempt for tracking"""
        now = datetime.now()
        
        # Update dev last attempt
        self.store.update_dev_last_attempt(dev_wallet)
        
        # Update hourly metrics
        self.hourly_trades.append(now)
        self.hourly_risk.append((now, sol_amount))
        
        self.logger.info(f"Recorded attempt: {dev_wallet[:8]}... for {sol_amount} SOL")
    
    def get_position_size(self, score: float, config: Dict) -> float:
        """Determine position size based on score"""
        size_tiers = config['trade']['size_by_score']
        
        for tier in size_tiers:
            if score >= tier['min_score']:
                return tier['size_sol']
        
        # Default to base size
        return config['trade']['base_position_size_sol']
    
    def get_priority_fee(self, score: float, congestion_level: int, config: Dict) -> float:
        """Calculate priority fee based on score and congestion"""
        fee_config = config['trade']['priority_fee_sol']
        
        # Get base fee for score tier
        base_fee = fee_config['base']
        max_fee = fee_config['max']
        
        for tier in fee_config.get('by_score', []):
            if score >= tier['min_score']:
                base_fee = tier['base']
                max_fee = tier['max']
                break
        
        # Apply congestion multiplier
        congestion_multipliers = config['trade']['congestion']['multipliers']
        multiplier = congestion_multipliers[min(congestion_level, len(congestion_multipliers) - 1)]
        
        # Calculate final fee
        priority_fee = min(base_fee * multiplier, max_fee)
        
        self.logger.debug(f"Priority fee: {priority_fee} SOL (score: {score:.1f}, congestion: {congestion_level})")
        
        return priority_fee
    
    def get_stats(self) -> Dict:
        """Get risk management statistics"""
        now = datetime.now()
        cutoff = now - timedelta(hours=1)
        
        # Clean old entries
        self.hourly_trades = [t for t in self.hourly_trades if t > cutoff]
        self.hourly_risk = [(t, amt) for t, amt in self.hourly_risk if t > cutoff]
        
        return {
            'hourly_trades': len(self.hourly_trades),
            'max_hourly_trades': self.max_trades_per_hour,
            'hourly_risk_sol': sum(amt for _, amt in self.hourly_risk),
            'max_hourly_risk_sol': self.max_risk_per_hour,
            'open_positions': len(self.store.get_open_positions()),
            'max_positions': self.max_open_positions
        }