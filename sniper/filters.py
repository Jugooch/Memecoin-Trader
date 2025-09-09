"""
Safety filters and whitelist/blacklist logic for token validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List


class SafetyFilters:
    """Comprehensive safety checks before entry"""
    
    def __init__(self, config: Dict, store, metrics):
        self.logger = logging.getLogger(__name__)
        self.config = config['safety']
        self.store = store
        self.metrics = metrics
        
        # Cache last rejection reason
        self.last_reason = None
        
        # Dev requirements
        self.dev_min_hold_pct = self.config['dev_min_hold_pct']
        self.dev_max_hold_pct = self.config['dev_max_hold_pct']
        self.dev_min_liq_sol = self.config['dev_min_liq_sol']
        self.dev_max_initial_buy = self.config['dev_max_initial_buy_sol']
        self.dev_no_rug_days = self.config['dev_no_rug_history_days']
        
        # Token requirements
        self.max_tax_bps = self.config['max_tax_bps']
        self.max_tokens_7d = self.config['max_tokens_launched_7d']
        self.min_buyers_5min = self.config['min_unique_buyers_first_5min']
        
        # Blocklist/allowlist
        self.blocked_wallets = set(self.config['blocklist']['wallets'])
        self.blocked_tickers = set(self.config['blocklist']['tickers'])
        self.allowed_wallets = set(self.config['allowlist']['wallets'])
        
        # Track auto-blacklist events
        self.auto_blacklist_reasons = []
    
    def passes_all(self, evt: Dict) -> bool:
        """Run all safety checks on candidate event"""
        checks = [
            self._not_blacklisted(evt),
            self._dev_history_clean(evt),
            self._dev_initials_ok(evt),
            self._lp_min_ok(evt),
            self._dev_hold_bounds(evt),
            self._tax_bounds(evt),
            self._breadth_ok(evt),
            self._ticker_not_blocked(evt)
        ]
        
        for passed, reason in checks:
            if not passed:
                self.last_reason = reason
                self.metrics.inc(f"safety.failed.{reason.lower().replace(' ', '_')}")
                self.logger.warning(f"Safety check failed for {evt.get('ticker', 'UNKNOWN')}: {reason}")
                return False
        
        self.metrics.inc("safety.passed_all")
        return True
    
    def _not_blacklisted(self, evt: Dict) -> Tuple[bool, str]:
        """Check if dev is not blacklisted"""
        dev_wallet = evt.get('dev_wallet')
        
        # Check store blacklist
        if self.store.is_blacklisted(dev_wallet):
            return False, "DEV_BLACKLISTED"
        
        # Check config blocklist
        if dev_wallet in self.blocked_wallets:
            return False, "DEV_BLOCKED"
        
        # Check if in allowlist or whitelisted by discovery (overrides other checks)
        if dev_wallet in self.allowed_wallets or self.store.is_whitelisted(dev_wallet):
            evt['allowlisted'] = True
        
        return True, ""
    
    def _dev_history_clean(self, evt: Dict) -> Tuple[bool, str]:
        """Check dev has no recent rugs"""
        # Skip if allowlisted
        if evt.get('allowlisted'):
            return True, ""
        
        profile = evt.get('profile', {})
        num_rugs = profile.get('num_rugs_90d', 0)
        
        if num_rugs > 0:
            # Auto-blacklist ruggers
            self._auto_blacklist(evt['dev_wallet'], f"Rugged {num_rugs} times in 90d")
            return False, "DEV_RUG_HISTORY"
        
        return True, ""
    
    def _dev_initials_ok(self, evt: Dict) -> Tuple[bool, str]:
        """Check dev's initial buy isn't too large (farming indicator)"""
        # Skip if allowlisted
        if evt.get('allowlisted'):
            return True, ""
        
        dev_initial_buy = evt.get('dev_initial_buy_sol', 0)
        
        if dev_initial_buy > self.dev_max_initial_buy:
            return False, "DEV_INITIAL_BUY_TOO_LARGE"
        
        return True, ""
    
    def _lp_min_ok(self, evt: Dict) -> Tuple[bool, str]:
        """Check minimum liquidity provided"""
        # Skip if allowlisted (whitelisted devs get more lenient liquidity requirements)
        if evt.get('allowlisted'):
            return True, ""
        
        lp_sol = evt.get('lp_init', 0)
        
        if lp_sol < self.dev_min_liq_sol:
            return False, "INSUFFICIENT_LIQUIDITY"
        
        return True, ""
    
    def _dev_hold_bounds(self, evt: Dict) -> Tuple[bool, str]:
        """Check dev holdings are within acceptable range"""
        # Skip if allowlisted
        if evt.get('allowlisted'):
            return True, ""
        
        dev_hold_pct = evt.get('dev_hold_pct', 0)
        
        if dev_hold_pct < self.dev_min_hold_pct:
            return False, "DEV_HOLDING_TOO_LOW"
        
        if dev_hold_pct > self.dev_max_hold_pct:
            return False, "DEV_HOLDING_TOO_HIGH"
        
        return True, ""
    
    def _tax_bounds(self, evt: Dict) -> Tuple[bool, str]:
        """Check combined tax is acceptable"""
        taxes = evt.get('taxes', {})
        buy_tax = taxes.get('buy_tax_bps', 0)
        sell_tax = taxes.get('sell_tax_bps', 0)
        combined_tax = buy_tax + sell_tax
        
        if combined_tax > self.max_tax_bps:
            # Auto-blacklist high tax tokens
            self._auto_blacklist(evt['dev_wallet'], f"High tax: {combined_tax/100}%")
            return False, "TAX_TOO_HIGH"
        
        return True, ""
    
    def _breadth_ok(self, evt: Dict) -> Tuple[bool, str]:
        """Check market breadth (unique buyers)"""
        # This check might be deferred for instant sniping
        # Only enforce if we're entering after initial launch
        
        launch_time = evt.get('timestamp')
        if launch_time and (datetime.now() - launch_time).seconds > 300:  # 5 minutes
            unique_buyers = evt.get('unique_buyers_5min', 0)
            if unique_buyers < self.min_buyers_5min:
                return False, "INSUFFICIENT_BREADTH"
        
        return True, ""
    
    def _ticker_not_blocked(self, evt: Dict) -> Tuple[bool, str]:
        """Check ticker isn't blocked"""
        ticker = evt.get('ticker', '').upper()
        
        if ticker in self.blocked_tickers:
            return False, "TICKER_BLOCKED"
        
        return True, ""
    
    def _auto_blacklist(self, dev_wallet: str, reason: str):
        """Automatically blacklist a dev"""
        self.store.add_to_blacklist(dev_wallet, reason)
        self.auto_blacklist_reasons.append({
            'dev': dev_wallet,
            'reason': reason,
            'timestamp': datetime.now()
        })
        self.logger.warning(f"Auto-blacklisted {dev_wallet[:8]}...: {reason}")
        self.metrics.inc("safety.auto_blacklist")
    
    def check_for_rug(self, evt: Dict, market_data: Dict) -> bool:
        """
        Check for rug conditions after entry
        Returns True if rug detected
        """
        # LP removed
        if market_data.get('lp_removed'):
            self._auto_blacklist(evt['dev_wallet'], "LP removed after launch")
            return True
        
        # Dev dumped
        dev_sold_pct = market_data.get('dev_sold_pct', 0)
        if dev_sold_pct > 50:  # Dev sold >50% of holdings
            self._auto_blacklist(evt['dev_wallet'], f"Dev dumped {dev_sold_pct}%")
            return True
        
        # Honeypot detected (can buy but not sell)
        if market_data.get('honeypot_detected'):
            self._auto_blacklist(evt['dev_wallet'], "Honeypot detected")
            return True
        
        # Tax changed to extreme levels
        new_tax = market_data.get('current_tax_bps', 0)
        if new_tax > self.max_tax_bps * 2:  # Tax doubled beyond limits
            self._auto_blacklist(evt['dev_wallet'], f"Tax increased to {new_tax/100}%")
            return True
        
        return False
    
    def get_filter_stats(self) -> Dict:
        """Get statistics on filter performance"""
        return {
            'auto_blacklisted': len(self.auto_blacklist_reasons),
            'blocked_wallets': len(self.blocked_wallets),
            'blocked_tickers': len(self.blocked_tickers),
            'allowed_wallets': len(self.allowed_wallets),
            'recent_rejections': self.last_reason
        }