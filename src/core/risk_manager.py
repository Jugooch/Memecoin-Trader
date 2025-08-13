"""
Adaptive Risk Manager for dynamic trading parameter adjustment
Adjusts selectivity based on daily P&L, never changes position size
"""

import logging
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class RiskLevel:
    """Risk level configuration"""
    name: str
    min_wallets: int
    min_weight: float
    max_daily_trades: int
    min_time_between: int  # seconds
    min_wallet_score: float
    description: str


class AdaptiveRiskManager:
    """
    Manages risk by adjusting trading selectivity based on performance
    Position size remains constant to avoid fee impact
    """
    
    def __init__(self, pnl_store=None, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.pnl_store = pnl_store
        self.config = config or {}
        
        # Track consecutive losses for circuit breakers
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.trades_today = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # Cooldown tracking
        self.cooldown_until = 0
        self.cooldown_reason = None
        
        # Define risk levels
        self.risk_levels = {
            'normal': RiskLevel(
                name='normal',
                min_wallets=3,
                min_weight=2.5,
                max_daily_trades=20,
                min_time_between=120,
                min_wallet_score=0.45,
                description='Normal trading parameters'
            ),
            'cautious': RiskLevel(
                name='cautious',
                min_wallets=4,
                min_weight=3.0,
                max_daily_trades=15,
                min_time_between=180,
                min_wallet_score=0.50,
                description='Down 2%, increased selectivity'
            ),
            'conservative': RiskLevel(
                name='conservative',
                min_wallets=5,
                min_weight=3.5,
                max_daily_trades=10,
                min_time_between=300,
                min_wallet_score=0.55,
                description='Down 4%, high selectivity'
            ),
            'defensive': RiskLevel(
                name='defensive',
                min_wallets=6,
                min_weight=4.0,
                max_daily_trades=5,
                min_time_between=600,
                min_wallet_score=0.60,
                description='Down 6%+, maximum selectivity'
            )
        }
        
        # Current risk level
        self.current_level = self.risk_levels['normal']
        
    def get_daily_pnl_percentage(self) -> float:
        """
        Get current daily P&L percentage
        
        Returns:
            Daily P&L as percentage (e.g., -2.5 for -2.5%)
        """
        if not self.pnl_store:
            return 0.0
            
        try:
            daily_pnl = self.pnl_store.get_daily_pnl()
            initial_capital = self.config.get('initial_capital', 500.0)
            
            if initial_capital > 0:
                return (daily_pnl / initial_capital) * 100
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting daily P&L: {e}")
            return 0.0
    
    def get_trading_params(self) -> Dict:
        """
        Get current trading parameters based on daily P&L
        
        Returns:
            Dictionary with current risk-adjusted parameters
        """
        # Check if we need to reset daily counters
        self._check_daily_reset()
        
        # Get current P&L
        daily_pnl_pct = self.get_daily_pnl_percentage()
        
        # Determine risk level based on P&L
        if daily_pnl_pct <= -6:
            new_level = self.risk_levels['defensive']
        elif daily_pnl_pct <= -4:
            new_level = self.risk_levels['conservative']
        elif daily_pnl_pct <= -2:
            new_level = self.risk_levels['cautious']
        else:
            new_level = self.risk_levels['normal']
        
        # Log if risk level changed
        if new_level.name != self.current_level.name:
            self.logger.warning(f"Risk level changed: {self.current_level.name} → {new_level.name} "
                              f"(Daily P&L: {daily_pnl_pct:+.1f}%)")
            self.current_level = new_level
        
        return {
            'min_wallets': self.current_level.min_wallets,
            'min_weight': self.current_level.min_weight,
            'max_daily_trades': self.current_level.max_daily_trades,
            'min_time_between': self.current_level.min_time_between,
            'min_wallet_score': self.current_level.min_wallet_score,
            'risk_level': self.current_level.name,
            'daily_pnl_pct': daily_pnl_pct,
            'trades_remaining': self.current_level.max_daily_trades - self.trades_today
        }
    
    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on risk parameters
        
        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.cooldown_until:
            remaining = int(self.cooldown_until - current_time)
            return False, f"Cooldown active ({self.cooldown_reason}): {remaining}s remaining"
        
        # Check daily trade limit
        params = self.get_trading_params()
        if self.trades_today >= params['max_daily_trades']:
            return False, f"Daily trade limit reached ({params['max_daily_trades']})"
        
        # Check time between trades
        if self.last_trade_time > 0:
            time_since_last = current_time - self.last_trade_time
            if time_since_last < params['min_time_between']:
                wait_time = int(params['min_time_between'] - time_since_last)
                return False, f"Too soon after last trade (wait {wait_time}s)"
        
        return True, None
    
    def record_trade_outcome(self, is_win: bool) -> None:
        """
        Record a trade outcome and update risk state
        
        Args:
            is_win: Whether the trade was profitable
        """
        self.trades_today += 1
        self.last_trade_time = time.time()
        
        if is_win:
            # Reset consecutive losses on a win
            self.consecutive_losses = 0
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            
            # Check for cooldown triggers
            if self.consecutive_losses >= 5:
                self.activate_cooldown(3600, "5 consecutive losses")  # 60 min
            elif self.consecutive_losses >= 3:
                self.activate_cooldown(1800, "3 consecutive losses")  # 30 min
    
    def activate_cooldown(self, duration_seconds: int, reason: str) -> None:
        """
        Activate a trading cooldown
        
        Args:
            duration_seconds: Cooldown duration in seconds
            reason: Reason for the cooldown
        """
        self.cooldown_until = time.time() + duration_seconds
        self.cooldown_reason = reason
        self.logger.warning(f"Trading cooldown activated: {reason} ({duration_seconds}s)")
    
    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily counters"""
        current_time = datetime.now()
        if current_time.date() > self.daily_reset_time.date():
            self.trades_today = 0
            self.consecutive_losses = 0
            self.daily_reset_time = current_time.replace(hour=0, minute=0, second=0)
            self.logger.info("Daily risk counters reset")
    
    def get_risk_summary(self) -> Dict:
        """
        Get current risk management status
        
        Returns:
            Dictionary with risk management metrics
        """
        params = self.get_trading_params()
        can_trade, reason = self.can_trade()
        
        return {
            'risk_level': self.current_level.name,
            'risk_description': self.current_level.description,
            'daily_pnl_pct': params['daily_pnl_pct'],
            'trades_today': self.trades_today,
            'trades_remaining': params['trades_remaining'],
            'consecutive_losses': self.consecutive_losses,
            'can_trade': can_trade,
            'block_reason': reason,
            'cooldown_active': time.time() < self.cooldown_until,
            'parameters': {
                'min_wallets': params['min_wallets'],
                'min_weight': params['min_weight'],
                'min_wallet_score': params['min_wallet_score'],
                'min_time_between': params['min_time_between']
            }
        }
    
    def should_skip_token(self, confidence_score: float, wallet_score: float) -> bool:
        """
        Check if a token should be skipped based on current risk parameters
        
        Args:
            confidence_score: Token confidence score
            wallet_score: Average wallet score for the signal
            
        Returns:
            True if token should be skipped
        """
        params = self.get_trading_params()
        
        # Check minimum wallet score
        if wallet_score < params['min_wallet_score']:
            self.logger.info(f"Token skipped: wallet score {wallet_score:.2f} < "
                           f"minimum {params['min_wallet_score']:.2f}")
            return True
            
        return False