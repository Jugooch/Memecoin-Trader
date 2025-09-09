"""
Dynamic exit manager with state machine for small positions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExitReason(Enum):
    """Exit trigger reasons"""
    LOSS_CAP = "loss_cap"
    WHALE_SINGLE = "whale_single_dump"
    WHALE_MULTI = "whale_multi_dump"
    TRAILING_STOP = "trailing_stop"
    LIQUIDITY_CLIFF = "liquidity_cliff"
    TIME_DECAY = "time_decay"
    PARTIAL_3X = "partial_3x"
    PARTIAL_6X = "partial_6x"
    RUG_DETECTED = "rug_detected"
    MANUAL = "manual"


class PositionState:
    """Track position state for exit management"""
    
    def __init__(self, position_id: str, event: Dict, entry_sol: float):
        self.position_id = position_id
        self.token_mint = event['token_mint']
        self.ticker = event.get('ticker', 'UNKNOWN')
        self.entry_sol = entry_sol
        self.entry_time = datetime.now()
        
        # Price tracking
        self.entry_price = 0.000001  # Will be updated
        self.current_price = self.entry_price
        self.peak_price = self.entry_price
        self.last_price_update = datetime.now()
        
        # Partial exit tracking
        self.partials_executed = {}  # multiplier -> bool
        self.remaining_pct = 100.0
        
        # Market data
        self.volume_10m = 0
        self.liquidity_depth = 0
        self.spread_bps = 0
        self.top_holders = []  # List of (wallet, pct) tuples
        
        # State flags
        self.is_active = True
        self.exit_triggered = False
        self.exit_reason = None


class ExitManager:
    """Manages exit lifecycle for all positions"""
    
    def __init__(self, config: Dict, store, metrics):
        self.logger = logging.getLogger(__name__)
        self.config = config['exit']
        self.store = store
        self.metrics = metrics
        
        # Exit parameters
        self.loss_cap_pct = self.config['loss_cap_pct']
        self.trailing_floor_min = self.config['trailing_floor_pct_min']
        self.trailing_floor_max = self.config['trailing_floor_pct_max']
        
        # Whale detection
        self.whale_threshold = self.config['whale_sell_threshold_pct_supply']
        self.whale_single_pct = self.config['whale_single_sale_pct']
        self.whale_multi_pct = self.config['whale_multi_sale_pct']
        self.whale_multi_window = self.config['whale_multi_window_sec']
        self.whale_top_n = self.config['whale_multi_topN']
        
        # Liquidity cliff
        self.vol_drop_pct = self.config['liq_cliff']['vol_drop_pct']
        self.depth_drop_pct = self.config['liq_cliff']['depth_drop_pct']
        self.spread_threshold = self.config['liq_cliff']['spread_bps']
        self.liq_window = self.config['liq_cliff']['window_min']
        
        # Time decay
        self.time_decay_min = self.config['time_decay_min_without_new_ath']
        
        # Partials
        self.partials_enabled = self.config['partials']['enabled']
        self.partial_steps = self.config['partials']['steps']
        
        # Active positions
        self.positions = {}  # position_id -> PositionState
        
        # Executor reference (will be set by orchestrator)
        self.executor = None
        
        # Market data providers (will be initialized)
        self.price_feed = None
        self.holder_tracker = None
    
    def register_position(self, event: Dict, position_id: str, entry_sol: float):
        """Register new position for exit management"""
        state = PositionState(position_id, event, entry_sol)
        self.positions[position_id] = state
        
        self.logger.info(f"Registered position {position_id} for {state.ticker}")
        self.metrics.inc("exits.positions_registered")
        
        # Start monitoring
        asyncio.create_task(self._monitor_position(state))
    
    async def _monitor_position(self, state: PositionState):
        """Monitor position and trigger exits"""
        self.logger.info(f"Starting exit monitor for {state.ticker}")
        
        while state.is_active:
            try:
                # Update market data
                await self._update_market_data(state)
                
                # Check exit conditions in priority order
                exit_signal = await self._check_exit_conditions(state)
                
                if exit_signal:
                    reason, sell_pct = exit_signal
                    await self._trigger_exit(state, reason, sell_pct)
                
                # Sleep before next check
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring {state.ticker}: {e}")
                await asyncio.sleep(5)
        
        self.logger.info(f"Exit monitor stopped for {state.ticker}")
    
    async def _update_market_data(self, state: PositionState):
        """Update market data for position"""
        try:
            # Get current price (mock for now - integrate with price feed)
            # In production, this would query Moralis/Birdeye/etc
            if self.price_feed:
                price_data = await self.price_feed.get_price(state.token_mint)
                state.current_price = price_data.get('price', state.current_price)
                
                # Update peak
                if state.current_price > state.peak_price:
                    state.peak_price = state.current_price
                    state.last_price_update = datetime.now()
            
            # Get holder data (mock for now)
            if self.holder_tracker:
                holders = await self.holder_tracker.get_top_holders(state.token_mint)
                state.top_holders = holders[:self.whale_top_n]
            
            # Get liquidity metrics (mock for now)
            # state.volume_10m = ...
            # state.liquidity_depth = ...
            # state.spread_bps = ...
            
        except Exception as e:
            self.logger.debug(f"Market data update error for {state.ticker}: {e}")
    
    async def _check_exit_conditions(self, state: PositionState) -> Optional[Tuple[ExitReason, float]]:
        """
        Check all exit conditions in priority order
        Returns (reason, sell_pct) if exit triggered
        """
        
        # 1. RUG DETECTION - Highest priority
        if await self._check_rug(state):
            return ExitReason.RUG_DETECTED, 100
        
        # 2. LOSS CAP - Stop loss
        if self._check_loss_cap(state):
            return ExitReason.LOSS_CAP, 100
        
        # 3. WHALE DUMPS
        whale_signal = await self._check_whale_activity(state)
        if whale_signal:
            return whale_signal
        
        # 4. PARTIAL PROFITS
        if self.partials_enabled:
            partial_signal = self._check_partials(state)
            if partial_signal:
                return partial_signal
        
        # 5. TRAILING STOP
        if self._check_trailing_stop(state):
            return ExitReason.TRAILING_STOP, state.remaining_pct
        
        # 6. LIQUIDITY CLIFF
        if self._check_liquidity_cliff(state):
            return ExitReason.LIQUIDITY_CLIFF, min(50, state.remaining_pct)
        
        # 7. TIME DECAY
        if self._check_time_decay(state):
            return ExitReason.TIME_DECAY, state.remaining_pct
        
        return None
    
    async def _check_rug(self, state: PositionState) -> bool:
        """Check for rug conditions"""
        # This would integrate with safety filters
        # Check LP removal, dev dump, honeypot, etc.
        return False  # Placeholder
    
    def _check_loss_cap(self, state: PositionState) -> bool:
        """Check if position hit loss cap"""
        if state.current_price <= 0:
            return False
        
        loss_pct = ((state.entry_price - state.current_price) / state.entry_price) * 100
        
        if loss_pct >= self.loss_cap_pct:
            self.logger.warning(f"{state.ticker} hit loss cap: -{loss_pct:.1f}%")
            return True
        
        return False
    
    async def _check_whale_activity(self, state: PositionState) -> Optional[Tuple[ExitReason, float]]:
        """Check for whale selling activity"""
        if not state.top_holders:
            return None
        
        # Track whale sales (would need historical comparison)
        # This is simplified - in production, track holder changes
        
        # Single whale dump check
        for wallet, holding_pct in state.top_holders:
            if holding_pct >= self.whale_threshold:
                # Check if this whale sold (needs historical data)
                # If sold > whale_single_pct of their holdings:
                # return ExitReason.WHALE_SINGLE, 100
                pass
        
        # Multi-whale dump check
        # Count whales that sold > whale_multi_pct in window
        # If count >= 2:
        #     return ExitReason.WHALE_MULTI, 100
        
        return None
    
    def _check_partials(self, state: PositionState) -> Optional[Tuple[ExitReason, float]]:
        """Check for partial profit taking triggers"""
        if state.current_price <= 0 or state.entry_price <= 0:
            return None
        
        multiplier = state.current_price / state.entry_price
        
        for step in self.partial_steps:
            trigger_mult = step['trigger_mult']
            sell_pct = step['sell_pct']
            
            # Check if this partial was already executed
            if trigger_mult in state.partials_executed:
                continue
            
            if multiplier >= trigger_mult:
                state.partials_executed[trigger_mult] = True
                actual_sell_pct = min(sell_pct, state.remaining_pct)
                state.remaining_pct -= actual_sell_pct
                
                self.logger.info(f"{state.ticker} hit {trigger_mult}x, selling {actual_sell_pct}%")
                
                if trigger_mult == 3.0:
                    return ExitReason.PARTIAL_3X, actual_sell_pct
                elif trigger_mult == 6.0:
                    return ExitReason.PARTIAL_6X, actual_sell_pct
        
        return None
    
    def _check_trailing_stop(self, state: PositionState) -> bool:
        """Check adaptive trailing stop"""
        if state.peak_price <= 0 or state.current_price <= 0:
            return False
        
        # Calculate adaptive trailing percentage based on volatility
        # Simplified: use fixed percentage for now
        trailing_pct = self.trailing_floor_min  # Could be dynamic based on ATR
        
        drawdown_pct = ((state.peak_price - state.current_price) / state.peak_price) * 100
        
        if drawdown_pct >= trailing_pct:
            self.logger.info(f"{state.ticker} trailing stop triggered: "
                           f"-{drawdown_pct:.1f}% from peak")
            return True
        
        return False
    
    def _check_liquidity_cliff(self, state: PositionState) -> bool:
        """Check for liquidity deterioration"""
        # Check volume drop
        # if state.volume_10m < (initial_volume * (1 - self.vol_drop_pct/100)):
        #     return True
        
        # Check depth drop
        # if state.liquidity_depth < (initial_depth * (1 - self.depth_drop_pct/100)):
        #     return True
        
        # Check spread widening
        if state.spread_bps > self.spread_threshold:
            self.logger.warning(f"{state.ticker} spread too wide: {state.spread_bps} bps")
            return True
        
        return False
    
    def _check_time_decay(self, state: PositionState) -> bool:
        """Check for time decay without new ATH"""
        time_since_ath = datetime.now() - state.last_price_update
        
        if time_since_ath.seconds >= (self.time_decay_min * 60):
            # Additional check: EMA crossover or other momentum indicator
            self.logger.info(f"{state.ticker} time decay: No new ATH for {self.time_decay_min} min")
            return True
        
        return False
    
    async def _trigger_exit(self, state: PositionState, reason: ExitReason, sell_pct: float):
        """Trigger exit execution"""
        if state.exit_triggered:
            return
        
        state.exit_triggered = True
        state.exit_reason = reason
        
        self.logger.info(f"EXIT TRIGGERED: {state.ticker} - {reason.value} - {sell_pct}%")
        self.metrics.inc(f"exits.triggered.{reason.value}")
        
        # Execute exit via executor
        if self.executor:
            success = await self.executor.execute_exit(
                state.position_id,
                sell_pct,
                reason.value
            )
            
            if success:
                self.metrics.inc("exits.executed")
                
                # Mark position inactive if fully exited
                if sell_pct >= state.remaining_pct:
                    state.is_active = False
                    del self.positions[state.position_id]
                else:
                    # Reset for continued monitoring
                    state.exit_triggered = False
                    state.remaining_pct -= sell_pct
            else:
                # Reset to retry
                state.exit_triggered = False
                self.metrics.inc("exits.failed")
    
    def set_executor(self, executor):
        """Set executor reference"""
        self.executor = executor
    
    def get_active_positions(self) -> List[Dict]:
        """Get list of active positions"""
        return [
            {
                'position_id': state.position_id,
                'ticker': state.ticker,
                'entry_sol': state.entry_sol,
                'current_multiple': state.current_price / state.entry_price if state.entry_price > 0 else 0,
                'remaining_pct': state.remaining_pct,
                'time_held': (datetime.now() - state.entry_time).seconds
            }
            for state in self.positions.values()
        ]
    
    def get_stats(self) -> Dict:
        """Get exit manager statistics"""
        return {
            'active_positions': len(self.positions),
            'partials_enabled': self.partials_enabled,
            'loss_cap_pct': self.loss_cap_pct,
            'trailing_range': f"{self.trailing_floor_min}-{self.trailing_floor_max}%"
        }