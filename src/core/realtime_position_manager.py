"""
Real-time Position Manager - Updates positions immediately from WebSocket events
Replaces slow blockchain polling with instant WebSocket updates
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Callable
import logging
from dataclasses import dataclass

@dataclass
class RealtimePosition:
    """Position updated in real-time from WebSocket events"""
    mint: str
    entry_time: datetime
    entry_price: float
    sol_invested: float
    buy_tx_signature: str
    
    # Updated from WebSocket events
    current_tokens: float = 0.0
    sol_recovered: float = 0.0
    last_update: datetime = None
    
    @property
    def unrealized_pnl_sol(self) -> float:
        """P&L in SOL terms"""
        return self.sol_recovered - self.sol_invested
    
    @property
    def is_active(self) -> bool:
        """Position has tokens remaining"""
        return self.current_tokens > 0

class RealtimePositionManager:
    """
    Manages positions with real-time updates from WebSocket events.
    No polling, no delays - positions update the moment trades execute.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.positions: Dict[str, RealtimePosition] = {}
        
        # Callbacks for position events
        self.on_position_opened: Optional[Callable] = None
        self.on_position_updated: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        
    def handle_trade_event(self, trade_event: Dict) -> None:
        """
        Handle incoming trade events from WebSocket.
        Updates positions immediately - no blockchain polling.
        """
        try:
            mint = trade_event.get('mint')
            action = trade_event.get('action', '').lower()
            tx_signature = trade_event.get('tx_signature')
            
            if not mint or not action:
                return
                
            if action == 'buy':
                self._handle_buy_event(trade_event)
            elif action == 'sell':
                self._handle_sell_event(trade_event)
                
        except Exception as e:
            self.logger.error(f"Error handling trade event: {e}")
    
    def _handle_buy_event(self, event: Dict) -> None:
        """Handle buy trade event - create new position"""
        mint = event['mint']
        tx_signature = event.get('tx_signature', f"ws_{mint}_{int(datetime.now().timestamp())}")
        
        # Create position immediately based on WebSocket event
        position = RealtimePosition(
            mint=mint,
            entry_time=datetime.now(),
            entry_price=event.get('price', 0.0),
            sol_invested=event.get('sol_amount', 0.0),
            buy_tx_signature=tx_signature,
            current_tokens=event.get('tokens_received', 0.0),
            last_update=datetime.now()
        )
        
        self.positions[mint] = position
        
        self.logger.info(f"🚀 Position opened: {mint[:8]}... {position.current_tokens:,.0f} tokens")
        
        if self.on_position_opened:
            self.on_position_opened(position)
    
    def _handle_sell_event(self, event: Dict) -> None:
        """Handle sell trade event - update existing position"""
        mint = event['mint']
        
        if mint not in self.positions:
            self.logger.warning(f"Sell event for unknown position: {mint[:8]}...")
            return
            
        position = self.positions[mint]
        
        # Update position from sell event
        tokens_sold = event.get('tokens_sold', 0.0)
        sol_received = event.get('sol_received', 0.0)
        
        position.current_tokens = max(0, position.current_tokens - tokens_sold)
        position.sol_recovered += sol_received
        position.last_update = datetime.now()
        
        self.logger.info(f"💰 Position updated: {mint[:8]}... sold {tokens_sold:,.0f} tokens, +{sol_received:.4f} SOL")
        
        if self.on_position_updated:
            self.on_position_updated(position)
            
        # Close position if no tokens left
        if position.current_tokens <= 1:  # Account for rounding
            self._close_position(mint)
    
    def _close_position(self, mint: str) -> None:
        """Close position and calculate final P&L"""
        if mint not in self.positions:
            return
            
        position = self.positions[mint]
        final_pnl = position.unrealized_pnl_sol
        
        self.logger.info(f"🏁 Position closed: {mint[:8]}... Final P&L: {final_pnl:+.4f} SOL")
        
        if self.on_position_closed:
            self.on_position_closed(position)
            
        del self.positions[mint]
    
    def get_position(self, mint: str) -> Optional[RealtimePosition]:
        """Get position for mint"""
        return self.positions.get(mint)
    
    def get_active_positions(self) -> Dict[str, RealtimePosition]:
        """Get all positions with tokens remaining"""
        return {mint: pos for mint, pos in self.positions.items() if pos.is_active}
    
    def can_sell(self, mint: str, percentage: float = 1.0) -> tuple[bool, float]:
        """
        Fast sell check - no blockchain queries needed.
        Returns: (can_sell, tokens_to_sell)
        """
        position = self.positions.get(mint)
        
        if not position or not position.is_active:
            return False, 0.0
            
        tokens_to_sell = position.current_tokens * percentage
        
        if tokens_to_sell < 1:  # Minimum sell threshold
            return False, 0.0
            
        return True, tokens_to_sell
    
    def get_performance_summary(self) -> Dict:
        """Get current performance metrics"""
        active_positions = self.get_active_positions()
        
        total_sol_invested = sum(pos.sol_invested for pos in active_positions.values())
        total_sol_recovered = sum(pos.sol_recovered for pos in active_positions.values())
        unrealized_pnl = total_sol_recovered - total_sol_invested
        
        return {
            'active_positions': len(active_positions),
            'total_sol_invested': total_sol_invested,
            'total_sol_recovered': total_sol_recovered,
            'unrealized_pnl_sol': unrealized_pnl,
            'positions': list(active_positions.keys())
        }