"""
Trading engine for executing buy/sell orders and managing positions
"""

import asyncio
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from src.clients.pumpfun_client import PumpFunClient
from src.clients.moralis_client import MoralisClient
from src.utils.discord_notifier import DiscordNotifier
from src.utils.pnl_store import PnLStore


@dataclass
class Position:
    mint: str
    entry_price: float
    amount: float
    sol_invested: float
    entry_time: datetime
    tp_price: float
    sl_price: float
    peak_price: float
    paper_mode: bool


class TradingEngine:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.pumpfun = PumpFunClient(config.quicknode_endpoint, config.quicknode_api_key)
        self.moralis = MoralisClient(config.moralis_key)
        
        # Initialize Discord notifier
        webhook_url = ""
        if hasattr(config, 'notifications'):
            webhook_url = config.notifications.get('discord_webhook_url', '')
            self.logger.info(f"DISCORD_DEBUG: Found notifications section, webhook_url length: {len(webhook_url) if webhook_url else 0}")
        elif hasattr(config, 'discord_webhook_url'):
            webhook_url = config.discord_webhook_url
            self.logger.info(f"DISCORD_DEBUG: Found root discord_webhook_url, length: {len(webhook_url) if webhook_url else 0}")
        else:
            self.logger.warning("DISCORD_DEBUG: No Discord webhook URL found in config")
        
        self.notifier = DiscordNotifier(webhook_url) if webhook_url else None
        self.logger.info(f"DISCORD_DEBUG: Notifier initialized: {self.notifier is not None}, enabled: {self.notifier.enabled if self.notifier else False}")
        
        # Initialize P&L store
        self.pnl_store = PnLStore(
            path="data/pnl_state.json",
            initial_capital=config.initial_capital
        )
        
        # Position tracking
        self.active_positions = {}
        self.paper_capital = self.pnl_store.current_equity  # Use P&L store's equity
        self.total_trades = 0
        self.winning_trades = 0

    async def buy_token(self, mint_address: str, usd_amount: float, paper_mode: bool = True, symbol: str = "UNKNOWN", confidence_score: float = None) -> Dict:
        """Execute a buy order for a token"""
        try:
            self.logger.info(f"Executing BUY for {mint_address}, amount: ${usd_amount}")
            
            if paper_mode:
                return await self._execute_paper_buy(mint_address, usd_amount, symbol, confidence_score)
            else:
                return await self._execute_real_buy(mint_address, usd_amount)
                
        except Exception as e:
            self.logger.error(f"Error executing buy: {e}")
            
            # Send error notification
            if self.notifier:
                await self.notifier.send_error_notification(
                    f"Failed to execute buy order: {str(e)}",
                    {"token": mint_address[:8], "amount": f"${usd_amount}"}
                )
            
            return {"success": False, "error": str(e)}

    async def sell_token(self, mint_address: str, percentage: float, paper_mode: bool = True, symbol: str = "UNKNOWN") -> Dict:
        """Execute a sell order for a token"""
        try:
            self.logger.info(f"Executing SELL for {mint_address}, percentage: {percentage}")
            
            if paper_mode:
                return await self._execute_paper_sell(mint_address, percentage, symbol)
            else:
                return await self._execute_real_sell(mint_address, percentage)
                
        except Exception as e:
            self.logger.error(f"Error executing sell: {e}")
            
            # Send error notification
            if self.notifier:
                await self.notifier.send_error_notification(
                    f"Failed to execute sell order: {str(e)}",
                    {"token": mint_address[:8], "percentage": f"{percentage*100}%"}
                )
            
            return {"success": False, "error": str(e)}

    async def _execute_paper_buy(self, mint_address: str, usd_amount: float, symbol: str = "UNKNOWN", confidence_score: float = None) -> Dict:
        """Execute a paper trading buy"""
        # Get current price
        price = await self.moralis.get_current_price(mint_address)
        if price <= 0:
            return {"success": False, "error": "Could not get price"}
        
        # Calculate SOL amount (assuming SOL = $20 for paper trading)
        sol_price = 20.0  # Mock SOL price
        sol_amount = usd_amount / sol_price
        
        # Calculate tokens received
        tokens_received = usd_amount / price
        
        # Update paper capital
        if self.paper_capital < usd_amount:
            return {"success": False, "error": "Insufficient paper capital"}
        
        self.paper_capital -= usd_amount
        
        # Create position
        position = Position(
            mint=mint_address,
            entry_price=price,
            amount=tokens_received,
            sol_invested=sol_amount,
            entry_time=datetime.now(),
            tp_price=price * self.config.tp_multiplier,
            sl_price=price * self.config.stop_loss_pct,
            peak_price=price,
            paper_mode=True
        )
        
        self.active_positions[mint_address] = position
        
        # Record in P&L store
        self.pnl_store.add_trade(
            action="BUY",
            symbol=symbol,
            mint_address=mint_address,
            amount=tokens_received,
            price=price,
            usd_value=usd_amount,
            paper_mode=True
        )
        
        # Send Discord notification
        if self.notifier:
            await self.notifier.send_trade_notification(
                side="BUY",
                symbol=symbol,
                mint_address=mint_address,
                quantity=tokens_received,
                price=price,
                usd_amount=usd_amount,
                equity=self.pnl_store.current_equity,
                confidence_score=confidence_score,
                paper_mode=True
            )
        
        self.logger.info(f"Paper buy executed: {tokens_received} tokens at ${price}")
        
        return {
            "success": True,
            "price": price,
            "tokens_received": tokens_received,
            "sol_amount": sol_amount,
            "usd_amount": usd_amount,
            "symbol": symbol,
            "paper_mode": True
        }

    async def _execute_real_buy(self, mint_address: str, usd_amount: float) -> Dict:
        """Execute a real buy transaction"""
        # This would require actual wallet integration
        # For now, implement paper trading logic
        self.logger.warning("Real trading not implemented, falling back to paper mode")
        return await self._execute_paper_buy(mint_address, usd_amount)

    async def _execute_paper_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN") -> Dict:
        """Execute a paper trading sell"""
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found"}
        
        position = self.active_positions[mint_address]
        
        # Get current price
        current_price = await self.moralis.get_current_price(mint_address)
        if current_price <= 0:
            return {"success": False, "error": "Could not get current price"}
        
        # Calculate sell amount
        tokens_to_sell = position.amount * percentage
        usd_received = tokens_to_sell * current_price
        
        # Update position
        position.amount -= tokens_to_sell
        self.paper_capital += usd_received
        
        # Calculate profit/loss
        cost_basis = (position.sol_invested * percentage) * 20  # Mock SOL price
        profit = usd_received - cost_basis
        profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0
        
        # Record in P&L store
        self.pnl_store.add_trade(
            action="SELL",
            symbol=symbol,
            mint_address=mint_address,
            amount=tokens_to_sell,
            price=current_price,
            usd_value=usd_received,
            realized_pnl=profit,
            paper_mode=True
        )
        
        # Send Discord notification
        if self.notifier:
            await self.notifier.send_trade_notification(
                side="SELL",
                symbol=symbol,
                mint_address=mint_address,
                quantity=tokens_to_sell,
                price=current_price,
                usd_amount=usd_received,
                equity=self.pnl_store.current_equity,
                realized_pnl=profit,
                paper_mode=True
            )
        
        self.logger.info(f"Paper sell executed: {tokens_to_sell} tokens at ${current_price}, profit: ${profit:.2f} ({profit_pct:.2f}%)")
        
        # Remove position if fully sold
        if position.amount <= 0.001:  # Small threshold for floating point precision
            del self.active_positions[mint_address]
            
            # Update stats
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
        
        return {
            "success": True,
            "tokens_sold": tokens_to_sell,
            "price": current_price,
            "usd_amount": usd_received,
            "usd_received": usd_received,
            "sol_amount": usd_received / 20,  # Mock SOL price
            "profit": profit,
            "profit_pct": profit_pct,
            "symbol": symbol,
            "paper_mode": True
        }

    async def _execute_real_sell(self, mint_address: str, percentage: float) -> Dict:
        """Execute a real sell transaction"""
        # This would require actual wallet integration
        self.logger.warning("Real trading not implemented, falling back to paper mode")
        return await self._execute_paper_sell(mint_address, percentage)

    async def check_exit_conditions(self, mint_address: str) -> Optional[str]:
        """Check if position should be closed"""
        if mint_address not in self.active_positions:
            return None
        
        position = self.active_positions[mint_address]
        
        # Get current price
        current_price = await self.moralis.get_current_price(mint_address)
        if current_price <= 0:
            return None
        
        # Update peak price
        if current_price > position.peak_price:
            position.peak_price = current_price
        
        # Check take profit
        if current_price >= position.tp_price:
            return "take_profit"
        
        # Check stop loss
        if current_price <= position.sl_price:
            return "stop_loss"
        
        # Check trailing stop (20% from peak)
        trailing_stop = position.peak_price * 0.8
        if current_price <= trailing_stop:
            return "trailing_stop"
        
        # Check time-based exit (4 hours max hold)
        hold_time = datetime.now() - position.entry_time
        if hold_time.total_seconds() > 14400:  # 4 hours
            return "time_limit"
        
        return None

    async def update_position_prices(self):
        """Update prices for all active positions"""
        for mint_address in list(self.active_positions.keys()):
            try:
                current_price = await self.moralis.get_current_price(mint_address)
                position = self.active_positions[mint_address]
                
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    
                # Check exit conditions
                exit_reason = await self.check_exit_conditions(mint_address)
                if exit_reason:
                    self.logger.info(f"Exit condition triggered for {mint_address}: {exit_reason}")
                    
                    # Determine sell percentage
                    if exit_reason == "take_profit":
                        sell_pct = 0.5  # Sell half on take profit
                    else:
                        sell_pct = 1.0  # Sell all on stop loss or other conditions
                    
                    await self.sell_token(mint_address, sell_pct, position.paper_mode)
                    
            except Exception as e:
                self.logger.error(f"Error updating position {mint_address}: {e}")

    def get_portfolio_summary(self) -> Dict:
        """Get summary of current portfolio"""
        total_invested = 0
        total_current_value = 0
        position_count = len(self.active_positions)
        
        for position in self.active_positions.values():
            invested = position.sol_invested * 20  # Mock SOL price
            total_invested += invested
            
            # Would need current price to calculate current value
            # For now, use entry price
            current_value = position.amount * position.entry_price
            total_current_value += current_value
        
        unrealized_pnl = total_current_value - total_invested
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "paper_capital": self.paper_capital,
            "total_invested": total_invested,
            "current_value": total_current_value,
            "unrealized_pnl": unrealized_pnl,
            "position_count": position_count,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate
        }

    def get_position_details(self, mint_address: str) -> Optional[Dict]:
        """Get details for a specific position"""
        if mint_address not in self.active_positions:
            return None
        
        position = self.active_positions[mint_address]
        
        return {
            "mint": position.mint,
            "entry_price": position.entry_price,
            "amount": position.amount,
            "sol_invested": position.sol_invested,
            "entry_time": position.entry_time.isoformat(),
            "tp_price": position.tp_price,
            "sl_price": position.sl_price,
            "peak_price": position.peak_price,
            "paper_mode": position.paper_mode
        }

    async def close_all_positions(self):
        """Close all active positions"""
        for mint_address in list(self.active_positions.keys()):
            try:
                position = self.active_positions[mint_address]
                await self.sell_token(mint_address, 1.0, position.paper_mode)
                self.logger.info(f"Closed position {mint_address}")
            except Exception as e:
                self.logger.error(f"Error closing position {mint_address}: {e}")

    async def send_error_notification(self, message: str, context: Dict = None):
        """Send error notification to Discord"""
        if self.notifier:
            await self.notifier.send_error_notification(message, context)
    
    async def send_summary(self):
        """Send portfolio summary to Discord"""
        if self.notifier:
            summary = self.pnl_store.get_summary()
            await self.notifier.send_summary(
                equity=summary["equity"],
                daily_pnl=summary["daily_pnl"],
                total_trades=summary["total_trades"],
                win_rate=summary["win_rate"],
                active_positions=len(self.active_positions)
            )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.notifier:
            await self.notifier.close()
        await self.pumpfun.close()
        await self.moralis.close()