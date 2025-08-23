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
    tokens_initial: float
    cost_usd_remaining: float
    avg_cost_per_token: float
    tp1_hit_time: Optional[datetime] = None  # Track when TP1 was hit
    tp1_percentage_sold: float = 0  # Track how much was sold at TP1
    break_even_armed: bool = False  # Track if break-even stop is armed
    break_even_armed_time: Optional[datetime] = None  # When break-even was armed
    trailing_stop_active: bool = False  # Track if trailing stop is active
    high_gain_peak: float = 0  # Track highest gain percentage achieved


class TradingEngine:
    def __init__(self, config, moralis_client: MoralisClient):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clients
        self.pumpfun = PumpFunClient(config.quicknode_endpoint, config.quicknode_api_key)
        self.moralis = moralis_client  # Use shared client instead of creating new one
        
        # Initialize Discord notifier
        webhook_url = None
        if hasattr(config, 'notifications') and config.notifications:
            webhook_url = config.notifications.get('discord_webhook_url', None)
            self.logger.info(f"DISCORD_DEBUG: Found notifications section, webhook_url: {webhook_url[:50] if webhook_url else 'None'}")
        elif hasattr(config, 'discord_webhook_url'):
            webhook_url = config.discord_webhook_url
            self.logger.info(f"DISCORD_DEBUG: Found root discord_webhook_url: {webhook_url[:50] if webhook_url else 'None'}")
        else:
            self.logger.warning("DISCORD_DEBUG: No Discord webhook URL found in config")
            self.logger.info(f"DISCORD_DEBUG: Config notifications attr exists: {hasattr(config, 'notifications')}, value: {getattr(config, 'notifications', 'NONE')}")
        
        # Check if webhook URL is actually set and not empty
        if webhook_url and webhook_url.strip():
            self.notifier = DiscordNotifier(webhook_url.strip())
            self.logger.info(f"DISCORD_DEBUG: Notifier initialized successfully, enabled: {self.notifier.enabled}")
        else:
            self.notifier = None
            self.logger.warning(f"DISCORD_DEBUG: Notifier not initialized - webhook_url is empty or None: '{webhook_url}'")
        
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

    async def sell_token(self, mint_address: str, percentage: float, paper_mode: bool = True, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a sell order for a token"""
        try:
            # ENHANCED: Comprehensive exit logging with position details
            position_info = ""
            if mint_address in self.active_positions:
                pos = self.active_positions[mint_address]
                hold_time = (datetime.now() - pos.entry_time).total_seconds()
                current_price = await self.moralis.get_current_price(mint_address, fresh=True) or pos.entry_price
                pnl_pct = ((current_price / pos.entry_price) - 1) * 100 if pos.entry_price > 0 else 0
                position_info = f"hold_sec={hold_time:.0f} pnl={pnl_pct:+.1f}% entry=${pos.entry_price:.8f} current=${current_price:.8f}"
            
            self.logger.info(f"Exit: mint={mint_address[:8]}... reason={exit_reason} percentage={percentage*100:.0f}% {position_info}")
            
            if paper_mode:
                return await self._execute_paper_sell(mint_address, percentage, symbol, exit_reason)
            else:
                return await self._execute_real_sell(mint_address, percentage, exit_reason)
                
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
        
        # Calculate SOL amount using current estimate
        sol_price = getattr(self.config, "paper_trading", {}).get("sol_price_estimate", 140)
        sol_amount = usd_amount / sol_price
        
        # Apply realistic fees for alpha-following strategy (not sniping)
        paper_config = getattr(self.config, "paper_trading", {})
        
        # DEX trading fee (pump.fun ~0.30%)
        dex_fee_bps = paper_config.get("dex_fee_bps", 30)
        
        # Network fees (less aggressive since we're following, not sniping)
        sol_price_usd = paper_config.get("sol_price_estimate", 140)  # Current SOL price estimate
        base_tx_fee_sol = paper_config.get("base_tx_fee_sol", 0.000005)  # Base Solana tx fee
        priority_fee_sol = paper_config.get("priority_fee_sol", 0.002)    # Medium priority (not high)
        network_fee_usd = (base_tx_fee_sol + priority_fee_sol) * sol_price_usd
        
        # Slippage (lower since we're not competing with MEV bots)
        buy_slip_bps = paper_config.get("buy_slippage_bps", 50)  # Reduced from 75
        max_slippage_bps = paper_config.get("max_slippage_bps", 150)
        
        # Slippage rejection gate
        if buy_slip_bps > max_slippage_bps:
            self.logger.warning(f"Trade rejected: estimated slippage {buy_slip_bps}bps > max {max_slippage_bps}bps")
            return {"success": False, "error": f"Slippage too high: {buy_slip_bps/100:.1f}%"}
        
        # Calculate fill price with slippage
        fill_price = price * (1 + buy_slip_bps/10000)
        
        # Calculate total fees: DEX fee + network fee
        dex_fee_usd = usd_amount * (dex_fee_bps/10000)
        total_fee_usd = dex_fee_usd + network_fee_usd
        
        # Calculate tokens received after all fees
        net_usd_amount = usd_amount - total_fee_usd
        if net_usd_amount <= 0:
            return {"success": False, "error": f"Trade too small: fees (${total_fee_usd:.2f}) exceed trade size"}
        
        tokens_received = net_usd_amount / fill_price
        
        # Log fee breakdown for transparency
        self.logger.debug(f"Buy fees: DEX ${dex_fee_usd:.3f} + Network ${network_fee_usd:.3f} = ${total_fee_usd:.3f} (on ${usd_amount})")
        
        # Update paper capital
        if self.paper_capital < usd_amount:
            return {"success": False, "error": "Insufficient paper capital"}
        
        self.paper_capital -= usd_amount
        
        # Create position with proper cost tracking (use fill_price as entry)
        position = Position(
            mint=mint_address,
            entry_price=fill_price,
            amount=tokens_received,
            sol_invested=sol_amount,
            entry_time=datetime.now(),
            tp_price=fill_price * self.config.tp_multiplier,
            sl_price=fill_price * self.config.stop_loss_pct,
            peak_price=fill_price,
            paper_mode=True,
            tokens_initial=tokens_received,
            cost_usd_remaining=usd_amount,
            avg_cost_per_token=usd_amount / tokens_received
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

    async def _execute_paper_sell(self, mint_address: str, percentage: float, symbol: str = "UNKNOWN", exit_reason: str = "unknown") -> Dict:
        """Execute a paper trading sell"""
        if mint_address not in self.active_positions:
            return {"success": False, "error": "No position found"}
        
        position = self.active_positions[mint_address]
        
        # Get current price
        current_price = await self.moralis.get_current_price(mint_address, fresh=True)
        if current_price <= 0:
            return {"success": False, "error": "Could not get current price"}
        
        # Calculate sell amount (percentage of current amount)
        tokens_to_sell = position.amount * percentage
        if tokens_to_sell <= 0:
            return {"success": False, "error": "Nothing to sell"}
        
        # Apply realistic fees for alpha-following strategy
        paper_config = getattr(self.config, "paper_trading", {})
        
        # DEX trading fee
        dex_fee_bps = paper_config.get("dex_fee_bps", 30)
        
        # Network fees (same as buy)
        sol_price_usd = paper_config.get("sol_price_estimate", 140)
        base_tx_fee_sol = paper_config.get("base_tx_fee_sol", 0.000005)
        priority_fee_sol = paper_config.get("priority_fee_sol", 0.002)
        network_fee_usd = (base_tx_fee_sol + priority_fee_sol) * sol_price_usd
        
        # Slippage (slightly higher on sells due to market impact)
        sell_slip_bps = paper_config.get("sell_slippage_bps", 75)  # Reduced from 100
        
        # Calculate fill price with slippage
        fill_price = current_price * (1 - sell_slip_bps/10000)
        
        # Calculate USD received after all fees
        cost_basis_usd = tokens_to_sell * position.avg_cost_per_token
        gross_usd = tokens_to_sell * fill_price
        dex_fee_usd = gross_usd * (dex_fee_bps/10000)
        total_fee_usd = dex_fee_usd + network_fee_usd
        usd_received = gross_usd - total_fee_usd
        
        # Log fee breakdown
        self.logger.debug(f"Sell fees: DEX ${dex_fee_usd:.3f} + Network ${network_fee_usd:.3f} = ${total_fee_usd:.3f} (on ${gross_usd:.2f})")
        
        # Update position with proper accounting
        position.amount -= tokens_to_sell
        position.cost_usd_remaining -= cost_basis_usd
        if position.amount > 0:
            position.avg_cost_per_token = position.cost_usd_remaining / position.amount
        else:
            position.avg_cost_per_token = 0.0
        
        self.paper_capital += usd_received
        
        # Calculate profit/loss with correct cost basis
        profit = usd_received - cost_basis_usd
        profit_pct = (profit / cost_basis_usd) * 100 if cost_basis_usd > 0 else 0
        
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
        if position.amount <= 1e-9:  # Small threshold for floating point precision
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

    async def _execute_real_sell(self, mint_address: str, percentage: float, exit_reason: str = "unknown") -> Dict:
        """Execute a real sell transaction"""
        # This would require actual wallet integration
        self.logger.warning("Real trading not implemented, falling back to paper mode")
        return await self._execute_paper_sell(mint_address, percentage, exit_reason=exit_reason)

    async def check_exit_conditions(self, mint_address: str, current_price: float = 0) -> Optional[tuple]:
        """
        Enhanced exit strategy with dynamic TP1 sizing and intelligent trailing stops
        Returns: (exit_reason, sell_percentage) or None
        """
        if mint_address not in self.active_positions:
            return None
        
        position = self.active_positions[mint_address]
        
        # Get current price
        if current_price <= 0:
            return None
        
        # Calculate current gain/loss percentage
        current_gain_pct = ((current_price / position.entry_price) - 1) * 100
        hold_time_seconds = (datetime.now() - position.entry_time).total_seconds()
        
        # Enhanced logging for exit decision monitoring
        self.logger.debug(f"EXIT CHECK {mint_address[:8]}... "
                         f"Price: ${current_price:.8f}, "
                         f"Gain: {current_gain_pct:+.1f}%, "
                         f"Hold: {hold_time_seconds:.0f}s, "
                         f"Peak: {position.high_gain_peak:.1f}%, "
                         f"TP1Hit: {position.tp1_hit_time is not None}")
        
        # Update peak price and high gain
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.high_gain_peak = max(position.high_gain_peak, current_gain_pct)
        
        # NEW: Check for scratch exit conditions (early weakness detection)
        scratch_enabled = getattr(self.config, 'scratch_rule', {}).get('enabled', True)
        if scratch_enabled and hold_time_seconds <= 45:
            scratch_result = self.should_scratch_exit(position, current_price)
            
            if scratch_result['should_scratch']:
                self.logger.info(f"🔸 SCRATCH EXIT triggered for {mint_address[:8]}...: {scratch_result['reason']}")
                self.logger.info(f"   Exiting at {current_gain_pct:.1f}% loss to prevent further drawdown")
                return ("scratch", 1.0)
        
        # PHASE 3.1: Enhanced Dynamic TP1 Sizing
        if current_price >= position.tp_price and position.tp1_hit_time is None:
            position.tp1_hit_time = datetime.now()
            time_to_tp1 = (position.tp1_hit_time - position.entry_time).total_seconds()
            
            # OPTIMIZED: Better TP1 logic for right tail capture
            if current_gain_pct >= 60:  # Already at 60%+ gain
                tp1_percentage = 0.0  # Don't take any - let it run with tight trail
                self.logger.info(f"MOONSHOT DETECTED: {current_gain_pct:.0f}% gain, skipping TP1 to capture tail")
                position.tp1_hit_time = datetime.now()  # Mark as hit but don't sell
                position.trailing_stop_active = True  # Activate trailing immediately
                return None  # Don't sell, just activate trailing
            elif time_to_tp1 < 30:  # Ultra-fast spike (< 30 seconds)
                tp1_percentage = 0.08  # Sell only 8% - likely strong momentum
            elif time_to_tp1 < 60:  # < 1 minute - very fast move
                tp1_percentage = 0.12  # Sell only 12%
            elif time_to_tp1 < 180:  # 1-3 minutes - fast move
                tp1_percentage = 0.20  # Sell 20%
            else:  # > 3 minutes - normal/slow move
                tp1_percentage = 0.25  # Sell 25% (reduced from 30%)
            
            position.tp1_percentage_sold = tp1_percentage
            self.logger.info(f"Dynamic TP1: Selling {tp1_percentage*100:.0f}% after {time_to_tp1:.0f}s to TP")
            return ("take_profit_partial", tp1_percentage)
        
        # PHASE 3.2: Optimized Intelligent Trailing Stops
        
        # MOONSHOT trailing for exceptional gains (looser to capture full runs)
        if position.high_gain_peak >= 100:  # Triple digit gains
            trailing_pct = 0.75  # Trail at 75% of peak - very loose for moonshots
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                self.logger.info(f"MOONSHOT EXIT: {position.high_gain_peak:.0f}% peak, exiting at {current_gain_pct:.0f}%")
                return ("trailing_stop_moonshot", 1.0)
        elif position.high_gain_peak >= 60:  # High gains
            trailing_pct = 0.80  # Trail at 80% of peak (loosened from 82%)
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                return ("trailing_stop_high_gain", 1.0)
        elif position.high_gain_peak >= 30 and hold_time_seconds < 120:  # +30% in first 2 min
            trailing_pct = 0.85  # Trail at 85% of peak
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                return ("trailing_stop_fast_gain", 1.0)
        
        # Break-even stop logic
        if current_gain_pct >= 8 and not position.break_even_armed:
            # Arm break-even stop at +8% gain
            position.break_even_armed = True
            position.break_even_armed_time = datetime.now()
            self.logger.info(f"Break-even stop armed for {mint_address[:8]}... at +{current_gain_pct:.1f}%")
        
        if position.break_even_armed:
            time_since_armed = (datetime.now() - position.break_even_armed_time).total_seconds()
            if time_since_armed <= 60:  # Break-even protection for 60 seconds
                if current_price <= position.entry_price * 1.01:  # Allow 1% buffer
                    return ("break_even_stop", 1.0)
            else:
                # After 60 seconds, switch to normal trailing
                position.trailing_stop_active = True
                position.break_even_armed = False
        
        # Standard trailing stop (less aggressive)
        if position.trailing_stop_active or current_gain_pct >= 15:
            trailing_pct = 0.85  # Trail at 85% of peak
            trailing_stop = position.peak_price * trailing_pct
            if current_price <= trailing_stop:
                return ("trailing_stop", 1.0)
        
        # VOLATILITY-BASED STOP LOSS
        # Instead of fixed 8%, adjust based on token volatility and time
        volatility_stop = await self._calculate_dynamic_stop_loss(mint_address, position, hold_time_seconds)
        
        if current_price <= volatility_stop:
            stop_pct = ((volatility_stop / position.entry_price) - 1) * 100
            self.logger.info(f"DYNAMIC STOP HIT: Volatility-adjusted stop at {stop_pct:.1f}%")
            return ("stop_loss", 1.0)
        
        # PHASE 3.3: Remove hard time exit (or extend it)
        # Only exit after max_hold_seconds as a safety measure
        max_hold = getattr(self.config, 'max_hold_seconds', 900)  # 15 minutes default
        if hold_time_seconds > max_hold:
            # But only if we're not in profit
            if current_gain_pct < 5:  # Less than 5% gain
                return ("time_limit", 1.0)
            # If profitable, let it run with trailing stop
        
        return None

    def should_scratch_exit(self, position: Position, current_price: float, 
                           recent_trades: list = None) -> Dict:
        """
        Determine if position should be scratched due to early weakness
        
        Args:
            position: Current position data
            current_price: Current token price
            recent_trades: Recent trade data for buyer acceleration
            
        Returns:
            Dictionary with scratch decision
        """
        current_time = datetime.now()
        entry_time = position.entry_time
        
        # Only consider scratch in first 45 seconds
        hold_time_seconds = (current_time - entry_time).total_seconds()
        if hold_time_seconds > 45:
            return {
                'should_scratch': False,
                'reason': 'Outside scratch window (>45s)'
            }
        
        if position.entry_price <= 0 or current_price <= 0:
            return {
                'should_scratch': False,
                'reason': 'Invalid price data'
            }
        
        # Calculate current P&L
        current_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Calculate peak-to-current drawdown
        peak_drawdown = (position.peak_price - current_price) / position.peak_price if position.peak_price > 0 else 0
        
        # Check if drawdown exceeds threshold
        drawdown_threshold = 0.04  # 4%
        if peak_drawdown < drawdown_threshold:
            return {
                'should_scratch': False,
                'reason': f'Drawdown {peak_drawdown:.1%} < {drawdown_threshold:.1%}'
            }
        
        # Check buyer acceleration (simplified - we'll estimate without recent_trades for now)
        buyer_accel = self._estimate_buyer_acceleration(position.mint)
        
        # Scratch conditions
        is_negative_accel = buyer_accel < 0
        scratch_threshold_low = -0.02   # -2%
        scratch_threshold_high = -0.03  # -3%
        
        should_scratch = (
            peak_drawdown >= drawdown_threshold and
            is_negative_accel and
            scratch_threshold_high <= current_pnl_pct <= scratch_threshold_low
        )
        
        return {
            'should_scratch': should_scratch,
            'reason': f'Drawdown: {peak_drawdown:.1%}, Accel: {buyer_accel:.1f}, P&L: {current_pnl_pct:.1%}',
            'peak_drawdown': peak_drawdown,
            'buyer_acceleration': buyer_accel,
            'current_pnl_pct': current_pnl_pct
        }

    def _estimate_buyer_acceleration(self, mint_address: str) -> float:
        """
        Estimate buyer acceleration without external data
        For now, return 0 (neutral) - can be enhanced later with real-time data
        
        Returns:
            Buyer acceleration (positive = accelerating, negative = decelerating)
        """
        # TODO: Implement real buyer acceleration calculation when real-time data is available
        # For now, return slightly negative to be conservative
        return -0.5
    
    async def _calculate_dynamic_stop_loss(self, mint_address: str, position: Position, hold_time_seconds: float) -> float:
        """
        Calculate volatility-based stop loss that adapts to market conditions
        
        Args:
            mint_address: Token mint address
            position: Current position
            hold_time_seconds: How long we've held the position
            
        Returns:
            Stop loss price level
        """
        try:
            # Time-based stop loss tightening
            if hold_time_seconds < 30:
                # First 30 seconds: Tight 4% stop (quick scratch)
                base_stop_pct = 0.96
            elif hold_time_seconds < 60:
                # 30-60 seconds: Get recent trades to calculate volatility
                try:
                    # Try to get recent trades for volatility calculation
                    recent_trades = await self.moralis.get_token_swaps(mint_address, limit=20, ttl_override=5)
                    
                    if recent_trades and len(recent_trades) >= 5:
                        # Calculate simple volatility from recent price moves
                        prices = [t.get('price', 0) for t in recent_trades if t.get('price', 0) > 0]
                        
                        if len(prices) >= 5:
                            # Calculate price volatility
                            price_changes = []
                            for i in range(1, len(prices)):
                                change = abs((prices[i] - prices[i-1]) / prices[i-1])
                                price_changes.append(change)
                            
                            # Average volatility
                            avg_volatility = sum(price_changes) / len(price_changes) if price_changes else 0.08
                            
                            # Scale stop loss to volatility (2x volatility, capped)
                            volatility_stop = min(max(avg_volatility * 2, 0.05), 0.15)  # 5-15% range
                            base_stop_pct = 1 - volatility_stop
                            
                            self.logger.debug(f"Volatility-based stop: {volatility_stop:.1%} "
                                            f"(avg volatility: {avg_volatility:.1%})")
                        else:
                            # Not enough price data, use default
                            base_stop_pct = 0.94  # 6% stop
                    else:
                        # No trade data available, use moderate stop
                        base_stop_pct = 0.94  # 6% stop
                        
                except Exception as e:
                    self.logger.debug(f"Could not calculate volatility: {e}")
                    base_stop_pct = 0.94  # 6% stop on error
                    
            elif hold_time_seconds < 180:
                # 1-3 minutes: Standard 8% stop
                base_stop_pct = 0.92
            else:
                # After 3 minutes: Wider 10% stop (more room for swings)
                base_stop_pct = 0.90
            
            # Calculate stop price
            stop_price = position.entry_price * base_stop_pct
            
            # Never let stop price go above entry (no positive stops)
            stop_price = min(stop_price, position.entry_price * 0.99)
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic stop loss: {e}")
            # Fallback to conservative fixed stop
            return position.entry_price * 0.92  # 8% stop on error

    # Note: update_position_prices removed - position monitoring is now handled by main.py:monitor_position
    # which calls check_exit_conditions directly for better integration

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