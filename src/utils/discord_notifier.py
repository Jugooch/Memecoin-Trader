"""
Discord webhook notifier for trading bot
Sends trade notifications and status updates without dealing with SQLite
"""

import asyncio
import json
import os
from typing import Optional, Dict, Any, List
import httpx
import logging
from datetime import datetime


class DiscordNotifier:
    """Handles Discord notifications via webhooks"""
    
    def __init__(self, webhook_url: str, username: str = "Memecoin Trading Bot"):
        self.webhook_url = webhook_url
        self.username = username
        self.logger = logging.getLogger(__name__)
        self._client = httpx.AsyncClient(timeout=10)
        self.enabled = bool(webhook_url and webhook_url.strip())
        
        if not self.enabled:
            self.logger.warning(f"Discord notifications disabled (webhook URL empty or invalid): '{webhook_url}'")
        else:
            self.logger.info(f"Discord notifier initialized with webhook: {webhook_url[:50]}...")
    
    async def _post(self, payload: Dict[str, Any]) -> bool:
        """Post to Discord webhook with retry logic"""
        if not self.enabled:
            self.logger.debug("DISCORD_DEBUG: Notifications disabled, skipping message")
            return False
        
        self.logger.debug(f"DISCORD_DEBUG: Attempting to send Discord message: {payload.get('content', 'embed')[:50]}...")
            
        for attempt in range(3):
            try:
                resp = await self._client.post(self.webhook_url, json=payload)
                if resp.status_code in [200, 204]:
                    return True
                    
                # Handle rate limiting
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    self.logger.warning(f"Discord rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                    
                # Handle server errors with retry
                if 500 <= resp.status_code < 600:
                    await asyncio.sleep(1 + attempt)
                    continue
                    
                # Log other errors
                self.logger.error(f"Discord webhook error {resp.status_code}: {resp.text}")
                return False
                
            except Exception as e:
                self.logger.error(f"Discord notification failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
                    
        return False
    
    async def send_text(self, content: str) -> bool:
        """Send a simple text message"""
        if not self.enabled:
            return False
            
        # Discord has a 2000 character limit
        content = content[:1900]
        
        return await self._post({
            "username": self.username,
            "content": content
        })
    
    async def send_embed(self, 
                        title: str, 
                        fields: Dict[str, str], 
                        color: int = 0x2b6cb0,
                        description: Optional[str] = None,
                        footer: Optional[str] = None,
                        thumbnail_url: Optional[str] = None) -> bool:
        """Send a rich embed message"""
        if not self.enabled:
            return False
            
        embed = {
            "title": title[:256],
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "fields": []
        }
        
        if description:
            embed["description"] = description[:2048]
            
        if footer:
            embed["footer"] = {"text": footer[:2048]}
            
        if thumbnail_url:
            embed["thumbnail"] = {"url": thumbnail_url}
        
        # Add fields
        for name, value in fields.items():
            embed["fields"].append({
                "name": str(name)[:256],
                "value": str(value)[:1024],
                "inline": True
            })
        
        return await self._post({
            "username": self.username,
            "embeds": [embed]
        })
    
    async def send_trade_notification(self,
                                     side: str,
                                     symbol: str,
                                     mint_address: str,
                                     quantity: float,
                                     price: float,
                                     usd_amount: float,
                                     equity: float,
                                     realized_pnl: Optional[float] = None,
                                     confidence_score: Optional[float] = None,
                                     paper_mode: bool = True) -> bool:
        """Send a formatted trade notification with blockchain verification"""
        
        # Choose color based on side
        if side.upper() == "BUY":
            color = 0x10B981  # Green
            title = f"🟢 BUY: {symbol}"
        else:
            color = 0xF59E0B  # Orange
            title = f"🟠 SELL: {symbol}"
            
        # Build fields
        fields = {
            "Token": f"`{mint_address[:8]}...`",
            "Quantity": f"{quantity:,.0f}",
            "Price": f"${price:.8f}",
            "USD Value": f"${usd_amount:.2f}",
        }
        
        if confidence_score is not None:
            fields["Confidence"] = f"{confidence_score:.0f}%"
        
        if realized_pnl is not None:
            pnl_emoji = "📈" if realized_pnl >= 0 else "📉"
            fields["Realized P&L"] = f"{pnl_emoji} ${realized_pnl:.2f}"
        elif side == "SELL":
            # For sells without verified P&L, show warning
            fields["Realized P&L"] = "⚠️ UNVERIFIED - Check Wallet"
            
        fields["Equity"] = f"💰 ${equity:.2f}"
        
        # Footer to indicate mode
        footer = "📝 Paper Trading" if paper_mode else "💸 Live Trading"
        
        return await self.send_embed(
            title=title,
            fields=fields,
            color=color,
            footer=footer
        )
    
    async def send_error_notification(self, 
                                     error_message: str,
                                     context: Optional[Dict[str, Any]] = None) -> bool:
        """Send an error notification"""
        
        fields = {
            "Error": error_message[:1024]
        }
        
        if context:
            for key, value in list(context.items())[:5]:  # Limit to 5 context fields
                fields[key] = str(value)[:200]
        
        return await self.send_embed(
            title="⚠️ Error Detected",
            fields=fields,
            color=0xEF4444  # Red
        )
    
    async def send_summary(self,
                          equity: float,
                          daily_pnl: float,
                          total_trades: int,
                          win_rate: float,
                          active_positions: int = 0) -> bool:
        """Send a portfolio summary"""
        
        # Choose emoji based on P&L
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        # Calculate P&L percentage
        starting_equity = equity - daily_pnl
        pnl_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0
        
        fields = {
            "Equity": f"${equity:.2f}",
            "Daily P&L": f"{pnl_emoji} ${daily_pnl:+.2f} ({pnl_pct:+.1f}%)",
            "Total Trades": str(total_trades),
            "Win Rate": f"{win_rate:.1f}%",
            "Active Positions": str(active_positions)
        }
        
        # Color based on daily performance
        if daily_pnl > 0:
            color = 0x10B981  # Green
        elif daily_pnl < 0:
            color = 0xEF4444  # Red
        else:
            color = 0x3B82F6  # Blue
        
        return await self.send_embed(
            title="📊 Daily Summary",
            fields=fields,
            color=color
        )
    
    async def send_heartbeat(self, 
                           status: str = "Running",
                           uptime_hours: float = 0,
                           tokens_scanned: int = 0,
                           alpha_checks: int = 0) -> bool:
        """Send a heartbeat/status update"""
        
        fields = {
            "Status": f"✅ {status}",
            "Uptime": f"{uptime_hours:.1f} hours",
            "Tokens Scanned": str(tokens_scanned),
            "Alpha Checks": str(alpha_checks)
        }
        
        return await self.send_embed(
            title="💓 Bot Heartbeat",
            fields=fields,
            color=0x3B82F6  # Blue
        )
    
    async def send_api_exhausted_alert(self, 
                                      api_name: str,
                                      remaining_keys: int = 0) -> bool:
        """Alert when API keys are exhausted"""
        
        if remaining_keys > 0:
            message = f"⚠️ **{api_name} API Key Exhausted**\n"
            message += f"Remaining keys: {remaining_keys}\n"
            message += "Rotating to next key..."
            color = 0xF59E0B  # Orange warning
        else:
            message = f"🚨 **ALL {api_name} API Keys Exhausted!**\n"
            message += "Bot functionality may be limited.\n"
            message += "Please add more API keys or wait for rate limit reset."
            color = 0xEF4444  # Red alert
            
        return await self.send_text(message)
    
    async def send_alpha_wallet_update(self,
                                     new_wallets_found: int,
                                     total_added: int,
                                     total_watching: int,
                                     discovery_time: float,
                                     trigger_reason: str) -> bool:
        """Send notification about alpha wallet discovery results"""
        
        if total_added > 0:
            title = "✅ Found New Alpha Wallets!"
            color = 0x10B981  # Green
            description = f"Bot discovered {total_added} new profitable wallets to follow for better trade signals!"
        else:
            title = "🔍 Alpha Wallet Search Complete"
            color = 0xF59E0B  # Orange
            description = "Discovery completed but no new profitable wallets met the criteria."
        
        fields = {
            "New Wallets Found": str(new_wallets_found),
            "Added to Bot": str(total_added),
            "Total Watching": str(total_watching),
            "Search Time": f"{discovery_time:.1f}s",
            "Triggered By": trigger_reason
        }
        
        return await self.send_embed(
            title=title,
            fields=fields,
            color=color,
            description=description
        )
    
    # NEW ENHANCED NOTIFICATIONS for Implementation Plan
    
    async def send_daily_summary(self,
                                equity: float,
                                daily_pnl: float,
                                total_trades: int,
                                wins: int,
                                losses: int,
                                best_trade: Optional[float] = None,
                                worst_trade: Optional[float] = None,
                                top_wallet: Optional[str] = None,
                                top_wallet_wr: Optional[float] = None,
                                sol_balance: Optional[float] = None,
                                sol_in: Optional[float] = None,
                                sol_out: Optional[float] = None,
                                fees_paid: Optional[float] = None) -> bool:
        """Send comprehensive daily summary with blockchain data"""
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        
        # Calculate P&L percentage
        starting_equity = equity - daily_pnl
        pnl_pct = (daily_pnl / starting_equity * 100) if starting_equity > 0 else 0
        
        fields = {
            "P&L": f"{pnl_emoji} ${daily_pnl:+.2f} ({pnl_pct:+.1f}%)",
            "Trades": f"{total_trades} ({wins}W, {losses}L)",
            "Win Rate": f"{win_rate:.1f}%",
            "Equity": f"${equity:.2f}"
        }
        
        if best_trade is not None:
            fields["Best Trade"] = f"+{best_trade:.1f}%"
            
        if worst_trade is not None:
            fields["Worst Trade"] = f"{worst_trade:.1f}%"
            
        if top_wallet and top_wallet_wr:
            fields["Top Wallet"] = f"{top_wallet[:8]}... ({top_wallet_wr:.0f}% WR)"
        
        # Color based on daily performance
        if daily_pnl > 0:
            color = 0x10B981  # Green
        elif daily_pnl < -10:
            color = 0xEF4444  # Red
        else:
            color = 0x3B82F6  # Blue
        
        return await self.send_embed(
            title="📊 Daily Summary",
            fields=fields,
            color=color
        )
    
    async def send_risk_management_alert(self,
                                        daily_pnl_pct: float,
                                        action: str,
                                        new_requirements: Dict[str, Any]) -> bool:
        """Send risk management trigger notification"""
        
        fields = {
            "Daily P&L": f"{daily_pnl_pct:+.1f}%",
            "Action": action
        }
        
        if "min_wallets" in new_requirements:
            fields["Min Wallets"] = str(new_requirements["min_wallets"])
        if "min_weight" in new_requirements:
            fields["Min Weight"] = f"{new_requirements['min_weight']:.1f}"
        if "max_daily_trades" in new_requirements:
            fields["Max Daily Trades"] = str(new_requirements["max_daily_trades"])
        
        return await self.send_embed(
            title="⚠️ Risk Management Activated",
            fields=fields,
            color=0xF59E0B  # Orange
        )
    
    async def send_big_winner_alert(self, symbol: str, mint: str, pnl_pct: float, hold_time: int) -> bool:
        """Send notification for exceptional winning trades"""
        
        fields = {
            "Token": f"`{mint[:8]}...`",
            "Gain": f"📈 +{pnl_pct:.1f}%",
            "Hold Time": f"{hold_time//60}m {hold_time%60}s",
            "Symbol": symbol if symbol != "UNKNOWN" else "N/A"
        }
        
        return await self.send_embed(
            title="🚀 Big Winner!",
            fields=fields,
            color=0x10B981  # Green
        )
    
    async def send_stop_loss_alert(self, symbol: str, mint: str, pnl_pct: float, hold_time: int) -> bool:
        """Send notification for stop loss hits"""
        
        fields = {
            "Token": f"`{mint[:8]}...`",
            "Loss": f"💀 {pnl_pct:.1f}%",
            "Hold Time": f"{hold_time//60}m {hold_time%60}s",
            "Symbol": symbol if symbol != "UNKNOWN" else "N/A"
        }
        
        return await self.send_embed(
            title="💀 Stop Loss Hit",
            fields=fields,
            color=0xEF4444  # Red
        )
    
    async def send_wallet_tier_change(self, wallet: str, old_tier: str, new_tier: str, win_rate: float, trades: int) -> bool:
        """Send notification for wallet tier changes"""
        
        if new_tier == "S":
            emoji = "⭐"
            title = "New S-Tier Wallet!"
            color = 0xFFD700  # Gold
        elif old_tier in ["A", "B", "C"] and new_tier == "Unknown":
            emoji = "📉"
            title = "Wallet Demoted"
            color = 0xEF4444  # Red
        else:
            emoji = "🔄"
            title = "Wallet Tier Updated"
            color = 0x3B82F6  # Blue
        
        fields = {
            "Wallet": f"`{wallet[:8]}...`",
            "Change": f"{old_tier} → {new_tier}",
            "Win Rate": f"{win_rate:.1f}%",
            "Trades": str(trades)
        }
        
        return await self.send_embed(
            title=f"{emoji} {title}",
            fields=fields,
            color=color
        )
    
    async def send_system_health_check(self,
                                      uptime_hours: int,
                                      moralis_usage: tuple,
                                      bitquery_usage: tuple,
                                      active_positions: int,
                                      memory_mb: int) -> bool:
        """Send periodic system health notification"""
        
        moralis_used, moralis_limit = moralis_usage
        bitquery_used, bitquery_limit = bitquery_usage
        
        fields = {
            "Uptime": f"{uptime_hours}h",
            "API Usage": f"Moralis: {moralis_used}/{moralis_limit}\nBitquery: {bitquery_used}/{bitquery_limit}",
            "Positions": str(active_positions),
            "Memory": f"{memory_mb} MB"
        }
        
        # Color based on health
        if moralis_used/moralis_limit > 0.9 or bitquery_used/bitquery_limit > 0.9:
            color = 0xF59E0B  # Orange - high usage
        elif memory_mb > 500:
            color = 0xF59E0B  # Orange - high memory
        else:
            color = 0x10B981  # Green - healthy
        
        return await self.send_embed(
            title="❤️ System Health Check",
            fields=fields,
            color=color
        )
    
    async def send_cooldown_alert(self, reason: str, duration_minutes: int) -> bool:
        """Send notification when cooldown is activated"""
        
        fields = {
            "Reason": reason,
            "Duration": f"{duration_minutes} minutes",
            "Status": "Trading temporarily reduced"
        }
        
        return await self.send_embed(
            title="⏰ Cooldown Activated",
            fields=fields,
            color=0xF59E0B  # Orange
        )
    
    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()