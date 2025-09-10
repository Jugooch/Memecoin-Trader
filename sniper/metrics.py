"""
Metrics collection and Discord alerting for sniper bot
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Optional
from collections import defaultdict
import aiohttp
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.discord_notifier import DiscordNotifier


class Metrics:
    """Prometheus-style metrics and Discord alerts"""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Metrics storage
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Discord notifier
        self.discord = None
        if config.get('alerts', {}).get('discord_webhook'):
            self.discord = DiscordNotifier(config['alerts']['discord_webhook'])
            self.alert_events = set(config['alerts'].get('send_on', []))
        else:
            self.alert_events = set()
        
        # Prometheus settings
        self.prometheus_enabled = config.get('metrics', {}).get('enabled', True)
        self.prometheus_port = config.get('metrics', {}).get('prometheus_port', 9109)
        
        # PnL tracking
        self.total_pnl_sol = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def inc(self, metric: str, value: int = 1):
        """Increment counter metric"""
        self.counters[metric] += value
    
    def set(self, metric: str, value: float):
        """Set gauge metric"""
        self.gauges[metric] = value
    
    def observe(self, metric: str, value: float):
        """Observe histogram metric"""
        self.histograms[metric].append(value)
        # Keep only last 1000 observations
        if len(self.histograms[metric]) > 1000:
            self.histograms[metric] = self.histograms[metric][-1000:]
    
    async def send_alert(self, event_type: str, data: Dict):
        """Send Discord alert if configured"""
        if not self.discord or event_type not in self.alert_events:
            return
        
        try:
            # Format message based on event type
            if event_type == "ENTRY_OK":
                await self._send_entry_alert(data, success=True)
            elif event_type == "ENTRY_ABORTED":
                await self._send_entry_alert(data, success=False)
            elif event_type == "EXIT_FILLED":
                await self._send_exit_alert(data)
            elif event_type == "RUG_ALERT":
                await self._send_rug_alert(data)
            elif event_type == "WHITELIST_UPDATE":
                await self._send_whitelist_alert(data)
            elif event_type == "BLACKLIST_UPDATE":
                await self._send_blacklist_alert(data)
            else:
                # Generic alert
                await self.discord.send_generic_alert(event_type, data)
        
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {e}")
    
    async def _send_entry_alert(self, data: Dict, success: bool):
        """Send entry alert with comprehensive tracking info"""
        if success:
            embed = {
                "title": "🎯 SNIPER ENTRY",
                "color": 0x00FF00,
                "fields": [
                    {"name": "Token", "value": data.get('ticker', 'UNKNOWN'), "inline": True},
                    {"name": "Name", "value": data.get('name', 'Unknown'), "inline": True},
                    {"name": "Dev Score", "value": f"{data.get('score', 0):.1f}", "inline": True},
                    {"name": "Mint Address", "value": f"`{data.get('token_mint', 'Unknown')}`", "inline": False},
                    {"name": "Dev Wallet", "value": f"`{data.get('dev_wallet', 'Unknown')}`", "inline": False},
                    {"name": "Position", "value": f"{data.get('position_sol', 0):.3f} SOL", "inline": True},
                    {"name": "Entry Price", "value": f"${data.get('entry_price', 0):.8f}", "inline": True},
                    {"name": "Liquidity", "value": f"{data.get('lp_sol', 0):.2f} SOL", "inline": True},
                    {"name": "Dev Hold%", "value": f"{data.get('dev_hold_pct', 0):.1f}%", "inline": True},
                    {"name": "TX", "value": f"`{data.get('tx_signature', '')[:16]}...`", "inline": True},
                    {"name": "Detection Time", "value": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), "inline": True}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Include full mint for aborted entries to track manually
            embed = {
                "title": "⚠️ ENTRY ABORTED",
                "color": 0xFF9900,
                "fields": [
                    {"name": "Token", "value": data.get('ticker', 'UNKNOWN'), "inline": True},
                    {"name": "Name", "value": data.get('name', 'Unknown'), "inline": True},
                    {"name": "Dev Score", "value": f"{data.get('score', 0):.1f}", "inline": True},
                    {"name": "Reason", "value": data.get('reason', 'Unknown'), "inline": False},
                    {"name": "Mint Address", "value": f"`{data.get('token_mint', 'Unknown')}`", "inline": False},
                    {"name": "Dev Wallet", "value": f"`{data.get('dev_wallet', 'Unknown')}`", "inline": False},
                    {"name": "Liquidity", "value": f"{data.get('lp_sol', 0):.2f} SOL", "inline": True},
                    {"name": "Dev Hold%", "value": f"{data.get('dev_hold_pct', 0):.1f}%", "inline": True},
                    {"name": "Detection Time", "value": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'), "inline": True}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        await self.discord.send_embed_object(embed)
    
    async def _send_exit_alert(self, data: Dict):
        """Send exit alert"""
        pnl = data.get('pnl', 0)
        pnl_pct = data.get('pnl_pct', 0)
        color = 0x00FF00 if pnl > 0 else 0xFF0000
        
        embed = {
            "title": "💰 POSITION CLOSED" if pnl > 0 else "📉 POSITION CLOSED",
            "color": color,
            "fields": [
                {"name": "Token", "value": data.get('ticker', 'UNKNOWN'), "inline": True},
                {"name": "Exit Reason", "value": data.get('reason', 'Unknown'), "inline": True},
                {"name": "PnL", "value": f"{pnl:+.4f} SOL ({pnl_pct:+.1f}%)", "inline": True},
                {"name": "Entry", "value": f"${data.get('entry_price', 0):.8f}", "inline": True},
                {"name": "Exit", "value": f"${data.get('exit_price', 0):.8f}", "inline": True},
                {"name": "Hold Time", "value": f"{data.get('hold_time_min', 0):.1f} min", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.discord.send_embed_object(embed)
        
        # Update PnL tracking
        self.total_pnl_sol += pnl
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    async def _send_rug_alert(self, data: Dict):
        """Send rug detection alert"""
        embed = {
            "title": "🚨 RUG DETECTED",
            "color": 0xFF0000,
            "fields": [
                {"name": "Token", "value": data.get('ticker', 'UNKNOWN'), "inline": True},
                {"name": "Dev", "value": f"`{data.get('dev_wallet', '')[:8]}...`", "inline": True},
                {"name": "Type", "value": data.get('rug_type', 'Unknown'), "inline": True},
                {"name": "Action", "value": "Emergency exit triggered", "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.discord.send_embed_object(embed)
    
    async def _send_whitelist_alert(self, data: Dict):
        """Send whitelist update alert"""
        embed = {
            "title": "✅ WHITELIST UPDATED",
            "color": 0x00FF00,
            "fields": [
                {"name": "Action", "value": data.get('action', 'Unknown'), "inline": True},
                {"name": "Dev", "value": f"`{data.get('dev_wallet', '')[:8]}...`", "inline": True},
                {"name": "Score", "value": f"{data.get('score', 0):.1f}", "inline": True},
                {"name": "Reason", "value": data.get('reason', 'Manual'), "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.discord.send_embed_object(embed)
    
    async def _send_blacklist_alert(self, data: Dict):
        """Send blacklist update alert"""
        embed = {
            "title": "🚫 BLACKLIST UPDATED",
            "color": 0xFF0000,
            "fields": [
                {"name": "Action", "value": data.get('action', 'Unknown'), "inline": True},
                {"name": "Dev", "value": f"`{data.get('dev_wallet', '')[:8]}...`", "inline": True},
                {"name": "Reason", "value": data.get('reason', 'Unknown'), "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.discord.send_embed_object(embed)
    
    async def serve_http(self):
        """Serve Prometheus metrics via HTTP"""
        if not self.prometheus_enabled:
            return
        
        from aiohttp import web
        
        async def metrics_handler(request):
            """Generate Prometheus metrics format"""
            lines = []
            
            # Counters
            for name, value in self.counters.items():
                lines.append(f"sniper_{name}_total {value}")
            
            # Gauges
            for name, value in self.gauges.items():
                lines.append(f"sniper_{name} {value}")
            
            # Histograms (simplified - just show count and sum)
            for name, values in self.histograms.items():
                if values:
                    lines.append(f"sniper_{name}_count {len(values)}")
                    lines.append(f"sniper_{name}_sum {sum(values)}")
                    lines.append(f"sniper_{name}_avg {sum(values)/len(values):.2f}")
            
            # PnL metrics
            lines.append(f"sniper_pnl_total_sol {self.total_pnl_sol:.4f}")
            lines.append(f"sniper_winning_trades_total {self.winning_trades}")
            lines.append(f"sniper_losing_trades_total {self.losing_trades}")
            
            total_trades = self.winning_trades + self.losing_trades
            if total_trades > 0:
                win_rate = (self.winning_trades / total_trades) * 100
                lines.append(f"sniper_win_rate_pct {win_rate:.1f}")
            
            return web.Response(text='\n'.join(lines), content_type='text/plain')
        
        async def health_handler(request):
            """Health check endpoint"""
            return web.Response(text='OK', status=200)
        
        app = web.Application()
        app.router.add_get('/metrics', metrics_handler)
        app.router.add_get('/health', health_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.prometheus_port)
        
        self.logger.info(f"Starting metrics server on port {self.prometheus_port}")
        await site.start()
        
        # Keep server running
        while True:
            await asyncio.sleep(3600)
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_pnl_sol': self.total_pnl_sol,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'entries_attempted': self.counters.get('execution.entry_attempts', 0),
            'entries_success': self.counters.get('execution.live_entry', 0) + 
                             self.counters.get('execution.paper_entry', 0),
            'exits_triggered': sum(v for k, v in self.counters.items() if k.startswith('exits.triggered')),
            'safety_rejected': sum(v for k, v in self.counters.items() if k.startswith('safety.failed')),
            'discovery_candidates': self.counters.get('discovery.candidate_emitted', 0)
        }