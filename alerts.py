"""
Alert system for manual trading execution
"""

import logging
import json
from datetime import datetime
from typing import Dict


class TradingAlerts:
    def __init__(self, output_file: str = "trading_alerts.json"):
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
        self.alerts = []

    async def send_buy_alert(self, mint_address: str, metadata: Dict, liquidity: Dict, alpha_data: Dict):
        """Generate buy alert for manual execution"""
        alert = {
            "type": "BUY_SIGNAL",
            "timestamp": datetime.now().isoformat(),
            "mint": mint_address,
            "token_name": metadata.get('name', 'Unknown'),
            "token_symbol": metadata.get('symbol', 'Unknown'),
            "liquidity_usd": liquidity.get('total_liquidity_usd', 0),
            "alpha_buyers": alpha_data.get('buyer_count', 0),
            "alpha_wallets": alpha_data.get('wallets', []),
            "suggested_amount_usd": 5.0,  # 5% of $100
            "take_profit_target": "2x (100% profit)",
            "stop_loss": "30% loss",
            "urgency": "HIGH",
            "instructions": {
                "1": f"Go to pump.fun and search for: {mint_address}",
                "2": "Verify liquidity and early buyers match alert data",
                "3": "Buy $5 worth if still under 2 minutes since launch",
                "4": "Set alerts for 2x price (take 50% profit) and 70% price (stop loss)"
            }
        }
        
        self.alerts.append(alert)
        await self._save_alerts()
        await self._display_alert(alert)

    async def send_sell_alert(self, mint_address: str, current_price: float, entry_price: float, reason: str):
        """Generate sell alert for manual execution"""
        profit_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        alert = {
            "type": "SELL_SIGNAL",
            "timestamp": datetime.now().isoformat(),
            "mint": mint_address,
            "reason": reason,
            "entry_price": entry_price,
            "current_price": current_price,
            "profit_pct": profit_pct,
            "urgency": "HIGH" if reason == "stop_loss" else "MEDIUM",
            "instructions": {
                "1": f"Go to pump.fun and find your position in: {mint_address}",
                "2": f"Current price: ${current_price:.6f} (Entry: ${entry_price:.6f})",
                "3": f"Sell immediately - {reason.replace('_', ' ').title()}",
                "4": f"Expected profit: {profit_pct:.1f}%"
            }
        }
        
        self.alerts.append(alert)
        await self._save_alerts()
        await self._display_alert(alert)

    async def _save_alerts(self):
        """Save alerts to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.alerts, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save alerts: {e}")

    async def _display_alert(self, alert: Dict):
        """Display alert in console"""
        print("\n" + "="*60)
        print(f"ALERT: {alert['type']} - {alert['urgency']} URGENCY")
        print("="*60)
        print(f"⏰ Time: {alert['timestamp']}")
        print(f"🪙 Token: {alert.get('token_name', 'Unknown')} ({alert.get('token_symbol', 'Unknown')})")
        print(f"Address: {alert['mint']}")
        
        if alert['type'] == 'BUY_SIGNAL':
            print(f"Liquidity: ${alert['liquidity_usd']:,.0f}")
            print(f"Alpha Buyers: {alert['alpha_buyers']}")
            print(f"Suggested Amount: ${alert['suggested_amount_usd']}")
            print(f"Take Profit: {alert['take_profit_target']}")
            print(f"Stop Loss: {alert['stop_loss']}")
        else:
            print(f"Entry Price: ${alert['entry_price']:.6f}")
            print(f"Current Price: ${alert['current_price']:.6f}")
            print(f"Profit: {alert['profit_pct']:.1f}%")
            print(f"Reason: {alert['reason'].replace('_', ' ').title()}")
        
        print("\nINSTRUCTIONS:")
        for step, instruction in alert['instructions'].items():
            print(f"  {step}. {instruction}")
        
        print("="*60)
        
        # Also log the alert
        self.logger.info(f"{alert['type']} alert generated for {alert['mint']}")

    async def get_recent_alerts(self, hours: int = 24) -> list:
        """Get recent alerts within specified hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        recent_alerts = []
        for alert in self.alerts:
            alert_time = datetime.fromisoformat(alert['timestamp']).timestamp()
            if alert_time >= cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts

    async def clear_old_alerts(self, days: int = 7):
        """Clear alerts older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']).timestamp() >= cutoff_time
        ]
        
        await self._save_alerts()
        self.logger.info(f"Cleared alerts older than {days} days")

    async def export_alerts(self, filename: str = None):
        """Export alerts to file"""
        if not filename:
            filename = f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.alerts, f, indent=2, default=str)
            
            print(f"OK - Alerts exported to {filename}")
            return filename
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
            return None