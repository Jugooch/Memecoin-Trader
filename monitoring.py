"""
Monitoring and metrics module for bot performance tracking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import json

from database import Database


class PerformanceMonitor:
    def __init__(self, database: Database):
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.daily_metrics = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0,
            'api_calls': 0,
            'api_errors': 0,
            'tokens_analyzed': 0,
            'alpha_signals': 0
        }
        
        self.start_time = datetime.now()

    async def log_trade_execution(self, mint: str, action: str, success: bool, profit: float = 0):
        """Log trade execution metrics"""
        self.daily_metrics['trades_executed'] += 1
        
        if success:
            self.daily_metrics['successful_trades'] += 1
            self.daily_metrics['total_profit'] += profit
        else:
            self.daily_metrics['failed_trades'] += 1
        
        self.logger.info(f"Trade {action} on {mint}: {'SUCCESS' if success else 'FAILED'}")

    async def log_api_call(self, service: str, endpoint: str, success: bool, response_time: float):
        """Log API call metrics"""
        self.daily_metrics['api_calls'] += 1
        
        if not success:
            self.daily_metrics['api_errors'] += 1
        
        if response_time > 5.0:  # Log slow API calls
            self.logger.warning(f"Slow API call: {service}/{endpoint} - {response_time:.2f}s")

    async def log_token_analysis(self, mint: str, passed_filters: bool, alpha_signal: bool):
        """Log token analysis metrics"""
        self.daily_metrics['tokens_analyzed'] += 1
        
        if alpha_signal:
            self.daily_metrics['alpha_signals'] += 1
        
        if not passed_filters:
            self.logger.debug(f"Token {mint} failed filters")

    async def get_daily_summary(self) -> Dict:
        """Get daily performance summary"""
        uptime = datetime.now() - self.start_time
        
        # Calculate rates
        success_rate = 0
        if self.daily_metrics['trades_executed'] > 0:
            success_rate = self.daily_metrics['successful_trades'] / self.daily_metrics['trades_executed'] * 100
        
        api_error_rate = 0
        if self.daily_metrics['api_calls'] > 0:
            api_error_rate = self.daily_metrics['api_errors'] / self.daily_metrics['api_calls'] * 100
        
        alpha_signal_rate = 0
        if self.daily_metrics['tokens_analyzed'] > 0:
            alpha_signal_rate = self.daily_metrics['alpha_signals'] / self.daily_metrics['tokens_analyzed'] * 100
        
        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'trades_executed': self.daily_metrics['trades_executed'],
            'trade_success_rate': success_rate,
            'total_profit': self.daily_metrics['total_profit'],
            'api_calls': self.daily_metrics['api_calls'],
            'api_error_rate': api_error_rate,
            'tokens_analyzed': self.daily_metrics['tokens_analyzed'],
            'alpha_signals': self.daily_metrics['alpha_signals'],
            'alpha_signal_rate': alpha_signal_rate
        }

    async def generate_performance_report(self, days: int = 7) -> Dict:
        """Generate comprehensive performance report"""
        # Get performance history from database
        performance_history = await self.db.get_performance_history(days)
        trade_history = await self.db.get_trade_history(limit=1000)
        
        if not trade_history:
            return {'error': 'No trade data available'}
        
        # Calculate metrics
        total_trades = len(trade_history)
        profitable_trades = sum(1 for trade in trade_history if trade.get('profit', 0) > 0)
        total_profit = sum(trade.get('profit', 0) for trade in trade_history)
        
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate max drawdown
        running_total = 0
        peak = 0
        max_drawdown = 0
        
        for trade in reversed(trade_history):  # Oldest first
            running_total += trade.get('profit', 0)
            if running_total > peak:
                peak = running_total
            else:
                drawdown = (peak - running_total) / peak * 100 if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
        
        # Get current day summary
        daily_summary = await self.get_daily_summary()
        
        return {
            'report_period_days': days,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit_per_trade,
            'max_drawdown': max_drawdown,
            'daily_summary': daily_summary,
            'performance_history': performance_history
        }

    async def check_performance_alerts(self) -> List[Dict]:
        """Check for performance alerts and warnings"""
        alerts = []
        summary = await self.get_daily_summary()
        
        # High API error rate
        if summary['api_error_rate'] > 20:
            alerts.append({
                'type': 'warning',
                'message': f"High API error rate: {summary['api_error_rate']:.1f}%",
                'metric': 'api_error_rate',
                'value': summary['api_error_rate']
            })
        
        # Low trade success rate
        if summary['trade_success_rate'] < 50 and summary['trades_executed'] > 5:
            alerts.append({
                'type': 'warning',
                'message': f"Low trade success rate: {summary['trade_success_rate']:.1f}%",
                'metric': 'trade_success_rate',
                'value': summary['trade_success_rate']
            })
        
        # No trades executed in last hour (if expected)
        if summary['uptime_hours'] > 1 and summary['trades_executed'] == 0:
            alerts.append({
                'type': 'info',
                'message': "No trades executed yet",
                'metric': 'trades_executed',
                'value': 0
            })
        
        # Negative profit
        if summary['total_profit'] < -50:  # More than $50 loss
            alerts.append({
                'type': 'critical',
                'message': f"Significant losses: ${summary['total_profit']:.2f}",
                'metric': 'total_profit',
                'value': summary['total_profit']
            })
        
        return alerts

    async def save_daily_metrics(self):
        """Save daily metrics to database"""
        summary = await self.get_daily_summary()
        
        performance_data = {
            'date': datetime.now().date(),
            'total_trades': summary['trades_executed'],
            'winning_trades': self.daily_metrics['successful_trades'],
            'total_profit': summary['total_profit'],
            'win_rate': summary['trade_success_rate'],
            'metadata': {
                'api_calls': summary['api_calls'],
                'api_error_rate': summary['api_error_rate'],
                'tokens_analyzed': summary['tokens_analyzed'],
                'alpha_signals': summary['alpha_signals'],
                'uptime_hours': summary['uptime_hours']
            }
        }
        
        await self.db.save_daily_performance(performance_data)
        self.logger.info("Daily metrics saved to database")

    def reset_daily_metrics(self):
        """Reset daily metrics (called at midnight)"""
        self.daily_metrics = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0,
            'api_calls': 0,
            'api_errors': 0,
            'tokens_analyzed': 0,
            'alpha_signals': 0
        }
        self.logger.info("Daily metrics reset")

    async def export_metrics(self, file_path: str):
        """Export metrics to JSON file"""
        report = await self.generate_performance_report(30)  # Last 30 days
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {file_path}")


class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def check_system_health(self) -> Dict:
        """Check overall system health"""
        import psutil
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        return {
            'memory_used_pct': memory.percent,
            'cpu_used_pct': cpu_percent,
            'disk_used_pct': disk.percent,
            'available_memory_gb': memory.available / (1024**3),
            'available_disk_gb': disk.free / (1024**3)
        }

    async def check_network_connectivity(self) -> Dict:
        """Check network connectivity to APIs"""
        import aiohttp
        
        endpoints = {
            'bitquery': 'https://graphql.bitquery.io',
            'moralis': 'https://solana-api.moralis.io',
            'solana_rpc': 'https://api.mainnet-beta.solana.com'
        }
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for name, url in endpoints.items():
                try:
                    start_time = asyncio.get_event_loop().time()
                    async with session.get(url, timeout=5) as response:
                        end_time = asyncio.get_event_loop().time()
                        
                        results[name] = {
                            'status': 'online',
                            'response_time': end_time - start_time,
                            'status_code': response.status
                        }
                except Exception as e:
                    results[name] = {
                        'status': 'offline',
                        'error': str(e)
                    }
        
        return results