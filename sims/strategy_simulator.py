#!/usr/bin/env python3
"""
Strategy Simulation Engine
Monitors real-time alpha buys and simulates 3 different trading strategies:
1. Current Implementation (with reconciliation delays, tight stops)
2. Buffer Strategy (60s volatility buffer, wider stops) 
3. Late Alpha Strategy (allows late entries, 4+ alpha threshold)
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.clients.pumpportal_client import PumpPortalClient
from src.clients.moralis_client import MoralisClient
from src.utils.config_loader import ConfigLoader
from src.utils.discord_notifier import DiscordNotifier

@dataclass
class SimulatedPosition:
    """Represents a simulated trading position"""
    mint_address: str
    symbol: str
    entry_time: datetime
    entry_price: float
    tokens: float
    usd_invested: float
    strategy: str
    alpha_count: int
    alpha_delay_seconds: float
    
    # Exit tracking
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    profit_usd: Optional[float] = None
    profit_pct: Optional[float] = None
    
    # Price tracking
    peak_price: float = 0.0
    peak_gain_pct: float = 0.0
    current_price: float = 0.0
    
    # Multi-tier tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    tp1_profit: float = 0.0
    tp2_profit: float = 0.0
    tp3_profit: float = 0.0

@dataclass
class StrategyResults:
    """Results for a specific strategy"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_usd: float = 0.0
    total_invested_usd: float = 0.0
    win_rate: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_hold_time_minutes: float = 0.0
    max_profit_trade: float = 0.0
    max_loss_trade: float = 0.0
    
    # Breakdown by exit reason
    stop_loss_count: int = 0
    take_profit_count: int = 0
    trailing_stop_count: int = 0
    time_based_count: int = 0

class StrategySimulator:
    """Main simulation engine"""
    
    def __init__(self, config_path: str):
        """Initialize simulator with config"""
        self.config = ConfigLoader.load_config(config_path)
        self.sims_dir = Path("sims")
        self.data_dir = self.sims_dir / "data"
        self.logs_dir = self.sims_dir / "logs"
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize clients
        self.pumpportal = None
        self.moralis = None
        self.discord = None
        
        # Simulation data
        self.active_positions: Dict[str, List[SimulatedPosition]] = {
            "current": {},
            "buffer": {},
            "late_alpha": {}
        }
        
        self.completed_positions: Dict[str, List[SimulatedPosition]] = {
            "current": [],
            "buffer": [], 
            "late_alpha": []
        }
        
        # Alpha tracking
        self.alpha_signals: Dict[str, List[Dict]] = {}  # mint -> alpha events
        self.watched_wallets = set()
        
        # Results storage
        self.results_files = {
            "current": self.data_dir / "strategy_sim_current.json",
            "buffer": self.data_dir / "strategy_sim_buffer.json",
            "late_alpha": self.data_dir / "strategy_sim_alphas.json"
        }
        
    def setup_logging(self):
        """Setup logging for simulation"""
        log_file = self.logs_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("StrategySimulator")
        
    async def initialize_clients(self):
        """Initialize API clients"""
        try:
            # PumpPortal for real-time data
            if hasattr(self.config, 'pumpportal'):
                self.pumpportal = PumpPortalClient(
                    api_key=self.config.pumpportal.get('api_key'),
                    ws_endpoint=self.config.pumpportal.get('ws_endpoint'),
                    trade_endpoint=self.config.pumpportal.get('trade_endpoint')
                )
                await self.pumpportal.initialize()
                self.logger.info("✅ PumpPortal client initialized")
            
            # Moralis for price data
            if hasattr(self.config, 'moralis_keys') and self.config.moralis_keys:
                self.moralis = MoralisClient(
                    api_keys=self.config.moralis_keys,
                    rate_limit_delay=0.1
                )
                self.logger.info("✅ Moralis client initialized")
            
            # Discord for notifications
            if hasattr(self.config, 'discord_webhook_url'):
                self.discord = DiscordNotifier(self.config.discord_webhook_url)
                self.logger.info("✅ Discord notifier initialized")
            
            # Load watched wallets
            if hasattr(self.config, 'watched_wallets'):
                self.watched_wallets = set(self.config.watched_wallets)
                self.logger.info(f"✅ Loaded {len(self.watched_wallets)} alpha wallets")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing clients: {e}")
            raise
    
    async def start_simulation(self, duration_hours: int = 24):
        """Start the simulation for specified duration"""
        self.logger.info(f"🚀 Starting strategy simulation for {duration_hours} hours...")
        self.logger.info("📊 Strategies: Current, Buffer (60s), Late Alpha (4+ alphas)")
        
        await self.initialize_clients()
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_alpha_trades()),
            asyncio.create_task(self.update_positions_loop()),
            asyncio.create_task(self.save_results_loop()),
            asyncio.create_task(self.send_progress_reports())
        ]
        
        # Run for specified duration
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks), 
                timeout=duration_hours * 3600
            )
        except asyncio.TimeoutError:
            self.logger.info(f"⏰ Simulation completed after {duration_hours} hours")
        except KeyboardInterrupt:
            self.logger.info("⛔ Simulation stopped by user")
        finally:
            await self.cleanup()
    
    async def monitor_alpha_trades(self):
        """Monitor real-time alpha wallet trades"""
        if not self.pumpportal:
            self.logger.error("❌ PumpPortal not available for monitoring")
            return
            
        try:
            # Subscribe to alpha wallet trades
            await self.pumpportal.subscribe_wallet_trades(list(self.watched_wallets))
            
            async for event in self.pumpportal.listen_events():
                try:
                    await self.process_alpha_event(event)
                except Exception as e:
                    self.logger.error(f"❌ Error processing event: {e}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error in alpha monitoring: {e}")
    
    async def process_alpha_event(self, event: Dict):
        """Process an alpha wallet trade event"""
        try:
            if event.get('type') != 'trade' or not event.get('is_buy'):
                return
                
            mint = event.get('mint')
            trader = event.get('traderPublicKey')
            
            if not mint or trader not in self.watched_wallets:
                return
                
            # Track alpha signal
            if mint not in self.alpha_signals:
                self.alpha_signals[mint] = []
                
            alpha_event = {
                'wallet': trader,
                'timestamp': datetime.now(),
                'price': event.get('price', 0),
                'amount_sol': event.get('sol_amount', 0)
            }
            
            self.alpha_signals[mint].append(alpha_event)
            alpha_count = len(self.alpha_signals[mint])
            
            self.logger.info(f"🔥 ALPHA BUY #{alpha_count}: {mint[:8]}... by {trader[:8]}...")
            
            # Check if we should simulate entries
            await self.check_entry_conditions(mint, alpha_count)
            
        except Exception as e:
            self.logger.error(f"❌ Error processing alpha event: {e}")
    
    async def check_entry_conditions(self, mint: str, alpha_count: int):
        """Check if any strategy should enter this token"""
        try:
            if mint not in self.alpha_signals or not self.alpha_signals[mint]:
                return
                
            first_alpha_time = self.alpha_signals[mint][0]['timestamp']
            delay_seconds = (datetime.now() - first_alpha_time).total_seconds()
            
            # Get current price
            current_price = await self.get_current_price(mint)
            if not current_price or current_price <= 0:
                return
                
            # Strategy 1: Current Implementation
            if (alpha_count >= 1 and delay_seconds <= 15 and 
                mint not in self.active_positions["current"]):
                await self.simulate_entry(mint, "current", current_price, alpha_count, delay_seconds)
            
            # Strategy 2: Buffer Strategy (same entry as current)
            if (alpha_count >= 1 and delay_seconds <= 15 and 
                mint not in self.active_positions["buffer"]):
                await self.simulate_entry(mint, "buffer", current_price, alpha_count, delay_seconds)
            
            # Strategy 3: Late Alpha Strategy
            if (alpha_count >= 4 and delay_seconds <= 300 and  # 5 minutes max
                mint not in self.active_positions["late_alpha"]):
                await self.simulate_entry(mint, "late_alpha", current_price, alpha_count, delay_seconds)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking entry conditions: {e}")
    
    async def simulate_entry(self, mint: str, strategy: str, price: float, alpha_count: int, delay_seconds: float):
        """Simulate entering a position"""
        try:
            # Get token info
            symbol = await self.get_token_symbol(mint) or f"Token_{mint[:8]}"
            
            # Fixed position size for simulation
            usd_amount = 30.0
            tokens = usd_amount / price
            
            position = SimulatedPosition(
                mint_address=mint,
                symbol=symbol,
                entry_time=datetime.now(),
                entry_price=price,
                tokens=tokens,
                usd_invested=usd_amount,
                strategy=strategy,
                alpha_count=alpha_count,
                alpha_delay_seconds=delay_seconds,
                current_price=price,
                peak_price=price
            )
            
            self.active_positions[strategy][mint] = position
            
            self.logger.info(f"📈 {strategy.upper()} ENTRY: {symbol} @ ${price:.8f} "
                           f"({alpha_count} alphas, {delay_seconds:.0f}s delay)")
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating entry: {e}")
    
    async def update_positions_loop(self):
        """Continuously update active positions"""
        while True:
            try:
                for strategy in ["current", "buffer", "late_alpha"]:
                    positions_to_remove = []
                    
                    for mint, position in self.active_positions[strategy].items():
                        # Update current price
                        current_price = await self.get_current_price(mint)
                        if current_price and current_price > 0:
                            position.current_price = current_price
                            position.peak_price = max(position.peak_price, current_price)
                            position.peak_gain_pct = max(position.peak_gain_pct, 
                                                       ((current_price / position.entry_price) - 1) * 100)
                        
                        # Check exit conditions
                        exit_signal = await self.check_exit_conditions(position, strategy)
                        if exit_signal:
                            await self.simulate_exit(position, exit_signal)
                            positions_to_remove.append(mint)
                    
                    # Remove completed positions
                    for mint in positions_to_remove:
                        completed_position = self.active_positions[strategy].pop(mint)
                        self.completed_positions[strategy].append(completed_position)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"❌ Error updating positions: {e}")
                await asyncio.sleep(5)
    
    async def check_exit_conditions(self, position: SimulatedPosition, strategy: str) -> Optional[Dict]:
        """Check if position should exit based on strategy"""
        try:
            hold_seconds = (datetime.now() - position.entry_time).total_seconds()
            current_gain_pct = ((position.current_price / position.entry_price) - 1) * 100
            
            # Strategy-specific exit logic
            if strategy == "current":
                return await self.check_current_strategy_exits(position, hold_seconds, current_gain_pct)
            elif strategy == "buffer":
                return await self.check_buffer_strategy_exits(position, hold_seconds, current_gain_pct)
            elif strategy == "late_alpha":
                return await self.check_late_alpha_strategy_exits(position, hold_seconds, current_gain_pct)
                
        except Exception as e:
            self.logger.error(f"❌ Error checking exit conditions: {e}")
            return None
    
    async def check_current_strategy_exits(self, position: SimulatedPosition, hold_seconds: float, gain_pct: float) -> Optional[Dict]:
        """Current strategy: tight stops, immediate exits"""
        # Immediate stop loss (current aggressive strategy)
        if gain_pct <= -15.0:  # -15% stop
            return {"reason": "stop_loss", "percentage": 1.0}
        
        # Take profit at 20%
        if gain_pct >= 20.0 and not position.tp1_hit:
            return {"reason": "take_profit", "percentage": 1.0}
            
        # Time-based exit after 30 minutes
        if hold_seconds > 1800:  # 30 minutes
            return {"reason": "time_based", "percentage": 1.0}
        
        return None
    
    async def check_buffer_strategy_exits(self, position: SimulatedPosition, hold_seconds: float, gain_pct: float) -> Optional[Dict]:
        """Buffer strategy: 60s buffer, wider stops"""
        # No stop losses in first 60 seconds
        if hold_seconds < 60 and gain_pct < 0:
            return None
            
        # Take profits anytime (no buffer on profits)
        if gain_pct >= 20.0 and not position.tp1_hit:
            position.tp1_hit = True
            position.tp1_profit = position.usd_invested * 0.45 * (gain_pct / 100)
            return {"reason": "tp1_partial", "percentage": 0.45}
        
        if gain_pct >= 50.0 and not position.tp2_hit and position.tp1_hit:
            position.tp2_hit = True
            position.tp2_profit = position.usd_invested * 0.35 * (gain_pct / 100)
            return {"reason": "tp2_partial", "percentage": 0.35}
        
        if gain_pct >= 100.0 and not position.tp3_hit and position.tp2_hit:
            position.tp3_hit = True
            position.tp3_profit = position.usd_invested * 0.15 * (gain_pct / 100)
            return {"reason": "tp3_partial", "percentage": 0.15}
        
        # Wider stop loss after buffer period
        if hold_seconds >= 60 and gain_pct <= -35.0:  # -35% stop (vs -15%)
            return {"reason": "stop_loss", "percentage": 1.0}
            
        # Trailing stop on remaining position
        if position.tp1_hit and position.peak_gain_pct > 30:
            drawdown_from_peak = ((position.peak_price - position.current_price) / position.peak_price) * 100
            if drawdown_from_peak >= 15:  # 15% drawdown from peak
                return {"reason": "trailing_stop", "percentage": 1.0}
        
        return None
    
    async def check_late_alpha_strategy_exits(self, position: SimulatedPosition, hold_seconds: float, gain_pct: float) -> Optional[Dict]:
        """Late alpha strategy: same as buffer (coming in late already)"""
        return await self.check_buffer_strategy_exits(position, hold_seconds, gain_pct)
    
    async def simulate_exit(self, position: SimulatedPosition, exit_signal: Dict):
        """Simulate exiting a position"""
        try:
            position.exit_time = datetime.now()
            position.exit_price = position.current_price
            position.exit_reason = exit_signal["reason"]
            
            percentage = exit_signal["percentage"]
            
            if percentage >= 1.0:  # Full exit
                position.profit_usd = (position.current_price - position.entry_price) * position.tokens
                position.profit_pct = ((position.current_price / position.entry_price) - 1) * 100
            else:  # Partial exit
                tokens_sold = position.tokens * percentage
                profit_on_sold = (position.current_price - position.entry_price) * tokens_sold
                
                # Update position for remaining tokens
                position.tokens *= (1 - percentage)
                position.usd_invested *= (1 - percentage)
                
                # Track partial profit
                if exit_signal["reason"].startswith("tp"):
                    if "tp1" in exit_signal["reason"]:
                        position.tp1_profit = profit_on_sold
                    elif "tp2" in exit_signal["reason"]:
                        position.tp2_profit = profit_on_sold
                    elif "tp3" in exit_signal["reason"]:
                        position.tp3_profit = profit_on_sold
                
                return  # Don't complete position yet for partial exits
            
            # Calculate total profit including partials
            total_profit = position.profit_usd or 0
            if position.tp1_profit:
                total_profit += position.tp1_profit
            if position.tp2_profit:
                total_profit += position.tp2_profit
            if position.tp3_profit:
                total_profit += position.tp3_profit
            
            position.profit_usd = total_profit
            position.profit_pct = (total_profit / position.usd_invested) * 100
            
            self.logger.info(f"📉 {position.strategy.upper()} EXIT: {position.symbol} "
                           f"{position.profit_pct:+.1f}% (${position.profit_usd:+.2f}) - {exit_signal['reason']}")
            
        except Exception as e:
            self.logger.error(f"❌ Error simulating exit: {e}")
    
    async def get_current_price(self, mint: str) -> Optional[float]:
        """Get current token price"""
        try:
            if self.moralis:
                return await self.moralis.get_current_price(mint, fresh=True)
            return None
        except Exception as e:
            self.logger.debug(f"⚠️ Could not get price for {mint[:8]}...: {e}")
            return None
    
    async def get_token_symbol(self, mint: str) -> Optional[str]:
        """Get token symbol"""
        try:
            if self.moralis:
                metadata = await self.moralis.get_token_metadata(mint)
                return metadata.get('symbol') if metadata else None
            return None
        except Exception as e:
            self.logger.debug(f"⚠️ Could not get symbol for {mint[:8]}...: {e}")
            return None
    
    def calculate_strategy_results(self, strategy: str) -> StrategyResults:
        """Calculate results for a strategy"""
        positions = self.completed_positions[strategy]
        
        if not positions:
            return StrategyResults(strategy_name=strategy)
        
        results = StrategyResults(strategy_name=strategy)
        results.total_trades = len(positions)
        
        total_hold_time = 0
        
        for pos in positions:
            if pos.profit_pct and pos.profit_pct > 0:
                results.winning_trades += 1
            else:
                results.losing_trades += 1
                
            if pos.profit_usd:
                results.total_profit_usd += pos.profit_usd
                results.max_profit_trade = max(results.max_profit_trade, pos.profit_usd)
                results.max_loss_trade = min(results.max_loss_trade, pos.profit_usd)
            
            results.total_invested_usd += pos.usd_invested
            
            if pos.exit_time and pos.entry_time:
                hold_time = (pos.exit_time - pos.entry_time).total_seconds() / 60
                total_hold_time += hold_time
            
            # Exit reason breakdown
            if pos.exit_reason:
                if "stop_loss" in pos.exit_reason:
                    results.stop_loss_count += 1
                elif "take_profit" in pos.exit_reason or "tp" in pos.exit_reason:
                    results.take_profit_count += 1
                elif "trailing" in pos.exit_reason:
                    results.trailing_stop_count += 1
                elif "time" in pos.exit_reason:
                    results.time_based_count += 1
        
        # Calculate derived metrics
        if results.total_trades > 0:
            results.win_rate = (results.winning_trades / results.total_trades) * 100
            results.avg_profit_per_trade = results.total_profit_usd / results.total_trades
            results.avg_hold_time_minutes = total_hold_time / results.total_trades
        
        return results
    
    async def save_results_loop(self):
        """Periodically save results to JSON files"""
        while True:
            try:
                for strategy in ["current", "buffer", "late_alpha"]:
                    # Calculate current results
                    results = self.calculate_strategy_results(strategy)
                    
                    # Prepare data for JSON
                    data = {
                        "last_updated": datetime.now().isoformat(),
                        "results": asdict(results),
                        "active_positions": len(self.active_positions[strategy]),
                        "completed_positions": [
                            asdict(pos) for pos in self.completed_positions[strategy]
                        ]
                    }
                    
                    # Save to file
                    with open(self.results_files[strategy], 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                
                self.logger.debug("💾 Results saved to JSON files")
                await asyncio.sleep(60)  # Save every minute
                
            except Exception as e:
                self.logger.error(f"❌ Error saving results: {e}")
                await asyncio.sleep(60)
    
    async def send_progress_reports(self):
        """Send periodic progress reports to Discord"""
        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes
                
                if self.discord:
                    report = await self.generate_progress_report()
                    await self.discord.send_notification(
                        "📊 Strategy Simulation Update", 
                        report
                    )
                    
            except Exception as e:
                self.logger.error(f"❌ Error sending progress report: {e}")
    
    async def generate_progress_report(self) -> str:
        """Generate progress report comparing all strategies"""
        try:
            report_lines = ["**Strategy Simulation Progress Report**\n"]
            
            for strategy in ["current", "buffer", "late_alpha"]:
                results = self.calculate_strategy_results(strategy)
                active_count = len(self.active_positions[strategy])
                
                strategy_name = {
                    "current": "Current Implementation",
                    "buffer": "Buffer Strategy (60s)",
                    "late_alpha": "Late Alpha Strategy (4+ alphas)"
                }[strategy]
                
                report_lines.append(f"**{strategy_name}:**")
                report_lines.append(f"• Total Trades: {results.total_trades} (Active: {active_count})")
                report_lines.append(f"• Win Rate: {results.win_rate:.1f}%")
                report_lines.append(f"• Total P&L: ${results.total_profit_usd:+.2f}")
                report_lines.append(f"• Avg P&L/Trade: ${results.avg_profit_per_trade:+.2f}")
                report_lines.append(f"• Avg Hold Time: {results.avg_hold_time_minutes:.1f}m")
                report_lines.append("")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.logger.error(f"❌ Error generating report: {e}")
            return "Error generating progress report"
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.pumpportal:
                await self.pumpportal.close()
            if self.moralis:
                await self.moralis.close()
            if self.discord:
                await self.discord.close()
                
            self.logger.info("✅ Simulation cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")

async def main():
    """Main entry point"""
    config_path = "config/config.yml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Please create a config file based on config.yml.example")
        return
    
    simulator = StrategySimulator(config_path)
    
    try:
        # Run simulation for 24 hours by default
        duration = int(input("Enter simulation duration in hours (default 24): ") or "24")
        await simulator.start_simulation(duration_hours=duration)
        
    except KeyboardInterrupt:
        print("\n⛔ Simulation stopped by user")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
    finally:
        await simulator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())