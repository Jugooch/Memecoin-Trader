#!/usr/bin/env python3
"""
Advanced Strategy A/B Testing Framework
Tests GPT5's recommendations vs current strategy using existing infrastructure
"""

import asyncio
import json
import logging
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing bot infrastructure
from src.clients.moralis_client import MoralisClient
from src.clients.realtime_client import RealtimeClient
from src.core.wallet_tracker import WalletTracker
from src.core.safety_checks import SafetyChecker
from src.discovery.alpha_discovery_v2 import ProvenAlphaFinder
from src.utils.config_loader import load_config
from src.utils.discord_notifier import DiscordNotifier
from src.utils.logger_setup import setup_logging


@dataclass
class MarketReality:
    """Models real trading conditions and execution failures"""
    latency_mean_ms: float = 150
    latency_std_ms: float = 50
    base_slippage_bps: int = 30
    price_impact_factor: float = 0.001
    
    # Failure rates by slippage tolerance
    fail_rate_2pct: float = 0.15    # 15% fail at 2% slippage
    fail_rate_5pct: float = 0.08    # 8% fail at 5% slippage
    fail_rate_10pct: float = 0.03   # 3% fail at 10% slippage
    
    # Network conditions
    congestion_multiplier: float = 1.0
    volatility_multiplier: float = 1.0

@dataclass
class TradeExecution:
    """Models realistic trade execution with failures"""
    signal_time: datetime
    attempted_time: datetime
    executed_time: Optional[datetime]
    target_price: float
    executed_price: Optional[float]
    slippage_tolerance: float
    actual_slippage: Optional[float]
    success: bool
    failure_reason: Optional[str] = None
    gas_cost_usd: float = 0.0
    priority_fee_usd: float = 0.0

@dataclass
class SimulatedPosition:
    """Enhanced position tracking with realistic execution"""
    mint_address: str
    symbol: str
    strategy_name: str
    
    # Entry details
    entry_signal_time: datetime
    entry_execution: TradeExecution
    alpha_wallets: List[str]
    alpha_entry_price: float  # What alpha wallet paid
    leader_price_delta: float  # How much higher we paid
    
    # Position details
    tokens_bought: float
    usd_invested: float
    cost_basis_with_fees: float
    
    # Exit tracking
    is_active: bool = True
    exit_executions: List[TradeExecution] = None
    total_usd_received: float = 0.0
    total_fees_paid: float = 0.0
    
    # Performance tracking
    peak_price: float = 0.0
    peak_gain_pct: float = 0.0
    final_pnl_usd: Optional[float] = None
    final_pnl_pct: Optional[float] = None
    
    # Strategy-specific tracking
    tp_levels_hit: List[int] = None
    exit_reason: Optional[str] = None
    hold_duration_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.exit_executions is None:
            self.exit_executions = []
        if self.tp_levels_hit is None:
            self.tp_levels_hit = []

@dataclass
class StrategyConfig:
    """Configuration for each strategy being tested"""
    name: str
    description: str
    
    # Entry rules
    min_alpha_wallets: int
    max_entry_delay_seconds: int
    require_confirmation: bool
    confirmation_timeout_seconds: int
    independence_check: bool
    
    # Execution settings  
    base_slippage_pct: float
    dynamic_slippage: Dict[str, float]  # Time-based slippage ladder
    max_leader_delta_pct: float
    position_size_pct: float
    
    # Exit rules
    stop_loss_pct: float
    buffer_seconds: int
    tp_levels: List[Dict]  # [{"level": 0.20, "sell_pct": 0.40}, ...]
    trailing_stop_pct: float
    max_hold_seconds: int
    
    # Safety settings
    require_sellability: bool
    min_liquidity_usd: float
    bypass_safety_for_s_tier: bool


class StrategyABTester:
    """Main A/B testing framework using existing infrastructure"""
    
    def __init__(self, config_path: str = "config.yml"):
        """Initialize using existing bot infrastructure"""
        print("🚀 Initializing Strategy A/B Tester")
        
        # Load main config using existing loader
        self.config_data = load_config(config_path)
        self.config = type('Config', (), self.config_data)()
        
        # Setup logging using existing system
        self.logger = setup_logging(
            getattr(self.config, 'logging_level', 'INFO'),
            'sims/logs/ab_test.log'
        )
        
        # Initialize existing clients with rotation
        self.moralis = MoralisClient(
            self.config_data.get('moralis_keys', []), 
            self.config_data.get('api_optimization', {})
        )
        
        self.realtime_client = RealtimeClient({
            'bitquery_tokens': self.config_data.get('bitquery_tokens', []),
            'pumpportal': self.config_data.get('pumpportal', {}),
            'realtime_source': self.config_data.get('realtime_source', 'pumpportal')
        })
        
        # Initialize wallet tracker with existing logic
        wallet_config = {
            'beta_prior_alpha': self.config_data.get('beta_prior_alpha', 3),
            'beta_prior_beta': self.config_data.get('beta_prior_beta', 5),
            'alpha_weight_min': self.config_data.get('alpha_weight_min', 0.6),
            'require_one_wallet_pge_55': self.config_data.get('require_one_wallet_pge_55', False),
            'alpha_enhanced': self.config_data.get('alpha_enhanced', {}),
            'api_optimization': self.config_data.get('api_optimization', {}),
            'safety': self.config_data.get('safety', {})
        }
        
        self.wallet_tracker = WalletTracker(
            self.config_data.get('watched_wallets', []), 
            config=wallet_config
        )
        
        # Initialize safety checker
        self.safety_checker = SafetyChecker(self.config_data.get('safety', {}))
        
        # Initialize alpha discovery (optional, not used in current simulation)
        # self.alpha_discovery = ProvenAlphaFinder(bitquery, moralis, database, self.config_data)
        
        # Discord notifications
        discord_webhook = self.config_data.get('notifications', {}).get('discord_webhook_url')
        self.notifier = DiscordNotifier(discord_webhook) if discord_webhook else None
        
        # Market reality modeling
        self.market = MarketReality()
        
        # Strategy configurations
        self.strategies = self.load_strategies()
        
        # Test results storage
        self.test_results = {strategy: [] for strategy in self.strategies.keys()}
        self.alpha_signals = {}  # Track all alpha signals detected
        
        # Metrics tracking
        self.start_time = None
        self.signals_detected = 0
        self.positions_opened = 0
        
    def load_strategies(self) -> Dict[str, StrategyConfig]:
        """Load strategy configurations"""
        strategies = {}
        
        # Current Aggressive Strategy
        strategies['current_aggressive'] = StrategyConfig(
            name='current_aggressive',
            description='Your current single-wallet aggressive strategy',
            min_alpha_wallets=1,
            max_entry_delay_seconds=15,
            require_confirmation=False,
            confirmation_timeout_seconds=0,
            independence_check=False,
            base_slippage_pct=0.03,
            dynamic_slippage={},
            max_leader_delta_pct=float('inf'),
            position_size_pct=0.03,
            stop_loss_pct=-0.35,
            buffer_seconds=60,
            tp_levels=[
                {"level": 0.20, "sell_pct": 0.40},
                {"level": 0.45, "sell_pct": 0.35}, 
                {"level": 0.90, "sell_pct": 0.20}
            ],
            trailing_stop_pct=0.30,
            max_hold_seconds=1800,
            require_sellability=False,
            min_liquidity_usd=5000,
            bypass_safety_for_s_tier=True
        )
        
        # GPT5 Recommended Strategy
        strategies['gpt5_recommended'] = StrategyConfig(
            name='gpt5_recommended',
            description='GPT5s recommended improvements',
            min_alpha_wallets=1,  # But with graduation requirement
            max_entry_delay_seconds=20,
            require_confirmation=True,  # 2-wallet confirmation
            confirmation_timeout_seconds=90,
            independence_check=True,
            base_slippage_pct=0.04,
            dynamic_slippage={
                "0-5": 0.04,
                "5-10": 0.06,
                "10-20": 0.10
            },
            max_leader_delta_pct=0.12,
            position_size_pct=0.008,
            stop_loss_pct=-0.15,
            buffer_seconds=0,
            tp_levels=[
                {"level": 0.20, "sell_pct": 0.25},
                {"level": 1.00, "sell_pct": 0.50}  # Derisk at 2x
            ],
            trailing_stop_pct=0.08,
            max_hold_seconds=2700,
            require_sellability=True,
            min_liquidity_usd=20000,
            bypass_safety_for_s_tier=False
        )
        
        # Hybrid Compromise Strategy
        strategies['hybrid_compromise'] = StrategyConfig(
            name='hybrid_compromise', 
            description='Compromise between current and GPT5',
            min_alpha_wallets=1,
            max_entry_delay_seconds=15,
            require_confirmation=False,  # Single wallet but graduated entry
            confirmation_timeout_seconds=60,
            independence_check=True,
            base_slippage_pct=0.035,
            dynamic_slippage={
                "0-5": 0.035,
                "5-10": 0.05,
                "10-15": 0.07
            },
            max_leader_delta_pct=0.08,
            position_size_pct=0.015,  # Between current and GPT5
            stop_loss_pct=-0.20,  # Between current and GPT5
            buffer_seconds=30,   # Shorter buffer
            tp_levels=[
                {"level": 0.20, "sell_pct": 0.30},
                {"level": 0.50, "sell_pct": 0.30},
                {"level": 1.00, "sell_pct": 0.30}
            ],
            trailing_stop_pct=0.12,
            max_hold_seconds=2100,
            require_sellability=True,
            min_liquidity_usd=12000,
            bypass_safety_for_s_tier=False  # No bypasses
        )
        
        return strategies
    
    async def run_ab_test(self, duration_hours: float = 24.0) -> Dict:
        """Run comprehensive A/B test"""
        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(hours=duration_hours)
        
        self.logger.info(f"🧪 Starting A/B test for {duration_hours} hours")
        self.logger.info(f"📊 Testing strategies: {list(self.strategies.keys())}")
        
        if self.notifier:
            await self.notifier.send_text(
                f"🧪 **Strategy A/B Test Started**\n"
                f"Duration: {duration_hours} hours\n"
                f"Strategies: {', '.join(self.strategies.keys())}\n"
                f"Alpha wallets: {len(self.wallet_tracker.watched_wallets)}"
            )
        
        # Initialize realtime client
        await self.realtime_client.initialize()
        
        # Run monitoring tasks
        tasks = [
            self.monitor_alpha_signals(end_time),
            self.update_positions(end_time),
            self.periodic_reporting(end_time),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("⚠️ Test interrupted by user")
        finally:
            # Generate final results
            results = await self.generate_final_results()
            return results
    
    async def monitor_alpha_signals(self, end_time: datetime):
        """Monitor for alpha wallet signals using existing PumpPortal"""
        self.logger.info("👀 Starting alpha signal monitoring")
        
        if not self.realtime_client.pumpportal_client:
            self.logger.error("❌ PumpPortal client not available")
            return
            
        watched_wallets = list(self.wallet_tracker.watched_wallets)
        self.logger.info(f"Monitoring {len(watched_wallets)} alpha wallets")
        
        async for event in self.realtime_client.pumpportal_client.subscribe_all_events(watched_wallets):
            if datetime.now() > end_time:
                break
                
            if event.get('event_type') == 'trade' and event.get('buyer'):
                await self.process_alpha_signal(event)
    
    async def process_alpha_signal(self, event: Dict):
        """Process alpha wallet buy signal"""
        mint = event.get('mint')
        buyer = event.get('buyer')
        timestamp = event.get('timestamp', time.time())
        price = event.get('price', 0)
        
        if not mint or not buyer or buyer not in self.wallet_tracker.watched_wallets:
            return
            
        signal_time = datetime.fromtimestamp(timestamp)
        self.signals_detected += 1
        
        # Track this alpha signal
        if mint not in self.alpha_signals:
            self.alpha_signals[mint] = {
                'first_signal_time': signal_time,
                'alpha_wallets': [],
                'alpha_prices': {},
                'symbol': event.get('symbol', 'UNKNOWN')
            }
        
        signal_data = self.alpha_signals[mint]
        if buyer not in signal_data['alpha_wallets']:
            signal_data['alpha_wallets'].append(buyer)
            signal_data['alpha_prices'][buyer] = price
            
            self.logger.info(f"📡 Alpha signal: {buyer[:8]}... bought {mint[:8]}... at ${price:.8f}")
            
            # Test each strategy's entry decision
            for strategy_name, strategy_config in self.strategies.items():
                should_enter = await self.evaluate_entry_conditions(
                    mint, signal_data, strategy_config, signal_time
                )
                
                if should_enter:
                    await self.simulate_entry(mint, signal_data, strategy_config, signal_time)
    
    async def evaluate_entry_conditions(self, mint: str, signal_data: Dict, 
                                       strategy: StrategyConfig, signal_time: datetime) -> bool:
        """Evaluate if strategy would enter this trade using existing logic"""
        
        # Check alpha wallet count requirement
        alpha_count = len(signal_data['alpha_wallets'])
        if alpha_count < strategy.min_alpha_wallets:
            return False
            
        # Check entry delay
        delay_seconds = (signal_time - signal_data['first_signal_time']).total_seconds()
        if delay_seconds > strategy.max_entry_delay_seconds:
            return False
        
        # For confirmation strategies, check if we have enough wallets
        if strategy.require_confirmation and alpha_count < 2:
            # Check if we're still in confirmation window
            if delay_seconds < strategy.confirmation_timeout_seconds:
                return False  # Wait for confirmation
            else:
                return False  # Timeout, no confirmation
        
        # Use existing safety checks
        try:
            # Get recent trade data for safety checks
            swaps = await self.moralis.get_token_swaps(mint, limit=50)
            
            # Skip safety checks if enabled for strategy
            if strategy.bypass_safety_for_s_tier:
                # Check if any alpha wallet is S-tier
                s_tier_present = any(
                    self.wallet_tracker.get_wallet_tier(wallet) == 'S'
                    for wallet in signal_data['alpha_wallets']
                )
                if s_tier_present:
                    return True  # Bypass all safety
            
            # Run safety checks using existing infrastructure
            safety_result = self.safety_checker.check_token_safety(
                mint=mint,
                order_size_usd=strategy.position_size_pct * 1000,  # Assume $1000 capital
                recent_trades=swaps,
                max_impact=0.05,
                bypass_sellability=not strategy.require_sellability
            )
            
            if not safety_result['safe']:
                self.logger.debug(f"❌ {strategy.name}: Safety check failed for {mint[:8]}...")
                return False
            
            # Check liquidity requirement
            # This would use existing liquidity checking logic
            # For simulation, we'll use a simplified version
            total_volume = sum(trade.get('amount_usd', 0) for trade in swaps)
            if total_volume < strategy.min_liquidity_usd:
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error evaluating entry for {mint[:8]}...: {e}")
            return False
    
    async def simulate_entry(self, mint: str, signal_data: Dict, 
                           strategy: StrategyConfig, signal_time: datetime):
        """Simulate realistic trade execution"""
        
        # Get alpha wallet entry price (use first wallet's price)
        alpha_entry_price = list(signal_data['alpha_prices'].values())[0]
        
        # Calculate execution delay (realistic latency)
        execution_delay_ms = np.random.normal(self.market.latency_mean_ms, self.market.latency_std_ms)
        execution_delay_ms = max(50, execution_delay_ms)  # Minimum 50ms
        execution_time = signal_time + timedelta(milliseconds=execution_delay_ms)
        
        # Price movement during execution delay
        # Simulate realistic price drift based on volatility
        price_drift_pct = np.random.normal(0, 0.02)  # 2% std dev price movement
        market_price = alpha_entry_price * (1 + price_drift_pct)
        
        # Calculate slippage needed
        time_since_signal = execution_delay_ms / 1000
        if time_since_signal <= 5:
            slippage_pct = strategy.dynamic_slippage.get("0-5", strategy.base_slippage_pct)
        elif time_since_signal <= 10:
            slippage_pct = strategy.dynamic_slippage.get("5-10", strategy.base_slippage_pct * 1.5)
        else:
            slippage_pct = strategy.dynamic_slippage.get("10-20", strategy.base_slippage_pct * 2)
        
        # Check leader price delta guard
        leader_delta = (market_price * (1 + slippage_pct) - alpha_entry_price) / alpha_entry_price
        if leader_delta > strategy.max_leader_delta_pct:
            self.logger.debug(f"❌ {strategy.name}: Leader delta too high for {mint[:8]}... ({leader_delta:.1%})")
            return
        
        # Simulate execution success/failure
        if slippage_pct <= 0.02:
            fail_rate = self.market.fail_rate_2pct
        elif slippage_pct <= 0.05:
            fail_rate = self.market.fail_rate_5pct
        else:
            fail_rate = self.market.fail_rate_10pct
            
        success = np.random.random() > fail_rate
        
        if not success:
            self.logger.debug(f"❌ {strategy.name}: Execution failed for {mint[:8]}... (slippage)")
            return
        
        # Calculate actual execution
        execution_price = market_price * (1 + slippage_pct)
        position_size_usd = strategy.position_size_pct * 1000  # Assume $1000 capital
        tokens_bought = position_size_usd / execution_price
        
        # Calculate fees
        gas_cost = np.random.normal(0.002, 0.0005) * 140  # SOL price assumption
        priority_fee = np.random.normal(0.001, 0.0005) * 140
        total_fees = gas_cost + priority_fee
        
        # Create position
        position = SimulatedPosition(
            mint_address=mint,
            symbol=signal_data['symbol'],
            strategy_name=strategy.name,
            entry_signal_time=signal_time,
            entry_execution=TradeExecution(
                signal_time=signal_time,
                attempted_time=signal_time,
                executed_time=execution_time,
                target_price=market_price,
                executed_price=execution_price,
                slippage_tolerance=slippage_pct,
                actual_slippage=slippage_pct,
                success=True,
                gas_cost_usd=gas_cost,
                priority_fee_usd=priority_fee
            ),
            alpha_wallets=signal_data['alpha_wallets'].copy(),
            alpha_entry_price=alpha_entry_price,
            leader_price_delta=leader_delta,
            tokens_bought=tokens_bought,
            usd_invested=position_size_usd,
            cost_basis_with_fees=position_size_usd + total_fees,
            peak_price=execution_price
        )
        
        # Add to strategy results
        self.test_results[strategy.name].append(position)
        self.positions_opened += 1
        
        self.logger.info(f"✅ {strategy.name}: Opened position in {mint[:8]}... at ${execution_price:.8f} "
                        f"(delta: {leader_delta:.1%}, slippage: {slippage_pct:.1%})")
    
    async def update_positions(self, end_time: datetime):
        """Update all active positions with current prices and handle exits"""
        while datetime.now() < end_time:
            active_positions = []
            
            # Collect all active positions
            for strategy_positions in self.test_results.values():
                active_positions.extend([p for p in strategy_positions if p.is_active])
            
            if active_positions:
                await self.update_position_prices(active_positions)
                await self.check_exit_conditions(active_positions)
            
            await asyncio.sleep(5)  # Update every 5 seconds
    
    async def update_position_prices(self, positions: List[SimulatedPosition]):
        """Update current prices for all positions"""
        # Group by mint for efficient API calls
        mints_to_update = {}
        for position in positions:
            if position.mint_address not in mints_to_update:
                mints_to_update[position.mint_address] = []
            mints_to_update[position.mint_address].append(position)
        
        # Batch price updates
        for mint, mint_positions in mints_to_update.items():
            try:
                current_price = await self.moralis.get_current_price(mint, fresh=True)
                if current_price > 0:
                    for position in mint_positions:
                        # Update peak price
                        if current_price > position.peak_price:
                            position.peak_price = current_price
                            entry_price = position.entry_execution.executed_price
                            position.peak_gain_pct = ((current_price - entry_price) / entry_price) * 100
                        
            except Exception as e:
                self.logger.error(f"❌ Error updating price for {mint[:8]}...: {e}")
    
    async def check_exit_conditions(self, positions: List[SimulatedPosition]):
        """Check exit conditions for all active positions"""
        for position in positions:
            if not position.is_active:
                continue
                
            strategy = self.strategies[position.strategy_name]
            current_time = datetime.now()
            hold_duration = (current_time - position.entry_execution.executed_time).total_seconds()
            
            try:
                current_price = await self.moralis.get_current_price(position.mint_address, fresh=True)
                if current_price <= 0:
                    continue
                    
                entry_price = position.entry_execution.executed_price
                current_gain_pct = ((current_price - entry_price) / entry_price) * 100
                
                exit_reason = None
                sell_percentage = 1.0
                
                # Check stop loss (respect buffer period)
                if strategy.buffer_seconds == 0 or hold_duration > strategy.buffer_seconds:
                    if current_gain_pct <= strategy.stop_loss_pct * 100:
                        exit_reason = "stop_loss"
                
                # Check take profit levels
                if not exit_reason:
                    for tp in strategy.tp_levels:
                        tp_level_pct = tp["level"] * 100
                        if current_gain_pct >= tp_level_pct and tp["level"] not in position.tp_levels_hit:
                            position.tp_levels_hit.append(tp["level"])
                            exit_reason = f"take_profit_{tp['level']}"
                            sell_percentage = tp["sell_pct"]
                            break
                
                # Check max hold time
                if not exit_reason and hold_duration >= strategy.max_hold_seconds:
                    exit_reason = "max_hold_time"
                
                # Check trailing stop (simplified)
                if not exit_reason and position.peak_price > entry_price * 1.1:  # Only after 10% gain
                    trailing_stop_price = position.peak_price * (1 - strategy.trailing_stop_pct)
                    if current_price <= trailing_stop_price:
                        exit_reason = "trailing_stop"
                
                if exit_reason:
                    await self.simulate_exit(position, current_price, exit_reason, sell_percentage)
                    
            except Exception as e:
                self.logger.error(f"❌ Error checking exit for {position.mint_address[:8]}...: {e}")
    
    async def simulate_exit(self, position: SimulatedPosition, exit_price: float, 
                          exit_reason: str, sell_percentage: float = 1.0):
        """Simulate realistic exit execution"""
        
        # Simulate exit slippage (higher for stop losses)
        if exit_reason == "stop_loss":
            exit_slippage = 0.03  # 3% slippage on panic sells
        else:
            exit_slippage = 0.02  # 2% normal exit slippage
        
        # Simulate exit failure (lower rate than entries)
        exit_fail_rate = 0.05  # 5% exit failure rate
        success = np.random.random() > exit_fail_rate
        
        if not success:
            self.logger.debug(f"❌ Exit failed for {position.mint_address[:8]}... (will retry)")
            return
            
        # Calculate actual exit price with slippage
        actual_exit_price = exit_price * (1 - exit_slippage)
        tokens_to_sell = position.tokens_bought * sell_percentage
        usd_received = tokens_to_sell * actual_exit_price
        
        # Calculate fees
        exit_gas = np.random.normal(0.002, 0.0005) * 140
        exit_priority = np.random.normal(0.001, 0.0005) * 140
        exit_fees = exit_gas + exit_priority
        
        net_usd_received = usd_received - exit_fees
        position.total_usd_received += net_usd_received
        position.total_fees_paid += exit_fees
        
        # Update position
        if sell_percentage >= 1.0:
            # Full exit
            position.is_active = False
            position.exit_reason = exit_reason
            position.hold_duration_seconds = (datetime.now() - position.entry_execution.executed_time).total_seconds()
            position.final_pnl_usd = position.total_usd_received - position.cost_basis_with_fees
            position.final_pnl_pct = (position.final_pnl_usd / position.cost_basis_with_fees) * 100
            
            self.logger.info(f"🚪 {position.strategy_name}: Exited {position.mint_address[:8]}... "
                           f"at ${actual_exit_price:.8f} ({exit_reason}) "
                           f"P&L: {position.final_pnl_pct:+.1f}%")
        else:
            # Partial exit
            position.tokens_bought *= (1 - sell_percentage)
            position.usd_invested *= (1 - sell_percentage)
            
            self.logger.info(f"📤 {position.strategy_name}: Partial exit {position.mint_address[:8]}... "
                           f"{sell_percentage:.0%} at ${actual_exit_price:.8f} ({exit_reason})")
    
    async def periodic_reporting(self, end_time: datetime):
        """Send periodic progress reports"""
        report_interval = timedelta(hours=2)
        next_report = datetime.now() + report_interval
        
        while datetime.now() < end_time:
            if datetime.now() >= next_report:
                await self.send_progress_report()
                next_report = datetime.now() + report_interval
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def send_progress_report(self):
        """Send progress report to Discord"""
        if not self.notifier:
            return
            
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        report = f"📊 **A/B Test Progress Report** ({elapsed_hours:.1f}h)\n\n"
        report += f"**Signals Detected:** {self.signals_detected}\n"
        report += f"**Positions Opened:** {self.positions_opened}\n\n"
        
        for strategy_name in self.strategies.keys():
            positions = self.test_results[strategy_name]
            active = len([p for p in positions if p.is_active])
            completed = len([p for p in positions if not p.is_active])
            
            if completed > 0:
                pnls = [p.final_pnl_pct for p in positions if p.final_pnl_pct is not None]
                win_rate = len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0
                avg_pnl = np.mean(pnls) if pnls else 0
            else:
                win_rate = 0
                avg_pnl = 0
            
            report += f"**{strategy_name}:**\n"
            report += f"• Trades: {completed} (Active: {active})\n"
            report += f"• Win Rate: {win_rate:.1f}%\n"
            report += f"• Avg P&L: {avg_pnl:+.1f}%\n\n"
        
        await self.notifier.send_text(report)
    
    async def generate_final_results(self) -> Dict:
        """Generate comprehensive final results"""
        results = {
            'test_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                'signals_detected': self.signals_detected,
                'positions_opened': self.positions_opened,
                'unique_tokens': len(self.alpha_signals)
            },
            'strategies': {}
        }
        
        for strategy_name, positions in self.test_results.items():
            strategy_results = self.analyze_strategy_performance(positions)
            results['strategies'][strategy_name] = strategy_results
        
        # Save detailed results
        results_path = f"sims/results/ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📄 Results saved to {results_path}")
        
        # Send final report
        if self.notifier:
            await self.send_final_report(results)
        
        return results
    
    def analyze_strategy_performance(self, positions: List[SimulatedPosition]) -> Dict:
        """Analyze performance metrics for a strategy"""
        completed_positions = [p for p in positions if not p.is_active]
        
        if not completed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl_pct': 0,
                'total_pnl_usd': 0,
                'fill_rate': 100,  # All simulated fills succeeded
                'median_leader_delta': 0,
                'tp_hit_rates': {},
                'loss_distribution': {},
                'median_hold_time_minutes': 0
            }
        
        # Basic performance
        pnls = [p.final_pnl_pct for p in completed_positions if p.final_pnl_pct is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        total_pnl_usd = sum(p.final_pnl_usd for p in completed_positions if p.final_pnl_usd)
        
        # GPT5's specific metrics
        leader_deltas = [p.leader_price_delta for p in completed_positions]
        median_leader_delta = np.median(leader_deltas) if leader_deltas else 0
        
        # TP hit rates
        tp_hit_rates = {}
        for level in [0.2, 0.45, 0.9, 1.0]:
            hits = len([p for p in completed_positions if level in p.tp_levels_hit])
            tp_hit_rates[f"tp_{level}"] = hits / len(completed_positions) * 100
        
        # Loss distribution
        loss_thresholds = [-0.15, -0.20, -0.25, -0.30, -0.35]
        loss_distribution = {}
        for threshold in loss_thresholds:
            count = len([p for p in pnls if p <= threshold * 100])
            loss_distribution[f"worse_than_{abs(threshold):.0%}"] = count / len(pnls) * 100 if pnls else 0
        
        # Hold times
        hold_times = [p.hold_duration_seconds / 60 for p in completed_positions if p.hold_duration_seconds]
        median_hold_time = np.median(hold_times) if hold_times else 0
        
        # Winner/Loser ratio for break-even calculation
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = abs(np.mean(losers)) if losers else 35  # Default to 35% loss
        winner_loser_ratio = avg_winner / avg_loser if avg_loser > 0 else 0
        required_win_rate = avg_loser / (avg_winner + avg_loser) * 100 if (avg_winner + avg_loser) > 0 else 50
        
        return {
            'total_trades': len(completed_positions),
            'active_trades': len([p for p in positions if p.is_active]),
            'win_rate': win_rate,
            'avg_pnl_pct': avg_pnl,
            'total_pnl_usd': total_pnl_usd,
            'avg_winner_pct': avg_winner,
            'avg_loser_pct': -avg_loser,  # Negative for display
            'winner_loser_ratio': winner_loser_ratio,
            'required_win_rate': required_win_rate,
            'fill_rate': 100,  # All simulated (would be lower in reality)
            'median_leader_delta': median_leader_delta * 100,  # As percentage
            'tp_hit_rates': tp_hit_rates,
            'loss_distribution': loss_distribution,
            'median_hold_time_minutes': median_hold_time,
            'sharpe_ratio': avg_pnl / np.std(pnls) if len(pnls) > 1 else 0,
            'max_drawdown': min(pnls) if pnls else 0
        }
    
    async def send_final_report(self, results: Dict):
        """Send final comprehensive report"""
        if not self.notifier:
            return
            
        summary = results['test_summary']
        
        report = f"🏁 **Final A/B Test Results**\n\n"
        report += f"**Duration:** {summary['duration_hours']:.1f} hours\n"
        report += f"**Signals:** {summary['signals_detected']} alpha signals\n"
        report += f"**Positions:** {summary['positions_opened']} opened\n\n"
        
        # Strategy comparison
        strategies_data = []
        for name, data in results['strategies'].items():
            strategies_data.append((name, data))
        
        # Sort by win rate
        strategies_data.sort(key=lambda x: x[1]['win_rate'], reverse=True)
        
        for name, data in strategies_data:
            report += f"**{name.upper()}:**\n"
            report += f"• Win Rate: {data['win_rate']:.1f}%\n"
            report += f"• Avg P&L: {data['avg_pnl_pct']:+.1f}%\n"
            report += f"• Total P&L: ${data['total_pnl_usd']:+.2f}\n"
            report += f"• Win/Loss Ratio: {data['winner_loser_ratio']:.2f}\n"
            report += f"• Required Win Rate: {data['required_win_rate']:.1f}%\n"
            report += f"• Leader Delta: {data['median_leader_delta']:+.1f}%\n\n"
        
        # The key verdict
        best_strategy = strategies_data[0]
        report += f"🏆 **Winner: {best_strategy[0].upper()}**\n"
        report += f"Achieved {best_strategy[1]['win_rate']:.1f}% win rate\n"
        
        if best_strategy[1]['win_rate'] > 40:
            report += "✅ Above profitable threshold!"
        else:
            report += "⚠️ Still below profitable threshold"
        
        await self.notifier.send_text(report)


# Main execution
async def main():
    """Run the A/B test"""
    try:
        tester = StrategyABTester("config.yml")
        results = await tester.run_ab_test(duration_hours=24)
        print(f"\n🎉 Test completed! Results: {json.dumps(results, indent=2, default=str)}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())